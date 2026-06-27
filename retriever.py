"""
Hybrid Retriever: BM25 + Dense Retrieval with Reciprocal Rank Fusion (RRF)
Implements production-grade document retrieval for RAG pipelines.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Install: pip install rank-bm25 faiss-cpu sentence-transformers")


@dataclass
class RetrievedDoc:
    doc_id: int
    content: str
    metadata: Dict = field(default_factory=dict)
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0


class HybridRetriever:
    """
    Combines BM25 (sparse) + dense retrieval with Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1/(k + rank_i(d))
    where k=60 is a stability constant (Cormack et al. 2009).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        top_k: int = 10,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        logger.info(f"Loading encoder: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self._embeddings: Optional[np.ndarray] = None

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Index documents for retrieval."""
        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]

        logger.info(f"Indexing {len(documents)} documents...")

        # BM25 (sparse)
        tokenized = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # FAISS (dense) — cosine similarity via normalized IP
        self._embeddings = self.encoder.encode(
            documents, show_progress_bar=True, batch_size=32, normalize_embeddings=True
        )
        dim = self._embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self._embeddings.astype(np.float32))
        logger.info(f"Indexed {len(documents)} docs | dim={dim}")

    def add_documents(self, new_docs: List[str], new_meta: Optional[List[Dict]] = None) -> None:
        """Incrementally add documents to an existing index."""
        if not self.documents:
            return self.index(new_docs, new_meta)

        new_meta = new_meta or [{} for _ in new_docs]
        all_docs = self.documents + new_docs
        all_meta = self.metadata + new_meta
        self.index(all_docs, all_meta)  # rebuild (FAISS doesn't support incremental easily)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDoc]:
        """Retrieve top-k documents using hybrid BM25 + dense + RRF."""
        if not self.documents:
            raise RuntimeError("No documents indexed. Call index() first.")

        k = top_k or self.top_k
        candidate_k = min(k * 3, len(self.documents))  # over-retrieve then fuse

        # BM25 ranking
        tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][:candidate_k]

        # Dense ranking
        q_emb = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        dense_scores, dense_top = self.faiss_index.search(q_emb, candidate_k)
        dense_top = dense_top[0]
        dense_scores = dense_scores[0]

        # Build score lookup
        bm25_score_map = {int(idx): float(bm25_scores[idx]) for idx in bm25_top}
        dense_score_map = {int(idx): float(s) for idx, s in zip(dense_top, dense_scores)}

        # RRF fusion
        rrf: Dict[int, float] = {}
        for rank, idx in enumerate(bm25_top):
            rrf[int(idx)] = rrf.get(int(idx), 0) + self.bm25_weight / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(dense_top):
            rrf[int(idx)] = rrf.get(int(idx), 0) + self.dense_weight / (self.rrf_k + rank + 1)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            RetrievedDoc(
                doc_id=idx,
                content=self.documents[idx],
                metadata=self.metadata[idx],
                bm25_score=bm25_score_map.get(idx, 0.0),
                dense_score=dense_score_map.get(idx, 0.0),
                rrf_score=score,
            )
            for idx, score in ranked
        ]

    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Pure dense similarity search (no BM25)."""
        q_emb = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.faiss_index.search(q_emb, top_k)
        return [(self.documents[i], float(s)) for i, s in zip(indices[0], scores[0])]

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization for BM25."""
        return text.lower().split()

    def save(self, path: str) -> None:
        """Persist index to disk."""
        import pickle, os
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.faiss_index, f"{path}/faiss.index")
        with open(f"{path}/meta.pkl", "wb") as f:
            pickle.dump({"documents": self.documents, "metadata": self.metadata}, f)
        logger.info(f"Index saved to {path}/")

    @classmethod
    def load(cls, path: str, model_name: str = "BAAI/bge-base-en-v1.5", **kwargs) -> "HybridRetriever":
        """Load a persisted index."""
        import pickle
        obj = cls(model_name=model_name, **kwargs)
        obj.faiss_index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/meta.pkl", "rb") as f:
            data = pickle.load(f)
        obj.documents = data["documents"]
        obj.metadata = data["metadata"]
        tokens = [cls._tokenize(d) for d in obj.documents]
        obj.bm25 = BM25Okapi(tokens)
        return obj

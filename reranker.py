"""
Cross-Encoder Reranker: re-scores retrieved candidates for higher precision.
Uses a cross-encoder model that jointly encodes query + document.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a cross-encoder model.

    Cross-encoders process (query, doc) pairs jointly — much more
    accurate than bi-encoders but slower (not scalable for first-stage).
    Typical pipeline: BM25/dense retrieve 50 → rerank → return top 5.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded cross-encoder: {self.model_name}")
            except ImportError:
                raise ImportError("pip install sentence-transformers")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = True,
    ) -> List[Tuple[int, str, float]]:
        """
        Rerank documents.

        Returns: List of (original_idx, document, score) sorted by score desc.
        """
        self._load()
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs, batch_size=self.batch_size)

        ranked = sorted(enumerate(zip(documents, scores)), key=lambda x: x[1][1], reverse=True)
        top_k = top_k or len(documents)

        return [(orig_idx, doc, float(score)) for orig_idx, (doc, score) in ranked[:top_k]]

    def rerank_with_threshold(
        self, query: str, documents: List[str], threshold: float = 0.5
    ) -> List[Tuple[int, str, float]]:
        """Rerank and filter by relevance threshold."""
        results = self.rerank(query, documents)
        return [(i, doc, s) for i, doc, s in results if s >= threshold]

    def score_pair(self, query: str, document: str) -> float:
        """Score a single (query, document) pair."""
        self._load()
        return float(self._model.predict([(query, document)])[0])

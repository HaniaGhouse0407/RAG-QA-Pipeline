"""
Embedding utilities: chunking, encoding, and vector management.
"""
from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: Dict
    chunk_id: int = 0
    doc_id: str = ""


class DocumentChunker:
    """
    Splits documents into overlapping chunks for dense retrieval.
    Supports sentence-aware splitting to avoid cutting mid-sentence.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, strategy: str = "sentence"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy  # "sentence" | "word" | "character"

    def chunk(self, text: str, doc_id: str = "", metadata: Optional[Dict] = None) -> List[Chunk]:
        meta = metadata or {}
        if self.strategy == "sentence":
            return self._sentence_chunk(text, doc_id, meta)
        elif self.strategy == "word":
            return self._word_chunk(text, doc_id, meta)
        return self._char_chunk(text, doc_id, meta)

    def _sentence_chunk(self, text: str, doc_id: str, metadata: Dict) -> List[Chunk]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks, current, current_len = [], [], 0
        chunk_id = 0

        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append(Chunk(chunk_text, {**metadata, "sentences": len(current)}, chunk_id, doc_id))
                chunk_id += 1
                # Overlap: keep last N words
                overlap_words = " ".join(current).split()[-self.chunk_overlap:]
                current = [" ".join(overlap_words)]
                current_len = len(overlap_words)
            current.append(sent)
            current_len += sent_len

        if current:
            chunks.append(Chunk(" ".join(current), metadata, chunk_id, doc_id))
        return chunks

    def _word_chunk(self, text: str, doc_id: str, metadata: Dict) -> List[Chunk]:
        words = text.split()
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for i, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start:start + self.chunk_size]
            if chunk_words:
                chunks.append(Chunk(" ".join(chunk_words), metadata, i, doc_id))
        return chunks

    def _char_chunk(self, text: str, doc_id: str, metadata: Dict) -> List[Chunk]:
        step = self.chunk_size - self.chunk_overlap
        return [
            Chunk(text[i:i+self.chunk_size], metadata, j, doc_id)
            for j, i in enumerate(range(0, len(text), step))
            if text[i:i+self.chunk_size].strip()
        ]


class EmbeddingCache:
    """LRU cache for embedding lookups to avoid redundant API calls."""

    def __init__(self, max_size: int = 1000):
        from collections import OrderedDict
        self._cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = self.misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

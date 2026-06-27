"""
Configuration management for RAG pipeline.
Loads from environment variables or config file.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrieverConfig:
    model_name: str = "BAAI/bge-base-en-v1.5"
    top_k: int = 10
    rrf_k: int = 60
    dense_weight: float = 0.6
    bm25_weight: float = 0.4
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunking_strategy: str = "sentence"   # sentence | word | character


@dataclass
class RerankerConfig:
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_after_rerank: int = 5
    score_threshold: float = 0.0


@dataclass
class LLMConfig:
    provider: str = "openai"            # openai | anthropic | local
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    api_key: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")


@dataclass
class EvaluationConfig:
    enabled: bool = False
    framework: str = "ragas"            # ragas | heuristic
    metrics: list = field(default_factory=lambda: [
        "faithfulness", "answer_relevancy", "context_precision"
    ])


@dataclass
class RAGConfig:
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    index_path: str = "./rag_index"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "RAGConfig":
        cfg = cls()
        cfg.retriever.model_name = os.getenv("EMBED_MODEL", cfg.retriever.model_name)
        cfg.retriever.top_k = int(os.getenv("RETRIEVER_TOP_K", cfg.retriever.top_k))
        cfg.llm.model = os.getenv("LLM_MODEL", cfg.llm.model)
        cfg.llm.temperature = float(os.getenv("LLM_TEMPERATURE", cfg.llm.temperature))
        return cfg

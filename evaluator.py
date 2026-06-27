"""
RAG Evaluation using RAGAS metrics.
Measures: faithfulness, answer relevancy, context precision, context recall.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    faithfulness: float = 0.0          # Is answer grounded in context?
    answer_relevancy: float = 0.0      # Does answer address the question?
    context_precision: float = 0.0     # Are retrieved docs relevant?
    context_recall: float = 0.0        # Did we retrieve all needed docs?
    answer_correctness: float = 0.0    # Overall factual correctness

    @property
    def aggregate_score(self) -> float:
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        scores = [self.faithfulness, self.answer_relevancy,
                  self.context_precision, self.context_recall,
                  self.answer_correctness]
        return sum(w * s for w, s in zip(weights, scores))

    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "answer_correctness": round(self.answer_correctness, 4),
            "aggregate": round(self.aggregate_score, 4),
        }


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using RAGAS framework.
    Falls back to lightweight heuristics if ragas not installed.
    """

    def __init__(self, llm=None, embeddings=None):
        self.llm = llm
        self.embeddings = embeddings
        self._ragas_available = self._check_ragas()

    def _check_ragas(self) -> bool:
        try:
            import ragas
            return True
        except ImportError:
            logger.warning("ragas not installed; using heuristic evaluation")
            return False

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> RAGASResult:
        if self._ragas_available:
            return self._eval_ragas(questions, answers, contexts, ground_truths)
        return self._eval_heuristic(questions, answers, contexts, ground_truths)

    def _eval_ragas(self, questions, answers, contexts, ground_truths) -> RAGASResult:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                    context_precision, context_recall)

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)
        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truths:
            metrics.append(context_recall)

        result = evaluate(dataset, metrics=metrics, llm=self.llm, embeddings=self.embeddings)
        df = result.to_pandas()

        return RAGASResult(
            faithfulness=float(df["faithfulness"].mean()),
            answer_relevancy=float(df["answer_relevancy"].mean()),
            context_precision=float(df["context_precision"].mean()),
            context_recall=float(df.get("context_recall", [0]).mean() if "context_recall" in df else 0),
        )

    def _eval_heuristic(self, questions, answers, contexts, ground_truths) -> RAGASResult:
        """Lightweight heuristic scoring when ragas unavailable."""
        from sentence_transformers import SentenceTransformer, util
        encoder = SentenceTransformer("all-MiniLM-L6-v2")

        faithfulness_scores, relevancy_scores, precision_scores = [], [], []

        for q, a, ctx_list in zip(questions, answers, contexts):
            if not ctx_list:
                faithfulness_scores.append(0.0)
                relevancy_scores.append(0.0)
                precision_scores.append(0.0)
                continue

            ctx_combined = " ".join(ctx_list)
            q_emb = encoder.encode(q, convert_to_tensor=True)
            a_emb = encoder.encode(a, convert_to_tensor=True)
            ctx_emb = encoder.encode(ctx_combined, convert_to_tensor=True)

            faithfulness_scores.append(float(util.cos_sim(a_emb, ctx_emb)))
            relevancy_scores.append(float(util.cos_sim(a_emb, q_emb)))

            ctx_embs = encoder.encode(ctx_list, convert_to_tensor=True)
            q_ctx_sims = util.cos_sim(q_emb, ctx_embs)[0]
            precision_scores.append(float(q_ctx_sims.mean()))

        return RAGASResult(
            faithfulness=float(sum(faithfulness_scores) / max(len(faithfulness_scores), 1)),
            answer_relevancy=float(sum(relevancy_scores) / max(len(relevancy_scores), 1)),
            context_precision=float(sum(precision_scores) / max(len(precision_scores), 1)),
        )

    def single_eval(self, question: str, answer: str, context: List[str]) -> RAGASResult:
        return self.evaluate([question], [answer], [context])

"""
RAGAS 0.4.3 metrics evaluation using DeepSeek via instructor-wrapped AsyncOpenAI.
Falls back to DSPy-based evaluation, then sentence-transformer similarity.
"""

import os
import json
import asyncio
import numpy as np


def _get_or_create_loop():
    """Get running event loop or create a new one."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class RAGASEvaluator:
    """Evaluates RAG pipeline using RAGAS 0.4.3 with DeepSeek."""

    def __init__(self):
        self._ragas_llm = None
        self._metrics = {}
        self._init_ragas()

    def _init_ragas(self):
        """Set up RAGAS 0.4.3 metrics with DeepSeek."""
        try:
            import openai
            import instructor
            from ragas.metrics import DiscreteMetric, NumericMetric
            from ragas.llms import LiteLLMStructuredLLM
            from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

            async_client = instructor.from_openai(
                openai.AsyncOpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                )
            )
            self._ragas_llm = LiteLLMStructuredLLM(
                client=async_client,
                model=DEEPSEEK_MODEL,
                provider="deepseek",
            )

            self._metrics = {
                "faithfulness": DiscreteMetric(
                    name="faithfulness",
                    allowed_values=["yes", "no"],
                    prompt=(
                        "Is the response fully supported by the context? "
                        "Answer 'yes' or 'no'.\n"
                        "Context: {context}\nResponse: {response}"
                    ),
                ),
                "answer_relevancy": NumericMetric(
                    name="answer_relevancy",
                    allowed_values=(0.0, 1.0),
                    prompt=(
                        "Score 0.0-1.0: how well does the response answer the question?\n"
                        "Question: {question}\nResponse: {response}"
                    ),
                ),
                "context_precision": NumericMetric(
                    name="context_precision",
                    allowed_values=(0.0, 1.0),
                    prompt=(
                        "Score 0.0-1.0: how much of the context is relevant to "
                        "answering the question?\n"
                        "Question: {question}\nContext: {context}"
                    ),
                ),
            }
            print(f"✓ RAGAS 0.4.3 initialized with metrics: {list(self._metrics.keys())}")
        except Exception as e:
            print(f"⚠️ RAGAS init failed ({e}), will use DSPy fallback")

    def evaluate(self, query: str, answer: str, contexts: list[str]) -> dict:
        """
        Evaluate RAG response.

        Args:
            query: User's question.
            answer: Generated answer from RAG.
            contexts: List of retrieved context documents.

        Returns:
            dict with faithfulness, answer_relevancy, context_precision,
            context_recall (None), overall_rag_score.
        """
        if self._ragas_llm and self._metrics:
            try:
                return _get_or_create_loop().run_until_complete(
                    self._evaluate_async(query, answer, contexts)
                )
            except Exception as e:
                print(f"⚠️ RAGAS evaluation failed ({e}), using DSPy fallback")

        # DSPy fallback
        try:
            import dspy
            if dspy.settings.lm is not None:
                return self._evaluate_with_dspy(query, answer, contexts)
        except Exception:
            pass

        return self._evaluate_with_similarity(query, answer, contexts)

    async def _evaluate_async(self, query: str, answer: str, contexts: list[str]) -> dict:
        """Run all three RAGAS metrics concurrently."""
        context_text = "\n\n".join(contexts) if contexts else ""

        faith_task = self._metrics["faithfulness"].ascore(
            context=context_text, response=answer, llm=self._ragas_llm
        )
        rel_task = self._metrics["answer_relevancy"].ascore(
            question=query, response=answer, llm=self._ragas_llm
        )
        prec_task = self._metrics["context_precision"].ascore(
            question=query, context=context_text, llm=self._ragas_llm
        )

        faith_r, rel_r, prec_r = await asyncio.gather(
            faith_task, rel_task, prec_task, return_exceptions=True
        )

        faithfulness = self._parse_result(faith_r, discrete=True)
        relevancy = self._parse_result(rel_r)
        precision = self._parse_result(prec_r)
        overall = float(np.mean([faithfulness, relevancy, precision]))

        return {
            "faithfulness": max(0.0, min(1.0, faithfulness)),
            "answer_relevancy": max(0.0, min(1.0, relevancy)),
            "context_precision": max(0.0, min(1.0, precision)),
            "context_recall": None,
            "overall_rag_score": max(0.0, min(1.0, overall)),
        }

    @staticmethod
    def _parse_result(result, discrete: bool = False) -> float:
        """Extract float from MetricResult or exception."""
        if isinstance(result, Exception):
            return 0.5
        try:
            value = result.value
            if discrete:
                return 1.0 if str(value).strip().lower() == "yes" else 0.0
            return float(value)
        except Exception:
            return 0.5

    # ── DSPy fallback ────────────────────────────────────────────────────────

    def _evaluate_with_dspy(self, query: str, answer: str, contexts: list[str]) -> dict:
        """Use DSPy signatures (same LM as the pipeline) to score each metric."""
        import dspy
        from crag.signatures import (
            FaithfulnessEvaluator,
            AnswerRelevancyEvaluator,
            ContextPrecisionEvaluator,
        )

        context_text = "\n\n".join(contexts) if contexts else ""

        faithfulness = self._safe_predict(
            dspy.Predict(FaithfulnessEvaluator), "faithfulness_score",
            question=query, answer=answer, context=context_text,
        )
        relevancy = self._safe_predict(
            dspy.Predict(AnswerRelevancyEvaluator), "relevancy_score",
            question=query, answer=answer,
        )
        precision = self._safe_predict(
            dspy.Predict(ContextPrecisionEvaluator), "precision_score",
            question=query, context=context_text,
        )
        overall = float(np.mean([faithfulness, relevancy, precision]))

        return {
            "faithfulness": max(0.0, min(1.0, faithfulness)),
            "answer_relevancy": max(0.0, min(1.0, relevancy)),
            "context_precision": max(0.0, min(1.0, precision)),
            "context_recall": None,
            "overall_rag_score": max(0.0, min(1.0, overall)),
        }

    @staticmethod
    def _safe_predict(predictor, output_field: str, **kwargs) -> float:
        import re
        try:
            result = predictor(**kwargs)
            value = getattr(result, output_field, None)
            if isinstance(value, (int, float)) and not np.isnan(float(value)):
                return float(value)
            if isinstance(value, str):
                m = re.search(r"[-+]?\d*\.?\d+", value)
                if m:
                    return float(m.group())
            for text in vars(result).values():
                if isinstance(text, str):
                    m = re.search(r"\b(0\.\d+|1\.0+|0|1)\b", text)
                    if m:
                        return float(m.group())
            return 0.5
        except Exception:
            return 0.5

    # ── Similarity fallback ──────────────────────────────────────────────────

    def _evaluate_with_similarity(self, query: str, answer: str, contexts: list[str]) -> dict:
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")
            q = model.encode(query, convert_to_tensor=True)
            a = model.encode(answer, convert_to_tensor=True)
            c = model.encode(contexts[0] if contexts else "", convert_to_tensor=True)
            faith = float(util.cos_sim(a, c)[0][0])
            rel   = float(util.cos_sim(q, a)[0][0])
            prec  = float(util.cos_sim(q, c)[0][0])
            overall = float(np.mean([faith, rel, prec]))
            return {
                "faithfulness": max(0.0, min(1.0, faith)),
                "answer_relevancy": max(0.0, min(1.0, rel)),
                "context_precision": max(0.0, min(1.0, prec)),
                "context_recall": None,
                "overall_rag_score": max(0.0, min(1.0, overall)),
            }
        except Exception:
            return {
                "faithfulness": 0.5, "answer_relevancy": 0.5,
                "context_precision": 0.5, "context_recall": None,
                "overall_rag_score": 0.5,
            }


class MetricsLogger:
    """Logs metrics for analysis."""

    def __init__(self, log_path: str = None):
        if log_path is None:
            from config.settings import BASE_DIR
            log_path = os.path.join(BASE_DIR, "logs", "ragas_metrics.json")
        self.log_path = log_path
        self.metrics_history = []

    def log_metrics(self, query: str, metrics: dict) -> None:
        entry = {
            "query": query,
            "metrics": metrics,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        }
        self.metrics_history.append(entry)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)

    def get_summary(self) -> dict:
        if not self.metrics_history:
            return {}
        all_metrics = {
            "faithfulness": [], "answer_relevancy": [],
            "context_precision": [], "overall_rag_score": [],
        }
        for entry in self.metrics_history:
            m = entry["metrics"]
            for k in all_metrics:
                all_metrics[k].append(m.get(k, 0))
        summary = {}
        for name, scores in all_metrics.items():
            if scores:
                summary[f"{name}_mean"] = float(np.mean(scores))
                summary[f"{name}_std"]  = float(np.std(scores))
                summary[f"{name}_min"]  = float(np.min(scores))
                summary[f"{name}_max"]  = float(np.max(scores))
        return summary

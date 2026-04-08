"""
RAGAS metrics evaluation for CRAG pipeline.

Evaluates RAG responses using:
- Faithfulness: Is answer grounded in retrieved context?
- Answer Relevance: Is answer relevant to the query?
- Context Precision: Fraction of retrieved context that is relevant
- Context Recall: Fraction of relevant context that was retrieved
"""

from typing import Optional
import numpy as np
from sentence_transformers import util
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class RAGASEvaluator:
    """Evaluates RAG pipeline using RAGAS-inspired metrics."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.1,
            timeout=60,
        )

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> dict:
        """
        Evaluate RAG response using RAGAS metrics.

        Args:
            query: User's question.
            answer: Generated answer from RAG.
            contexts: List of retrieved context documents.
            ground_truth: Optional ground truth for context recall.

        Returns:
            dict containing all metrics and scores.
        """
        metrics = {}

        # 1. Faithfulness: Answer grounded in context?
        metrics["faithfulness"] = self._evaluate_faithfulness(answer, contexts)

        # 2. Answer Relevance: Answer relevant to query?
        metrics["answer_relevance"] = self._evaluate_answer_relevance(query, answer)

        # 3. Context Precision: Fraction of context relevant to query?
        metrics["context_precision"] = self._evaluate_context_precision(query, contexts)

        # 4. Context Recall: Fraction of relevant context retrieved? (if ground truth available)
        if ground_truth:
            metrics["context_recall"] = self._evaluate_context_recall(
                ground_truth, contexts
            )
        else:
            metrics["context_recall"] = None

        # 5. Overall RAG score (average of available metrics)
        available_scores = [v for k, v in metrics.items() if v is not None and k != "context_recall"]
        metrics["overall_rag_score"] = np.mean(available_scores) if available_scores else 0.0

        return metrics

    def _evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        Faithfulness: Does the answer contain only information grounded in context?

        Returns score 0-1 where 1 = fully grounded, 0 = unsupported claims.
        """
        if not contexts or not answer.strip():
            return 0.0

        context_text = "\n\n".join(contexts)
        prompt = (
            f"You are a fact-checking evaluator. Your job is to determine if the answer "
            f"is grounded in the provided context.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"Instructions:\n"
            f"1. Read the answer carefully.\n"
            f"2. Check if EVERY factual claim in the answer is supported by the context.\n"
            f"3. Mark unsupported or contradicted claims.\n"
            f"4. Give a faithfulness score from 0 to 1, where:\n"
            f"   - 1.0 = All claims are grounded in context\n"
            f"   - 0.5 = Some claims are grounded, some are not\n"
            f"   - 0.0 = No claims are grounded in context\n\n"
            f"Respond with ONLY a single decimal number between 0 and 1."
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            score_str = response.content.strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"Faithfulness evaluation error: {e}")
            return 0.5

    def _evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """
        Answer Relevance: Does the answer address the user's query?

        Returns score 0-1 where 1 = perfectly relevant, 0 = irrelevant.
        """
        if not query.strip() or not answer.strip():
            return 0.0

        prompt = (
            f"You are a relevance evaluator. Determine if the answer addresses the query.\n\n"
            f"QUERY:\n{query}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"Instructions:\n"
            f"1. Does the answer directly address the query?\n"
            f"2. Does it provide the information asked for?\n"
            f"3. Is the answer on-topic and relevant?\n\n"
            f"Give a relevance score from 0 to 1 where:\n"
            f"   - 1.0 = Answer perfectly addresses the query\n"
            f"   - 0.5 = Answer partially addresses the query\n"
            f"   - 0.0 = Answer is completely irrelevant\n\n"
            f"Respond with ONLY a single decimal number between 0 and 1."
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            score_str = response.content.strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"Answer relevance evaluation error: {e}")
            return 0.5

    def _evaluate_context_precision(self, query: str, contexts: list[str]) -> float:
        """
        Context Precision: What fraction of retrieved contexts are relevant to the query?

        Returns score 0-1 where 1 = all contexts relevant, 0 = no contexts relevant.
        """
        if not contexts or not query.strip():
            return 0.0

        relevant_count = 0
        for context in contexts:
            if self._is_context_relevant(query, context):
                relevant_count += 1

        return relevant_count / len(contexts) if contexts else 0.0

    def _evaluate_context_recall(
        self, ground_truth: str, contexts: list[str]
    ) -> float:
        """
        Context Recall: What fraction of relevant information (ground truth) was retrieved?

        Returns score 0-1 where 1 = all relevant info retrieved, 0 = none retrieved.
        """
        if not ground_truth.strip() or not contexts:
            return 0.0

        prompt = (
            f"You are a recall evaluator. Determine what fraction of the ground truth "
            f"information is covered by the retrieved contexts.\n\n"
            f"GROUND TRUTH:\n{ground_truth}\n\n"
            f"RETRIEVED CONTEXTS:\n"
            + "\n---\n".join(contexts)
            + f"\n\nInstructions:\n"
            f"1. Identify key facts/claims in the ground truth.\n"
            f"2. Check which ones appear in the retrieved contexts.\n"
            f"3. Calculate fraction: (facts found / total facts)\n\n"
            f"Give a recall score from 0 to 1 where:\n"
            f"   - 1.0 = All ground truth information is in contexts\n"
            f"   - 0.5 = Half of ground truth is in contexts\n"
            f"   - 0.0 = No ground truth information is in contexts\n\n"
            f"Respond with ONLY a single decimal number between 0 and 1."
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            score_str = response.content.strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"Context recall evaluation error: {e}")
            return 0.5

    def _is_context_relevant(self, query: str, context: str) -> bool:
        """Quick check: Is context relevant to query?"""
        prompt = (
            f"Is this context relevant to the query? Answer YES or NO only.\n\n"
            f"QUERY: {query}\n\n"
            f"CONTEXT: {context}\n\n"
            f"Respond with ONLY 'YES' or 'NO'."
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip().upper()
            return answer.startswith("YES")
        except Exception as e:
            print(f"Context relevance check error: {e}")
            return False


class MetricsLogger:
    """Logs RAGAS metrics for analysis."""

    def __init__(self, log_path: str = "logs/ragas_metrics.json"):
        self.log_path = log_path
        self.metrics_history = []

    def log_metrics(self, query: str, metrics: dict) -> None:
        """Log metrics for a single query."""
        entry = {
            "query": query,
            "metrics": metrics,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        }
        self.metrics_history.append(entry)

    def get_summary(self) -> dict:
        """Get summary statistics of all logged metrics."""
        if not self.metrics_history:
            return {}

        all_metrics = {
            "faithfulness": [],
            "answer_relevance": [],
            "context_precision": [],
            "context_recall": [],
            "overall_rag_score": [],
        }

        for entry in self.metrics_history:
            metrics = entry["metrics"]
            all_metrics["faithfulness"].append(metrics.get("faithfulness", 0))
            all_metrics["answer_relevance"].append(metrics.get("answer_relevance", 0))
            all_metrics["context_precision"].append(metrics.get("context_precision", 0))
            if metrics.get("context_recall") is not None:
                all_metrics["context_recall"].append(metrics["context_recall"])
            all_metrics["overall_rag_score"].append(metrics.get("overall_rag_score", 0))

        summary = {}
        for metric_name, scores in all_metrics.items():
            if scores:
                summary[f"{metric_name}_mean"] = np.mean(scores)
                summary[f"{metric_name}_std"] = np.std(scores)
                summary[f"{metric_name}_min"] = np.min(scores)
                summary[f"{metric_name}_max"] = np.max(scores)

        return summary

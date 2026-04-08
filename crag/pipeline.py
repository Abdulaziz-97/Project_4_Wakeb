import json
from threading import Thread
from tavily import TavilyClient
from config.settings import TAVILY_API_KEY, HIGH_RELEVANCE, LOW_RELEVANCE
from vectorstore.chroma_store import ChromaStore
from crag.modules import Evaluator, Refiner, Rewriter, Generator
from crag.metrics import RAGASEvaluator, MetricsLogger
from agent.logger import get_logger


class CRAGPipeline:
    """
    Corrective RAG pipeline.

    Flow:
    1. Retrieve documents from ChromaDB.
    2. Evaluate relevance of each document.
    3. Based on max relevance score:
       - CORRECT (>0.7): Use retrieved document directly for high-confidence answers.
       - AMBIGUOUS (0.3-0.7): Combine best document with web search results.
       - INCORRECT (<0.3): Discard local retrieval and perform web search via Tavily.
    4. Generate a cited response.
    5. (Optional) Evaluate with RAGAS independently at the end.
    """

    def __init__(self, store: ChromaStore = None, evaluate_metrics: bool = False):
        self.store = store or ChromaStore()
        self.evaluator = Evaluator()
        self.refiner = Refiner()
        self.rewriter = Rewriter()
        self.generator = Generator()
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        self.evaluate_metrics = evaluate_metrics
        self.ragas_evaluator = RAGASEvaluator()
        self.metrics_logger = MetricsLogger()
        self.logger = get_logger()

    def run(self, query: str, evaluate_ragas_async: bool = False) -> dict:
        """
        Run the full CRAG pipeline on a query (FAST).

        Args:
            query: User question/query
            evaluate_ragas_async: If True, run RAGAS evaluation in background thread
            
        Returns:
            dict with keys:
                - answer: Generated answer
                - sources: List of sources
                - action: "correct", "ambiguous", "incorrect", or "web_only"
                - scores: Relevance scores from evaluation
                - ragas_metrics: (Optional) Only if evaluate_metrics=True during __init__
        """
        # Step 1: Retrieve
        retrieved = self.store.query(query)
        if not retrieved:
            return self._web_search_path(query)

        # Step 2: Evaluate
        scores = []
        for doc in retrieved:
            score = self.evaluator(query=query, document=doc["text"])
            scores.append(score)

        if not scores:
            return self._web_search_path(query)

        max_score = max(scores)
        best_idx = scores.index(max_score)
        best_doc = retrieved[best_idx]

        # Step 3: Route
        if max_score > HIGH_RELEVANCE:
            result = self._correct_path(query, best_doc, scores)
        elif max_score < LOW_RELEVANCE:
            result = self._incorrect_path(query, scores)
        else:
            result = self._ambiguous_path(query, best_doc, scores)

        # Step 4: (Optional) Run RAGAS evaluation independently in background
        if evaluate_ragas_async:
            self._evaluate_ragas_async(query, result)

        return result

    def _correct_path(self, query: str, best_doc: dict, scores: list) -> dict:
        """High relevance: use retrieved document."""
        metadata = best_doc.get("metadata", {})
        sources = self._extract_sources(metadata)
        is_ingested = "ingested_at" in metadata

        if is_ingested:
            answer = best_doc["text"]
        else:
            ref_str = f"[1] {best_doc['text']}"
            answer = self.generator(query=query, references=ref_str)

        result = {
            "answer": answer,
            "sources": sources,
            "action": "correct",
            "scores": scores,
            "is_ingested": is_ingested,
            "ingested_query": metadata.get("query", ""),
            "ingested_at": metadata.get("ingested_at", ""),
        }
        
        # Add RAGAS metrics ONLY if inline evaluation enabled
        if self.evaluate_metrics:
            ragas_metrics = self.ragas_evaluator.evaluate(
                query=query,
                answer=answer,
                contexts=[best_doc['text']],
            )
            result["ragas_metrics"] = ragas_metrics
            self.metrics_logger.log_metrics(query, ragas_metrics)
        
        return result

    def _incorrect_path(self, query: str, scores: list) -> dict:
        """Low relevance: web search only."""
        return self._web_search_path(query, scores=scores, action="incorrect")

    def _ambiguous_path(self, query: str, best_doc: dict, scores: list) -> dict:
        """Ambiguous relevance: combine retrieval with web search."""
        refined_local = self.refiner(document=best_doc["text"])
        web_contents, web_sources = self._do_web_search(query)

        local_sources = self._extract_sources(best_doc.get("metadata", {}))
        all_sources = local_sources + web_sources

        refs = [f"[1] {refined_local}"]
        for i, content in enumerate(web_contents, 2):
            refs.append(f"[{i}] {content}")
        ref_str = "\n\n".join(refs)

        answer = self.generator(query=query, references=ref_str)
        
        result = {
            "answer": answer,
            "sources": all_sources,
            "action": "ambiguous",
            "scores": scores,
            "is_ingested": False
        }
        
        if self.evaluate_metrics:
            all_contexts = [refined_local] + web_contents
            ragas_metrics = self.ragas_evaluator.evaluate(
                query=query,
                answer=answer,
                contexts=all_contexts,
            )
            result["ragas_metrics"] = ragas_metrics
            self.metrics_logger.log_metrics(query, ragas_metrics)
        
        return result

    def _web_search_path(self, query: str, scores: list = None, action: str = "web_only") -> dict:
        """Perform web search and generate answer."""
        web_contents, web_sources = self._do_web_search(query)

        refs = []
        for i, content in enumerate(web_contents, 1):
            refs.append(f"[{i}] {content}")
        ref_str = "\n\n".join(refs) if refs else "No references found."

        answer = self.generator(query=query, references=ref_str)
        
        result = {
            "answer": answer,
            "sources": web_sources,
            "action": action,
            "scores": scores or [],
            "is_ingested": False
        }
        
        if self.evaluate_metrics:
            ragas_metrics = self.ragas_evaluator.evaluate(
                query=query,
                answer=answer,
                contexts=web_contents,
            )
            result["ragas_metrics"] = ragas_metrics
            self.metrics_logger.log_metrics(query, ragas_metrics)
        
        return result

    def _do_web_search(self, query: str) -> tuple[list[str], list[str]]:
        """Rewrite query, search via Tavily. Returns (list of content strings, list of source URLs)."""
        rewritten = self.rewriter(query=query)
        try:
            results = self.tavily.search(query=rewritten, max_results=3)
            contents = [r.get("content", "") for r in results.get("results", [])]
            sources = [r.get("url", "") for r in results.get("results", [])]
        except Exception as e:
            self.logger.error(f"[CRAG] Web search failed: {e}")
            contents = [f"Web search failed: {e}"]
            sources = []

        return contents, sources

    # ===== INDEPENDENT RAGAS EVALUATION =====
    
    def _evaluate_ragas_async(self, query: str, result: dict):
        """
        Run RAGAS evaluation in background thread WITHOUT blocking pipeline.
        
        Results stored in result dict with key 'ragas_metrics_async'.
        """
        thread = Thread(
            target=self._ragas_worker,
            args=(query, result),
            daemon=True
        )
        thread.start()
        self.logger.debug("[CRAG] RAGAS evaluation queued in background")

    def _ragas_worker(self, query: str, result: dict):
        """Background worker: evaluate answer with RAGAS and store results."""
        try:
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            
            # Extract context from sources (simplified)
            contexts = [s.replace("[web] ", "").replace("[doc] ", "") for s in sources[:3]]
            contexts = [c for c in contexts if c.strip()]
            
            if not contexts:
                contexts = [answer[:500]]  # Fallback
            
            self.logger.debug(f"[CRAG RAGAS Worker] Evaluating answer for query: {query[:50]}...")
            ragas_metrics = self.ragas_evaluator.evaluate(
                query=query,
                answer=answer,
                contexts=contexts,
            )
            
            # Store results in result dict
            result["ragas_metrics_async"] = ragas_metrics
            self.metrics_logger.log_metrics(query, ragas_metrics)
            
            self.logger.info(
                f"[CRAG RAGAS] Completed: "
                f"faith={ragas_metrics.get('faithfulness', 0):.2f}, "
                f"rel={ragas_metrics.get('answer_relevance', 0):.2f}, "
                f"prec={ragas_metrics.get('context_precision', 0):.2f}, "
                f"score={ragas_metrics.get('overall_rag_score', 0):.2f}"
            )
        except Exception as e:
            self.logger.error(f"[CRAG RAGAS Worker] Failed: {e}")
            result["ragas_metrics_async"] = None

    def evaluate_result_ragas(self, query: str, result: dict) -> dict:
        """
        Synchronous RAGAS evaluation for a CRAG result.
        Call this to get immediate RAGAS metrics (BLOCKING).
        
        Args:
            query: Original query
            result: Result dict from run()
            
        Returns:
            result dict with added 'ragas_metrics' key
        """
        try:
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            contexts = [s.replace("[web] ", "").replace("[doc] ", "") for s in sources[:3]]
            contexts = [c for c in contexts if c.strip()]
            
            if not contexts:
                contexts = [answer[:500]]
            
            self.logger.info(f"[CRAG] Running RAGAS evaluation synchronously...")
            ragas_metrics = self.ragas_evaluator.evaluate(
                query=query,
                answer=answer,
                contexts=contexts,
            )
            
            result["ragas_metrics"] = ragas_metrics
            self.metrics_logger.log_metrics(query, ragas_metrics)
            
            return result
        except Exception as e:
            self.logger.error(f"[CRAG RAGAS] Sync evaluation failed: {e}")
            return result

    @staticmethod
    def _extract_sources(metadata: dict) -> list[str]:
        """Extract source URLs or citations from document metadata."""
        sources = []
        if "source" in metadata:
            sources.append(f"[doc] {metadata['source']}")
        if "url" in metadata:
            sources.append(f"[web] {metadata['url']}")
        return sources
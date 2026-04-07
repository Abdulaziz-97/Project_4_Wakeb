import json
from tavily import TavilyClient
from config.settings import TAVILY_API_KEY, HIGH_RELEVANCE, LOW_RELEVANCE
from vectorstore.chroma_store import ChromaStore
from crag.modules import Evaluator, Refiner, Rewriter, Generator


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
    """

    def __init__(self, store: ChromaStore = None):
        self.store = store or ChromaStore()
        self.evaluator = Evaluator()
        self.refiner = Refiner()
        self.rewriter = Rewriter()
        self.generator = Generator()
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)

    def run(self, query: str) -> dict:
        """
        Run the full CRAG pipeline on a query.

        Returns:
            dict with keys: answer, sources, action, scores
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

        max_score = max(scores)
        best_idx = scores.index(max_score)
        best_doc = retrieved[best_idx]

        # Step 3: Route
        if max_score > HIGH_RELEVANCE:
            return self._correct_path(query, best_doc, scores)
        elif max_score < LOW_RELEVANCE:
            return self._incorrect_path(query, scores)
        else:
            return self._ambiguous_path(query, best_doc, scores)

    def _correct_path(self, query: str, best_doc: dict, scores: list) -> dict:
        """High relevance: use retrieved document."""
        metadata = best_doc.get("metadata", {})
        sources = self._extract_sources(metadata)
        is_ingested = "ingested_at" in metadata
        ref_str = f"[1] {best_doc['text']}"
        answer = self.generator(query=query, references=ref_str)
        return {
            "answer": answer,
            "sources": sources,
            "action": "correct",
            "scores": scores,
            "is_ingested": is_ingested,
            "ingested_query": metadata.get("query", ""),
            "ingested_at": metadata.get("ingested_at", ""),
        }

    def _incorrect_path(self, query: str, scores: list) -> dict:
        """Low relevance: web search only."""
        return self._web_search_path(query, scores=scores, action="incorrect")

    def _ambiguous_path(self, query: str, best_doc: dict, scores: list) -> dict:
        """Ambiguous relevance: combine retrieval with web search."""
        refined_local = self.refiner(document=best_doc["text"])
        web_contents, web_sources = self._do_web_search(query)

        local_sources = self._extract_sources(best_doc["metadata"])
        all_sources = local_sources + web_sources

        # Build numbered references
        refs = [f"[1] {refined_local}"]
        for i, content in enumerate(web_contents, 2):
            refs.append(f"[{i}] {content}")
        ref_str = "\n\n".join(refs)

        answer = self.generator(query=query, references=ref_str)
        return {"answer": answer, "sources": all_sources, "action": "ambiguous", "scores": scores, "is_ingested": False}

    def _web_search_path(self, query: str, scores: list = None, action: str = "web_only") -> dict:
        """Perform web search and generate answer."""
        web_contents, web_sources = self._do_web_search(query)

        # Build numbered references from web results
        refs = []
        for i, content in enumerate(web_contents, 1):
            refs.append(f"[{i}] {content}")
        ref_str = "\n\n".join(refs) if refs else "No references found."

        answer = self.generator(query=query, references=ref_str)
        return {"answer": answer, "sources": web_sources, "action": action, "scores": scores or [], "is_ingested": False}

    def _do_web_search(self, query: str) -> tuple[list[str], list[str]]:
        """Rewrite query, search via Tavily. Returns (list of content strings, list of source URLs)."""
        rewritten = self.rewriter(query=query)
        try:
            results = self.tavily.search(query=rewritten, max_results=3)
            contents = [r.get("content", "") for r in results.get("results", [])]
            sources = [r.get("url", "") for r in results.get("results", [])]
        except Exception as e:
            contents = [f"Web search failed: {e}"]
            sources = []

        return contents, sources

    @staticmethod
    def _extract_sources(metadata: dict) -> list[str]:
        """
        Extract source list from document metadata.

        Handles two cases:
        - Ingested answers: metadata has "sources" key with a JSON-encoded list
        - Original documents: metadata has "source" / "source_type" keys
        """
        # Case 1: Ingested answer — "sources" is a JSON string of a list
        raw_sources = metadata.get("sources")
        if raw_sources and isinstance(raw_sources, str):
            try:
                parsed = json.loads(raw_sources)
                if isinstance(parsed, list) and parsed:
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass

        # Case 2: Original document metadata
        source = metadata.get("source", "unknown")
        source_type = metadata.get("source_type", "")
        page = metadata.get("page", "")
        section = metadata.get("section", "")
        parts = [f"[{source_type}] {source}"]
        if page:
            parts.append(f"page {page}")
        if section:
            parts.append(f"section {section}")
        return [" | ".join(parts)]
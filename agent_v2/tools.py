"""
Tools for the Deep Agent v2 pipeline.

Wraps the existing CRAG pipeline and weather tools as plain functions
compatible with create_deep_agent's tools parameter.
"""

import json
import logging
from datetime import datetime

from langchain_core.tools import tool, ToolException

logger = logging.getLogger("weather_agent_v2")

# ---------------------------------------------------------------------------
#  Tool 1: CRAG retrieval
# ---------------------------------------------------------------------------

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from crag.pipeline import CRAGPipeline
        _pipeline = CRAGPipeline()
    return _pipeline


@tool
def crag_retrieve(query: str) -> str:
    """Retrieve weather information using the Corrective RAG pipeline.

    This searches the local ChromaDB vector store first. Based on relevance:
    - High relevance (>0.7): uses local documents directly
    - Medium (0.3-0.7): combines local + web search
    - Low (<0.3): falls back to web search via Tavily

    Returns the CRAG answer with action, scores, and sources as JSON.
    Always call this FIRST for any weather query.
    """
    pipeline = _get_pipeline()
    try:
        result = pipeline.run(query)
    except Exception as e:
        return json.dumps({
            "error": f"CRAG pipeline failed: {type(e).__name__}: {e}",
            "suggestion": "Use web_search tool directly as fallback.",
        })

    return json.dumps({
        "answer": result["answer"],
        "sources": result["sources"],
        "action": result["action"],
        "scores": result["scores"],
        "max_score": max(result["scores"]) if result["scores"] else 0.0,
    })


# ---------------------------------------------------------------------------
#  Tool 2: Temperature conversion
# ---------------------------------------------------------------------------

@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature to Fahrenheit.

    Use this for EVERY temperature conversion in weather reports.
    Never guess Fahrenheit values -- always call this tool.
    Raises an error if the value is below absolute zero (-273.15C).
    """
    if celsius < -273.15:
        raise ToolException(
            f"{celsius}C is below absolute zero. "
            "Provide a valid temperature above -273.15C."
        )
    return round((celsius * 9 / 5) + 32, 2)


# ---------------------------------------------------------------------------
#  Tool 3: Web search (with smart caching)
# ---------------------------------------------------------------------------

_CACHE_COLLECTION = "web_search_cache"
_CACHE_SIMILARITY_THRESHOLD = 0.25
_CACHE_TTL_HOURS = 6


def _get_cache_store():
    from vectorstore.chroma_store import ChromaStore
    return ChromaStore(
        persist_dir="data/chroma_db",
        collection_name=_CACHE_COLLECTION,
    )


@tool
def web_search(query: str) -> str:
    """Search the web for current weather information using Tavily.

    Checks ChromaDB cache first (6-hour TTL); saves new results for reuse.
    Use this when CRAG data is outdated, insufficient, or when you need
    current/live weather data for a specific location.
    """
    # 1. Check cache
    try:
        store = _get_cache_store()
        if store.count() > 0:
            hits = store.query(query, k=3)
            now = datetime.utcnow()
            fresh = []
            for h in hits:
                if h["distance"] >= _CACHE_SIMILARITY_THRESHOLD:
                    continue
                cached_at = h["metadata"].get("cached_at")
                if cached_at:
                    age = now - datetime.fromisoformat(cached_at)
                    if age.total_seconds() > _CACHE_TTL_HOURS * 3600:
                        continue
                fresh.append(h)
            if fresh:
                output = []
                for i, hit in enumerate(fresh, 1):
                    content = hit["text"]
                    source = hit["metadata"].get("source", "cached")
                    output.append(f"[{i}] {content}\nSource: {source}")
                return "\n\n".join(output)
    except Exception:
        pass

    # 2. Live Tavily search
    from tavily import TavilyClient
    from config.settings import TAVILY_API_KEY

    client = TavilyClient(api_key=TAVILY_API_KEY)
    results = client.search(query=query, max_results=3)
    entries = results.get("results", [])

    # 3. Save to cache
    try:
        store = _get_cache_store()
        chunks = []
        now_str = datetime.utcnow().isoformat()
        for r in entries:
            chunks.append({
                "text": r.get("content", ""),
                "metadata": {
                    "source": r.get("url", ""),
                    "query": query,
                    "cached_at": now_str,
                    "type": "web_search_cache",
                },
            })
        if chunks:
            store.add_documents(chunks)
    except Exception:
        pass

    # 4. Format output
    output = []
    for i, r in enumerate(entries, 1):
        content = r.get("content", "")
        source = r.get("url", "")
        output.append(f"[{i}] {content}\nSource: {source}")

    return "\n\n".join(output) if output else "No results found."


ALL_TOOLS = [crag_retrieve, celsius_to_fahrenheit, web_search]

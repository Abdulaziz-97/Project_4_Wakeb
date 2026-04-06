import logging
from datetime import datetime

from langchain_core.tools import tool, ToolException

logger = logging.getLogger("weather_agent")


@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit. Used for verifying temperature values in weather reports."""
    if celsius < -273.15:
        raise ToolException(
            f"RDEM[celsius_to_fahrenheit]: {celsius}\u00b0C is below absolute zero. "
            "Suggestion: Provide a valid temperature value above -273.15\u00b0C."
        )
    result = (celsius * 9 / 5) + 32
    logger.debug(f"[tool] celsius_to_fahrenheit({celsius}) -> {result}")
    return result


# ---------------------------------------------------------------------------
#  Smart-cached web search
# ---------------------------------------------------------------------------
_CACHE_COLLECTION = "web_search_cache"
_CACHE_SIMILARITY_THRESHOLD = 0.25  # cosine distance; lower = more similar
_CACHE_TTL_HOURS = 6


def _get_cache_store():
    """Lazy-load the cache store to avoid import-time ChromaDB init."""
    from vectorstore.chroma_store import ChromaStore
    return ChromaStore(
        persist_dir="data/chroma_db",
        collection_name=_CACHE_COLLECTION,
    )


def _check_cache(query: str):
    """Return cached results if a sufficiently similar query exists and is fresh."""
    try:
        store = _get_cache_store()
        if store.count() == 0:
            return None
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
                    logger.debug(
                        f"[web_search] Skipping stale cache entry "
                        f"(age={age.total_seconds()/3600:.1f}h, query='{query}')"
                    )
                    continue
            fresh.append(h)
        if fresh:
            logger.info(
                f"[web_search] CACHE HIT for '{query}' "
                f"(best distance={fresh[0]['distance']:.3f}, "
                f"cached={fresh[0]['metadata'].get('cached_at','?')})"
            )
            return fresh
    except Exception as e:
        logger.debug(f"[web_search] Cache lookup failed (non-fatal): {e}")
    return None


def _save_to_cache(query: str, entries: list[dict]):
    """Persist Tavily search results into ChromaDB for future reuse."""
    try:
        store = _get_cache_store()
        chunks = []
        now = datetime.utcnow().isoformat()
        for r in entries:
            chunks.append({
                "text": r.get("content", ""),
                "metadata": {
                    "source": r.get("url", ""),
                    "query": query,
                    "cached_at": now,
                    "type": "web_search_cache",
                },
            })
        if chunks:
            store.add_documents(chunks)
            logger.info(
                f"[web_search] Cached {len(chunks)} results for '{query}'"
            )
    except Exception as e:
        logger.debug(f"[web_search] Cache save failed (non-fatal): {e}")


@tool
def web_search(query: str) -> str:
    """Search the web for current weather information using Tavily.
    Checks ChromaDB cache first; saves new results for future reuse."""

    # 1. Check cache
    cached = _check_cache(query)
    if cached:
        output = []
        for i, hit in enumerate(cached, 1):
            content = hit["text"]
            source = hit["metadata"].get("source", "cached")
            output.append(f"[{i}] {content}\nSource: {source}")
        logger.info(f"[web_search] Returning {len(output)} cached results")
        return "\n\n".join(output)

    # 2. Live Tavily search
    from tavily import TavilyClient
    from config.settings import TAVILY_API_KEY

    logger.info(f"[web_search] Tavily API call: '{query}'")
    client = TavilyClient(api_key=TAVILY_API_KEY)
    results = client.search(query=query, max_results=3)
    entries = results.get("results", [])

    # 3. Save to cache for future reuse
    _save_to_cache(query, entries)

    # 4. Format output
    output = []
    for i, r in enumerate(entries, 1):
        content = r.get("content", "")
        source = r.get("url", "")
        output.append(f"[{i}] {content}\nSource: {source}")

    return "\n\n".join(output) if output else "No results found."

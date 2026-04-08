"""
Answer Ingester — saves generated answers into ChromaDB as documents.
Allows future CRAG retrievals to find previously-generated answers.
Uses background threading to avoid blocking the main pipeline.
"""
import json
import logging
from datetime import datetime
from threading import Thread
from vectorstore.chroma_store import ChromaStore
from agent.state import CRAGOutput

logger = logging.getLogger("weather_agent")


class AnswerIngester:
    """Saves CRAG answers back into the vector store for future retrieval."""

    def __init__(self):
        self.store = ChromaStore()

    def _ingest_answer_sync(
        self, query: str, crag_output: CRAGOutput, node_name: str = "unknown"
    ) -> bool:
        """Internal: Synchronous ingestion (blocks)."""
        try:
            chunks = [
                {
                    "text": crag_output.answer,
                    "metadata": {
                        "sources": json.dumps(crag_output.sources),
                        "query": query,
                        "node": node_name,
                        "action": crag_output.action,
                        "ingested_at": datetime.utcnow().isoformat(),
                    },
                }
            ]
            self.store.add_documents(chunks)
            logger.info(
                f"[answer_ingester] ✓ Ingested answer into vector store "
                f"(node={node_name}, sources={len(crag_output.sources)})"
            )
            return True
        except Exception as e:
            logger.error(f"[answer_ingester] Failed to ingest answer: {e}")
            return False

    def ingest_answer(
        self, query: str, crag_output: CRAGOutput, node_name: str = "unknown", async_mode: bool = True
    ) -> bool:
        """
        Save a generated answer into the vector store.

        Args:
            query: The original user query
            crag_output: The CRAGOutput with answer, sources, action
            node_name: Which node generated this
            async_mode: If True, run in background thread (non-blocking but waited on)

        Returns:
            True if successfully queued/ingested, False otherwise
        """
        if async_mode:
            # Run in background thread — NOT daemon so it finishes even if
            # the main pipeline completes before the write is done.
            thread = Thread(
                target=self._ingest_answer_sync,
                args=(query, crag_output, node_name),
                daemon=False,
                name=f"ingest-{node_name}"
            )
            thread.start()
            logger.debug(f"[answer_ingester] Background ingestion started for {node_name}")
            return True
        else:
            # Synchronous mode (blocks caller)
            return self._ingest_answer_sync(query, crag_output, node_name)


# Global instance
_ingester = AnswerIngester()


def ingest_answer(
    query: str, crag_output: CRAGOutput, node_name: str = "unknown", async_mode: bool = True
) -> bool:
    """Public API to ingest answer into vector store (non-blocking by default)."""
    return _ingester.ingest_answer(query, crag_output, node_name, async_mode=async_mode)
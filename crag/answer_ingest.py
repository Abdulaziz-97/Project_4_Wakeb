"""
Answer Ingester — saves generated answers into ChromaDB as documents.
Allows future CRAG retrievals to find previously-generated answers.
"""
import json
import logging
from datetime import datetime
from vectorstore.chroma_store import ChromaStore
from agent.state import CRAGOutput

logger = logging.getLogger("weather_agent")


class AnswerIngester:
    """Saves CRAG answers back into the vector store for future retrieval."""

    def __init__(self):
        self.store = ChromaStore()

    def ingest_answer(
        self, query: str, crag_output: CRAGOutput, node_name: str = "unknown"
    ) -> bool:
        """
        Save a generated answer into the vector store.

        Args:
            query: The original user query
            crag_output: The CRAGOutput with answer, sources, action
            node_name: Which node generated this

        Returns:
            True if successfully ingested, False otherwise
        """
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


# Global instance
_ingester = AnswerIngester()


def ingest_answer(
    query: str, crag_output: CRAGOutput, node_name: str = "unknown"
) -> bool:
    """Public API to ingest answer into vector store."""
    return _ingester.ingest_answer(query, crag_output, node_name)
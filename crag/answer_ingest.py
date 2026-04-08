\
\
\
\
   
import json
import logging
from datetime import datetime
from threading import Thread

from vectorstore.chroma_store import ChromaStore
from agent.state import CRAGOutput

logger = logging.getLogger("weather_agent")


class AnswerIngester:
                                                                             

    def __init__(self):
        self.store = ChromaStore()

    def _ingest_sync(self, query: str, crag_output: CRAGOutput, node_name: str) -> bool:
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
                f"[answer_ingester] Ingested answer "
                f"(node={node_name}, sources={len(crag_output.sources)})"
            )
            return True
        except Exception as e:
            logger.error(f"[answer_ingester] Failed to ingest answer: {e}")
            return False

    def ingest_answer(
        self,
        query: str,
        crag_output: CRAGOutput,
        node_name: str = "unknown",
        async_mode: bool = True,
    ) -> bool:
                                                                                      
        if async_mode:
            thread = Thread(
                target=self._ingest_sync,
                args=(query, crag_output, node_name),
                daemon=False,
                name=f"ingest-{node_name}",
            )
            thread.start()
            return True
        return self._ingest_sync(query, crag_output, node_name)


_ingester = AnswerIngester()


def ingest_answer(
    query: str, crag_output: CRAGOutput, node_name: str = "unknown", async_mode: bool = True
) -> bool:
                                                             
    return _ingester.ingest_answer(query, crag_output, node_name, async_mode=async_mode)

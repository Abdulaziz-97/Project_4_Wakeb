import chromadb
from sentence_transformers import SentenceTransformer
from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, RETRIEVAL_K

_shared_embedder = None


def _get_embedder():
    global _shared_embedder
    if _shared_embedder is None:
        _shared_embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _shared_embedder


class ChromaStore:
                                                                           

    def __init__(self, persist_dir: str = None, collection_name: str = None):
        persist_dir = persist_dir or CHROMA_PERSIST_DIR
        collection_name = collection_name or CHROMA_COLLECTION
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = _get_embedder()

    def add_documents(self, chunks: list[dict]) -> None:
                                                      
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        start = self.collection.count()
        ids = [f"doc_{i}" for i in range(start, start + len(chunks))]
        embeddings = self.embedder.encode(texts).tolist()
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_text: str, k: int = None) -> list[dict]:
                                                                                               
        k = k or RETRIEVAL_K
        embedding = self.embedder.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return docs

    def count(self) -> int:
        return self.collection.count()

    def get_where(self, where_filter: dict, limit: int = 10000) -> dict:
\
\
\
           
        return self.collection.get(
            where=where_filter,
            limit=limit,
            include=["metadatas", "documents"],
        )

    def delete_where(self, where_filter: dict) -> int:
\
\
\
           
        results = self.collection.get(where=where_filter, include=[])
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
                                                     
        self.client.delete_collection(CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

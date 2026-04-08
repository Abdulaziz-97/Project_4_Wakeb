from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list[dict], chunk_size: int = None, overlap: int = None) -> list[dict]:
\
\
\
\
\
\
\
\
\
\
       
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    chunks = []
    for doc in documents:
        text = doc["text"]
        meta = doc["metadata"]
        start = 0
        idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": {**meta, "chunk_index": idx},
                })
                idx += 1
            start += chunk_size - overlap
    return chunks

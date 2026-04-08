import os
import requests
from pypdf import PdfReader


def load_pdf(source: str, save_dir: str = "data") -> list[dict]:
                                                                                               
    if source.startswith("http"):
        filename = os.path.join(save_dir, source.split("/")[-1])
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(filename):
            resp = requests.get(source, timeout=30)
            resp.raise_for_status()
            with open(filename, "wb") as f:
                f.write(resp.content)
        path = filename
    else:
        path = source

    reader = PdfReader(path)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append({
                "text": text,
                "metadata": {
                    "source": path,
                    "source_type": "pdf",
                    "page": i + 1,
                },
            })
    return documents

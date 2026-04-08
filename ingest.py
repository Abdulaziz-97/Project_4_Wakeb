\
\
\
\
\
   

import sys
import os

                                
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import DOCUMENT_SOURCES
from loaders.pdf_loader import load_pdf
from loaders.markdown_loader import load_markdown
from loaders.web_loader import load_web_page
from loaders.file_loader import load_json_url, load_csv_url
from chunking.chunker import chunk_documents
from vectorstore.chroma_store import ChromaStore


def ingest_all():
    store = ChromaStore()
    total = 0

                  
    print("--- Ingesting PDFs ---")
    for name, url in DOCUMENT_SOURCES["pdf"].items():
        print(f"  Loading: {name}")
        try:
            docs = load_pdf(url)
            chunks = chunk_documents(docs)
            store.add_documents(chunks)
            total += len(chunks)
            print(f"  Added {len(chunks)} chunks from {name}")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

                      
    print("--- Ingesting Markdown ---")
    for name, url in DOCUMENT_SOURCES["markdown"].items():
        print(f"  Loading: {name}")
        try:
            docs = load_markdown(url)
            chunks = chunk_documents(docs)
            store.add_documents(chunks)
            total += len(chunks)
            print(f"  Added {len(chunks)} chunks from {name}")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

                       
    print("--- Ingesting Web Pages ---")
    for name, url in DOCUMENT_SOURCES["web"].items():
        print(f"  Loading: {name}")
        try:
            docs = load_web_page(url)
            chunks = chunk_documents(docs)
            store.add_documents(chunks)
            total += len(chunks)
            print(f"  Added {len(chunks)} chunks from {name}")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

                                                                                 
    print("--- Downloading Data Files ---")
    for name, url in DOCUMENT_SOURCES["files"].items():
        print(f"  Downloading: {name}")
        try:
            if "json" in name:
                load_json_url(url)
            elif "csv" in name:
                load_csv_url(url)
            print(f"  Saved {name}")
        except Exception as e:
            print(f"  Failed to download {name}: {e}")

    print(f"\nDone. Total chunks indexed: {total}")
    print(f"ChromaDB document count: {store.count()}")

                                                                     
    print("\n--- Checking Saudi City Forecast TTL ---")
    try:
        from forecast_manager import check_and_refresh
        refreshed = check_and_refresh()
        if refreshed:
            print("  Saudi city forecasts ingested (7-day TTL).")
        else:
            print("  Saudi city forecasts are still fresh — skipped.")
    except Exception as e:
        print(f"  Failed to refresh forecasts: {e}")

    print(f"Final ChromaDB document count: {store.count()}")


if __name__ == "__main__":
    ingest_all()

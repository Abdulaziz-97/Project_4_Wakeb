import os
import requests


def load_markdown(source: str, save_dir: str = "data") -> list[dict]:
    """Load a Markdown file from URL or local path. Returns list of {text, metadata} dicts per section."""
    if source.startswith("http"):
        filename = os.path.join(save_dir, source.split("/")[-1])
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(filename):
            resp = requests.get(source, timeout=30)
            resp.raise_for_status()
            with open(filename, "w", encoding="utf-8") as f:
                f.write(resp.text)
        with open(filename, "r", encoding="utf-8") as f:
            raw = f.read()
        path = filename
    else:
        path = source
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

    sections = _split_by_headers(raw)
    documents = []
    for i, section in enumerate(sections):
        if section.strip():
            documents.append({
                "text": section.strip(),
                "metadata": {
                    "source": path,
                    "source_type": "markdown",
                    "section": i,
                },
            })
    return documents


def _split_by_headers(text: str) -> list[str]:
    """Split markdown text at ## headers. Keeps header with its content."""
    lines = text.split("\n")
    sections = []
    current = []
    for line in lines:
        if line.startswith("## ") and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections

import requests
from bs4 import BeautifulSoup


def load_web_page(url: str) -> list[dict]:
                                                                                            
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

                                         
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
                                   
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

    if not cleaned:
        return []

    return [{
        "text": cleaned,
        "metadata": {
            "source": url,
            "source_type": "web",
            "title": soup.title.string if soup.title else url,
        },
    }]

import os
from dotenv import load_dotenv

load_dotenv()

# --- Project root (absolute) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- API Keys ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- DeepSeek (OpenAI-compatible) ---
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- ChromaDB ---
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
CHROMA_COLLECTION = "weather_docs"

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- CRAG thresholds ---
HIGH_RELEVANCE = 0.7
LOW_RELEVANCE = 0.3
RETRIEVAL_K = 3

# --- Document sources ---
DOCUMENT_SOURCES = {
    "pdf": {
        "noaa_briefing": "https://www.ncei.noaa.gov/access/monitoring/monthly-report/briefings/20250110.pdf",
    },
    "web": {
        "open_meteo_forecast": "https://open-meteo.com/en/docs",
        "open_meteo_historical": "https://open-meteo.com/en/docs/historical-weather-api",
    },
    "markdown": {},
    "files": {
        "berlin_json": "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current_weather=true&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m",
        "berlin_csv": "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&hourly=temperature_2m&format=csv",
    },
}

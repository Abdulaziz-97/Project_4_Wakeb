import os
from dotenv import load_dotenv

load_dotenv()

                                 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

                  
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

                                      
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

                   
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

                  
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
CHROMA_COLLECTION = "weather_docs"

                  
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

                         
HIGH_RELEVANCE = 0.7
LOW_RELEVANCE = 0.3
RETRIEVAL_K = 3

                      
FORECAST_TTL_DAYS = 7

                                               
SAUDI_CITIES = {
    "Riyadh":             {"lat": 24.6877, "lon": 46.7219},
    "Jeddah":             {"lat": 21.5433, "lon": 39.1728},
    "Mecca":              {"lat": 21.3891, "lon": 39.8579},
    "Medina":             {"lat": 24.5247, "lon": 39.5692},
    "Dammam":             {"lat": 26.3927, "lon": 49.9777},
    "Al-Khobar":          {"lat": 26.2172, "lon": 50.1971},
    "Dhahran":            {"lat": 26.2361, "lon": 50.0393},
    "Tabuk":              {"lat": 28.3998, "lon": 36.5715},
    "Abha":               {"lat": 18.2164, "lon": 42.5053},
    "Taif":               {"lat": 21.2854, "lon": 40.4157},
    "Buraydah":           {"lat": 26.3260, "lon": 43.9750},
    "Khamis Mushait":     {"lat": 18.3059, "lon": 42.7307},
    "Hail":               {"lat": 27.5236, "lon": 41.7022},
    "Al-Ula":             {"lat": 26.6199, "lon": 37.9224},
    "Najran":             {"lat": 17.4924, "lon": 44.1277},
    "Jazan":              {"lat": 16.8892, "lon": 42.5611},
    "Arar":               {"lat": 30.9762, "lon": 41.0183},
    "Al-Baha":            {"lat": 20.0126, "lon": 41.4656},
    "Yanbu":              {"lat": 24.0891, "lon": 38.0618},
    "Al-Hofuf":           {"lat": 25.3839, "lon": 49.5867},
    "Sakaka":             {"lat": 29.9697, "lon": 40.2061},
    "Diriyah":            {"lat": 24.7339, "lon": 46.5752},
    "Al-Kharj":           {"lat": 24.1539, "lon": 47.3085},
    "Ras Tanura":         {"lat": 26.6478, "lon": 50.1632},
    "Dawmat al-Jandal":   {"lat": 29.8167, "lon": 39.8661},
    "Alhada":             {"lat": 21.4297, "lon": 40.3533},
    "Ash Shafa":          {"lat": 21.1297, "lon": 40.3697},
    "Uqair":              {"lat": 25.6331, "lon": 50.2082},
}

                          
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

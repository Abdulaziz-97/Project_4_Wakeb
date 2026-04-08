\
\
\
\
\
\
   

import time
import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger("weather_agent")

                                                                      
WMO_CODES = {
    0:  "Clear sky",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

_DAILY_PARAMS = ",".join([
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "weathercode",
    "uv_index_max",
    "sunrise",
    "sunset",
])


def _wmo_label(code: int) -> str:
    return WMO_CODES.get(code, f"Weather code {code}")


def fetch_city_forecast(
    city_name: str,
    lat: float,
    lon: float,
    forecast_days: int = 7,
    ingested_at_ts: int = None,
    ttl_days: int = 7,
) -> list[dict]:
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
\
       
    if ingested_at_ts is None:
        ingested_at_ts = int(time.time())

    expires_at_ts = ingested_at_ts + ttl_days * 86400

    now_dt = datetime.fromtimestamp(ingested_at_ts, tz=timezone.utc)
    iso_week = f"{now_dt.isocalendar()[0]}-W{now_dt.isocalendar()[1]:02d}"
    ingested_at_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": _DAILY_PARAMS,
        "timezone": "auto",
        "forecast_days": forecast_days,
    }

    try:
        resp = requests.get(_OPEN_METEO_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error(f"[forecast_loader] HTTP error fetching {city_name}: {exc}")
        return []
    except Exception as exc:
        logger.error(f"[forecast_loader] Unexpected error for {city_name}: {exc}")
        return []

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        logger.warning(f"[forecast_loader] No daily data returned for {city_name}")
        return []

    chunks = []
    for idx, date_str in enumerate(dates):
        def _val(key, default="N/A"):
            arr = daily.get(key, [])
            v = arr[idx] if idx < len(arr) else None
            return default if v is None else v

        t_max  = _val("temperature_2m_max")
        t_min  = _val("temperature_2m_min")
        precip = _val("precipitation_sum", 0)
        precip_prob = _val("precipitation_probability_max", 0)
        wind   = _val("wind_speed_10m_max")
        wind_dir = _val("wind_direction_10m_dominant")
        hum_max = _val("relative_humidity_2m_max")
        hum_min = _val("relative_humidity_2m_min")
        wcode  = _val("weathercode", 0)
        uv     = _val("uv_index_max")
        sunrise = _val("sunrise")
        sunset  = _val("sunset")

        condition = _wmo_label(int(wcode) if isinstance(wcode, (int, float)) else 0)

                                                                            
                                                         
        raw_text = (
            f"weather forecast for {city_name.lower()}, saudi arabia on {date_str}:\n"
            f"  condition: {condition.lower()}\n"
            f"  temperature: max {t_max}°c, min {t_min}°c\n"
            f"  precipitation: {precip} mm (probability {precip_prob}%)\n"
            f"  wind: max {wind} km/h, dominant direction {wind_dir}°\n"
            f"  relative humidity: max {hum_max}%, min {hum_min}%\n"
            f"  uv index (max): {uv}\n"
            f"  sunrise: {sunrise}  |  sunset: {sunset}"
        )
        text = " ".join(raw_text.split())

        metadata = {
            "doc_type":       "forecast",
            "city":           city_name,
            "country":        "Saudi Arabia",
            "forecast_date":  date_str,
            "source":         "open-meteo",
            "ingested_at":    ingested_at_iso,
            "ingested_week":  iso_week,
            "ingested_at_ts": ingested_at_ts,
            "expires_at_ts":  expires_at_ts,
            "latitude":       lat,
            "longitude":      lon,
        }

        chunks.append({"text": text, "metadata": metadata})

    logger.info(
        f"[forecast_loader] {city_name}: {len(chunks)} day-chunks fetched "
        f"(expires {datetime.fromtimestamp(expires_at_ts, tz=timezone.utc).strftime('%Y-%m-%d')})"
    )
    return chunks

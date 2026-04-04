import os
import json
import requests
import pandas as pd


def load_json_url(url: str, save_dir: str = "data") -> dict:
    """Download JSON from URL, save locally, return parsed dict."""
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, "berlin_forecast.json")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return data


def load_csv_url(url: str, save_dir: str = "data") -> pd.DataFrame:
    """Download CSV from URL, save locally, return DataFrame."""
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, "berlin_hourly.csv")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(filename, "w") as f:
        f.write(resp.text)
    # Open-Meteo CSV has comment lines starting with #
    lines = [l for l in resp.text.splitlines() if not l.startswith("#")]
    cleaned = "\n".join(lines)
    from io import StringIO
    df = pd.read_csv(StringIO(cleaned))
    return df

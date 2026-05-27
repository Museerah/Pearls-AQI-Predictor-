import os
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str) -> str:
    """Read a required environment variable. Raises clear error if missing."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing environment variable: {key}")
    return value


# --- API Config ---
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "")  

# --- City Config ---
CITY     = get_env("CITY")   # karachi
CITY_LAT = 24.8607           # Karachi latitude
CITY_LON = 67.0011           # Karachi longitude

# --- OpenMeteo URLs ---
WEATHER_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL      = "https://air-quality-api.open-meteo.com/v1/air-quality"

# --- DagHub / MLflow Config ---
DAGSHUB_USERNAME  = get_env("DAGSHUB_USERNAME")
DAGSHUB_REPO      = get_env("DAGSHUB_REPO")

# --- Feature Store Config (local parquet file) ---
FEATURE_STORE_PATH = "data/features.parquet"

# --- Feature columns used by ML models ---
FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "temperature", "humidity", "wind_speed",
    "precipitation", "surface_pressure",
    "pm2_5", "pm10", "no2", "ozone"
]

TARGET_COLUMN = "aqi"
import os
from dotenv import load_dotenv

load_dotenv()

KARACHI_CITY = "karachi"
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

# --- API Config ---
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "")

# --- OpenMeteo URLs ---
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# --- DagHub / MLflow Config ---
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "")

# --- Feature Store Config ---
FEATURE_STORE_PATH = os.getenv("FEATURE_STORE_PATH", "data/features.parquet")

# --- Feature columns used by ML models ---
FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "temperature", "humidity", "wind_speed",
    "precipitation", "surface_pressure",
    "pm2_5", "pm10", "no2", "ozone"
]

TARGET_COLUMN = "aqi"


def get_required_env(key: str) -> str:
    """Read a required environment variable at runtime."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing environment variable: {key}")
    return value


def get_city_config() -> tuple[str, float, float]:
    """Return supported city config. Karachi-only for now."""
    city = os.getenv("CITY", KARACHI_CITY).strip().lower()
    if city != KARACHI_CITY:
        raise ValueError("Only Karachi is currently supported. Set CITY=karachi")
    return city, KARACHI_LAT, KARACHI_LON


def get_mlflow_tracking_uri() -> str:
    username = get_required_env("DAGSHUB_USERNAME")
    repo = get_required_env("DAGSHUB_REPO")
    return f"https://dagshub.com/{username}/{repo}.mlflow"


def get_dagshub_dvc_remote_url() -> str:
    username = get_required_env("DAGSHUB_USERNAME")
    repo = get_required_env("DAGSHUB_REPO")
    return f"https://dagshub.com/{username}/{repo}.dvc"

"""
Feature Pipeline — runs every hour via GitHub Actions.

What it does:
1. Fetches current weather from OpenMeteo
2. Fetches current AQI + pollutants from OpenMeteo Air Quality API
3. Combines into one feature row
4. Appends to local parquet feature store
"""

import os
import requests
import pandas as pd
from datetime import datetime

from src.utils import (
    CITY, CITY_LAT, CITY_LON,
    WEATHER_FORECAST_URL, AIR_QUALITY_URL,
    FEATURE_STORE_PATH
)


def fetch_current_weather() -> dict:
    """Fetch current hour's weather from OpenMeteo."""
    params = {
        "latitude":      CITY_LAT,
        "longitude":     CITY_LON,
        "hourly":        "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,surface_pressure",
        "forecast_days": 1,
        "timezone":      "UTC",
    }

    response = requests.get(WEATHER_FORECAST_URL, params=params, timeout=10)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    now_hour = datetime.utcnow().hour
    return {
        "temperature":      float(hourly["temperature_2m"][now_hour]),
        "humidity":         float(hourly["relative_humidity_2m"][now_hour]),
        "wind_speed":       float(hourly["wind_speed_10m"][now_hour]),
        "precipitation":    float(hourly["precipitation"][now_hour]),
        "surface_pressure": float(hourly["surface_pressure"][now_hour]),
    }


def fetch_current_air_quality() -> dict:
    """Fetch current hour's AQI + pollutants from OpenMeteo."""
    params = {
        "latitude":      CITY_LAT,
        "longitude":     CITY_LON,
        "hourly":        "pm2_5,pm10,nitrogen_dioxide,ozone,us_aqi",
        "forecast_days": 1,
        "timezone":      "UTC",
    }

    response = requests.get(AIR_QUALITY_URL, params=params, timeout=10)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    now_hour = datetime.utcnow().hour
    return {
        "aqi":   float(hourly["us_aqi"][now_hour]           or 0),
        "pm2_5": float(hourly["pm2_5"][now_hour]            or 0),
        "pm10":  float(hourly["pm10"][now_hour]             or 0),
        "no2":   float(hourly["nitrogen_dioxide"][now_hour] or 0),
        "ozone": float(hourly["ozone"][now_hour]            or 0),
    }


def compute_features(weather: dict, air: dict) -> pd.DataFrame:
    """Combine weather + air quality into a single feature row."""
    now = datetime.utcnow()

    row = {
        "city":             CITY,
        "timestamp":        pd.Timestamp(now),
        "hour":             int(now.hour),
        "day":              int(now.day),
        "month":            int(now.month),
        "day_of_week":      int(now.weekday()),
        "aqi":              air["aqi"],
        "pm2_5":            air["pm2_5"],
        "pm10":             air["pm10"],
        "no2":              air["no2"],
        "ozone":            air["ozone"],
        "temperature":      weather["temperature"],
        "humidity":         weather["humidity"],
        "wind_speed":       weather["wind_speed"],
        "precipitation":    weather["precipitation"],
        "surface_pressure": weather["surface_pressure"],
    }

    return pd.DataFrame([row])


def append_to_feature_store(df: pd.DataFrame) -> None:
    """
    Append new row to the parquet feature store.
    If file exists, load it and append. Otherwise create new.
    """
    os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)

    if os.path.exists(FEATURE_STORE_PATH):
        existing = pd.read_parquet(FEATURE_STORE_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
        # Remove duplicates based on timestamp
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    else:
        combined = df

    combined.to_parquet(FEATURE_STORE_PATH, index=False)
    print(f"[OK] Feature store now has {len(combined)} rows.")


def run():
    """Fetch → Compute → Append. Called every hour by GitHub Actions."""
    print(f"[INFO] Running feature pipeline for {CITY}...")

    print("[INFO] Fetching weather...")
    weather = fetch_current_weather()
    print(f"[INFO] Weather: {weather}")

    print("[INFO] Fetching air quality...")
    air = fetch_current_air_quality()
    print(f"[INFO] Air quality: {air}")

    print("[INFO] Computing features...")
    df = compute_features(weather, air)
    print(df)

    print("[INFO] Appending to feature store...")
    append_to_feature_store(df)
    print("[DONE] Feature pipeline completed.")


if __name__ == "__main__":
    run()
"""
Backfill Script — run ONCE to populate feature store with historical data.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import requests

from src.utils import (
    WEATHER_ARCHIVE_URL,
    AIR_QUALITY_URL,
    FEATURE_STORE_PATH,
    get_city_config,
)


def fetch_historical_weather(start_date: str, end_date: str, city_lat: float, city_lon: float) -> pd.DataFrame:
    """Fetch hourly weather data from OpenMeteo Weather Archive."""
    print(f"[INFO] Fetching weather from {start_date} to {end_date}...")

    params = {
        "latitude": city_lat,
        "longitude": city_lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,surface_pressure",
        "timezone": "UTC",
    }

    response = requests.get(WEATHER_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature": pd.array(hourly["temperature_2m"], dtype="Float64"),
            "humidity": pd.array(hourly["relative_humidity_2m"], dtype="Float64"),
            "wind_speed": pd.array(hourly["wind_speed_10m"], dtype="Float64"),
            "precipitation": pd.array(hourly["precipitation"], dtype="Float64"),
            "surface_pressure": pd.array(hourly["surface_pressure"], dtype="Float64"),
        }
    )

    print(f"[INFO] Got {len(df)} hourly weather rows.")
    return df


def fetch_historical_air_quality(start_date: str, end_date: str, city_lat: float, city_lon: float) -> pd.DataFrame:
    """Fetch hourly AQI + pollutants from OpenMeteo Air Quality API."""
    print(f"[INFO] Fetching air quality from {start_date} to {end_date}...")

    params = {
        "latitude": city_lat,
        "longitude": city_lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,us_aqi",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }

    response = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "pm2_5": pd.array(hourly["pm2_5"], dtype="Float64"),
            "pm10": pd.array(hourly["pm10"], dtype="Float64"),
            "no2": pd.array(hourly["nitrogen_dioxide"], dtype="Float64"),
            "ozone": pd.array(hourly["ozone"], dtype="Float64"),
            "aqi": pd.array(hourly["us_aqi"], dtype="Float64"),
        }
    )

    print(f"[INFO] Got {len(df)} hourly air quality rows.")
    return df


def build_features(weather_df: pd.DataFrame, air_df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Merge weather + air quality and add time-based features."""
    df = pd.merge(weather_df, air_df, on="timestamp", how="inner")

    df["city"] = city
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["day"] = df["timestamp"].dt.day.astype(int)
    df["month"] = df["timestamp"].dt.month.astype(int)
    df["day_of_week"] = df["timestamp"].dt.weekday.astype(int)

    df = df.dropna(subset=["aqi"])
    df = df[
        [
            "city",
            "timestamp",
            "hour",
            "day",
            "month",
            "day_of_week",
            "aqi",
            "pm2_5",
            "pm10",
            "no2",
            "ozone",
            "temperature",
            "humidity",
            "wind_speed",
            "precipitation",
            "surface_pressure",
        ]
    ]

    print(f"[INFO] Final dataset: {len(df)} rows.")
    return df


def save_to_feature_store(df: pd.DataFrame) -> None:
    """Save features to parquet file (for DVC tracking)."""
    os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)
    df.to_parquet(FEATURE_STORE_PATH, index=False)
    print(f"[OK] Saved {len(df)} rows to {FEATURE_STORE_PATH}")


def run(days_back: int = 90):
    """Main — fetch historical data and save to feature store."""
    city, city_lat, city_lon = get_city_config()
    print(f"[INFO] Starting backfill — last {days_back} days for {city}...")

    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    weather_df = fetch_historical_weather(start_date, end_date, city_lat, city_lon)
    air_df = fetch_historical_air_quality(start_date, end_date, city_lat, city_lon)
    df = build_features(weather_df, air_df, city)

    print("\n[INFO] Sample data:")
    print(df.head(3))

    save_to_feature_store(df)
    print("[DONE] Backfill completed.")


if __name__ == "__main__":
    run(days_back=90)

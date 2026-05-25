"""
Predict Script — called by the Streamlit web app.

What it does:
1. Loads the 3 best models from MLflow on DagHub
2. Fetches current weather + air quality from OpenMeteo
3. Predicts AQI for next 3 days
4. Returns predictions as a clean DataFrame
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.sklearn
import dagshub
from datetime import datetime, timedelta

from src.utils import (
    CITY, CITY_LAT, CITY_LON,
    WEATHER_FORECAST_URL, AIR_QUALITY_URL,
    DAGSHUB_USERNAME, DAGSHUB_REPO,
    FEATURE_COLUMNS
)


# ── 1. Load Models from MLflow ────────────────────────────────────────────────

def load_model(forecast_day: int):
    """Load best model + scaler for a given forecast day from MLflow."""
    model_dir = f"tmp/models/day{forecast_day}"

    model_info = joblib.load(f"{model_dir}/model_info.pkl")
    scaler     = joblib.load(f"{model_dir}/scaler.pkl")

    if model_info["type"] == "pytorch":
        from src.training_pipeline import AQINet
        model = AQINet(input_size=len(FEATURE_COLUMNS))
        model.load_state_dict(torch.load(f"{model_dir}/model.pt"))
        model.eval()
    else:
        model = joblib.load(f"{model_dir}/model.pkl")

    return model, scaler, model_info


# ── 2. Fetch Current Conditions ───────────────────────────────────────────────

def fetch_current_conditions() -> dict:
    """Fetch current weather + air quality for Karachi."""
    import requests

    now_hour = datetime.utcnow().hour

    # Weather
    weather_params = {
        "latitude":      CITY_LAT,
        "longitude":     CITY_LON,
        "hourly":        "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,surface_pressure",
        "forecast_days": 1,
        "timezone":      "UTC",
    }
    weather_resp = requests.get(WEATHER_FORECAST_URL, params=weather_params, timeout=10)
    weather_resp.raise_for_status()
    weather = weather_resp.json()["hourly"]

    # Air quality
    air_params = {
        "latitude":      CITY_LAT,
        "longitude":     CITY_LON,
        "hourly":        "pm2_5,pm10,nitrogen_dioxide,ozone,us_aqi",
        "forecast_days": 1,
        "timezone":      "UTC",
    }
    air_resp = requests.get(AIR_QUALITY_URL, params=air_params, timeout=10)
    air_resp.raise_for_status()
    air = air_resp.json()["hourly"]

    return {
        "temperature":      float(weather["temperature_2m"][now_hour]),
        "humidity":         float(weather["relative_humidity_2m"][now_hour]),
        "wind_speed":       float(weather["wind_speed_10m"][now_hour]),
        "precipitation":    float(weather["precipitation"][now_hour]),
        "surface_pressure": float(weather["surface_pressure"][now_hour]),
        "pm2_5":            float(air["pm2_5"][now_hour]            or 0),
        "pm10":             float(air["pm10"][now_hour]             or 0),
        "no2":              float(air["nitrogen_dioxide"][now_hour] or 0),
        "ozone":            float(air["ozone"][now_hour]            or 0),
        "aqi":              float(air["us_aqi"][now_hour]           or 0),
    }


# ── 3. Build Input Row ────────────────────────────────────────────────────────

def build_input(conditions: dict) -> pd.DataFrame:
    """Build a single feature row from current conditions."""
    now = datetime.utcnow()

    row = {
        "hour":             int(now.hour),
        "day":              int(now.day),
        "month":            int(now.month),
        "day_of_week":      int(now.weekday()),
        "temperature":      conditions["temperature"],
        "humidity":         conditions["humidity"],
        "wind_speed":       conditions["wind_speed"],
        "precipitation":    conditions["precipitation"],
        "surface_pressure": conditions["surface_pressure"],
        "pm2_5":            conditions["pm2_5"],
        "pm10":             conditions["pm10"],
        "no2":              conditions["no2"],
        "ozone":            conditions["ozone"],
    }

    return pd.DataFrame([row])[FEATURE_COLUMNS]


# ── 4. AQI Category ───────────────────────────────────────────────────────────

def get_aqi_category(aqi: float) -> tuple:
    """
    Return AQI category label and color based on US AQI scale.
    Used by Streamlit dashboard to color code predictions.
    """
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"


# ── 5. Run Predictions ────────────────────────────────────────────────────────

def predict_next_3_days() -> pd.DataFrame:
    """
    Load models, fetch current conditions, predict AQI for next 3 days.
    Returns a clean DataFrame with predictions.
    """
    print("[INFO] Loading current conditions...")
    conditions = fetch_current_conditions()
    X = build_input(conditions)

    predictions = []

    for forecast_day in [1, 2, 3]:
        model, scaler, model_info = load_model(forecast_day)

        # Scale if needed
        X_input = scaler.transform(X) if model_info["needs_scaling"] else X.values

        # Predict
        if model_info["type"] == "pytorch":
            X_tensor = torch.tensor(X_input, dtype=torch.float32)
            with torch.no_grad():
                aqi_pred = float(model(X_tensor).numpy()[0])
        else:
            aqi_pred = float(model.predict(X_input)[0])

        # Clip to valid AQI range
        aqi_pred = max(0, min(500, aqi_pred))

        date        = (datetime.utcnow() + timedelta(days=forecast_day)).strftime("%Y-%m-%d")
        category, color = get_aqi_category(aqi_pred)

        predictions.append({
            "day":      f"Day {forecast_day}",
            "date":     date,
            "aqi":      round(aqi_pred, 1),
            "category": category,
            "color":    color,
        })

        print(f"[INFO] Day {forecast_day} ({date}): AQI = {round(aqi_pred, 1)} — {category}")

    return pd.DataFrame(predictions)


if __name__ == "__main__":
    # Initialize DagHub
    dagshub.init(
        repo_owner=DAGSHUB_USERNAME,
        repo_name=DAGSHUB_REPO,
        mlflow=True
    )
    df = predict_next_3_days()
    print(df)
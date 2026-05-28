"""
Prediction module for Karachi AQI day1/day2/day3 forecasts.
Loads models from MLflow Model Registry on DagsHub.
"""

from datetime import datetime, timedelta
import os

import mlflow
import pandas as pd
import requests
from mlflow.tracking import MlflowClient

from src.utils import AIR_QUALITY_URL, FEATURE_COLUMNS, WEATHER_FORECAST_URL, get_city_config, get_mlflow_tracking_uri


def setup_mlflow() -> None:
    tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


def load_model(forecast_day: int, city: str):
    """Load registered model for a forecast day."""
    model_name = f"aqi_{city}_day{forecast_day}"
    client = MlflowClient()

    alias_uri = f"models:/{model_name}@production"
    try:
        client.get_model_version_by_alias(model_name, "production")
        model_uri = alias_uri
    except Exception:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No registered versions found for {model_name}")
        latest = max(versions, key=lambda version: int(version.version))
        model_uri = f"models:/{model_name}/{latest.version}"

    print(f"[INFO] Loading model: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def fetch_current_conditions(city_lat: float, city_lon: float) -> dict:
    """Fetch current weather + air quality for Karachi."""
    now_hour = datetime.utcnow().hour

    weather_params = {
        "latitude": city_lat,
        "longitude": city_lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,surface_pressure",
        "forecast_days": 1,
        "timezone": "UTC",
    }
    weather_resp = requests.get(WEATHER_FORECAST_URL, params=weather_params, timeout=10)
    weather_resp.raise_for_status()
    weather = weather_resp.json()["hourly"]

    air_params = {
        "latitude": city_lat,
        "longitude": city_lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,us_aqi",
        "forecast_days": 1,
        "timezone": "UTC",
    }
    air_resp = requests.get(AIR_QUALITY_URL, params=air_params, timeout=10)
    air_resp.raise_for_status()
    air = air_resp.json()["hourly"]

    return {
        "temperature": float(weather["temperature_2m"][now_hour]),
        "humidity": float(weather["relative_humidity_2m"][now_hour]),
        "wind_speed": float(weather["wind_speed_10m"][now_hour]),
        "precipitation": float(weather["precipitation"][now_hour]),
        "surface_pressure": float(weather["surface_pressure"][now_hour]),
        "pm2_5": float(air["pm2_5"][now_hour] or 0),
        "pm10": float(air["pm10"][now_hour] or 0),
        "no2": float(air["nitrogen_dioxide"][now_hour] or 0),
        "ozone": float(air["ozone"][now_hour] or 0),
        "aqi": float(air["us_aqi"][now_hour] or 0),
    }


def build_input(conditions: dict) -> pd.DataFrame:
    """Build one feature row from current conditions."""
    now = datetime.utcnow()

    row = {
        "hour": int(now.hour),
        "day": int(now.day),
        "month": int(now.month),
        "day_of_week": int(now.weekday()),
        "temperature": conditions["temperature"],
        "humidity": conditions["humidity"],
        "wind_speed": conditions["wind_speed"],
        "precipitation": conditions["precipitation"],
        "surface_pressure": conditions["surface_pressure"],
        "pm2_5": conditions["pm2_5"],
        "pm10": conditions["pm10"],
        "no2": conditions["no2"],
        "ozone": conditions["ozone"],
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def get_aqi_category(aqi: float) -> tuple[str, str]:
    """Return AQI category label and badge color (US AQI scale)."""
    if aqi <= 50:
        return "Good", "#00e400"
    if aqi <= 100:
        return "Moderate", "#ffff00"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    if aqi <= 200:
        return "Unhealthy", "#ff0000"
    if aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    return "Hazardous", "#7e0023"


def predict_next_3_days() -> pd.DataFrame:
    """Predict AQI for next day1/day2/day3 using latest registry models."""
    city, city_lat, city_lon = get_city_config()
    setup_mlflow()

    conditions = fetch_current_conditions(city_lat, city_lon)
    X = build_input(conditions)

    predictions = []
    for forecast_day in [1, 2, 3]:
        model = load_model(forecast_day, city)
        aqi_pred = float(model.predict(X)[0])
        aqi_pred = max(0.0, min(500.0, aqi_pred))

        date = (datetime.utcnow() + timedelta(days=forecast_day)).strftime("%Y-%m-%d")
        category, color = get_aqi_category(aqi_pred)

        predictions.append(
            {
                "day": f"Day {forecast_day}",
                "date": date,
                "aqi": round(aqi_pred, 1),
                "category": category,
                "color": color,
            }
        )

    return pd.DataFrame(predictions)


if __name__ == "__main__":
    print(predict_next_3_days())

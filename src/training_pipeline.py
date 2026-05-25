"""
Training Pipeline — runs every 24 hours via GitHub Actions.

What it does:
1. Loads features from local parquet feature store
2. For each forecast day (1, 2, 3), trains 5 models:
   - Random Forest
   - Ridge Regression
   - Gradient Boosting
   - XGBoost
   - PyTorch Neural Network
3. Best model per day (lowest RMSE) gets saved to MLflow on DagHub
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.utils import (
    DAGSHUB_USERNAME, DAGSHUB_REPO,
    FEATURE_STORE_PATH, FEATURE_COLUMNS, TARGET_COLUMN
)


# ── PyTorch Model ─────────────────────────────────────────────────────────────

class AQINet(nn.Module):
    """
    Small feedforward neural network for AQI regression.
    3 layers with ReLU + Dropout for regularization.
    """
    def __init__(self, input_size: int):
        super(AQINet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ── 1. Load Features ──────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """Load features from local parquet feature store."""
    if not os.path.exists(FEATURE_STORE_PATH):
        raise FileNotFoundError(f"Feature store not found at {FEATURE_STORE_PATH}. Run backfill first.")

    df = pd.read_parquet(FEATURE_STORE_PATH)
    print(f"[INFO] Loaded {len(df)} rows from feature store.")
    return df


# ── 2. Prepare Data ───────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, forecast_day: int):
    """
    Prepare features and target for a given forecast day.
    Shifts AQI by forecast_day * 24 hours so model learns
    to predict future AQI from current conditions.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    shift = forecast_day * 24
    df["target"] = df[TARGET_COLUMN].shift(-shift)
    df = df.dropna(subset=["target"] + FEATURE_COLUMNS)

    X = df[FEATURE_COLUMNS].astype(float)
    y = df["target"].astype(float)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ── 3. Evaluate Model ─────────────────────────────────────────────────────────

def evaluate(name: str, y_test, y_pred) -> dict:
    """Compute RMSE, MAE, R² and print results."""
    metrics = {
        "model": name,
        "rmse":  round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "mae":   round(float(mean_absolute_error(y_test, y_pred)), 2),
        "r2":    round(float(r2_score(y_test, y_pred)), 4),
    }
    print(f"  [{name}] RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | R²: {metrics['r2']}")
    return metrics


# ── 4. Train PyTorch ──────────────────────────────────────────────────────────

def train_pytorch(X_train, y_train, X_test, y_test, input_size: int):
    """Train small neural network, return model + predictions."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    model     = AQINet(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(50):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()

    return model, y_pred


# ── 5. Train All 5 Models ─────────────────────────────────────────────────────

def train_for_day(df: pd.DataFrame, forecast_day: int):
    """
    Train 5 models for a given forecast day.
    Returns best model, scaler, metrics, model name.
    """
    print(f"\n[INFO] Training models for Day {forecast_day} forecast...")

    X_train, X_test, y_train, y_test = prepare_data(df, forecast_day)

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["RandomForest"] = (rf, evaluate("RandomForest", y_test, rf.predict(X_test)), False)

    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    results["Ridge"] = (ridge, evaluate("Ridge", y_test, ridge.predict(X_test_scaled)), True)

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    results["GradientBoosting"] = (gb, evaluate("GradientBoosting", y_test, gb.predict(X_test)), False)

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    results["XGBoost"] = (xgb, evaluate("XGBoost", y_test, xgb.predict(X_test)), False)

    # PyTorch
    pt_model, pt_pred = train_pytorch(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        input_size=len(FEATURE_COLUMNS)
    )
    results["PyTorch"] = (pt_model, evaluate("PyTorch", y_test, pt_pred), True)

    # Pick best by RMSE
    best_name = min(results, key=lambda k: results[k][1]["rmse"])
    best_model, best_metrics, needs_scaling = results[best_name]

    print(f"  → Best for Day {forecast_day}: {best_name} (RMSE: {best_metrics['rmse']})")
    return best_model, scaler, best_metrics, best_name, needs_scaling


# ── 6. Save to MLflow on DagHub ───────────────────────────────────────────────

def save_model(model, scaler, metrics: dict, forecast_day: int, model_name: str, needs_scaling: bool):
    """Log model + metrics to MLflow tracking on DagHub."""
    save_dir = f"tmp/models/day{forecast_day}"
    os.makedirs(save_dir, exist_ok=True)

    # Save scaler
    scaler_path = f"{save_dir}/scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Save model info
    info = {"type": "pytorch" if model_name == "PyTorch" else "sklearn",
            "needs_scaling": needs_scaling}
    joblib.dump(info, f"{save_dir}/model_info.pkl")

    with mlflow.start_run(run_name=f"day{forecast_day}_{model_name}"):
        # Log metrics
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae",  metrics["mae"])
        mlflow.log_metric("r2",   metrics["r2"])
        mlflow.log_param("model_type",    model_name)
        mlflow.log_param("forecast_day",  forecast_day)

        # Save model
        if model_name == "PyTorch":
            torch.save(model.state_dict(), f"{save_dir}/model.pt")
            mlflow.log_artifact(f"{save_dir}/model.pt")
        else:
            joblib.dump(model, f"{save_dir}/model.pkl")
            mlflow.sklearn.log_model(model, f"model_day{forecast_day}")

        mlflow.log_artifact(scaler_path)

    print(f"[OK] Day {forecast_day} model ({model_name}) saved to MLflow.")


# ── 7. Main ───────────────────────────────────────────────────────────────────

def run():
    """Load data → train 5 models x 3 days → save best 3 to MLflow."""
    print("[INFO] Starting training pipeline...")

    # Set DagHub token for CI/CD — must be set before dagshub.init
    dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
    if dagshub_token:
        os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
        mlflow.set_tracking_uri(
            f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('DAGSHUB_REPO')}.mlflow"
        )
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    else:
        dagshub.init(
            repo_owner=DAGSHUB_USERNAME,
            repo_name=DAGSHUB_REPO,
            mlflow=True
        )

    df = load_features()

    for forecast_day in [1, 2, 3]:
        model, scaler, metrics, model_name, needs_scaling = train_for_day(df, forecast_day)
        save_model(model, scaler, metrics, forecast_day, model_name, needs_scaling)

    print("\n[DONE] Training pipeline completed. 3 best models saved to DagHub.")


if __name__ == "__main__":
    run()
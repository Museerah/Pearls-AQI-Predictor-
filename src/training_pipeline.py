"""
Training Pipeline — runs every 24 hours via GitHub Actions.

- Pulls features from DVC-managed parquet
- Trains multiple regressors for day1/day2/day3
- Logs metrics to MLflow on DagsHub
- Registers best model for each horizon in MLflow Model Registry
"""

import os
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.tracking import MlflowClient

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from src.utils import (
    FEATURE_COLUMNS,
    FEATURE_STORE_PATH,
    TARGET_COLUMN,
    get_city_config,
    get_mlflow_tracking_uri,
)


def setup_mlflow() -> None:
    """Set MLflow tracking/registry URI."""

    tracking_uri = get_mlflow_tracking_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    dagshub_token = os.getenv("DAGSHUB_TOKEN", "")

    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
            "DAGSHUB_USERNAME",
            "",
        )

        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


def load_features() -> pd.DataFrame:
    """Load features from parquet feature store."""

    if not os.path.exists(FEATURE_STORE_PATH):
        raise FileNotFoundError(
            f"Feature store not found at {FEATURE_STORE_PATH}. "
            f"Run feature pipeline/backfill first."
        )

    df = pd.read_parquet(FEATURE_STORE_PATH)

    print(df.head())
    print(df.shape)
    print(df.columns.tolist())


    if len(df) < 100:
        print(
            f"[WARNING] Dataset too small ({len(df)} rows). "
            f"Skipping training run."
        )
        return pd.DataFrame()

    return df


def prepare_data(
    df: pd.DataFrame,
    forecast_day: int,
):

    df = df.sort_values("timestamp").reset_index(drop=True)

    shift = forecast_day * 24

    df["target"] = df[TARGET_COLUMN].shift(-shift)

    df = df.dropna(
        subset=["target"] + FEATURE_COLUMNS
    )

    if len(df) < 50:
        raise ValueError(
            f"Not enough data: {len(df)} rows. "
            f"Need at least 50."
        )

    X = df[FEATURE_COLUMNS].astype(float)

    y = df["target"].astype(float)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


def evaluate(
    name: str,
    y_test,
    y_pred,
) -> dict:
    """Compute evaluation metrics."""

    metrics = {
        "model": name,
        "rmse": round(
            float(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        y_pred,
                    )
                )
            ),
            4,
        ),
        "mae": round(
            float(
                mean_absolute_error(
                    y_test,
                    y_pred,
                )
            ),
            4,
        ),
        "r2": round(
            float(
                r2_score(
                    y_test,
                    y_pred,
                )
            ),
            6,
        ),
    }

    print(
        f"  [{name}] "
        f"RMSE={metrics['rmse']} "
        f"MAE={metrics['mae']} "
        f"R2={metrics['r2']}"
    )

    return metrics


def train_for_day(
    df: pd.DataFrame,
    forecast_day: int,
):
    """Train candidate models."""

    print(
        f"\n[INFO] Training models for Day {forecast_day}..."
    )

    X_train, X_test, y_train, y_test = prepare_data(
        df,
        forecast_day,
    )

    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            random_state=42,
        ),

        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
    }

    results = {}

    for model_name, model in candidates.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        results[model_name] = (
            model,
            evaluate(
                model_name,
                y_test,
                preds,
            ),
        )

    best_name = min(
        results,
        key=lambda key: results[key][1]["rmse"],
    )

    best_model, best_metrics = results[best_name]

    print(
        f"  -> Best model for Day {forecast_day}: "
        f"{best_name}"
    )

    return best_name, best_model, best_metrics


def save_model(
    best_model,
    metrics: dict,
    forecast_day: int,
    best_name: str,
    city: str,
    feature_rows: int,
) -> None:
    """Log run and register model."""

    run_stamp = datetime.utcnow().strftime(
        "%Y-%m-%d_%H%M"
    )

    train_date = datetime.utcnow().strftime(
        "%Y-%m-%d"
    )

    registry_name = (
        f"aqi_{city}_day{forecast_day}"
    )

    with mlflow.start_run(
        run_name=(
            f"{city}_day{forecast_day}_"
            f"{best_name}_{run_stamp}"
        )
    ) as run:

        mlflow.log_params(
            {
                "city": city,
                "forecast_day": forecast_day,
                "training_date": train_date,
                "winner_model": best_name,
                "feature_rows": feature_rows,
            }
        )

        mlflow.log_metrics(
            {
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )

        
        info = mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            registered_model_name=registry_name,
        )

        print(
            f"[INFO] Registered model URI: "
            f"{info.model_uri}"
        )

        client = MlflowClient()

        versions = client.search_model_versions(
            f"name='{registry_name}'"
        )

        if versions:

            latest = max(
                versions,
                key=lambda version: int(
                    version.version
                ),
            )

            client.set_model_version_tag(
                registry_name,
                latest.version,
                "city",
                city,
            )

            client.set_model_version_tag(
                registry_name,
                latest.version,
                "forecast_day",
                str(forecast_day),
            )

            client.set_model_version_tag(
                registry_name,
                latest.version,
                "training_date",
                train_date,
            )

            client.set_model_version_tag(
                registry_name,
                latest.version,
                "run_id",
                run.info.run_id,
            )

            client.set_registered_model_alias(
                registry_name,
                "production",
                latest.version,
            )

    print(
        f"[OK] Day {forecast_day} "
        f"registered as "
        f"{registry_name}@production"
    )


def run() -> None:
    """Main training runner."""

    city, _, _ = get_city_config()

    setup_mlflow()

    df = load_features()

    # ---------- FIX ----------
    # Prevent crash if dataset is too small
    if df.empty:
        print(
            "[INFO] Training skipped because "
            "dataset is too small."
        )
        return

    for forecast_day in [1, 2, 3]:

        best_name, best_model, best_metrics = (
            train_for_day(
                df,
                forecast_day,
            )
        )

        save_model(
            best_model,
            best_metrics,
            forecast_day,
            best_name,
            city,
            len(df),
        )

    print(
        "\n[DONE] Training pipeline completed."
    )


if __name__ == "__main__":
    run()
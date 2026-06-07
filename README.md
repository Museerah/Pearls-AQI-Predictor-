# Pearls AQI Predictor — Karachi

> **A fully serverless, end-to-end MLOps system that forecasts Air Quality Index (AQI) for Karachi, Pakistan across a 3-day horizon powered by Open-Meteo APIs, DVC, MLflow on DagsHub, and a live Gradio UI on Hugging Face Spaces.**

**Live App:** [https://huggingface.co/spaces/MuseerahFatima/Karachi-aqi-predictor](https://huggingface.co/spaces/MuseerahFatima/Karachi-aqi-predictor)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipelines](#pipelines)
  - [Hourly Feature Pipeline](#1-hourly-feature-pipeline)
  - [Daily Training Pipeline](#2-daily-training-pipeline)
  - [Inference UI](#3-inference-ui--hugging-face-spaces)
- [Dataset & Features](#dataset--features)
- [ML Models](#ml-models)
- [Setup & Local Development](#setup--local-development)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Limitations](#limitations)

---

## Overview

The **Pearls AQI Predictor** forecasts the US Air Quality Index (AQI) for **Karachi, Pakistan** for the next **Day 1**, **Day 2**, and **Day 3**. It is built as a fully serverless system — no dedicated server or database is required. All orchestration runs through **GitHub Actions**, data versioning through **DVC** on **DagsHub**, and model registry through **MLflow** on **DagsHub**. The Gradio front-end is deployed on **Hugging Face Spaces** and loads production models at startup.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          GitHub Actions (CI/CD)                         │
│                                                                         │
│  ┌──────────────────────┐         ┌────────────────────────────────┐   │
│  │  Hourly Feature      │         │    Daily Training Pipeline     │   │
│  │  Pipeline            │         │                                │   │
│  │  (every hour)        │         │  (daily @ 02:00 UTC)           │   │
│  │                      │         │                                │   │
│  │  Open-Meteo Weather  │         │  DVC pull features.parquet     │   │
│  │  Open-Meteo Air      │         │  Train RF / GBR / Ridge        │   │
│  │  Quality APIs        │         │  Log metrics → MLflow          │   │
│  │        ↓             │         │  Register best model           │   │
│  │  Append features     │         │  Set @production alias         │   │
│  │  to parquet          │         └────────────────────────────────┘   │
│  │        ↓             │                                               │
│  │  DVC add + push      │                                               │
│  │  (DagsHub Storage)   │                                               │
│  │        ↓             │                                               │
│  │  Git commit metadata │                                               │
│  └──────────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌────────────────────────────────┐
                    │   DagsHub (MLflow + DVC)        │
                    │   - features.parquet (DVC)      │
                    │   - Model Registry (MLflow)      │
                    │     aqi_karachi_day1 @production│
                    │     aqi_karachi_day2 @production│
                    │     aqi_karachi_day3 @production│
                    └────────────────────────────────┘
                                      │
                                      ▼
                    ┌────────────────────────────────┐
                    │  Hugging Face Spaces (Gradio)   │
                    │  app.py                         │
                    │  - Load model from registry     │
                    │  - Fetch live Karachi conditions│
                    │  - Render 3-day AQI forecast    │
                    │  - Health gauge + trend chart   │
                    └────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Data Source** | [Open-Meteo Weather Forecast API](https://open-meteo.com/) · [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) |
| **Feature Storage** | Apache Parquet + [DVC](https://dvc.org/) |
| **Remote Storage** | [DagsHub Storage](https://dagshub.com/) (DVC remote) |
| **ML Tracking** | [MLflow](https://mlflow.org/) on [DagsHub](https://dagshub.com/) |
| **Model Registry** | MLflow Model Registry (DagsHub) |
| **ML Models** | Scikit-learn (RandomForest, GradientBoosting, Ridge) |
| **Orchestration** | [GitHub Actions](https://github.com/features/actions) (cron-based) |
| **Inference UI** | [Gradio](https://gradio.app/) + [Plotly](https://plotly.com/) |
| **Hosting** | [Hugging Face Spaces](https://huggingface.co/spaces) |
| **Language** | Python 3.11 |

---

## Project Structure

```
aqi-predictor/
│
├── .dvc/                        # DVC configuration
│   └── config                   # Remote: DagsHub DVC storage
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml # Hourly feature ingestion
│       ├── training_pipeline.yml# Daily model training
│       └── smoke_tests.yml      # PR / push tests
│
├── data/
│   └── features.parquet.dvc     # DVC pointer — actual data on DagsHub
│
├── notebooks/
│   └── EDA.ipynb                # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Shared config (URLs, feature cols, env vars)
│   ├── backfill.py              # One-time historical data loader (90-day default)
│   ├── feature_pipeline.py      # Hourly: fetch → compute → append
│   ├── training_pipeline.py     # Daily: load → train → log → register
│   └── predict.py               # Inference: load model → predict → return DataFrame
│
├── tests/
│   └── test_runtime_imports.py  # Smoke tests: import safety + city validation
│
├── app.py                       # Gradio UI entry point (Hugging Face Spaces)
├── requirements.txt             # App + inference dependencies
├── requirements-pipeline.txt    # Pipeline (training / feature) dependencies
├── .env.example                 # Environment variable template
└── README.md
```

---

## Pipelines

### 1. Hourly Feature Pipeline

**File:** `src/feature_pipeline.py`  
**Trigger:** GitHub Actions cron `0 * * * *` (every hour)

**What it does:**

1. Fetches the current hour's weather from the Open-Meteo Forecast API (temperature, humidity, wind speed, precipitation, surface pressure).
2. Fetches the current hour's air quality from the Open-Meteo Air Quality API (PM2.5, PM10, NO₂, Ozone, US AQI).
3. Combines into a single feature row with time-based features (hour, day, month, day-of-week).
4. Appends the row to `data/features.parquet` (deduplicating on timestamp).
5. Runs `dvc add data/features.parquet` and `dvc push` to upload the updated parquet to DagsHub storage.
6. Commits and pushes the lightweight `.dvc` metadata file back to the Git repository.

**Bootstrap (one-time backfill):**

```bash
python -m src.backfill          # defaults to 90 days
```

---

### 2. Daily Training Pipeline

**File:** `src/training_pipeline.py`  
**Trigger:** GitHub Actions cron `0 2 * * *` (02:00 UTC daily)

**What it does:**

1. Pulls the latest `features.parquet` from DagsHub via DVC.
2. For each forecast horizon (**Day 1**, **Day 2**, **Day 3**):
   - Shifts the AQI target by `N × 24` hours to create the label.
   - Splits 80/20 train/test.
   - Trains three candidate models: `RandomForestRegressor`, `GradientBoostingRegressor`, and a `Ridge` pipeline with `StandardScaler`.
   - Selects the model with the lowest RMSE on the test split.
   - Logs parameters + metrics (RMSE, MAE, R²) to MLflow on DagsHub.
   - Registers the best model as `aqi_karachi_day{N}` and sets the `@production` alias.

**Model naming convention:**

| Model Name | Description |
|---|---|
| `aqi_karachi_day1` | Next-day AQI forecast |
| `aqi_karachi_day2` | 2-day AQI forecast |
| `aqi_karachi_day3` | 3-day AQI forecast |

---

### 3. Inference UI — Hugging Face Spaces

**File:** `app.py`  
**URL:** [https://huggingface.co/spaces/MuseerahFatima/Karachi-aqi-predictor](https://huggingface.co/spaces/MuseerahFatima/Karachi-aqi-predictor)

**What it does at runtime:**

1. Loads `aqi_karachi_day1/2/3 @production` models from the MLflow Model Registry on DagsHub.
2. Fetches live conditions for Karachi from Open-Meteo.
3. Builds a feature vector and calls `.predict()` for each horizon.
4. Renders a Gradio dashboard with:
   - **AQI Gauge** (current day) with colour-coded category.
   - **Health Advisory** banner.
   - **Pollutant Cards** (PM2.5, PM10, NO₂, Ozone).
   - **3-Day Forecast Cards** (date, AQI value, category, advice).
   - **Trend Chart** (Plotly line chart with AQI category bands).

---

## Dataset & Features

**Data source:** Open-Meteo (free, no API key required)  
**City:** Karachi — latitude `24.8607`, longitude `67.0011`  
**Granularity:** Hourly  
**History in EDA notebook:** March 2026 → June 2026 (~2,184 rows)

### Feature Columns

| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23) |
| `day` | Day of month |
| `month` | Month of year |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `temperature` | 2m air temperature (°C) |
| `humidity` | Relative humidity (%) |
| `wind_speed` | 10m wind speed (km/h) |
| `precipitation` | Hourly precipitation (mm) |
| `surface_pressure` | Surface pressure (hPa) |
| `pm2_5` | PM2.5 concentration (µg/m³) |
| `pm10` | PM10 concentration (µg/m³) |
| `no2` | Nitrogen dioxide (µg/m³) |
| `ozone` | Ozone (µg/m³) |

**Target:** `aqi` — US Air Quality Index, shifted N×24 hours for each forecast day.

---

## ML Models

Three candidate regressors are evaluated per forecast day. The best by RMSE on the test set is registered to MLflow.

| Candidate | Notes |
|---|---|
| `RandomForestRegressor` (100 trees) | Ensemble, handles non-linearity, robust to outliers |
| `GradientBoostingRegressor` (100 trees) | Sequential boosting, often wins on tabular data |
| `Ridge` + `StandardScaler` pipeline | Linear baseline, fast, interpretable |

**Sample performance (EDA notebook):**

| Horizon | Best Model | RMSE | MAE | R² |
|---|---|---|---|---|
| Day 1 | XGBoost* | 7.54 | 5.13 | 0.851 |
| Day 2 | RandomForest | 6.58 | 4.33 | 0.897 |
| Day 3 | XGBoost* | 5.62 | 3.77 | 0.919 |

*XGBoost was used in EDA experiments; the production pipeline uses the scikit-learn candidates above.

**SHAP analysis** (EDA notebook) shows that `pm2_5`, `pm10`, and `ozone` are the top predictors across all horizons.

---

## Setup & Local Development

### Prerequisites

- Python 3.11
- Git + DVC
- A DagsHub account (free)

### Clone & install

```bash
git clone https://github.com/<your-org>/aqi-predictor.git
cd aqi-predictor
pip install -r requirements-pipeline.txt   # for pipelines
pip install -r requirements.txt            # for Gradio app
```

### Configure environment

```bash
cp .env.example .env
# Fill in CITY, DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN
```

### Bootstrap feature store (one-time)

```bash
python -m src.backfill           # fetches ~90 days of historical data
```

### Run pipelines manually

```bash
# Feature pipeline (appends current hour)
python -m src.feature_pipeline

# Training pipeline (trains models, logs to MLflow)
python -m src.training_pipeline

# Run Gradio app locally
python app.py
```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `CITY` | City identifier — must be `karachi` | Yes |
| `DAGSHUB_USERNAME` | Your DagsHub username | Yes |
| `DAGSHUB_REPO` | Your DagsHub repository name | Yes |
| `DAGSHUB_TOKEN` | DagsHub personal access token (for DVC + MLflow auth) | Yes |
| `FEATURE_STORE_PATH` | Override default parquet path | No |

Set these as **GitHub Actions Secrets** for the workflows and as **Hugging Face Space Secrets** for the app.

---

## Deployment

### GitHub Actions

The repository contains three workflows:

| Workflow | Schedule | Purpose |
|---|---|---|
| `feature_pipeline.yml` | `0 * * * *` (hourly) | Ingest features, push to DagsHub via DVC |
| `training_pipeline.yml` | `0 2 * * *` (daily) | Retrain models, register in MLflow |
| `smoke_tests.yml` | On PR + push to main | Validate imports + city guard |

Secrets required in GitHub:  
`CITY`, `DAGSHUB_USERNAME`, `DAGSHUB_REPO`, `DAGSHUB_TOKEN`

### Hugging Face Spaces

1. Create a new **Gradio** Space.
2. Push repository files (at minimum: `app.py`, `src/`, `requirements.txt`).
3. Set Space secrets: `CITY`, `DAGSHUB_USERNAME`, `DAGSHUB_REPO`, `DAGSHUB_TOKEN`.
4. The Space auto-builds and serves on every push.

---

## Limitations

- **Karachi-only** — hardcoded coordinates and city guard; multi-city support would require parameterisation across all pipelines.
- **No satellite / geospatial data** — features rely solely on meteorological and pollutant readings from Open-Meteo.
- **Open-Meteo API continuity** — forecast quality depends on uninterrupted upstream data; API outages will create gaps in the feature store.
- **Tabular models only** — no time-series-specific architectures (e.g., LSTM, Transformer) are used.
- **Training on shifted AQI** — the target is the raw AQI value N days ahead rather than a true multi-step sequence model.
- **Static model selection** — the best model is chosen by test-set RMSE once per training run, without cross-validation.

---

## Data Sources

- **Open-Meteo Weather Forecast API** — [https://open-meteo.com/](https://open-meteo.com/)
- **Open-Meteo Air Quality API** — [https://open-meteo.com/en/docs/air-quality-api](https://open-meteo.com/en/docs/air-quality-api)

Both APIs are free and do not require an API key.

---

## License

This project is for educational and research purposes.

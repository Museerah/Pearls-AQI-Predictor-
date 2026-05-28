---
title: Pearls AQI Predictor
sdk: gradio
app_file: app.py
pinned: false
---

# Pearls AQI Predictor (Karachi) — 100% Serverless

A serverless end-to-end AQI prediction system for **Karachi only** that forecasts AQI for the next 3 days.

## Architecture (text diagram)

1. **Hourly Feature Pipeline (GitHub Actions)**
   - Pulls latest weather + air-quality observations from Open-Meteo APIs.
   - Appends latest feature row to `data/features.parquet`.
   - Tracks parquet with **DVC** and pushes data blob to **DagsHub Storage**.
   - Commits only lightweight metadata (`data/features.parquet.dvc`, `.dvc/config`) to Git.

2. **Daily Training Pipeline (GitHub Actions)**
   - Pulls latest DVC-tracked features from DagsHub Storage.
   - Trains candidate regressors for day1/day2/day3 horizon.
   - Logs metrics to **MLflow on DagsHub**.
   - Registers best model per horizon in MLflow Model Registry with names:
     - `aqi_karachi_day1`
     - `aqi_karachi_day2`
     - `aqi_karachi_day3`
   - Updates `@production` alias to latest successful version.

3. **Serverless Inference UI (Hugging Face Spaces + Gradio)**
   - `app.py` loads latest production model versions from MLflow registry on startup/refresh.
   - Fetches live Karachi conditions from Open-Meteo.
   - Produces next 3-day AQI forecast and renders:
     - values + categories
     - trend chart
     - hazardous AQI alert panel

## Karachi configuration (explicit)

- Supported city: **Karachi only**
- Coordinates: **24.8607, 67.0011**
- Runtime guard: if `CITY` is not `karachi`, the entrypoint fails fast.

## Pipelines

### Hourly feature pipeline
Workflow: `.github/workflows/feature_pipeline.yml`
- Schedule: `0 * * * *`
- Output: DVC metadata commit + DVC push to DagsHub storage

### Daily training pipeline
Workflow: `.github/workflows/training_pipeline.yml`
- Schedule: `0 2 * * *`
- Output: MLflow metrics and model versions in DagsHub registry

### Smoke tests
Workflow: `.github/workflows/smoke_tests.yml`
- Validates imports/config runtime behavior (no import-time env crashes)

## Environment variables

Copy `.env.example` for local use.

### GitHub Actions secrets
- `CITY` = `karachi`
- `DAGSHUB_USERNAME`
- `DAGSHUB_REPO`
- `DAGSHUB_TOKEN`

### Hugging Face Spaces secrets
- `CITY` = `karachi`
- `DAGSHUB_USERNAME`
- `DAGSHUB_REPO`
- `DAGSHUB_TOKEN` (recommended for private/authenticated registry calls)

### Notes on public DagsHub repo
Because the DagsHub repo is public, metadata/artifacts are visible according to DagsHub permissions. Keep using token-based auth in CI/HF for reliability and rate-limit handling.

## Local run

```bash
pip install -r requirements-pipeline.txt
cp .env.example .env
```

Backfill once (optional bootstrap):
```bash
python -m src.backfill
```

Run feature pipeline manually:
```bash
python -m src.feature_pipeline
```

Run training manually:
```bash
python -m src.training_pipeline
```

Run Gradio app:
```bash
pip install -r requirements.txt
python app.py
```

## Deploy on Hugging Face Spaces

1. Create a new **Gradio** Space.
2. Push repository files (at minimum: `app.py`, `src/`, `requirements.txt`).
3. Add Space secrets (`CITY`, `DAGSHUB_USERNAME`, `DAGSHUB_REPO`, `DAGSHUB_TOKEN`).
4. Space builds and serves `app.py` automatically.

## Data sources

- Open-Meteo Weather Forecast API
- Open-Meteo Air Quality API

## Limitations

- Karachi-only deployment currently.
- Forecast quality depends on available recent historical data and API continuity.
- Current model uses tabular regressors with hourly features; no explicit geospatial or satellite data.
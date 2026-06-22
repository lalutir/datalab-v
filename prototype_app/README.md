# Jijiga Flood & Drought Risk Prediction — Prototype App

## Overview

This is a prototype/demo interface for the hybrid ensemble model from the Datalab portfolio project. It shows how the model predicts drought and flood risk for the next 1, 3, or 7 days based on real-time weather forecasts.

**This is not a production system.** It is intended as a demonstration of model functionality.

## For the client (non-technical)

The easiest way to use this app is via the hosted version — no installation required:

**Just open this URL in your browser:**

> https://jijiga-prediction.streamlit.app

*(Note for the development team: this URL becomes active after deploying to Streamlit Community Cloud — see deployment instructions below.)*

All you need is:
- A web browser (Chrome, Firefox, Edge, Safari)
- An internet connection

No Python, no GitHub, no technical knowledge required.

---

## For the development team

### Quick start (local)

```bash
# 1. Install dependencies (from the prototype_app/ folder)
pip install -r requirements.txt

# 2. Start the app
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

### Deploy to Streamlit Community Cloud (free)

This makes the app available to anyone via a public URL:

1. Push this repository to GitHub (if not already done)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click "New app" and select:
   - Repository: your repo
   - Branch: main
   - Main file path: `prototype_app/app.py`
4. Click "Deploy"
5. After ~2 minutes, you get a public URL to share with the client

**Requirements for deployment:**
- The repo must be public (or you use a paid Streamlit tier for private repos)
- Total repo size must be under 1 GB (ours is ~25 MB — well within limits)
- The app needs internet access to call the Open-Meteo API (available on Streamlit Cloud)

---

## What it does

1. **Fetches weather forecast** via the free Open-Meteo API (no API key needed)
2. **XGBoost component**: predicts risk from historical lag features (40 features)
3. **GenCast-style component**: computes forward API from forecast precipitation to get flood score
4. **Hybrid ensemble**: `max(XGBoost, GenCast-style)` — takes the highest risk from both models

## Required files

The app reads the following files from the repository (relative to project root):

| File | Purpose |
|------|---------|
| `src/data/processed/era5_labeled.parquet` | ERA5 data + risk labels (Phase 3 output) |
| `src/data/processed/feature_matrix.parquet` | 40 lag features for XGBoost (Phase 4 output) |
| `src/data/processed/xgb_models/flood_+{1,3,7}d.ubj` | XGBoost flood models (Phase 5) |
| `src/data/processed/xgb_models/drought_+{1,3,7}d.ubj` | XGBoost drought models (Phase 5) |

## Model / notebook source

- **Primary source**: `notebooks/phase6_foundation_model.ipynb`
- **Model**: Hybrid ensemble (Section 5 of that notebook)
- **Ensemble logic**: `ensemble_risk = max(XGBoost_pred, GenCast_flood_score)`
- **XGBoost models**: trained in `notebooks/phase5_xgboost.ipynb`
- **Risk labels**: defined in `notebooks/phase3_index_eda.ipynb`

## Prototype limitations

1. **SPEI-6 is frozen**: the 6-month drought index cannot be updated from a 7-day forecast. The last known ERA5 value is used.
2. **SMI and total_ro are frozen**: Open-Meteo does not provide exact equivalents of ERA5 soil moisture/runoff.
3. **lag_365 features**: XGBoost uses features from 1 year ago — these come from stored ERA5 data, not from live data.
4. **Single grid point**: the model represents only Jijiga (42.75E, 9.25N). Conditions elsewhere in the region may differ.
5. **Open-Meteo vs ERA5**: forecast variables from Open-Meteo are not identical to ERA5 reanalysis. For a demo this is acceptable.
6. **No upstream floods**: the model cannot detect floods from the Wabi Shabelle river (structural limitation).
7. **GenCast is simulated**: the GenCast component uses the same flood-score computation with forecast precipitation, not the actual GenCast foundation model (which requires an A100 GPU).

## Technical architecture

```
Open-Meteo API --> forecast precipitation (7 days)
     |
[GenCast-style]: forward API --> flood score --> risk label
     |
[XGBoost]: feature_matrix (lag features) --> risk label
     |
[Ensemble]: max(XGBoost, GenCast-style) --> final prediction
     |
Streamlit UI: risk indicator + charts + table
```

## Troubleshooting

- **"Could not connect to Open-Meteo"**: check your internet connection. The app needs internet to fetch the weather forecast.
- **"FileNotFoundError"**: make sure you start the app from the `prototype_app/` folder and that the `src/data/processed/` files exist.
- **Model loads slowly**: first load takes a few seconds (parquet files + XGBoost models). After that it is cached.

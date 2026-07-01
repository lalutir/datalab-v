# Jijiga Early Warning Dashboard - Prototype App

## For the opdrachtgever

This app is meant to be opened through a hosted Streamlit link. No Python, GitHub, terminal, or local installation is needed.

Hosted demo URL after deployment:

> https://jijiga-prediction.streamlit.app

You only need:

- A web browser
- An internet connection

**Important:** this is a Datalab prototype/demo. It is not an official warning system and must not be used for operational flood or drought decisions.

## What the app shows

The dashboard estimates flood and drought risk for the Jijiga ERA5 grid point near 9.25N, 42.75E. It supports +1, +3, and +7 day forecast horizons.

The live demo combines:

- Stored XGBoost models trained from ERA5-derived lag features
- A forecast-index component using Open-Meteo precipitation
- A flood ensemble that takes the maximum of XGBoost and forecast-index risk
- A drought output based on XGBoost, because SPEI-6 cannot be updated from a 7-day forecast

The app does **not** run live GenCast GPU inference. It demonstrates the forecast-index step from the larger foundation-model pipeline.

## Local run for the development team

From the `prototype_app` folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

Recommended low-cost/free route:

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Create a new app.
4. Select the repository and branch.
5. Set the main file path to `prototype_app/app.py`.
6. Deploy the app.
7. Share the generated public Streamlit URL with the opdrachtgever.

### Dependency note

The prototype dependencies are listed in `prototype_app/requirements.txt`. If Streamlit Community Cloud installs only the repository-root `requirements.txt` for your setup, use a deployment branch where the root `requirements.txt` contains the app dependencies from `prototype_app/requirements.txt`.

## Required repository files

The app expects these files to exist in the repository:

| File | Purpose |
|------|---------|
| `src/data/processed/era5_labeled.parquet` | ERA5 data and risk labels |
| `src/data/processed/feature_matrix.parquet` | XGBoost lag feature matrix |
| `src/data/processed/xgb_models/flood_+1d.ubj` | Flood model for +1 day |
| `src/data/processed/xgb_models/flood_+3d.ubj` | Flood model for +3 days |
| `src/data/processed/xgb_models/flood_+7d.ubj` | Flood model for +7 days |
| `src/data/processed/xgb_models/drought_+1d.ubj` | Drought model for +1 day |
| `src/data/processed/xgb_models/drought_+3d.ubj` | Drought model for +3 days |
| `src/data/processed/xgb_models/drought_+7d.ubj` | Drought model for +7 days |

Paths are resolved relative to the repository root, so the app can run locally or in a hosted Streamlit environment.

## Prototype limitations

- Single ERA5 grid point near Jijiga only
- No upstream Wabi Shabelle river flood detection
- No live GenCast GPU inference in the hosted demo
- Open-Meteo forecast variables are not identical to ERA5 variables
- SMI, SPEI-6, and total runoff are not dynamically forecast by the live demo
- Predictions are indicative and for portfolio demonstration only

## Troubleshooting

- **Weather forecast error:** check that the hosted app has internet access to Open-Meteo.
- **Missing file error:** confirm the required processed data and model artifacts are committed or otherwise available to the hosted app.
- **Slow first load:** Streamlit loads parquet files and XGBoost models on first use, then caches the predictor.

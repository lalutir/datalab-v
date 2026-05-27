# Jijiga Flood & Drought Risk Prediction

GenAI portfolio project. Single grid-point (42.75°E, 9.25°N) early-warning system for Jijiga, Ethiopia. Compares three approaches: ERA5 threshold rules, XGBoost, and a foundation model (GenCast).

## Project layout

```
src/
  data/
    raw/            ERA5 6-hourly NetCDF files (2000–2025, monthly zips)
    processed/
      era5_2000_2025.parquet   9497 rows × 23 cols — read-only source of truth
      emdat.xlsx               EM-DAT disaster events — validation only, never for training
      era5_labeled.parquet     output of phase3
      feature_matrix.parquet   output of phase4
      baseline_scores.csv      output of phase4
  aggregate.py / indices.py / ...   ERA5 download & preprocessing scripts
notebooks/
  phase3_index_eda.ipynb       SPEI-6/12, API(k=0.92), SMI(FC), risk labels → era5_labeled.parquet
  phase4_feature_engineering.ipynb  40 lag features, leakage checks, naive baseline → feature_matrix.parquet
  phase5_xgboost.ipynb         8 XGBoost classifiers, walk-forward CV, SHAP → xgb_models/
  phase6_foundation_model.ipynb     GenCast pipeline, simulated FM results, three-way table
report/
  final_report.md              Full written report (~2500 words)
```

## Run order

Notebooks must be run 3 → 4 → 5 → 6. Each reads the previous phase's parquet output.

## Key methodology

**SPEI**: Recomputed as SPEI-6 and SPEI-12 (monthly log-logistic / Fisk distribution, calibrated on 2000–2020, forward-filled to daily). The existing `spei` column in `era5_2000_2025.parquet` is a 30-day rolling approximation — do not use it for analysis.

**API**: `k=0.92` decay (not the `k=0.85` in the raw parquet). Represents ~12-day memory for semi-arid soils.

**SMI**: Via field capacity (`FC1 = FC2 = 0.323 m³/m³`), clipped to [0, 1].

**Risk labels**: 5-class (0=None … 4=Extreme). Drought uses McKee SPEI thresholds + API/SMI modifier; flood uses a weighted composite score (40% API, 35% SMI, 25% total_ro) thresholded at training-set percentiles (p65/p80/p90/p97). Training mask is 2001–2022; test is 2023–2025.

**Shifted targets**: `drought_risk_t_plus_{1,3,7,14}` and `flood_risk_t_plus_{1,3,7,14}` — shift by `-n` rows (forecast horizons).

**Feature matrix**: 8 vars × 5 lags (1, 3, 7, 14, 365 days) = 40 features. First 365 rows dropped.

**Walk-forward CV folds**:
- Fold 1: train 2001–2014, val 2015–2016
- Fold 2: train 2001–2016, val 2017–2018
- Fold 3: train 2001–2018, val 2019–2020

**XGBoost**: `objective='multi:softmax'`, `num_class=5`, class-balanced sample weights. Models saved under `xgb_models/`.

**Foundation model**: GenCast (Google DeepMind) requires an Azure NC24ads A100 v4 GPU VM for inference. Phase 6 notebook documents the full pipeline and uses ERA5 + Gaussian noise as a simulation stand-in.

## Dependencies

```
pandas numpy scipy matplotlib seaborn xgboost shap scikit-learn openpyxl pyarrow
```

Install shap separately if missing: `pip install shap` (version 0.51.0 tested).

## Important constraints

- `emdat.xlsx` is validation-only — never used as training labels or features.
- The raw parquet `api` column uses `k=0.85`; phase 3 recomputes with `k=0.92`.
- Single grid-point cannot detect upstream Wabi Shabelle river floods — a known limitation in the report.
- EMDAT events contain ` ` (narrow no-break space) — avoid printing raw DataFrame rows to the terminal to sidestep cp1252 encoding errors on Windows.

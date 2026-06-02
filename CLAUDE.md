# Jijiga Flood & Drought Risk Prediction

## What this project is

Datalab portfolio project for a student group (4 people). We are building a proof-of-concept early-warning system for flood and drought risk at a single ERA5 grid point near Jijiga city, Ethiopia (42.75°E, 9.25°N) in the Somali region.

The core scientific contribution is a **three-way comparison** of increasingly sophisticated approaches:

| Approach | What it does | Requires GPU |
|---|---|---|
| 1 — ERA5 index thresholding | Applies SPEI/flood-score thresholds directly to ERA5-derived indices | No |
| 2 — XGBoost multiclass classifier | Predicts future risk level from lagged index and weather features | No |
| 3 — Foundation model pipeline (flood) | GenCast/HRES forecast → compute indices → apply fitted classifiers | Yes (Azure A100) for GenCast |
| 3 — Foundation model pipeline (drought) | SEAS5 51-member seasonal forecast → recompute SPEI-6 → McKee thresholds | No |

Each approach should improve on the previous. The comparison table (built across phases 3–6) is the centrepiece of the final report.

## Why Jijiga

The Horn of Africa experiences some of the most severe and recurrent drought and flood events in the world. Jijiga sits in the Ethiopian Somali region, strongly influenced by ENSO and the Indian Ocean Dipole. The region has a bimodal rainy season (April–May and October–November) separated by two dry seasons. A 14-day advance warning of elevated drought or flood risk would allow field teams to pre-position resources and activate contingency plans.

**Known major drought years to validate against**: 2002, 2006, 2011, 2016–17, 2022–23.

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
      seas5_jijiga_YYYY_MM.nc  SEAS5 system 51 forecast data (6 init months, also in Azure)
      seas5_drought_results.csv  SEAS5 drought risk predictions (36 rows, also in Azure)
  aggregate.py / indices.py / ...   ERA5 download & preprocessing scripts
  download_seas5_inputs.py     Download SEAS5 from CDS (system 51); --simulate flag available
  postprocess_seas5_results.py SPEI-6 pipeline on SEAS5; writes seas5_drought_results.csv
notebooks/
  phase3_index_eda.ipynb       SPEI-6/12, API(k=0.92), SMI(FC), risk labels → era5_labeled.parquet
  phase4_feature_engineering.ipynb  40 lag features, leakage checks, naive baseline → feature_matrix.parquet
  phase5_xgboost.ipynb         8 XGBoost classifiers, walk-forward CV, SHAP → xgb_models/
  phase6_foundation_model.ipynb     GenCast pipeline, HRES pipeline, SEAS5 drought pipeline, three-way table
report/
  final_report.md              Full written report (~2500 words)
```

## Run order

Notebooks must be run 3 → 4 → 5 → 6. Each reads the previous phase's parquet output.

---

## Key methodology

### Data resolution

The raw ERA5 data is 6-hourly. It is aggregated to **daily** resolution before any index computation. Aggregation rules per variable:

- `tp`, `e`, `pev`, `sro`, `lsp` — **sum** the four 6-hourly values
- `t2m`, `swvl1`, `swvl2` — **mean** of the four 6-hourly values

Daily resolution is chosen because flood events at Jijiga can develop within 24–48 hours. Monthly aggregation would smooth these events into noise.

### Indices

Four indices are computed. These are the inputs to both the risk classifiers and the XGBoost feature matrix.

**SPEI** (Standardized Precipitation Evapotranspiration Index):
- Recomputed as SPEI-6 and SPEI-12 using the log-logistic (Fisk) distribution
- Calibrated on the 2000–2020 reference period only
- Monthly accumulations, then forward-filled to daily resolution
- Captures drought driven by both rainfall deficit AND high-temperature evapotranspiration — critical for the Horn of Africa
- The existing `spei` column in `era5_2000_2025.parquet` is a 30-day rolling approximation — **do not use it for analysis**

**API** (Antecedent Precipitation Index):
- Formula: `API(t) = k × API(t-1) + P(t)` where `P(t)` is daily total precipitation
- Decay constant `k = 0.92` (~12-day soil memory, appropriate for semi-arid soils near Jijiga)
- Initialised at 0 on 2000-01-01, run forward through the full record
- Captures soil saturation state — recent rain counts more than older rain
- The raw parquet `api` column uses `k = 0.85` — **phase 3 recomputes with k = 0.92**
- Do NOT use API in the drought classifier — near-zero values during dry seasons trigger spurious modifier rule firing

**SMI** (Soil Moisture Index):
- Formula: `SMI = (swvl1 + swvl2) / (FC1 + FC2)`, clipped to [0, 1]
- Field capacity values: `FC1 = FC2 = 0.323 m³/m³` (loamy sand, Somali region)
- Provides a direct normalised measure of root-zone soil moisture independent of precipitation history

**total_ro** (total runoff):
- `total_ro = sro + lsp` (surface runoff + large-scale precipitation as subsurface proxy)
- Near-zero at this arid grid point for most of the year — limits flood model sensitivity to upstream Wabi Shabelle dynamics

### Risk label construction

All percentile cutoffs and normalisation parameters are **fitted on training data (2001–2022) only** and then applied to the full dataset. Never refit on test data.

**Drought risk** (4 classes: 0=Low, 1=Moderate, 2=High, 3=Extreme):

Primary classification using McKee (1993) SPEI thresholds:
- SPEI ≥ −0.5 → Low (0)
- −1.0 to −0.5 → Moderate (1)
- −1.5 to −1.0 → High (2)
- < −1.5 → Extreme (3)

Secondary modifier: if base risk is Moderate or High AND (API < training 25th percentile OR SMI < training 25th percentile) → elevate one level. Do not apply to Low or Extreme.

**Flood risk** (4 classes: 0=Low, 1=Moderate, 2=High, 3=Extreme):

Composite score: `flood_score = 0.40 × norm_API + 0.35 × norm_SMI + 0.25 × norm_total_ro`

Where norm_X = (X - train_min) / (train_max - train_min), clamped to [0, 1].

Thresholds from training composite score percentiles:
- < p65 → Low (0)
- p65–p80 → Moderate (1)
- p80–p90 → High (2)
- > p90 → Extreme (3)

SPEI is explicitly excluded from the flood composite — during flash flood events (dry conditions preceding sudden rainfall) SPEI can be negative, which would suppress the flood score.

**Expected class distribution**: Low ~65%, Moderate ~15%, High ~10%, Extreme ~10%.

### Shifted targets

8 forecast tasks: 2 hazards × 4 horizons.

```python
drought_risk_t_plus_{1,3,7,14}   # shift drought_risk by -n rows
flood_risk_t_plus_{1,3,7,14}     # shift flood_risk by -n rows
```

Positive horizon = forecast further into the future = harder task. The model at +14 days is predicting what will happen two weeks from now.

### Feature matrix

40 features = 8 variables × 5 lag values.

Variables: `SPEI`, `API`, `SMI`, `total_ro`, `tp`, `t2m`, `e`, `pev`

Lags: 1, 3, 7, 14, 365 days

The lag_365 features encode seasonal awareness implicitly — the model learns "at this point last year, the indices were at these values" without needing explicit calendar date inputs.

First 365 rows are dropped after lag creation (NaN values for lag_365). Final feature matrix has approximately 8766 rows.

**Do not include the non-lagged risk label columns as features** — they are derived from the indices already present in the lags and including them would constitute leakage for t+n prediction.

### Train / test split

- **Train**: 2001-01-01 → 2022-12-31 (after lag dropping)
- **Test**: 2023-01-01 → 2025-12-31

Never shuffle. Never use data from after 2022 during training, hyperparameter tuning, or any threshold fitting.

### Walk-forward CV folds

Used for hyperparameter tuning within the training period only.

| Fold | Train | Validate |
|---|---|---|
| 1 | 2001–2014 | 2015–2016 |
| 2 | 2001–2016 | 2017–2018 |
| 3 | 2001–2018 | 2019–2020 |

### XGBoost

8 separate multiclass classifiers — one per hazard × horizon combination. Separate models per task because optimal feature weights differ across horizons (lag_1 dominates at +1 day; lag_365 gains importance at +14 days).

```python
XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=300,       # tuned via walk-forward CV
    max_depth=6,            # tuned: try 4, 6, 8
    learning_rate=0.05,     # tuned: try 0.01, 0.05, 0.1
    subsample=0.8,
    colsample_bytree=0.8,
)
```

Handle class imbalance with sample weights: `weight = total_samples / (4 × count_of_that_class)`.

Models saved under `xgb_models/`.

**Naive persistence baseline**: for horizon +n, predict risk(t+n) = risk(t). XGBoost must beat this baseline on macro recall to demonstrate it adds value. If it does not beat the baseline on any task, that model adds no value and should be investigated.

**SHAP analysis**: use `shap.TreeExplainer`. Expected pattern — at +1 day: lag_1 features dominate. At +14 days: lag_365 and lag_14 gain relative importance. Flood models: API and tp lags should rank high. Drought models: SPEI and t2m lags should rank high. SHAP findings are a key scientific output of the project.

**Key evaluation metric**: macro recall (equal weight to each class regardless of frequency). Weighted F1 was rejected because the dominant Low class drives the score — a model predicting Low every day achieves high weighted F1 while detecting no hazard events. For a risk warning system, **recall on the Extreme class (class 3) is the single most important per-class metric** — missing a real extreme event is more costly than a false alarm. Always use `average='macro', zero_division=0, labels=range(4)` when computing recall.

### Foundation model (Phase 6)

GenCast (Google DeepMind) runs on the Azure spot GPU VM (NC24ads A100 v4, France Central region). The pipeline is:

```
Current ERA5 atmospheric state
        ↓
GenCast 1° inference → 10-day ensemble forecast of weather variables
        ↓
Aggregate 6-hourly forecast to daily
        ↓
Compute SPEI, API, SMI, total_ro from forecasted variables
        ↓
Apply fitted risk classifiers from Phase 3 (no retraining)
        ↓
Risk level for each forecast day
```

Critical detail: when computing API from foundation model forecasts, initialise from the last known ERA5 API value (not from zero) so the forecast API continues from the correct antecedent soil state.

For SPEI computation from forecast data: use the training-period normalisation parameters (do not refit on forecast output).

The Phase 6 notebook documents the full pipeline and uses ERA5 + Gaussian noise as a simulation stand-in for when the GPU VM is not available.

---

## EM-DAT usage

`emdat.xlsx` is **validation only**. It is never used as training labels or input features.

Role: light spot-check validation. For each documented EM-DAT event in the Jijiga/Somali region, the risk classifiers should show at least Moderate risk during the window from 30 days before event start to 30 days after event end. Most major events should show Elevated or above.

Accept unexplained elevated risk periods with no EM-DAT entry — small local events never reach EM-DAT's humanitarian documentation threshold. These are not false positives.

---

## Known limitations (for the report)

- **Single grid point**: represents conditions near Jijiga city specifically. Conditions 100+ km away in the Somali region may differ significantly.
- **Wabi Shabelle river floods**: upstream catchment flooding from the Wabi Shabelle river basin cannot be detected at the local grid point — local ERA5 precipitation may be near-zero during a downstream flood event. This is a core structural limitation.
- **Monthly SPEI disaggregation artefact**: SPEI is computed on monthly accumulations then forward-filled to daily. This means SPEI value is constant within each month, which is a coarse approximation.
- **GenCast 1° resolution**: a 1° grid cell (~100 km) is a very coarse representation of Jijiga micro-climate.
- **Forecast horizon limit**: GenCast produces forecasts up to 10 days. The +14 day XGBoost target has no foundation model equivalent and relies on climatological extrapolation.
- **No ENSO/IOD features**: El Niño and Indian Ocean Dipole conditions are primary drivers of multi-year drought in the Horn of Africa. Their absence from the feature set limits seasonal drought skill.

---

## Azure infrastructure

- **Blob Storage**: containers and contents:
  - `era5-data` — raw ERA5 6-hourly NetCDF zips (2000–2025, monthly)
  - `processed-data` — processed outputs: `seas5_drought_results.csv`
  - `model-artifacts` — model weights and forecast inputs:
    - `gencast_inputs/` — ERA5 pressure-level files for GenCast inference
    - `seas5_inputs/` — SEAS5 system 51 NetCDF files (6 init months, 2023–2024)
  - `era5-africa` — ERA5 Africa-domain NetCDF files (future expansion)
- **GPU VM**: NC24ads A100 v4, France Central, spot instance only. Deallocate (not just stop) immediately after inference. Auto-shutdown configured to 3 hours.
- **All CPU work runs locally** on team members' Windows laptops. No CPU VM.
- Connection string stored in `.env` file (gitignored). Load with `python-dotenv`.
- **CDS API credentials** also in `.env` as `CDS_API_URL` and `CDS_API_KEY`. SEAS5 requires `system: '51'` (SEAS5.1 real-time, 2020-present); `system: '5'` covers pre-2020 hindcasts only. Licence for `seasonal-monthly-single-levels` must be accepted at cds.climate.copernicus.eu before first download.

---

## Dependencies

```
pandas numpy scipy matplotlib seaborn xgboost shap scikit-learn openpyxl pyarrow
```

Install shap separately if missing: `pip install shap` (version 0.51.0 tested).

---

## Important constraints

- `emdat.xlsx` is validation-only — never used as training labels or features.
- The raw parquet `api` column uses `k=0.85`; phase 3 recomputes with `k=0.92`.
- The raw parquet `spei` column is a 30-day rolling approximation; phase 3 recomputes proper SPEI-6/12.
- All scalers, percentile thresholds, and normalisation parameters must be fit on 2001–2022 training data only.
- Temporal split is strict — never shuffle time series data.
- Single grid-point cannot detect upstream Wabi Shabelle river floods — document prominently in limitations.
- EMDAT events contain `\u202f` (narrow no-break space) — avoid printing raw DataFrame rows to the terminal to sidestep cp1252 encoding errors on Windows.
- Risk labels use **4 classes** (0=Low, 1=Moderate, 2=High, 3=Extreme). Do not use 5 classes — the old 5-class scheme (which included a separate "Elevated" class) was discarded in Phase 3. Use `num_class=4` in XGBoost and `labels=range(4)` in all recall computations.
- `.env` file is gitignored — credentials are never committed to the repository.
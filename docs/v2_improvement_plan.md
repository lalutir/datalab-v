# V2 Improvement Plan
## Jijiga Flood & Drought Risk Prediction

This document captures every methodological improvement identified after completing the v1
portfolio submission, with the scientific reasoning behind each, exact implementation steps,
and a ready-to-paste Claude Code prompt for each task.

---

## Guiding principles

**Preserve all existing work.** Every improvement adds new notebook cells or new scripts — it
does not overwrite or delete existing ones. The v1 results remain visible in every notebook.

**Label everything.** New cells begin with a markdown header of the form
`## [V2] <Improvement Name>` so the progression from v1 to v2 is immediately legible to a
reader or examiner.

**Version control strategy.** Before starting any improvement:

```bash
git tag v1.0-portfolio -m "Phase 6 complete: three-way comparison with GenCast hybrid ensemble"
git checkout -b v2-improvements
```

Work entirely on the `v2-improvements` branch. The `main` branch stays untouched as the v1
archive. When improvements are complete, merge `v2-improvements` into `main` and tag `v2.0`.

---

## Improvement 1 — ENSO/IOD features for drought XGBoost

### What
Add monthly El Niño (Nino3.4 SST anomaly) and Indian Ocean Dipole (Dipole Mode Index) indices
as new input features for the four drought XGBoost models. Retrain drought models only.

### Why
Horn of Africa drought is primarily driven by the Indian Ocean Dipole (positive phase suppresses
the short rains) and ENSO (La Niña reduces the long rains). These are the established scientific
explanation for every major drought year in the dataset (2002, 2006, 2011, 2016–17, 2022–23).
The v1 feature matrix has no seasonal climate signal — it relies entirely on lag-365 ERA5
features to encode seasonality. Adding Nino3.4 and DMI at lag-30, lag-90, and lag-180 days
gives the model explicit access to the large-scale circulation state that precedes drought onset
by one to six months. This is the single highest return-on-investment change in this plan.

### Files changed
- `notebooks/phase4_feature_engineering.ipynb` — new cells appended
- `notebooks/phase5_xgboost.ipynb` — new cells appended
- `src/data/processed/feature_matrix_v2_drought.parquet` — new file (40 original + 6 ENSO/IOD features)
- `src/data/processed/xgb_models/drought_v2_+1d.ubj` (and +3d, +7d, +14d) — new model files
- `report/final_report.md` — new subsection in Section 6 and updated comparison table

### Estimated effort
4–6 hours

### Implementation steps

1. Download the monthly Nino3.4 SST anomaly index (NOAA CPC ersst5 dataset) and save as
   `src/data/raw/nino34_monthly.csv`.
2. Download the monthly Indian Ocean Dipole Mode Index (NOAA PSL or KNMI) and save as
   `src/data/raw/dmi_monthly.csv`.
3. In Phase 4 notebook, add a new section after the existing feature matrix cells:
   - Load both monthly index files, parse dates, forward-fill to daily frequency to match ERA5.
   - Merge onto the existing feature matrix by date.
   - Create lag columns: `nino34_lag_30`, `nino34_lag_90`, `nino34_lag_180`,
     `dmi_lag_30`, `dmi_lag_90`, `dmi_lag_180`.
   - Save as `feature_matrix_v2_drought.parquet`.
4. In Phase 5 notebook, add a new section after the existing XGBoost evaluation:
   - Load `feature_matrix_v2_drought.parquet`.
   - Define feature columns = original 40 lag features + 6 ENSO/IOD lag features.
   - Retrain drought_+1d, +3d, +7d, +14d models using the same walk-forward CV and
     hyperparameter search as v1 (reuse the best v1 hyperparameters as starting point).
   - Compare v2 vs v1 drought macro recall on the 2023–2025 test set in a side-by-side table.
   - Save v2 models as `drought_v2_+{horizon}d.ubj`.
   - Run SHAP analysis on v2 models — verify ENSO/IOD features appear in top importance at
     longer horizons (+14d).
5. Add a paragraph to report Section 6 documenting the improvement and results.

### Claude Code prompt

```
This is a drought risk prediction project for Jijiga, Ethiopia. I have:
- `src/data/processed/feature_matrix.parquet` — 51 columns: time, 2 risk labels, 8 shifted
  targets, and 40 lag features (8 ERA5 variables × lags 1,3,7,14,365 days). Train period 2001–2022.
- `src/data/processed/xgb_models/drought_+{1,3,7,14}d.ubj` — four existing drought XGBoost models.
- `notebooks/phase4_feature_engineering.ipynb` and `notebooks/phase5_xgboost.ipynb` — existing notebooks.

Task: Improve drought prediction by adding ENSO and Indian Ocean Dipole (IOD) features.

Step 1 — Download data:
Search for and download the monthly Nino3.4 SST anomaly index from NOAA CPC (ersst5 series)
and the monthly Dipole Mode Index (DMI) from NOAA PSL or an equivalent authoritative source.
Save as `src/data/raw/nino34_monthly.csv` and `src/data/raw/dmi_monthly.csv`.

Step 2 — Add to Phase 4 notebook:
Append new cells (DO NOT modify existing cells) labeled `## [V2] ENSO/IOD Features`.
- Load both CSVs, parse dates, resample to daily frequency (forward-fill).
- Merge onto the existing feature matrix by date.
- Create: `nino34_lag_30`, `nino34_lag_90`, `nino34_lag_180`, `dmi_lag_30`, `dmi_lag_90`, `dmi_lag_180`.
- Save the extended feature matrix as `src/data/processed/feature_matrix_v2_drought.parquet`.
- Print shape and confirm no NaNs in the lag columns for the training period (2001–2022).

Step 3 — Add to Phase 5 notebook:
Append new cells labeled `## [V2] Drought Models with ENSO/IOD`.
- Load `feature_matrix_v2_drought.parquet`.
- Train/test split: train ≤ 2022-12-31, test ≥ 2023-01-01 (NEVER shuffle time series).
- Feature columns = 40 original lag features + 6 new ENSO/IOD lag features.
- Retrain drought_+1d, drought_+3d, drought_+7d, drought_+14d using the same walk-forward
  CV folds as v1 (fold 1: train 2001–2014 val 2015–2016; fold 2: 2001–2016 val 2017–2018;
  fold 3: 2001–2018 val 2019–2020). Use same class weights: total_samples / (5 × class_count).
- Select best hyperparameters from the same grid as v1 (max_depth: 4,6,8; lr: 0.01,0.05,0.1).
- Evaluate on test set with macro recall. Print a side-by-side comparison: v1 vs v2 for each model.
- Save v2 models as `src/data/processed/xgb_models/drought_v2_+{horizon}d.ubj`.
- Run SHAP TreeExplainer on v2 drought_+14d — plot mean absolute SHAP values, verify ENSO/IOD
  features appear in the top 10.
```

---

## Improvement 2 — Fix the persistence baseline comparison

### What
Verify and correct the claim in the report that "XGBoost consistently beats naive persistence
across all eight tasks." From Phase 5 output, drought models appear to score below persistence
on the test set. Produce an explicit comparison table and update the report accurately.

### Why
Scientific correctness. If drought models do not beat persistence (which is expected given that
SPEI-6 is a monthly step function rarely changing within 14 days), the report should say so
clearly. Acknowledging this honestly is stronger than a wrong claim — it also motivates the
ENSO/IOD improvement (Improvement 1) by showing why persistence is nearly unbeatable for a
slowly-varying drought label.

### Files changed
- `notebooks/phase5_xgboost.ipynb` — new cells appended
- `report/final_report.md` — Section 8.2 XGBoost paragraph and Section 6.4

### Estimated effort
2–3 hours

### Implementation steps

1. In Phase 5 notebook, add a new section after the existing evaluation:
   - For each of the 8 tasks, compute: macro recall for the XGBoost model, macro recall for the
     naive persistence baseline (predict risk(t) for all future horizons), and the difference.
   - Present as a table: Task | XGBoost Macro Recall | Persistence Macro Recall | Δ | Beats Baseline?
2. Update Section 8.2 of the report: remove the blanket "beats persistence on all eight tasks"
   claim and replace with the accurate per-task findings, noting which tasks XGBoost genuinely
   adds value on and which it does not.
3. Add a brief interpretation: for drought, persistence is near-perfect because SPEI-6 steps
   monthly — this makes it an easy baseline, not a sign XGBoost is good. This motivates ENSO/IOD.

### Claude Code prompt

```
This is a flood/drought risk prediction project at Jijiga, Ethiopia. XGBoost models are saved
at `src/data/processed/xgb_models/` (8 files: flood_+{1,3,7,14}d.ubj and drought_+{1,3,7,14}d.ubj).
The feature matrix is at `src/data/processed/feature_matrix.parquet` with columns:
time, drought_risk, flood_risk, drought_risk_t_plus_{1,3,7,14}, flood_risk_t_plus_{1,3,7,14},
and 40 lag features. Test period: 2023-01-01 onwards.

Task: Verify whether each XGBoost model actually beats the naive persistence baseline on the
test set, and fix the report accordingly.

The naive persistence baseline for horizon +n is: predict risk(t+n) = risk(t).
Metric: macro recall (sklearn, average='macro', zero_division=0).

Step 1 — Add to `notebooks/phase5_xgboost.ipynb` (new cells ONLY, do not modify existing):
Label the section `## [V2] Baseline Comparison — Explicit Per-Task Results`.
- Load the test set (time >= 2023-01-01) from feature_matrix.parquet.
- Load each of the 8 XGBoost models. Generate predictions on the test set.
- For each task compute: XGBoost macro recall, persistence macro recall.
- Print a formatted table with all results and a "Beats persistence?" column.
- Note: for drought models the SPEI-6 label changes only ~12 times per year (monthly steps),
  so persistence is expected to be nearly perfect for all drought horizons.

Step 2 — Update `report/final_report.md`:
Find the paragraph in Section 8.2 beginning "Approach 2 (XGBoost):" and update it to reflect
the actual per-task findings from Step 1. Be precise: state which specific tasks beat
persistence (if any), note that drought persistence is near-perfect due to monthly SPEI steps,
and explain this motivates the ENSO/IOD and transition-target improvements (referenced by name).
Do not change any other part of the report.
```

---

## Improvement 3 — Probabilistic output and calibration

### What
Change XGBoost evaluation to output calibrated class probabilities instead of hard risk-level
predictions. Add isotonic regression calibration fitted on the validation fold.

### Why
A 5-class hard prediction "High risk" is less useful operationally than "65% probability of
Elevated or higher, 20% probability of High or higher." Calibrated probabilities allow decision-
makers to apply their own cost thresholds rather than using a single fixed cut-off. They also
expose whether the model is overconfident in majority-class predictions (a common XGBoost issue
with class-weighted training). This improvement does not require retraining the models.

### Files changed
- `notebooks/phase5_xgboost.ipynb` — new cells appended
- `report/final_report.md` — new paragraph in Section 6.4

### Estimated effort
4–6 hours

### Implementation steps

1. In Phase 5 notebook, add a new section:
   - For each of the 8 models, use `predict_proba()` on the test set.
   - Fit isotonic regression calibration (sklearn `CalibratedClassifierCV` with
     `method='isotonic'`) using the last walk-forward fold as the calibration set.
   - Plot reliability diagrams (calibration curves) for the Elevated, High, and Extreme
     classes for flood_+7d and drought_+7d as representative examples.
   - Show: before calibration vs after calibration on each reliability diagram.
2. Update Section 6.4 of the report to mention calibrated probabilities as the recommended
   output format for operational use.

### Claude Code prompt

```
This is a flood/drought risk prediction project at Jijiga, Ethiopia. I have 8 XGBoost models
at `src/data/processed/xgb_models/` (flood/drought × +1d/+3d/+7d/+14d). Feature matrix is at
`src/data/processed/feature_matrix.parquet`. Train period 2001–2022, test period 2023+.
Walk-forward fold 3 (train 2001–2018, val 2019–2020) should be used as the calibration set.

Task: Add probability calibration to the existing XGBoost models and add reliability diagrams.

Add new cells ONLY to `notebooks/phase5_xgboost.ipynb` labeled `## [V2] Calibrated Probabilities`.

- For each of the 8 models:
  - Generate raw predicted class probabilities on the test set using predict_proba().
  - Fit a CalibratedClassifierCV(cv='prefit', method='isotonic') on the fold-3 validation set
    (2019-01-01 to 2020-12-31). Note: since the base classifiers are already fitted, use
    cv='prefit' mode.
  - Generate calibrated probabilities on the test set.
- For flood_+7d and drought_+7d specifically, plot reliability diagrams (fraction of positives
  vs mean predicted probability) for classes 2 (Elevated), 3 (High), and 4 (Extreme).
  Show pre- vs post-calibration on the same axes using matplotlib.
- Print a summary table: for each model, show the mean predicted probability for the Extreme
  class (pre and post calibration) vs the actual base rate of Extreme in the test set.
  This diagnoses overconfidence.
- Save the calibrated flood_+7d and drought_+7d calibration objects as
  `src/data/processed/xgb_models/flood_+7d_calibrated.pkl` and drought equivalent.

Also add one paragraph to Section 6.4 of `report/final_report.md` after the existing test set
evaluation paragraph, explaining that calibrated probabilities are available and briefly
describing the calibration method and what the reliability diagrams show.
```

---

## Improvement 4 — ECMWF HRES for Approach 3 flood (replacing GenCast)

### What
Download ECMWF HRES open data medium-range forecasts for the same 8 init dates used for
GenCast. HRES includes soil moisture and runoff outputs, so the full flood composite score
(API + SMI + total_ro) can be computed without freezing any component. Evaluate flood risk
forecast quality and compare against GenCast and XGBoost.

### Why
The fundamental weakness of GenCast for this project is that it outputs only precipitation and
temperature — the 60% of the flood composite driven by SMI and total_ro must be frozen at the
init-date ERA5 value. ECMWF HRES is the world's leading operational medium-range forecast
and publishes daily open data including soil moisture layers (swvl1, swvl2) and runoff. With
HRES, the full flood composite score can vary across the forecast horizon exactly as it does
with ERA5 actuals. No GPU is needed. HRES data is available free via the ECMWF open data
stream at 0.25° resolution up to 15 days lead time.

### Files changed
- `src/download_hres_inputs.py` — new script
- `src/postprocess_hres_results.py` — new script
- `src/data/processed/hres_forecast_results.csv` — new file
- `notebooks/phase6_foundation_model.ipynb` — new cells appended
- `report/final_report.md` — new subsection 7.7 and updated comparison table

### Estimated effort
1.5–2 days

### Implementation steps

1. Write `src/download_hres_inputs.py`:
   - Use the `ecmwf-opendata` Python library (pip install ecmwf-opendata) to download HRES
     forecasts for each of the 8 init dates.
   - Variables needed: `tp` (total precipitation, 6-hourly), `swvl1`, `swvl2` (soil water),
     `sro`, `ssro` (runoff), `t2m` (2m temperature).
   - Download for all available lead steps (0–15 days at 6-hourly).
   - Extract the Jijiga grid point (42.75°E, 9.25°N).
   - Save per init date as `src/data/processed/hres_jijiga_{date}.nc`.

2. Write `src/postprocess_hres_results.py`:
   - Mirror the structure of `src/postprocess_gencast_results.py`.
   - Load each HRES NetCDF file and aggregate to daily (tp: sum, swvl1/swvl2: mean, sro/ssro: sum).
   - Compute API: initialise from ERA5 api_92 at init date, then run forward using HRES daily tp.
   - Compute SMI: use HRES swvl1 and swvl2 forecast values (FC1=FC2=0.323 m³/m³).
   - Compute total_ro: HRES sro + ssro.
   - Apply the full flood composite score (all three components now vary over the forecast).
   - Apply training-period percentile thresholds to assign risk labels.
   - Compute macro recall at leads 1, 3, 7, 10 vs ERA5 actual flood risk.
   - Save results as `src/data/processed/hres_forecast_results.csv`.

3. Add new cells to `notebooks/phase6_foundation_model.ipynb` labeled `## [V2] HRES Flood Forecast`:
   - Load `hres_forecast_results.csv`.
   - Plot the same 8-panel figure as GenCast (HRES predicted vs actual flood risk per init date).
   - Print metrics table: HRES vs GenCast vs XGBoost macro recall at leads 1, 3, 7.
   - Discuss: does HRES improve on GenCast now that SMI and runoff are no longer frozen?

4. Add subsection 7.7 to `report/final_report.md` documenting the HRES approach, methodology,
   and results. Update the three-way comparison table to include HRES as Approach 3b.

### Claude Code prompt

```
This is a flood/drought risk prediction project at Jijiga, Ethiopia (42.75°E, 9.25°N).
ERA5 labeled data is at `src/data/processed/era5_labeled.parquet` with columns including
api_92, smi_fc, total_ro, flood_risk. Training-period normalisation parameters and flood score
percentile thresholds are computed in `src/postprocess_gencast_results.py` — use the same logic.

The existing GenCast Approach 3 has a critical flaw: it only outputs tp and t2m, so SMI and
total_ro are frozen at the init-date ERA5 value (60% of the flood composite signal is static).
I want to add ECMWF HRES forecasts as a parallel Approach 3b, since HRES outputs soil moisture
and runoff — enabling the full flood composite to vary over the forecast period.

The 8 init dates are:
2023-01-15, 2023-03-01, 2023-04-01, 2023-07-01, 2023-10-15, 2024-01-15, 2024-04-01, 2024-10-01

Flood composite: flood_score = 0.40*norm_API + 0.35*norm_SMI + 0.25*norm_total_ro
Thresholds: p65→None(0), p65–p80→Moderate(1), p80–p90→Elevated(2), p90–p97→High(3), >p97→Extreme(4)
All normalisation params and percentiles must be fitted on training data (years ≤ 2022) only.
API decay: k=0.92. API must be initialised from ERA5 api_92 at each init date (not from zero).
SMI field capacity: FC1=FC2=0.323 m³/m³.

Step 1 — Create `src/download_hres_inputs.py`:
Use the `ecmwf-opendata` Python library to download HRES 15-day forecasts for each init date.
Variables: tp, swvl1, swvl2, sro, ssro (and t2m for completeness). Extract only the grid point
nearest to 42.75°E, 9.25°N. Save each as `src/data/processed/hres_jijiga_{date}.nc`.
Include error handling for dates where HRES open data may not be available (it has a ~2 year
rolling archive, so 2023 dates may or may not be available — print a clear message if missing).

Step 2 — Create `src/postprocess_hres_results.py`:
Mirror the structure of `src/postprocess_gencast_results.py`. For each init date:
- Aggregate HRES 6-hourly outputs to daily (tp: sum, swvl: mean, sro+ssro: sum).
- Compute API forward from ERA5 init-date value using HRES daily tp.
- Compute SMI from HRES swvl1 and swvl2.
- Compute total_ro from HRES sro + ssro.
- Apply full flood composite — all three components vary.
- Assign risk labels using training-period thresholds.
- Compare to actual ERA5 flood risk at each forecast date.
- Save results to `src/data/processed/hres_forecast_results.csv`.
- Print macro recall at leads 1, 3, 7, 10 compared to GenCast in `gencast_forecast_results.csv`.

Step 3 — Add to `notebooks/phase6_foundation_model.ipynb` (new cells only, labeled `## [V2] HRES Flood Forecast`):
- Load hres_forecast_results.csv. Plot 8-panel figure (same style as the GenCast figure already
  in the notebook). Print side-by-side metrics: HRES macro recall vs GenCast macro recall vs
  XGBoost macro recall (from comparison_table.csv) at leads 1, 7, 10.
- Add a markdown explanation cell noting that HRES fully resolves the frozen-SMI problem.

Step 4 — Add subsection 7.7 to `report/final_report.md` after the existing Section 7.6.
Title it "7.7 ECMWF HRES as Approach 3b (Full Flood Composite)". Describe the methodology,
note why it resolves the GenCast SMI limitation, and report the results.
```

---

## Improvement 5 — SEAS5 seasonal forecasts for drought (Approach 3 addition)

### What
Download ECMWF SEAS5 seasonal forecasts from the Copernicus Climate Data Store for a set of
init dates spanning 2023–2024. Compute SPEI from the 51-member ensemble precipitation
forecasts at 1-month to 6-month lead. Apply the fitted Phase 3 drought classifier. This is the
first genuine forward-looking drought forecast in the project — GenCast cannot do this at all.

### Why
GenCast is a 10-day atmospheric model. SPEI-6 requires six months of accumulated climatic
water balance. These are incompatible — drought forecasting from GenCast is definitionally
impossible. SEAS5 is ECMWF's seasonal forecast system (51 ensemble members, 6-month lead,
monthly steps) and is specifically designed for this timescale. It is available free via the
Copernicus CDS `cdsapi` library — the same tool already used for ERA5 downloads. This
improvement answers research question 3 for drought, which is currently marked N/A.

### Files changed
- `src/download_seas5_inputs.py` — new script
- `src/postprocess_seas5_results.py` — new script
- `src/data/processed/seas5_drought_results.csv` — new file
- `notebooks/phase6_foundation_model.ipynb` — new cells appended
- `report/final_report.md` — updated Section 7 and comparison table

### Estimated effort
2–3 days

### Implementation steps

1. Write `src/download_seas5_inputs.py`:
   - Use `cdsapi` to download SEAS5 hindcast monthly total precipitation and 2m temperature
     for init months in the test period: Jan 2023, Apr 2023, Oct 2023, Jan 2024, Apr 2024,
     Oct 2024 (6 init dates corresponding to the main rainy season starts and ends).
   - Download all 51 ensemble members, 1–6 month lead, for the Jijiga region (40–45°E, 7–12°N).
   - Save as `src/data/processed/seas5_jijiga_{year}_{month}.nc`.

2. Write `src/postprocess_seas5_results.py`:
   - For each init month and each ensemble member, compute monthly climatic water balance
     (CWB = tp + pev) for the forecast period.
   - Compute rolling 6-month CWB accumulation (requires ERA5 historical CWB for the prior
     5 months as warmup — read from `era5_labeled.parquet`).
   - Apply the SPEI-6 fitted log-logistic distribution parameters from Phase 3 (calibrated
     on 2000–2020 reference period) to convert CWB accumulations to SPEI scores.
   - Apply the drought risk classifier (McKee thresholds + API/SMI modifier rule from CLAUDE.md)
     using ERA5 API and SMI at the init date for the modifier check.
   - Compute ensemble-mean drought risk level for each lead month.
   - Compare to actual ERA5 drought risk for the corresponding future months.
   - Save results to `src/data/processed/seas5_drought_results.csv`.

3. Add new cells to `notebooks/phase6_foundation_model.ipynb` labeled `## [V2] SEAS5 Drought Forecast`:
   - Load `seas5_drought_results.csv`.
   - Plot ensemble-mean drought risk vs actual drought risk for each init date (6-panel figure).
   - Print macro recall at leads 1, 2, 3, 6 months.
   - Update the three-way comparison table to fill in Approach 3 drought rows (previously N/A).

4. Update Section 7 of the report: add a new subsection documenting SEAS5 as the correct
   Approach 3 tool for drought. Update the comparison table with actual SEAS5 metrics.

### Claude Code prompt

```
This is a drought risk prediction project at Jijiga, Ethiopia (42.75°E, 9.25°N). In the
current version (v1), Approach 3 (foundation model) drought forecasting is marked N/A because
GenCast only runs 10 days while SPEI-6 requires 6 months. I want to add ECMWF SEAS5 seasonal
forecasts as a proper drought Approach 3.

Existing relevant data:
- `src/data/processed/era5_labeled.parquet`: daily ERA5 data including spei_6, api_92, smi_fc,
  drought_risk. SPEI-6 is fitted on 2000–2020 reference period using log-logistic (Fisk)
  distribution, separately per calendar month.
- `src/data/processed/comparison_table.csv`: the three-way comparison table.

SEAS5 is available via cdsapi (same library used for ERA5 downloads). Variable name in CDS:
dataset "seasonal-monthly-single-levels", variable "total_precipitation". 51 ensemble members,
lead months 1–6, monthly time step.

Init dates to use: 2023-01 (Jan), 2023-04, 2023-10, 2024-01, 2024-04, 2024-10.

Step 1 — Create `src/download_seas5_inputs.py`:
Use cdsapi to download SEAS5 monthly precipitation (and 2m temperature if available) for the
6 init months above, all 51 ensemble members, leads 1–6 months, over the Jijiga area
(lon 40–45°E, lat 7–12°N). Save each as `src/data/processed/seas5_jijiga_{year}_{month:02d}.nc`.
Handle the case where a SEAS5 init month is not yet archived (print clear message).

Step 2 — Create `src/postprocess_seas5_results.py`:
For each init month and each ensemble member:
- Read the 5 prior months of ERA5 CWB (tp + pev) from era5_labeled.parquet to use as warmup
  for the 6-month rolling accumulation.
- Compute monthly CWB from SEAS5 precipitation (and use ERA5 pev forward-filled as constant
  since SEAS5 monthly pev is not straightforward — document this assumption).
- Compute 6-month rolling CWB accumulation and transform to SPEI-6 using the Phase 3
  log-logistic parameters. These parameters are stored implicitly in the Phase 3 notebook —
  re-derive them from era5_labeled.parquet by fitting scipy.stats.fisk on monthly CWB
  accumulations from 2000–2020, per calendar month.
- Apply McKee drought risk thresholds (SPEI ≥ -0.5 → 0, -1.0 to -0.5 → 1, etc.)
- For the modifier rule: use ERA5 api_92 and smi_fc at the init date (not forecast values).
  Training p25 thresholds: compute from era5_labeled.parquet years ≤ 2022.
- Compute ensemble-mean and ensemble-max drought risk per lead month.
- Compare against actual ERA5 drought_risk for the corresponding calendar months.
- Save to `src/data/processed/seas5_drought_results.csv` with columns:
  init_date, lead_months, seas5_drought_mean, seas5_drought_max, actual_drought_risk.
- Print macro recall at leads 1, 2, 3, 6 months.

Step 3 — Add new cells to `notebooks/phase6_foundation_model.ipynb` ONLY (do not modify
existing cells). Label `## [V2] SEAS5 Seasonal Drought Forecast`.
- Load seas5_drought_results.csv. Plot 6-panel figure: ensemble spread and mean drought risk
  vs actual, one panel per init month. Style consistent with existing GenCast panels in notebook.
- Print metrics table: SEAS5 macro recall by lead month.
- Update the comparison_table.csv to replace "N/A (drought)" in the Approach 3 column with
  SEAS5 results.

Step 4 — Add subsection 7.8 to `report/final_report.md` (after section 7.7) titled
"7.8 SEAS5 Seasonal Drought Forecast (Approach 3 — Drought)". Document the methodology,
the CWB warmup approach, and results. Update the abstract to remove the statement that
drought forecasting is N/A for Approach 3.
```

---

## Improvement 6 — Multi-point Wabi Shabelle catchment features

### What
Download ERA5 for a spatial grid covering the Wabi Shabelle upstream catchment (~5×5 grid
cells centered upstream of Jijiga) and create catchment-aggregated precipitation features.
Retrain the four flood XGBoost models with these additional upstream features.

### Why
The most systematic miss in the current model is Wabi Shabelle river flooding, which is driven
by precipitation upstream in the catchment — not at the Jijiga local grid point. ERA5 already
has this data; it just hasn't been downloaded for those grid points. Adding upstream cumulative
precipitation as a feature directly captures the physical mechanism that causes the most severe
floods at Jijiga. This is the change most likely to improve detection of EMDAT-documented flood
events that the current model completely misses.

### Files changed
- `src/download_era5_upstream.py` — new script
- `src/data/raw/era5_upstream/` — new directory with upstream NetCDF files
- `src/data/processed/era5_upstream_daily.parquet` — aggregated upstream daily data
- `notebooks/phase4_feature_engineering.ipynb` — new cells appended
- `src/data/processed/feature_matrix_v2_flood.parquet` — new file
- `notebooks/phase5_xgboost.ipynb` — new cells appended
- `src/data/processed/xgb_models/flood_v2_+{1,3,7,14}d.ubj` — new model files
- `report/final_report.md` — updated sections 2, 5, 6, 8

### Estimated effort
1 week (dominated by CDS API download time — ~200 GB across 25 grid points)

### Implementation steps

1. Identify the upstream catchment grid points:
   - The Wabi Shabelle river originates near Gara Muleta mountain (~42°E, 9°N) and flows
     southeast. Key upstream grid points at ERA5 1° resolution: approximately a 3×3 or 5×5
     grid covering 40–44°E, 7–11°N (25 points at 1° resolution).
   - Within these points, define a distance-weighted or equal-weight upstream aggregate.

2. Write `src/download_era5_upstream.py`:
   - Use `cdsapi` to download ERA5 daily tp, swvl1, swvl2, sro, ssro for all upstream grid
     points, 2000–2025. Variables needed for flood risk only.
   - Save as `src/data/raw/era5_upstream/era5_upstream_{year}.nc`.

3. Aggregate upstream ERA5:
   - For each day, compute upstream catchment average tp and the area-weighted API.
   - Add to a new parquet: `src/data/processed/era5_upstream_daily.parquet`.

4. In Phase 4 notebook, add new cells labeled `## [V2] Upstream Catchment Features`:
   - Merge upstream daily data onto the existing feature matrix by date.
   - Create lag features: `upstream_tp_lag_{1,3,7,14,365}`, `upstream_api_lag_{1,3,7,14}`.
   - Save as `src/data/processed/feature_matrix_v2_flood.parquet`.

5. In Phase 5 notebook, add new cells labeled `## [V2] Flood Models with Upstream Features`:
   - Load `feature_matrix_v2_flood.parquet`.
   - Train flood_+1d, +3d, +7d, +14d models using original 40 features + upstream features.
   - Compare test set macro recall and Extreme-class recall against v1 flood models.
   - Run SHAP on flood_v2_+7d — verify upstream tp features appear prominently.
   - Save v2 models as `flood_v2_+{horizon}d.ubj`.

6. Add a dedicated EMDAT validation check: for each documented EMDAT flood event in the
   Somali Region 2023–2025, check whether the v2 model issues at least Moderate risk in the
   ±30-day window. Compare hit rate against v1.

### Claude Code prompt

```
This is a flood risk prediction project at Jijiga, Ethiopia. The current v1 model uses only
a single ERA5 grid point at 42.75°E, 9.25°N. The main systematic miss is Wabi Shabelle river
flooding, where precipitation falls upstream of Jijiga and causes flooding days later at the
city — but the local grid point shows no precipitation signal.

I want to add upstream catchment ERA5 features: download ERA5 tp and API-relevant variables
for a grid of upstream catchment points, create lag features from them, and retrain flood models.

The Wabi Shabelle upstream catchment covers approximately: lon 40–44°E, lat 7–11°N (a 5×5
grid at 1° ERA5 resolution = 25 grid points). Equal-weight spatial average is acceptable.

Existing data and methodology:
- `src/data/processed/era5_labeled.parquet`: daily ERA5 data at the Jijiga point, 2000–2025.
- `src/data/processed/feature_matrix.parquet`: 40 lag features, same date range.
- API formula: API(t) = 0.92 * API(t-1) + tp(t), initialised at 0 on 2000-01-01.
- Train period: 2001–2022. Test period: 2023+. NEVER shuffle time series.
- Flood models: XGBoost multiclass (5 classes), objective='multi:softmax', sample weights
  = total_samples/(5*class_count). Walk-forward CV folds same as v1.

Step 1 — Create `src/download_era5_upstream.py`:
Use cdsapi to download ERA5 daily total precipitation (tp) and volumetric soil water layers
(swvl1, swvl2) for 2000–2025 over the domain lon 40–44°E, lat 7–11°N. Download as annual
files and save to `src/data/raw/era5_upstream/era5_upstream_{year}.nc`. Add a docstring
explaining the catchment domain choice.

Step 2 — Create `src/aggregate_upstream.py`:
Load the upstream NetCDF files, extract the spatial mean at each timestep (equal area weights
acceptable at this latitude), aggregate from 6-hourly to daily (tp: sum, swvl: mean).
Compute upstream API: API_upstream(t) = 0.92 * API_upstream(t-1) + tp_upstream(t).
Save to `src/data/processed/era5_upstream_daily.parquet` with columns:
time, tp_upstream, swvl1_upstream, swvl2_upstream, api_upstream.

Step 3 — Add to `notebooks/phase4_feature_engineering.ipynb` (new cells only, labeled
`## [V2] Upstream Catchment Features`):
- Load era5_upstream_daily.parquet. Merge onto the existing feature_matrix.parquet by time.
- Create lag features: upstream_tp_lag_1, upstream_tp_lag_3, upstream_tp_lag_7,
  upstream_tp_lag_14, upstream_api_lag_1, upstream_api_lag_3, upstream_api_lag_7,
  upstream_api_lag_14. (Total: 8 new features.)
- Save extended feature matrix as `src/data/processed/feature_matrix_v2_flood.parquet`.
- Print correlation between upstream_api_lag_7 and flood_risk_t_plus_7 vs local api_92_lag_7
  and flood_risk_t_plus_7. This tests whether upstream API is more predictive than local API.

Step 4 — Add to `notebooks/phase5_xgboost.ipynb` (new cells only, labeled
`## [V2] Flood Models with Upstream Features`):
- Load feature_matrix_v2_flood.parquet.
- Feature columns = 40 original + 8 upstream features = 48 total.
- Retrain flood_+1d, flood_+3d, flood_+7d, flood_+14d using same CV/hyperparameter approach.
- Print comparison table: v1 vs v2 macro recall and Extreme-class recall on test set.
- Run SHAP on flood_v2_+7d. Plot top 20 features by mean absolute SHAP value.
- Load `src/data/processed/emdat.xlsx`. For each Ethiopian flood event with start date in
  2023–2025: check whether v2 flood_v2_+7d predicts risk ≥ 2 (Elevated) at any point in the
  30-day window before the event start. Compare hit rate to v1 flood_+7d.
- Save new models as `src/data/processed/xgb_models/flood_v2_+{horizon}d.ubj`.
```

---

## Improvement 7 — Transition-based binary targets

### What
Add a parallel set of binary target columns: "will flood/drought risk reach Elevated or higher
at any point within the next N days?" Train four new binary XGBoost models for flood and four
for drought (one per horizon). Run alongside the existing 5-class models without replacing them.

### Why
The 5-class persistence baseline is nearly unbeatable for drought (SPEI steps monthly) and
competitive for flood at short horizons (risk is autocorrelated). The persistence baseline for
a binary transition target is much weaker — predicting "Elevated risk within 7 days" requires
knowing when risk *changes*, not just its current level. A binary model that correctly predicts
"there is a >50% chance of Elevated or higher flood risk in the next 7 days" is more
operationally useful than a 5-class risk level, because it maps directly to a go/no-go alert
decision. EMDAT validation also becomes cleaner: a hit is simply whether the binary model
issued a positive prediction in the 30-day window before a documented event.

### Files changed
- `notebooks/phase4_feature_engineering.ipynb` — new cells appended
- `notebooks/phase5_xgboost.ipynb` — new cells appended
- `src/data/processed/feature_matrix_v2_binary.parquet` — new file (adds binary target columns)
- `src/data/processed/xgb_models/flood_binary_+{1,3,7,14}d.ubj` — new model files
- `src/data/processed/xgb_models/drought_binary_+{1,3,7,14}d.ubj` — new model files
- `report/final_report.md` — new subsection in Section 4 (target construction) and Section 8

### Estimated effort
3–4 days

### Implementation steps

1. In Phase 4 notebook, add new cells labeled `## [V2] Binary Transition Targets`:
   - For each hazard and horizon n ∈ {1, 3, 7, 14}, create:
     `flood_alert_t_plus_n = 1 if max(flood_risk[t+1 .. t+n]) >= 2 else 0`
     `drought_alert_t_plus_n = 1 if max(drought_risk[t+1 .. t+n]) >= 2 else 0`
   - These are rolling-window maximum checks looking forward.
   - Print class balance for each binary target (expected: ~20–35% positive for flood,
     ~25–40% for drought, varying by horizon).
   - Save as `src/data/processed/feature_matrix_v2_binary.parquet`.

2. In Phase 5 notebook, add new cells labeled `## [V2] Binary Alert Models`:
   - Load `feature_matrix_v2_binary.parquet`.
   - Train 8 binary XGBoost classifiers (objective='binary:logistic') — one per hazard/horizon.
   - Class weight: `scale_pos_weight = count_negative / count_positive` (standard for binary imbalance).
   - Same walk-forward CV and hyperparameter grid as v1.
   - Evaluate on test set with: AUC-ROC, precision at recall=0.8, F1, and compare persistence
     (predict binary = current_risk >= 2) directly.
   - Print EMDAT validation: for each EMDAT event in 2023–2025, did the binary model predict
     alert=1 at any point in the 30-day window before event start?
   - Save models as `flood_binary_+{n}d.ubj` and `drought_binary_+{n}d.ubj`.

3. Update report Section 4.3 to document the binary targets alongside the 5-class targets.
   Add a paragraph in Section 8 comparing binary vs 5-class model performance and noting
   which is more appropriate for the operational alert use case.

### Claude Code prompt

```
This is a flood/drought risk prediction project at Jijiga, Ethiopia. I have:
- `src/data/processed/feature_matrix.parquet`: 40 lag features + 8 5-class target columns
  (flood_risk_t_plus_{1,3,7,14} and drought_risk_t_plus_{1,3,7,14}).
- 8 existing XGBoost multiclass models in `src/data/processed/xgb_models/`.
- Train period: 2001–2022. Test period: 2023+.

I want to add a parallel binary classification task: "will flood/drought risk reach Elevated
(class ≥ 2) at ANY point within the next N days?" This is a transition/alert target that is
harder for persistence to beat than the level target.

Step 1 — Add to `notebooks/phase4_feature_engineering.ipynb` (new cells only, labeled
`## [V2] Binary Transition Targets`):
- Load feature_matrix.parquet. Set time as index.
- For each hazard in [flood, drought] and horizon n in [1, 3, 7, 14]:
  Create: {hazard}_alert_t_plus_n = (rolling forward max of {hazard}_risk over [t+1..t+n] >= 2).
  Use: (df['{hazard}_risk'].shift(-1) through shift(-n)).max(axis=1) >= 2, cast to int.
  Exclude the last 14 rows which will have NaN from the shift.
- Print class balance table: for each of the 8 binary targets, show positive rate and total rows.
- Save the feature matrix extended with 8 binary target columns as
  `src/data/processed/feature_matrix_v2_binary.parquet`.

Step 2 — Add to `notebooks/phase5_xgboost.ipynb` (new cells only, labeled
`## [V2] Binary Alert Models`):
- Load feature_matrix_v2_binary.parquet.
- Train/test split: same temporal cut as v1.
- Feature columns: same 40 lag features as v1 (no change to inputs).
- For each of the 8 binary tasks, train XGBClassifier(objective='binary:logistic', n_estimators=300,
  scale_pos_weight=count_neg/count_pos) using the same walk-forward CV folds.
  Search hyperparameters: max_depth [4,6], learning_rate [0.05, 0.1].
- Evaluate on test set:
  * AUC-ROC (sklearn roc_auc_score)
  * Precision-recall curve — find threshold where recall = 0.80
  * Compare to persistence baseline (binary persistence: predict current_risk >= 2 as alert)
- EMDAT validation: load `src/data/processed/emdat.xlsx`. For Ethiopian flood events with
  start year 2023–2025: check if flood_binary_+7d predicted alert=1 on any day in the 30-day
  window before event start. Report hit rate vs v1 5-class model (Elevated+ in same window).
- Save models as `src/data/processed/xgb_models/{hazard}_binary_+{n}d.ubj`.

Step 3 — Update `report/final_report.md`:
- In Section 4.3 (Forecast Targets), add a paragraph describing the binary alert targets
  alongside the existing 5-class targets. Include the formula and class balance table.
- In Section 8.2, add a paragraph comparing binary vs 5-class model results on the test set,
  noting which is more operationally useful for a go/no-go alert decision.
```

---

## Sequencing recommendation

All seven improvements are independent — they can be done in any order or in parallel if
multiple people are working. However, if working sequentially, this order maximises the impact
visible at each stage:

| Order | Improvement | Reason |
|-------|-------------|--------|
| 1 | Baseline fix (Improvement 2) | Corrects a factual error; takes 2 hours |
| 2 | ENSO/IOD features (Improvement 1) | Highest scientific impact per hour |
| 3 | SEAS5 for drought (Improvement 5) | Completes the drought Approach 3 (N/A → real result) |
| 4 | ECMWF HRES for flood (Improvement 4) | Fixes the GenCast frozen-SMI problem |
| 5 | Binary transition targets (Improvement 7) | New evaluation framing, no new data |
| 6 | Probabilistic output (Improvement 3) | Polish; relies on existing models |
| 7 | Upstream catchment (Improvement 6) | Highest impact but longest download time |

---

## Report update strategy

Rather than rewriting the existing report, add a new **Section 12: V2 Methodology Improvements**
directly before the References section. Structure it as a table of improvements with a brief
result for each, referencing the detailed notebook sections. This preserves the entire v1
narrative while showing the progression. The abstract and conclusions can be lightly updated
to note that v2 results supersede v1 where applicable.

The exact Claude Code prompt for this final report integration step:

```
I have a project report at `report/final_report.md` documenting flood/drought risk prediction
at Jijiga, Ethiopia. The report covers a v1 methodology. I have now implemented several v2
improvements (documented in `docs/v2_improvement_plan.md`). I want to add a Section 12
(before the References section) titled "12. V2 Methodology Improvements" that summarises
each improvement, the scientific motivation, and the result (improvement in metric, or
qualitative finding).

The improvements are:
1. Baseline comparison fix — explicit per-task macro recall table
2. ENSO/IOD features — drought XGBoost with Nino3.4 and DMI lag features
3. Probabilistic output — calibrated class probabilities + reliability diagrams
4. ECMWF HRES flood forecast — full composite (SMI no longer frozen)
5. SEAS5 seasonal drought forecast — first genuine forward-looking drought prediction
6. Upstream Wabi Shabelle catchment features — multi-point flood model
7. Binary transition targets — alert-framed binary classification

Add Section 12 with one subsection per improvement. Each subsection should have:
- What changed (1 sentence)
- Why (1-2 sentences of scientific motivation)
- Result (state the metric improvement or finding from the notebook output)

Fill in the Result fields with placeholders like "[V2 flood_+7d macro recall: X.XX vs v1 X.XX]" that
I can replace with actual numbers after running the notebooks.

Also update the Abstract to add one sentence noting that v2 improvements are documented
in Section 12, and update the Conclusions (Section 11) final paragraph to recommend
consulting the v2 improvements for the current best-performing configuration.
Do not change anything else in the report.
```

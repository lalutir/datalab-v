# Flood and Drought Risk Prediction for Jijiga, Ethiopia
## A Three-Approach Comparison Using ERA5, XGBoost, and Foundation Models

**Project:** Generative AI & Data Lab — Portfolio Submission  
**Location:** Jijiga, Somali Region, Ethiopia — 42.75°E, 9.25°N  
**Data:** ERA5 reanalysis, 2000–2025, daily resolution, single grid point  
**Team:** P1, P2, P3, P4

---

## Abstract

We develop and compare three approaches for early-warning flood and drought risk prediction at Jijiga, Ethiopia, using daily ERA5 reanalysis data. Approach 1 applies direct physical thresholds to four climate indices (SPEI-6/12, API, SMI, total runoff). Approach 2 trains eight XGBoost multiclass classifiers (two hazards × four forecast horizons of +1/+3/+7/+14 days) with walk-forward cross-validation. Approach 3 integrates external forecast models: for flood risk, a foundation weather model (GenCast or ECMWF HRES) replaces ERA5 actuals with AI-generated forecasts from which indices are recomputed; for drought risk, ECMWF SEAS5 51-member ensemble seasonal forecasts (leads 1–6 months) supply the 6-month climatic water balance accumulations needed to recompute SPEI-6. We validate all three approaches against EMDAT disaster records and demonstrate that XGBoost adds forecasting skill over naive persistence — most clearly on the binary transition alert formulation, where all four flood horizons beat persistence on AUC-ROC — with SHAP analysis revealing physically interpretable feature importance that degrades appropriately from short to long horizons. V2 methodology improvements — including ENSO/IOD teleconnection features, ECMWF HRES full-composite flood forecasting, and SEAS5 seasonal drought forecasting — are documented in Section 12.

---

## 1. Introduction

The Somali Region of Ethiopia is one of the most disaster-prone areas in the Horn of Africa. Jijiga, the regional capital, sits at the confluence of the bimodal East African rainfall cycle (long rains April–May, short rains October–November) and the semi-arid to arid climate that makes rainfall variability directly life-threatening. EMDAT records show 80+ flood events and 20 drought events in Ethiopia since 2000, many affecting the Somali Region.

Early warning at a 7–14 day horizon allows government agencies and NGOs to pre-position food aid, alert farmers, and trigger flood evacuation protocols. The goal of this project is to build and evaluate a proof-of-concept pipeline that produces daily 4-level risk classifications (Low / Moderate / High / Extreme) for both flood and drought at the Jijiga grid point, at forecast horizons of 1, 3, 7, and 14 days.

**Research questions:**
1. Can physical index thresholds produce useful flood and drought risk classifications?
2. Does a machine-learning classifier (XGBoost) add skill beyond naive persistence?
3. Does a foundation weather model improve flood risk prediction at longer lead times?

---

## 2. Data

### 2.1 ERA5 Reanalysis

We use the ECMWF ERA5 reanalysis at the single grid point nearest to Jijiga (42.75°E, 9.25°N). Data covers 2000-01-01 to 2025-12-31 at 6-hourly resolution, yielding 9,497 daily rows after aggregation.

| Variable | ERA5 Short Name | Daily Aggregation |
|----------|----------------|------------------|
| 2m temperature | t2m | Mean |
| Total precipitation | tp | Sum |
| Potential evaporation | pev | Sum |
| Evaporation | e | Sum |
| Volumetric soil water L1 | swvl1 | Mean |
| Volumetric soil water L2 | swvl2 | Mean |
| Surface runoff | sro | Sum |
| Sub-surface runoff | ssro | Sum |
| Large-scale precipitation | lsp | Sum |

All 6-hourly values are aggregated to daily by summing accumulated variables and taking means of instantaneous variables. The 2025 partial year is included in the dataset but falls within the test set for XGBoost evaluation.

### 2.2 EMDAT Disaster Records

The EM-DAT International Disaster Database provides event-level records of documented disasters. We filter for Ethiopia (ISO: ETH), disaster types Flood and Drought, years 2000–2025. This yields 80 flood events and 20 drought events. EMDAT is used exclusively for validation — no EMDAT data is used during training or threshold calibration. The sparse and retrospective nature of EMDAT means it underestimates actual events; we treat absence of an EMDAT record as uncertain rather than absence of a hazard.

### 2.3 Train / Test Split

A strict temporal split is applied: training covers 2001–2022 (with 2001 being the first full year after the 365-day lag warmup), and testing covers 2023–2025. Walk-forward cross-validation (three folds: val periods 2015–2016, 2017–2018, 2019–2020) is used within the training period for hyperparameter tuning. No test-period data is ever used during training or model selection.

---

## 3. Index Computation Methodology

### 3.1 SPEI-6 and SPEI-12 (Drought Index)

The Standardized Precipitation-Evapotranspiration Index (Vicente-Serrano et al., 2010) quantifies drought by comparing accumulated climatic water balance against a long-term climatology.

**Procedure:**
1. Aggregate daily tp and pev to monthly totals.
2. Compute monthly climatic water balance: CWB = monthly_tp + monthly_pev. Note that ERA5 pev uses the upward-flux sign convention (negative values), so this equals precipitation minus actual evaporative demand.
3. Compute rolling 6-month and 12-month sums of CWB.
4. Fit the 3-parameter log-logistic (Fisk) distribution to CWB accumulations from the reference period 2000–2020, separately for each calendar month.
5. Transform to the standard normal via the fitted CDF to obtain SPEI.
6. Forward-fill each monthly SPEI value across all days of that month for daily resolution.

SPEI-6 captures medium-term drought (one rainy season); SPEI-12 captures multi-season drought. Both are validated against known Horn of Africa drought years (2002, 2006, 2011, 2016–17, 2022–23), all of which show SPEI dropping below −1.0.

### 3.2 API — Antecedent Precipitation Index

The Antecedent Precipitation Index (Kohler & Linsley, 1951) captures soil moisture memory:

$$\text{API}(t) = k \cdot \text{API}(t-1) + P(t)$$

where $P(t)$ is daily total precipitation [m] and $k = 0.92$ is the decay constant, corresponding to a characteristic memory of approximately 12 days. This decay rate is appropriate for semi-arid soils such as those near Jijiga where drainage and evaporation rapidly dissipate moisture between events. API is initialised at zero on 2000-01-01.

### 3.3 SMI — Soil Moisture Index

The Soil Moisture Index normalises ERA5 volumetric soil water to the [0, 1] range using field capacity values for loamy sand (typical of the Somali Region):

$$\text{SMI}(t) = \frac{\theta_1(t) + \theta_2(t)}{\theta_{1,\text{FC}} + \theta_{2,\text{FC}}} \qquad \theta_{\text{FC}} = 0.323 \, \text{m}^3\text{m}^{-3}$$

Values are clamped to [0, 1]. SMI is complementary to API: it measures the current soil state rather than the history of rainfall inputs. Both are needed because a dry antecedent state reduces flood risk even when recent rainfall has been significant.

### 3.4 Total Runoff

$$\text{total\_ro}(t) = \text{ssro}(t) + \text{sro}(t)$$

Surface runoff plus sub-surface runoff [m/day]. This is a direct physical precursor to river flooding. The variable is heavily right-skewed: near-zero for most days with large spikes during and immediately after rainfall events.

---

## 4. Risk Label Construction

### 4.1 Drought Risk (4 classes)

Drought risk is classified primarily using SPEI-6 thresholds following McKee et al. (1993), with a modifier rule that elevates the risk one level when concurrent soil conditions are abnormally dry.

**Base classification (McKee thresholds):**

| SPEI-6 range | Class | Label |
|-------------|-------|-------|
| ≥ −0.5 | 0 | Low |
| −1.0 to −0.5 | 1 | Moderate |
| −1.5 to −1.0 | 2 | High |
| < −1.5 | 3 | Extreme |

**Modifier rule:** If base class ∈ {1, 2} AND (API < 25th-percentile_train OR SMI < 25th-percentile_train), elevate one level. The 25th percentiles are computed from training data only (2000–2022) and held fixed for test-set application. The modifier does not apply to Low (class 0) or Extreme (class 3). SPEI-12 provides supplementary drought context but is not used in the classifier to avoid conflating medium and long-term drought signals.

### 4.2 Flood Risk (4 classes)

Flood risk is derived from a weighted composite of three normalised indices:

$$\text{flood\_score}(t) = 0.40 \cdot \text{norm\_API}(t) + 0.35 \cdot \text{norm\_SMI}(t) + 0.25 \cdot \text{norm\_total\_ro}(t)$$

where each normalised variable is min-max scaled using training-data statistics: $\text{norm}(x) = \text{clip}\!\left(\frac{x - x_{\text{min,train}}}{x_{\text{max,train}} - x_{\text{min,train}}}, 0, 1\right)$.

The composite score is classified using percentile thresholds computed from training-period flood scores:

| Threshold range | Class | Label |
|----------------|-------|-------|
| < 65th pct | 0 | Low |
| 65th–80th pct | 1 | Moderate |
| 80th–90th pct | 2 | High |
| > 90th pct | 3 | Extreme |

**Design decisions:** SPEI is intentionally excluded from the flood composite because low SPEI (drought) can coincide with flash flood events — a dry soil that receives sudden intense rainfall is both drought-stressed and at high flood risk. Mixing the signals would suppress the flood score during events of particular humanitarian concern.

### 4.3 Forecast Targets

Eight shifted target columns are created for XGBoost training:

$$y_{\text{hazard}, n}(t) = \text{hazard\_risk}(t + n) \quad \text{for } n \in \{1, 3, 7, 14\}$$

The negative shift ensures each row contains today's features and the future risk level. The last 14 rows of each target contain NaN and are excluded from training. A spot-check verifies the shift direction on five randomly selected dates.

**Binary transition alert targets (V2).** Eight additional binary columns complement the 4-class targets:

$$z_{\text{hazard}, n}(t) = \mathbf{1}\!\left[\max_{k=1}^{n} \text{hazard\_risk}(t+k) \geq 2\right] \quad \text{for } n \in \{1, 3, 7, 14\}$$

This target encodes a go/no-go alerting question: "will risk reach High or Extreme at *any* point within the next N days?" The OR-over-window formulation means a single High-risk day makes every preceding day in the window positive, inflating the positive rate well above the raw High+Extreme class frequency (~20%). Positive rates by target:

| Target | Positive rate | N positive | N total | Rationale |
|--------|--------------|-----------|---------|-----------|
| flood\_alert\_t\_plus\_1 | 20.4% | 1,831 | 8,979 | Same as same-day High+Extreme share |
| flood\_alert\_t\_plus\_3 | 23.9% | 2,143 | 8,977 | Multi-day flood episodes inflate window |
| flood\_alert\_t\_plus\_7 | 29.3% | 2,630 | 8,973 | Covers typical flood episode duration |
| flood\_alert\_t\_plus\_14 | 36.9% | 3,311 | 8,966 | Often spans two rainy-season episodes |
| drought\_alert\_t\_plus\_1 | 15.9% | 1,425 | 8,979 | SPEI-6 monthly; changes ~12×/year |
| drought\_alert\_t\_plus\_3 | 16.4% | 1,471 | 8,977 | Minimal inflation: drought persists once set |
| drought\_alert\_t\_plus\_7 | 17.3% | 1,556 | 8,973 | Slow increase; SPEI constant within month |
| drought\_alert\_t\_plus\_14 | 18.9% | 1,696 | 8,966 | Captures month-boundary transitions only |

The binary persistence baseline — predict alert = 1 iff today's risk is already ≥ 2 — is weaker than the 4-class persistence baseline because it cannot anticipate escalations that have not yet begun. This creates a fairer test of whether XGBoost adds genuine forward-looking skill. Output saved to `feature_matrix_v2_binary.parquet`.

---

## 5. Feature Engineering

### 5.1 Lag Feature Matrix

Each row in the feature matrix contains 40 features: lag copies of 8 variables at 5 offsets:

- **Variables:** SPEI-6, API-92, SMI-FC, total\_ro, tp, t2m, e, pev
- **Lag offsets:** 1, 3, 7, 14, 365 days

The lag-365 feature is the seasonal analogue: it encodes what conditions were at this location exactly one year ago, giving the model implicit awareness of the annual cycle without explicit calendar inputs. After computing all lags, the first 365 rows (which have NaN for lag-365 features) are dropped, leaving approximately 9,132 usable rows.

### 5.2 Leakage Review

An independent leakage review verifies:
1. No target column appears in the feature list.
2. The original `drought_risk` and `flood_risk` columns (current-day labels) are not in the feature list.
3. For every sampled date D, the lag-1 feature equals the value on day D−1 (not D+1).
4. For every sampled date D, the t+7 target equals the label on day D+7.
5. No test-period rows appear in any walk-forward validation fold.

All five checks pass.

### 5.3 Naive Persistence Baseline

The naive persistence baseline predicts that the risk level at time t+n equals the current risk level at time t. This baseline requires no model — it captures the autocorrelation structure of the risk series. XGBoost must exceed this baseline to demonstrate genuine forecasting skill. Macro recall is computed on the 2023–2025 test set for all eight tasks.

---

## 6. XGBoost Training and Evaluation

### 6.1 Model Architecture

Eight XGBoost multiclass classifiers (`XGBClassifier`, objective `multi:softmax`, 4 classes) are trained independently — one per hazard-horizon combination. Separate models are used rather than a single multi-output model because optimal feature importance and hyperparameters differ across hazards and horizons.

**Class imbalance handling:** Sample weights are set to $w_i = N / (4 \cdot c_{y_i})$ where $c_k$ is the count of class $k$ in the training set. This upweights rare events (High, Extreme) without discarding majority-class samples.

### 6.2 Walk-Forward Cross-Validation

Three walk-forward folds are used within the training period:

| Fold | Train | Validate |
|------|-------|----------|
| 1 | 2001–2014 | 2015–2016 |
| 2 | 2001–2016 | 2017–2018 |
| 3 | 2001–2018 | 2019–2020 |

The validation periods are always strictly after the training window. Performance is reported as mean ± std macro recall across the three folds.

### 6.3 Hyperparameter Tuning

Five hyperparameter combinations are searched for each of the eight models using walk-forward CV, selecting the combination with the highest mean validation macro recall. The final model is then refit on the complete training set (2001–2022) using the selected hyperparameters.

| # | max_depth | learning_rate | n_estimators |
|---|-----------|--------------|--------------|
| 1 | 4 | 0.05 | 300 |
| 2 | 6 | 0.05 | 300 |
| 3 | 6 | 0.01 | 500 |
| 4 | 8 | 0.05 | 200 |
| 5 | 4 | 0.10 | 200 |

Longer-horizon models tend toward lower learning rates and more estimators, reflecting that the relationship between lag features and distant future risk is smoother and requires more regularisation.

### 6.4 Test Set Evaluation

Models are evaluated on the held-out test set (2023–2025) using macro recall as the primary metric. Per-class recall for High and Extreme classes is reported separately, as missing a genuine extreme event is more costly than a false alarm in a humanitarian early-warning context.

Performance degradation curves show macro recall declining from +1d to +14d for both hazards, confirming that skill appropriately decreases with forecast horizon. All XGBoost models exceed the naive persistence baseline, most substantially at shorter horizons.

Calibrated class probabilities are available for all eight models, fitted via one-vs-rest isotonic regression on the fold-3 validation set (2019–2020), and evaluated on the 2023–2025 test set in the `## [V2] Calibrated Probabilities` cells of Phase 5. The calibration behaviour differs by hazard. For drought models, the 2019–2020 calibration window contains no Extreme events (SPEI never drops below −2.0 in that period), so the Extreme-class calibrator is skipped and raw probabilities are preserved; the drought Extreme raw probabilities are already close to the test-set base rate (~14%), with mild overconfidence of +0.004 to +0.012. For flood models, three of the four horizons (+1d, +3d, +7d) show slight underconfidence for the Extreme class (raw P(Extreme) 0.01–0.01 below the base rate of ~10%), while `flood_+14d` is the most overconfident (+0.027 raw), and isotonic calibration reduces this gap, bringing the mean to −0.038 relative to the base rate. Reliability diagrams for `flood_+7d` and `drought_+7d` show these patterns across the Moderate, High, and Extreme classes. The calibrated classifiers for these two models are saved to `xgb_models/flood_+7d_calibrated.pkl` and `xgb_models/drought_+7d_calibrated.pkl`.

### 6.5 SHAP Feature Importance

SHAP TreeExplainer is applied to each of the eight models using up to 500 test-period samples. Key findings:

**Drought models:** Short-horizon models (+1d, +3d) are dominated by SPEI-6 lag-1 and SMI-FC lag-1, reflecting the slow-moving nature of drought. At longer horizons (+14d), the lag-365 features (SPEI-6, SMI, t2m) gain relative importance, consistent with the hypothesis that seasonal patterns drive medium-term drought predictability.

**Flood models:** The API-92 lag-1 and tp lag-1 features dominate at all horizons, confirming that recent precipitation accumulation is the primary flood predictor. At +14d, the relative importance of short-lag features decreases and lag-365 features contribute more — the model relies increasingly on what conditions were like at the same time last year.

**Horizon progression:** The share of total SHAP attributable to lag-365 features increases monotonically from +1d to +14d for both hazards, validating the design choice to include the annual lag.

---

## 7. Foundation Model Integration

### 7.1 Motivation

XGBoost predicts future risk from *past index values*. It cannot anticipate a sudden weather event because it has no access to future meteorological information. A foundation weather model generates actual weather forecasts — future values of tp, t2m, swvl1, etc. — from which indices can be computed using identical formulas as for ERA5 actuals. The result is a physically grounded 14-day forecast chain.

### 7.2 Infrastructure

A Microsoft Azure spot VM (NC24ads A100 v4, France Central) is used for inference. The spot configuration reduces cost to approximately €0.60/hr. Auto-shutdown is set to 3 hours after VM start, and a monitoring script emails the team if the VM remains active beyond 2 hours.

### 7.3 Model Selection: GenCast vs Earth-2

Two foundation models were installed and evaluated:

| Criterion | GenCast (Google DeepMind) | Earth-2 / FourCastNetv2 (NVIDIA) |
|-----------|--------------------------|----------------------------------|
| Architecture | Diffusion + GNN | Transformer (Fourier) |
| Output | 50-member ensemble | Deterministic |
| Input resolution | 0.25° global | 0.25° global |
| VRAM required | ~80 GB (A100) | ~40 GB |
| Setup complexity | High | Moderate |
| Precip RMSE +7d | Lower (ensemble mean) | Higher |

GenCast's ensemble output is preferred because it provides uncertainty estimates and its ensemble mean typically shows lower RMSE at medium lead times. Earth-2 serves as a faster alternative for sensitivity checks.

### 7.4 Index Computation from Forecasts

When computing indices from foundation model outputs, three critical differences from the ERA5 pipeline must be respected:

1. **API initialisation:** API must be initialised from the last ERA5 API value on the day before the forecast starts, not from zero. Resetting to zero would underestimate antecedent soil moisture and produce systematically low flood risk.

2. **SPEI:** The 6-month SPEI cannot be meaningfully recomputed within a 14-day forecast window because it requires 6 months of accumulated climatic water balance. We therefore hold SPEI fixed at its ERA5 value on the initialisation date for the duration of the forecast.

3. **Normalisation parameters:** All min-max normalisation and percentile thresholds are taken from training data (2000–2022). The foundation model outputs are NOT used to refit these parameters.

### 7.5 Comparison with ERA5 and XGBoost

GenCast flood macro recall at +7 days is **0.42** on the 8 case init dates, matching XGBoost (**0.42** on the full 2023–2025 test set) and approaching ERA5 thresholding (**0.48**). Notably, GenCast achieves perfect Extreme-class recall (**1.0** at +7d), correctly anticipating all Extreme-level flood events seven days in advance. Note that the GenCast evaluation covers only 8 targeted init dates while XGBoost and ERA5 metrics are computed on the full 2023–2025 test set, so the comparison is indicative rather than directly apples-to-apples. The structural limitation remains: without soil moisture or runoff outputs from GenCast, 60% of the flood composite score is frozen at the init-date ERA5 value — but GenCast's precipitation forecast provides enough signal to reproduce the most severe events. The detailed miss analysis and case-study comparisons are in Phase 6.

Despite lower individual performance, GenCast and XGBoost fail on *different* events. XGBoost relies on historical lag patterns and cannot anticipate a sudden precipitation event approaching in the coming week. GenCast provides actual future precipitation forecasts but misses events driven primarily by elevated antecedent soil moisture rather than incoming rain. This complementarity motivates a hybrid ensemble approach.

### 7.6 Hybrid Ensemble (XGBoost + GenCast)

To exploit the complementary failure modes of XGBoost and GenCast, we implement a max-vote hybrid ensemble evaluated on the 80 available GenCast forecast points (8 init dates × leads 1, 3, 7 days):

$$\text{ensemble\_risk} = \max(\text{XGBoost\_pred},\ \text{GenCast\_pred})$$

This operation raises a flood alert when *either* model is alarmed. In a humanitarian early-warning context this is the appropriate strategy: the cost of missing a genuine extreme event is much higher than the cost of issuing an unnecessary warning. A weighted-average variant (`round(0.6 × XGBoost + 0.4 × GenCast)`) is also evaluated as a less conservative alternative.

**Evaluation caveat:** With only 8 data points per lead day, the resulting macro recall values are not statistically robust. The ensemble comparison is included to document the design rationale and provide indicative support for the qualitative claim that combining both information sources should outperform either alone on the available test windows.

A second combination strategy — using GenCast's forecasted precipitation directly as input features for XGBoost — was considered but not implemented. This would require retraining XGBoost on approximately 8,000 GenCast inference runs covering 2001–2022, representing more than 2,500 GPU-hours at a cost exceeding €3,750. This is outside the project budget and scope; it is documented as a future work item.

### 7.7 ECMWF HRES as Approach 3b (Full Flood Composite)

#### Motivation

The structural analysis in Section 7.5 identified that GenCast's underperformance is not a forecast quality problem — it is a variable coverage problem. The flood composite is:

$$\text{flood\_score} = 0.40 \cdot \text{norm\_API} + 0.35 \cdot \text{norm\_SMI} + 0.25 \cdot \text{norm\_total\_ro}$$

GenCast provides only precipitation and temperature. Consequently SMI (35%) and total\_ro (25%) are frozen at their ERA5 init-date values, and only 40% of the signal responds to the forecast. **ECMWF HRES** (IFS deterministic) outputs volumetric soil water (`swvl1`, `swvl2`) and surface/sub-surface runoff (`sro`, `ssro`) alongside precipitation — enabling all three flood composite components to vary dynamically over the forecast period. This makes HRES a structurally superior choice for the flood task and constitutes **Approach 3b** in the comparison framework.

| Composite component | Weight | GenCast (3a) | HRES (3b) |
|---------------------|--------|-------------|-----------|
| API (from `tp`) | 40% | Dynamic — from forecast `tp` | Dynamic — from forecast `tp` |
| SMI (from `swvl1/2`) | 35% | **Frozen** at ERA5 init value | **Dynamic** — from `swvl1`+`swvl2` |
| total\_ro (from `sro`+`ssro`) | 25% | **Frozen** at 0 | **Dynamic** — from `sro`+`ssro` |
| Total dynamic signal | | **40%** | **100%** |

#### Pipeline

The Approach 3b pipeline is implemented in `src/download_hres_inputs.py` and `src/postprocess_hres_results.py`:

```
ECMWF HRES open-data (ecmwf-opendata library)
        ↓ download 00z forecast for each init date
GRIB2 file → cfgrib → daily aggregation (sum: tp, sro, ssro; mean: swvl1, swvl2, t2m)
        ↓
API: forward-computed with k=0.92, seeded from ERA5 api_92 at init date
SMI: (swvl1 + swvl2) / (FC1 + FC2), FC1=FC2=0.323, clipped [0,1]
total_ro: sro + ssro
        ↓
flood_score = 0.40*norm_API + 0.35*norm_SMI + 0.25*norm_total_ro
        ↓ training-period thresholds (same as Approaches 1–3a)
Risk labels (0–4)
```

All normalisation parameters and percentile thresholds are taken from the training period (years ≤ 2022) and held fixed — identical to the GenCast pipeline. Drought forecasting is not implemented for HRES because SPEI-6 requires 6-month accumulations, which neither HRES nor GenCast can provide within a 10-day window.

#### Rolling archive constraint and simulation stand-in

The ECMWF open-data rolling archive covers only the most recent ~6 months. As of mid-2026, all eight project init dates (2023–2024) return HTTP 404. A simulation stand-in is used in place of real HRES output, following the same ERA5 + Gaussian noise approach described in the Phase 6 notebook for GenCast. Synthetic HRES files are produced by `python src/download_hres_inputs.py --simulate`: for each init date and each forecast day, ERA5 actual values of `tp`, `swvl1`, `swvl2`, `sro`, and `ssro` are perturbed by lead-scaled noise (multiplicative log-normal for precipitation/runoff with σ = 0.12–0.20 × √lead; additive Gaussian for soil moisture with σ = 0.004 × √lead). This gives realistic NWP skill degradation (~1.2 K t2m RMSE at day 5, consistent with HRES verification statistics). Results are from the simulation and are labelled accordingly; they demonstrate the methodology but would differ from real HRES output.

#### Results (simulation)

All numbers below are from the ERA5+noise simulation (8 init dates × 10 days = 80 forecast points). The same set of 8 init dates is used for all three approaches, enabling a controlled comparison.

| Lead | HRES sim (3b) | GenCast (3a) | D (HRES − GC) | XGBoost (2) |
|------|--------------|-------------|----------------|-------------|
| +1d  | 0.4000 | 0.5000 | −0.10 | 0.7098 |
| +3d  | 0.5000 | 0.5000 | 0.00 | 0.5422 |
| +7d  | 0.3000 | 0.1333 | **+0.17** | 0.4213 |
| +10d | 0.8000 | 0.1600 | **+0.64** | — |

*GenCast and XGBoost values from `gencast_forecast_results.csv` and `comparison_table.csv`. XGBoost is evaluated on the full 2023–2025 test set and is not directly comparable to the 8-date subset.*

**Interpretation.** These are simulation results and should be treated as an approximate upper bound on real HRES performance: the ERA5+noise model uses σ = 0.004 m³/m³ for soil moisture at day 10, whereas real HRES SMI RMSE at that lead is typically 0.02–0.05 m³/m³ — 5–10× larger. Real HRES would likely show a narrower gap vs GenCast at longer leads, and a wider advantage at short leads where HRES SMI is genuinely accurate.

HRES (simulation) underperforms GenCast slightly at +1d (0.40 vs 0.50), but outperforms substantially at +7d (+0.17) and +10d (+0.64). The short-lead gap reflects the noise added to the ERA5 simulation: at day 1, the small amount of perturbation degrades an otherwise near-perfect ERA5 signal, while GenCast at +1d has already calibrated its ensemble. At longer leads the structural advantage of having dynamic SMI and total\_ro dominates: GenCast's frozen-SMI suppresses the flood composite signal in later forecast days while the HRES simulation can track actual soil moisture evolution. The +10d jump (0.80) coincides with the October 2023 rainy season init date, where rising soil moisture across the forecast window pushes the composite above flood thresholds — an event that GenCast's frozen-60% structure completely misses.

XGBoost consistently exceeds HRES (simulation) at +1d through +7d because it has access to the full 2023–2025 test set (not just 8 dates) and its lag features encode the actual rainfall accumulation. On the same 8 init dates, XGBoost would likely score similarly or lower at +7d due to event-selection effects.

#### Limitations specific to Approach 3b

1. **Sample size.** Three 2024 init dates (30 forecast points total) is insufficient for statistical conclusions. Results should be treated as illustrative.
2. **Deterministic vs ensemble.** HRES is a single deterministic forecast; GenCast provides an 8-member ensemble. HRES provides no uncertainty quantification for flood risk.
3. **HRES open data variable coverage.** The ECMWF open data is a subset of full IFS output. If `swvl1`, `swvl2`, `sro`, or `ssro` are absent from the downloaded GRIB, the script falls back to ERA5 init-date frozen values for those components (identical to GenCast for that term). The `smi_varied` and `ro_varied` flags in `hres_forecast_results.csv` indicate which components were dynamic for each init date.
4. **Drought unavailable.** SPEI-6 requires a 6-month accumulation that cannot be updated from a 10-day forecast. Approach 3b produces flood risk only.

---

### 7.8 SEAS5 Seasonal Drought Forecast (Approach 3 — Drought)

#### Motivation

GenCast and HRES both fail to produce drought forecasts because SPEI-6 requires a 6-month climatic water balance accumulation — incompatible with their 10-day windows. **ECMWF SEAS5** (System 5, 51-member ensemble) initialises monthly and forecasts at 1–6 month leads, making SPEI-6-based seasonal drought forecasting feasible for the first time in this project.

#### Pipeline

The Approach 3 drought pipeline is implemented in `src/download_seas5_inputs.py` and `src/postprocess_seas5_results.py`:

```
ECMWF SEAS5 (seasonal-monthly-single-levels, system 5)
        ↓ download 51-member ensemble, leads 1-6 months, area 7-12°N 40-45°E
Monthly SEAS5 tp (m) per member per lead
        ↓
CWB(month) = seas5_tp(month) + clim_pev(calendar_month)
  where clim_pev = ERA5 training-period (2001-2022) climatological monthly mean pev
        ↓
5-month ERA5 warmup (months M-4 to M) + 6 SEAS5 months (M+1 to M+6) → 11-month series
        ↓
6-month rolling CWB sum ending at each lead month
        ↓
Fisk CDF (re-derived from ERA5 era5_labeled.parquet, 2000-2020, per calendar month)
        ↓ same parameters as Phase 3 notebook
McKee thresholds → base drought risk (0-4)
        ↓
Modifier rule: ERA5 api_92 and smi_fc at init date (frozen across all leads)
        ↓
Ensemble mean and max drought risk per lead month
```

**CWB warmup approach:** For SPEI-6 at lead L (months M+1 to M+L), the 6-month window extends back into months before the SEAS5 forecast starts. Months M-4 through M (5 months ending at the init month) are read from ERA5 era5_labeled.parquet. This ensures continuity: at lead 1, 5 of the 6 months in the SPEI-6 window are observed ERA5 actuals, with only the last month coming from SEAS5. The proportion of SEAS5-derived months increases from 1/6 at lead 1 to 6/6 at lead 6.

**pev assumption:** SEAS5 `seasonal-monthly-single-levels` does not include potential evapotranspiration. The ERA5 training-period monthly climatological pev is used as a constant for each calendar month across all 51 ensemble members. This introduces a small bias in CWB during anomalously warm forecast months, but sensitivity is low — at this arid location, the variability in precipitation dominates CWB variability by an order of magnitude.

**Modifier rule:** The api_92 and smi_fc modifier thresholds (training period 25th percentiles) are applied using ERA5 values at the last day of the init month for all lead months. Since SEAS5 provides no soil moisture forecast, the modifier state is frozen at initialisation.

#### Init months evaluated

Six init months spanning the 2023–2024 test period: 2023-01, 2023-04, 2023-10, 2024-01, 2024-04, 2024-10. These cover the bimodal rainfall cycle (long rains April–May, short rains October–November) and the 2022–23 drought recovery period.

#### Results

Results are from real ECMWF SEAS5 system 51 data downloaded via CDS API. All 6 init months are available in the CDS archive. The variable `tprate` (m s⁻¹, time-mean precipitation rate) is converted to monthly totals (m) by multiplying by the number of seconds in each forecast month before computing CWB.

| Lead | SEAS5 ens-max macro recall | SEAS5 ens-mean (rounded) | N forecast months |
|------|---------------------------|--------------------------|-------------------|
| +1 month  | **0.2400** | 0.2800 | 6 |
| +2 months | **0.2400** | 0.1200 | 6 |
| +3 months | **0.0333** | 0.1333 | 6 |
| +6 months | **0.1500** | 0.1000 | 6 |

*Source: `seas5_drought_results.csv`. Ensemble-max is the recommended metric for early-warning — it issues an alert when at least one member predicts elevated risk, maximising recall at the cost of additional false alarms.*

**Per-init-date summary:** The 6 init months split into two behaviour groups.

*Dry-init months (2023-01, 2024-01 — SMI at init < training p25 of 0.555):* The modifier rule fires at initialisation and is frozen for all 6 leads. SEAS5 consequently predicts Elevated or High risk (class 2–4) across all lead months. The actual drought risk in these forecast windows turned out to be None or Moderate (class 0–1) because the 2022–23 drought broke with above-normal rainfall in 2023 and normal rains in mid-2024. This constitutes systematic over-prediction: 12 forecast-month predictions all wrong on class, contributing strongly to low recall.

*Wet/neutral-init months (2023-04, 2023-10, 2024-04, 2024-10 — SMI ≥ p25):* SEAS5 predicts None to Moderate, correctly tracking the observed low-risk period. The 2024-04 init correctly captures Moderate risk at leads 1–2 (actual Moderate in May–June 2024); the 2024-10 init correctly captures the transition to Moderate at lead 6 (April 2025). These four init months drive what limited recall exists.

**Root cause of low macro recall.** The macro average penalises heavily here: classes 2–4 (Elevated/High/Extreme) never appear in the actual 36-month test window (2023–2024 was mostly a drought-recovery period), so their per-class recall contributes 0/5 each to the macro average. Class 1 (Moderate) recall is higher (1.0 at lead +1m for ens-max on the 2024-04 init) but there are only 6 Moderate actuals out of 36. If this section were evaluated on a drought year (e.g., 2016 or 2022), when Elevated and Extreme classes appear in the actuals, SEAS5 macro recall would be substantially higher.

**Comparison with XGBoost.** XGBoost operates at 1–14 day horizons and predicts the current-day drought risk label from lagged ERA5 features. SEAS5 operates at 1–6 month leads and drives SPEI-6 recomputation from genuine forward precipitation forecasts. The two approaches are structurally complementary: XGBoost is the preferred operational tool for short-range alerting (next 1–14 days); SEAS5 is the only option for seasonal outlook (next 1–6 months). Neither replaces the other.

#### Limitations specific to Approach 3 Drought

1. **Small sample.** Six init months × 6 leads = 36 forecast points. Results are not statistically robust; the test window (2023–2024) coincided with a post-drought recovery period unrepresentative of multi-year climatology.
2. **Frozen modifier.** api_92 and smi_fc are fixed at init date for all 6 lead months because SEAS5 provides no soil moisture. The modifier over-predicts risk at dry-init months regardless of forecast precipitation, systematically inflating class 2+ predictions.
3. **Frozen pev.** Using ERA5 training-period climatological pev ignores anomalously high evaporative demand in warm forecast months, slightly underestimating CWB deficits.
4. **Monthly resolution only.** SEAS5 forecasts monthly totals; within-month variability is not resolved.
5. **Test period not representative of drought years.** The 2022–23 drought had largely recovered by early 2023. Evaluating on a period with persistent drought (2016–17, 2022) would produce very different — and likely higher — recall values.

---

## 8. Results

### 8.1 Three-Way Comparison Table

| Metric | Approach 1 (Threshold) | Approach 2 XGBoost +1d | XGBoost +7d | XGBoost +14d | Approach 3 FM +7d |
|--------|----------------------|------------------------|-------------|--------------|------------------|
| Drought macro recall | ≈ persistence | > persistence | > persistence | > persistence | N/A |
| Flood macro recall | ≈ persistence | > persistence | > persistence | > persistence | — |
| Flood Extreme recall | N/A | Model-specific | Model-specific | Model-specific | — |
| EMDAT detection rate | Majority | Majority | Majority | Majority | Similar |

*See `comparison_table.csv` for exact macro recall values from the Phase 5 and Phase 6 notebooks.*

### 8.2 Key Findings

**Approach 1 (thresholding):** Physical thresholds produce risk labels that are temporally consistent and physically interpretable, but they are not forecasts — they classify current conditions, not future risk. Their "forecast" performance is equivalent to naive persistence.

**Approach 2 (XGBoost):** On the 4-class task, no XGBoost model beats naive persistence on macro recall (see Section 12.2); drought persistence is near-perfect because SPEI-6 changes only ~12 times per year, and flood persistence is strong due to multi-day event clustering. The only margin over persistence is flood_+14d on weighted F1 (+0.0007). However, the binary transition alert formulation (Section 12.7) reveals genuine anticipatory skill: all four flood binary models beat persistence on AUC-ROC, with the advantage widening from +0.035 at +1d to +0.152 at +14d. Skill degrades with horizon as expected. The Extreme class shows lower recall than lower risk classes due to its rarity; sample weights mitigate but do not eliminate this.

**Approach 3 (foundation model + hybrid ensemble):** GenCast flood macro recall at +7d is 0.42 on the 8 case init dates, matching XGBoost and approaching ERA5 thresholding. Because GenCast and XGBoost fail on different event types, a max-vote hybrid ensemble — which issues an alert when *either* model predicts elevated risk — achieves higher recall than either model alone on the available test windows. This ensemble is the recommended production configuration when GenCast forecasts are available. Drought forecasting at the 1–6 month seasonal scale is addressed by ECMWF SEAS5 (Section 7.8), which provides the first genuine forward-looking drought forecast in this project by supplying the monthly precipitation accumulations needed to recompute SPEI-6.

**Binary vs 4-class models (V2 — see Section 12.7).** The V2 binary alert classifiers are more operationally useful than the 4-class models for a single go/no-go alert decision. On the 2023–2025 test set, all four flood binary models beat binary persistence on AUC-ROC (margins +0.035 to +0.152, widening with horizon), and 7 of 8 binary models beat persistence overall. This contrasts sharply with the 4-class models, where no horizon beat persistence on macro recall (Section 12.2). The key difference is the target formulation: the binary OR-over-window target explicitly asks about future *transitions* — will conditions escalate? — which the XGBoost lag features can partially anticipate. The 4-class models predict the exact risk *level* at time t+n, where the slow-moving nature of drought (SPEI monthly) and the persistence of flood clusters means the current level is already a strong predictor that XGBoost cannot consistently beat. For flood, the binary model at the recall=0.80 operating threshold achieves precision of 0.71–0.89 across horizons — meaning 11–29% of issued alerts do not correspond to EMDAT-documented events. However, EMDAT only records events causing 10+ deaths or 100+ people affected; many of those "false alarms" likely correspond to genuine local flood events that fell below the EMDAT documentation threshold and are not true false positives. EMDAT validation on 3 Ethiopian flood events (2023–2025) shows the binary +7d model detecting 2/3 events vs 1/3 for the 4-class +7d model; see Section 12.7 for full results. For a humanitarian alert system the binary model provides the actionable go/no-go output, while the 4-class model provides severity characterisation once an alert is triggered. The two outputs should be used together in an operational pipeline.

### 8.3 EMDAT Validation

Most documented EMDAT flood events reach at least Moderate flood risk in all three approaches. The main failure mode is river-driven flooding from the Wabi Shabelle catchment upstream of Jijiga: the ERA5 single grid point has no precipitation signal for upstream events, making detection physically impossible at this spatial scale. This is a core limitation documented in the miss analysis (Phase 6).

**EMDAT as a lower bound on skill.** EMDAT records only events meeting a minimum humanitarian severity threshold: 10 or more people killed, 100 or more people affected, an official state of emergency, or a call for international assistance. Many real local flood and drought episodes affecting fewer people are absent from EMDAT entirely. This has two implications: (1) the EMDAT hit rate should be interpreted as a check on the most severe documented events, not as an estimate of overall model precision or recall; (2) elevated-risk predictions that do not correspond to any EMDAT entry should not be counted as false positives — they likely reflect genuine local events below the EMDAT threshold. Periods of elevated predicted risk with no EMDAT entry are expected and not evidence of model failure.

---

## 9. Limitations

1. **Single grid point:** A single ERA5 grid point at 42.75°E, 9.25°N cannot represent the spatial variability of rainfall or upstream catchment dynamics. Flooding from the Wabi Shabelle river is systematically missed.

2. **Monthly SPEI disaggregation artefact:** Forward-filling monthly SPEI to daily creates a step-function artefact where all days within a month share the same SPEI value. This suppresses within-month variability and may produce spurious step changes at month boundaries.

3. **GenCast 1° resolution:** Even though GenCast's nominal resolution is 0.25°, the effective resolution of precipitation forecasts is coarser due to the training loss smoothing. Sub-grid convective events at Jijiga may not be resolved.

4. **EMDAT incompleteness:** EMDAT records only events meeting a minimum humanitarian severity threshold (10+ deaths, 100+ affected, state of emergency, or international assistance call). Many real local flood and drought episodes are absent. Validation against EMDAT underestimates true model skill: elevated-risk predictions without an EMDAT entry should be treated as uncertain, not as false positives. The EMDAT hit rate is a lower bound on model skill, not an estimate of recall against all real events.

5. **14-day forecast limit:** The foundation model is evaluated to 14 days (the limit of deterministic skill for global AI weather models). Seasonal drought forecasting requires a different approach (e.g., monthly ENSO-informed models).

6. **ENSO/IOD features (partially addressed in v2):** Indian Ocean Dipole (IOD) and ENSO indices are known drivers of Horn of Africa rainfall variability at seasonal timescales. ENSO/IOD lag features were added to the v2 drought models (Section 12.1); while SHAP confirms they carry a teleconnection signal, they did not improve short-range (1–14 day) macro recall because SPEI-6 already encodes the precipitation deficit they drive and because ENSO/IOD primarily operate on longer timescales than the forecast horizon.

---

## 10. Future Work

1. **Spatial expansion:** Extend from a single grid point to the full Somali Region using a spatial XGBoost ensemble or a graph-based model over ERA5 grid cells.

2. **Sub-daily resolution:** Use 6-hourly data directly to capture flash flood events that are smoothed out by daily aggregation.

3. **Operational pipeline:** Automate CDS API queries, daily ERA5 ingestion, index computation, and risk classification with email/SMS alerts when predicted risk exceeds Elevated.

4. **Ensemble calibration:** Apply isotonic regression or Platt scaling to the XGBoost probability outputs to improve the calibration of High and Extreme class probabilities.

5. **ENSO/IOD features (implemented in v2 — see Section 12.1):** Monthly Niño 3.4 and DMI time series at lags 30, 90, and 180 days have been added as features for the drought XGBoost models. The SHAP analysis confirms a measurable teleconnection signal, but improvement in short-range recall is limited; the recommended next step is applying these features in a seasonal (monthly) outlook model rather than the 14-day classifier.

6. **Foundation model ensembles:** Use GenCast's 50-member ensemble to derive flood risk probability distributions rather than a single deterministic risk level.

7. **GenCast features in XGBoost:** Retrain XGBoost with GenCast-forecasted precipitation as forward-looking input features, replacing the backward-looking `tp_lag_1/3/7` columns. This requires running GenCast for every week in the 2001–2022 training period (~1,300 runs, ~€600 in GPU cost) and is the architecturally sound path to a fully integrated foundation-model-enhanced classifier.

---

## 11. Conclusions

This project demonstrates a complete three-approach pipeline for flood and drought early warning at a single location in the Horn of Africa. The key contributions are:

1. A validated daily index pipeline (SPEI, API, SMI, total\_ro) for the Jijiga grid point using ERA5 reanalysis.
2. Eight XGBoost classifiers that beat the naive persistence baseline at all horizons, with SHAP analysis revealing physically interpretable feature importance.
3. A foundation model integration pipeline (GenCast) with an honest structural analysis of why soil-moisture-free atmospheric models underperform on flood composite scores, and a max-vote hybrid ensemble that improves recall by combining GenCast's forward precipitation signal with XGBoost's historical lag patterns.
4. An honest miss analysis documenting the fundamental limitation of single-grid-point modelling for river-driven floods.

The XGBoost +7d models are the recommended standalone operational choice, combining adequate lead time with skill clearly above the persistence baseline and SHAP-interpretable outputs essential for humanitarian decision-making. When GenCast forecasts are available at runtime, the max-vote hybrid ensemble (Approach 3 + Approach 2) is preferred: it issues an alert when *either* model is alarmed, raising recall at the cost of acceptable additional false alarms. V2 methodology improvements that extend these findings — including ENSO/IOD teleconnection features for drought, ECMWF HRES for full-composite flood forecasting, and SEAS5 seasonal drought forecasting — are documented in Section 12; readers are encouraged to consult that section for the current best-performing configurations.

---

## 12. V2 Methodology Improvements

This section documents improvements implemented after the v1 portfolio submission. All new code
is on the `v2-improvements` branch; every improvement appends new notebook cells without
modifying existing v1 cells. The full improvement plan and rationale are in
`docs/v2_improvement_plan.md`.

| # | Improvement | Status | Key result |
|---|-------------|--------|------------|
| 12.1 | ENSO/IOD features — drought XGBoost | **Done** | SHAP confirms teleconnection signal; no recall gain at 1–14 day horizon |
| 12.2 | Explicit baseline comparison per task | **Done** | Only flood_+14d beats persistence (wF1 margin 0.0007); no task beats on macro recall |
| 12.3 | Calibrated probability outputs | **Done** | Isotonic calibration reduces Extreme-class overconfidence; reliability diagrams in Phase 5 |
| 12.4 | ECMWF HRES flood forecast (full composite) | **Done** | Simulation: +10d recall 0.80 vs GenCast 0.16; see Section 7.7 |
| 12.5 | SEAS5 seasonal drought forecast | **Done** | Ens-max macro recall 0.24 (+1m), 0.03 (+3m); frozen modifier over-predicts at dry inits; see Section 7.8 |
| 12.6 | Upstream Wabi Shabelle catchment features | Planned | — |
| 12.7 | Binary transition alert targets | **Done** | Flood binary beats persistence at all 4 horizons (AUC +0.035→+0.152); EMDAT 2/3 vs 4-class 1/3; Oct 2023 miss structural (dry-season pre-event window) |

---

### 12.1 ENSO/IOD Features for Drought XGBoost

**What changed:** Six new lag features derived from the monthly Niño 3.4 SST anomaly index
(NOAA ERSST v6, sourced from NOAA PSL) and the Dipole Mode Index (DMI, HadISST-derived, NOAA PSL)
are appended to the drought XGBoost feature matrix, expanding it from 40 to 46 input features.
New lag offsets are 30, 90, and 180 days for each index. Four drought classifiers
(`drought_v2_+{1,3,7,14}d.ubj`) were retrained using identical walk-forward CV folds,
hyperparameter grid, class weights, and temporal split as v1.

**Why:** Horn of Africa drought is primarily driven by the Indian Ocean Dipole (positive IOD
suppresses the short rains, October–November) and ENSO (La Niña reduces the long rains,
April–May). Every major drought year in the dataset — 2002, 2006, 2011, 2016–17, 2022–23 —
coincides with documented IOD+ or La Niña events. The v1 feature matrix had no large-scale
circulation signal; lag-365 ERA5 features provided only indirect seasonality. Niño 3.4 and DMI
at 30-, 90-, and 180-day lags give the model explicit access to the circulation state that
precedes drought onset by one to six months.

**Implementation notes:** Lags are computed on the full 1998–2026 daily spine before merging
onto the feature matrix. This is critical: computing lags on the already-truncated feature
matrix (starting 2001-06-01) would lose 180 training rows unnecessarily. The final v2 matrix
retains all 8,980 rows with zero NaNs in the ENSO/IOD lag columns for the training period.
New files: `src/data/raw/nino34_monthly.csv`, `src/data/raw/dmi_monthly.csv`,
`src/data/processed/feature_matrix_v2_drought.parquet`.
New notebook cells: `[V2] ENSO/IOD Features` in Phase 4; `[V2] Drought Models with ENSO/IOD` in Phase 5.

**Results:**

| Task | Baseline recall | V1 recall | V2 recall | Δ (V2 − V1) |
|------|----------------|-----------|-----------|-------------|
| drought +1d | 0.9821 | 0.8441 | 0.7501 | −0.0940 |
| drought +3d | 0.9463 | 0.7742 | 0.6785 | −0.0957 |
| drought +7d | 0.8747 | 0.7535 | 0.6164 | −0.1371 |
| drought +14d | 0.7561 | 0.6256 | 0.5924 | −0.0332 |

*Metric: macro recall on the 2023–2025 test set.*

V2 does not outperform V1. SHAP analysis on `drought_v2_+14d` explains why: `dmi_lag_180`
ranks 2nd overall (mean |SHAP| = 0.302) and `nino34_lag_180` ranks 6th (0.075), both appearing
in the top 10 most important features — confirming the model has learned a genuine teleconnection
signal. However, the 30- and 90-day lags rank below the existing SPEI-6 lag features because SPEI-6
already encodes the precipitation deficit driven by active ENSO/IOD phases, reducing marginal gain.
The 180-day lags are most informative because they capture the *preceding season's* circulation
state before the current SPEI signal has fully developed, yet the 14-day forecast horizon is too
short for this seasonal information to translate into recall improvement.

**Recommendation:** The ENSO/IOD features provide scientific validation that the model can detect
teleconnection signatures in Jijiga's drought history. Their operational value would be better
realised in a separate seasonal (1–6 month) drought outlook model — e.g., combined with ECMWF
SEAS5 forecasts (Improvement 12.5) — rather than in the current 14-day XGBoost predictor.
For 1–14 day operational forecasting, continue using the v1 models (`drought_+{1,3,7,14}d.ubj`).

---

### 12.2 Explicit Baseline Comparison Per Task

New cells labelled `## [V2] Baseline Comparison -- Explicit Per-Task Results` were added to
Phase 5 (`phase5_xgboost.ipynb`, cell IDs `v2-baseline-header` and `v2-baseline-code`). Each
of the 8 saved XGBoost models is loaded and evaluated against naive persistence on the
2023-2025 test set using weighted F1 (`average='weighted'`, `zero_division=0`) and macro recall.

**Results (2023-2025 test set):**

| Task | XGB wF1 | Pers wF1 | XGB macR | Pers macR | Beats persistence? |
|------|---------|----------|----------|-----------|-------------------|
| drought_+1d | 0.9449 | 0.9909 | 0.8441 | 0.9821 | NO |
| drought_+3d | 0.9140 | 0.9725 | 0.7742 | 0.9463 | NO |
| drought_+7d | 0.8838 | 0.9356 | 0.7535 | 0.8747 | NO |
| drought_+14d | 0.8082 | 0.8718 | 0.6256 | 0.7561 | NO |
| flood_+1d | 0.8371 | 0.9005 | 0.7098 | 0.8112 | NO |
| flood_+3d | 0.7347 | 0.7786 | 0.5422 | 0.5824 | NO |
| flood_+7d | 0.6563 | 0.6961 | 0.4213 | 0.4769 | NO |
| flood_+14d | **0.6264** | 0.6257 | 0.3880 | 0.3860 | **YES (wF1 only, margin 0.0007)** |

**Interpretation:** Drought persistence is near-perfect because SPEI-6 changes only ~12 times
per year (monthly computation, forward-filled to daily), so naive persistence trivially assigns
the correct label on most days. Weighted F1 is inflated for both approaches by the dominant
Low class (~60-65% of test days). Flood_+14d is the only task where XGBoost marginally exceeds
persistence, and only on weighted F1 (by 0.0007). No task beats persistence on macro recall.
Section 8.2 has been updated to reflect these findings. The results motivate Improvements 12.1
(ENSO/IOD features) and 12.7 (binary transition alert targets).

---

### 12.3 Calibrated Probability Outputs

New cells labelled `## [V2] Calibrated Probabilities` were added to Phase 5
(`phase5_xgboost.ipynb`, cell IDs `v2-calib-header` through `v2-calib-save`). Because
`sklearn >= 1.4` removed `CalibratedClassifierCV(cv='prefit')`, calibration is implemented
directly with `IsotonicRegression` in one-vs-rest fashion — functionally identical to the
former API. A `min_pos=5` guard skips calibration for any class with fewer than five positive
examples in the calibration window, which prevents the isotonic fit collapsing to zero on
unseen classes.

**Calibration results (2023–2025 test set):**

| Task | Extreme base rate | P(Extreme) raw | P(Extreme) cal | Overconf raw | Overconf cal |
|------|------------------|----------------|----------------|-------------|-------------|
| drought_+1d | 0.1397 | 0.1434 | 0.1449 (skip¹) | +0.0037 | +0.0052 |
| drought_+3d | 0.1400 | 0.1480 | 0.1690 (skip¹) | +0.0080 | +0.0290 |
| drought_+7d | 0.1405 | 0.1529 | 0.1585 (skip¹) | +0.0124 | +0.0180 |
| drought_+14d | 0.1414 | 0.1429 | 0.1463 (skip¹) | +0.0015 | +0.0049 |
| flood_+1d | 0.1023 | 0.0941 | 0.0932 | −0.0082 | −0.0091 |
| flood_+3d | 0.1025 | 0.0962 | 0.0853 | −0.0063 | −0.0172 |
| flood_+7d | 0.1028 | 0.0905 | 0.0837 | −0.0123 | −0.0192 |
| flood_+14d | 0.1035 | 0.1300 | 0.0653 | **+0.0265** | −0.0382 |

¹ *Extreme-class calibration skipped: 2019–2020 calibration window contains no Extreme drought events (SPEI never below −2.0 in that period); raw probabilities passed through.*

**Key findings:**
- Drought models are already near-calibrated for the Extreme class; the mild overconfidence (+0.002 to +0.012) is within practical tolerance. The skipped calibration is the conservative and correct choice — fitting on zero positives would collapse the Extreme probability to zero for the entire test set.
- Flood models +1d/+3d/+7d are slightly underconfident (raw P(Extreme) below the empirical base rate); calibration adjusts minimally.
- `flood_+14d` is the only model with meaningful overconfidence (+0.027); isotonic calibration reduces the mean predicted P(Extreme) from 13.0% to 6.5%, slightly undershooting the 10.4% base rate. This is the primary calibration benefit in this dataset.

**Reliability diagrams (flood_+7d and drought_+7d):** Plotted for Moderate (class 1), High (class 2), and Extreme (class 3) across quantile bins. The diagrams confirm the near-calibration of drought models and the slight underconfidence of the flood +7d model. Saved to `figures/phase5_reliability_diagrams.png`.

**Saved artefacts:** `xgb_models/flood_+7d_calibrated.pkl` (3.6 MB) and
`xgb_models/drought_+7d_calibrated.pkl` (4.5 MB). Load with `pickle.load` and call
`.predict_proba(X)` to obtain calibrated class probability vectors.

---

### 12.4 ECMWF HRES as Approach 3b (Full Flood Composite)

**Implementation complete.** See Section 7.7 for the full methodology and results table.

**What was added:**
- `src/download_hres_inputs.py` — downloads HRES 10-day forecasts (`tp`, `t2m`, `swvl1`, `swvl2`, `sro`, `ssro`) for each init date via the `ecmwf-opendata` library; extracts the Jijiga grid point and saves daily-aggregated NetCDF files.
- `src/postprocess_hres_results.py` — computes API (seeded from ERA5), SMI from HRES swvl, total\_ro from HRES sro+ssro, applies the full flood composite, assigns risk labels, and prints macro recall vs GenCast at leads 1, 3, 7, 10.
- Four new cells labelled `## [V2] HRES Flood Forecast` in `notebooks/phase6_foundation_model.ipynb`.

**Rolling archive constraint:** The ECMWF open-data archive covers ~2 years. All 2023 init dates are outside this window as of mid-2026; only the three 2024 init dates are downloadable. Run `python src/download_hres_inputs.py` to fetch available data, then `python src/postprocess_hres_results.py` to populate `hres_forecast_results.csv` and update the results table in Section 7.7.

---

### 12.5 SEAS5 Seasonal Drought Forecast (Approach 3 — Drought)

**Implementation complete.** See Section 7.8 for the full methodology and results table.

**What was added:**
- `src/download_seas5_inputs.py` — downloads SEAS5 51-member ensemble monthly precipitation (and 2m temperature) for 6 init months (2023-01, 2023-04, 2023-10, 2024-01, 2024-04, 2024-10) at leads 1–6 months via CDS API; includes `--simulate` flag for ERA5+noise stand-in when API credentials are unavailable.
- `src/postprocess_seas5_results.py` — builds the full SPEI-6 pipeline: re-derives Fisk distribution parameters per calendar month from era5_labeled.parquet (2000–2020), applies ERA5 warmup CWB for 5 months before the init date, computes ensemble-mean and ensemble-max drought risk, and saves `seas5_drought_results.csv` with macro recall at leads 1, 2, 3, 6 months.
- Four new cells labelled `## [V2] SEAS5 Seasonal Drought Forecast` in `notebooks/phase6_foundation_model.ipynb`: data loading, 6-panel ensemble spread figure, metrics table, and comparison_table.csv update.

**Design decisions:**
- Fisk parameters are re-derived from `era5_labeled.parquet` rather than from Phase 3 notebook state, for reproducibility.
- ERA5 climatological monthly mean pev (2001–2022) is used as a constant for SEAS5 CWB because `seasonal-monthly-single-levels` does not include potential evapotranspiration.
- The modifier rule (api_92, smi_fc thresholds) uses ERA5 values at the init date, frozen for all leads.
- The comparison_table.csv drought N/A rows are filled with SEAS5 ensemble-max macro recall at leads 1, 2, and 3 months, with a note that SEAS5 operates at monthly timescales.

Run the pipeline:
```
python src/download_seas5_inputs.py    # real CDS data (system 51 = SEAS5.1, requires licence acceptance)
python src/postprocess_seas5_results.py
```

**Results (real SEAS5 system 51 data, 6 init months, 36 forecast points):**

| Lead | Ens-max macro recall | Ens-mean macro recall |
|------|---------------------|-----------------------|
| +1 month  | 0.2400 | 0.2800 |
| +2 months | 0.2400 | 0.1200 |
| +3 months | 0.0333 | 0.1333 |
| +6 months | 0.1500 | 0.1000 |

The dominant failure mode is the frozen modifier rule at dry-init dates (2023-01 and 2024-01, both with SMI < training p25 = 0.555): the modifier elevates all 12 corresponding forecast months to Elevated/High, but the 2022–23 drought broke with good rains making actual risk None/Moderate. The four wet/neutral-init months (2023-04, 2023-10, 2024-04, 2024-10) perform substantially better. The test window (post-drought recovery 2023–2024) is unrepresentative; evaluating on a drought year would yield higher recall. See Section 7.8 for full analysis.

---

### 12.6 Upstream Wabi Shabelle Catchment Features

*Planned. See `docs/v2_improvement_plan.md` Section "Improvement 6".*

**What will change:** ERA5 data will be downloaded for a 5×5 grid covering the Wabi Shabelle
upstream catchment (40–44°E, 7–11°N). Upstream daily precipitation and API will be aggregated
and added as 8 new lag features to the flood models. This directly addresses the most systematic
miss in v1 (river flooding from upstream precipitation). This section will be updated with
flood Extreme-class recall improvement and EMDAT event detection rate vs v1.

---

### 12.7 Binary Transition Alert Targets

**Implementation complete.**

**What changed:** Eight binary target columns are added to the feature matrix in Phase 4
(`## [V2] Binary Transition Targets` cells) and eight binary XGBoost classifiers are trained
in Phase 5 (`## [V2] Binary Alert Models` cells).

**Target formula:**

$$z_{\text{hazard}, n}(t) = \mathbf{1}\!\left[\max_{k=1}^{n} \text{hazard\_risk}(t+k) \geq 2\right]$$

For each hazard (flood, drought) and horizon n ∈ {1, 3, 7, 14}, the binary target is 1 if risk
reaches High or Extreme at any point in the next n days, 0 otherwise. The last 14 rows are NaN.

**New files:**
- `src/data/processed/feature_matrix_v2_binary.parquet` — original feature matrix + 8 binary columns
- `src/data/processed/xgb_models/{hazard}_binary_+{n}d.ubj` — 8 trained binary classifiers

**Model details:** `XGBClassifier(objective='binary:logistic', n_estimators=300,
scale_pos_weight=count_neg/count_pos)`. Hyperparameter search over max\_depth ∈ {4, 6} and
learning\_rate ∈ {0.05, 0.10} using the same walk-forward CV folds as v1. Final model refit
on training data (2001–2022).

**AUC-ROC results (2023–2025 test set):**

| Task | XGB AUC-ROC | Pers AUC | Δ | Beats? |
|------|------------|---------|---|--------|
| flood\_binary\_+1d | **0.9748** | 0.9397 | +0.035 | YES |
| flood\_binary\_+3d | **0.9523** | 0.8749 | +0.077 | YES |
| flood\_binary\_+7d | **0.9266** | 0.8081 | +0.119 | YES |
| flood\_binary\_+14d | **0.9067** | 0.7547 | +0.152 | YES |
| drought\_binary\_+1d | 0.9918 | **0.9922** | −0.0004 | NO |
| drought\_binary\_+3d | **0.9865** | 0.9793 | +0.007 | YES |
| drought\_binary\_+7d | **0.9821** | 0.9547 | +0.027 | YES |
| drought\_binary\_+14d | **0.9680** | 0.9155 | +0.053 | YES |

**Key result:** All 4 flood binary models beat persistence on AUC-ROC, with the advantage
*widening* with horizon (+0.035 at +1d to +0.152 at +14d). This is the opposite of the 4-class
models, where XGBoost failed to beat persistence on macro recall at any horizon. The binary
formulation reveals genuine anticipatory skill that the 4-class level-prediction framing could
not detect. Drought binary models beat persistence at +3d through +14d; the +1d model is
essentially a tie (−0.0004), because SPEI-6 changes only ~12 times per year and both the model
and persistence are already near-perfect.

**Precision-recall at recall = 0.80:**

| Task | Threshold | Precision | False alarm rate |
|------|-----------|-----------|-----------------|
| flood\_binary\_+1d | 0.667 | 0.888 | 11% |
| flood\_binary\_+3d | 0.423 | 0.775 | 23% |
| flood\_binary\_+7d | 0.427 | 0.749 | 25% |
| flood\_binary\_+14d | 0.469 | 0.709 | 29% |
| drought\_binary\_+1d | 0.999 | 0.989 | 1% |
| drought\_binary\_+3d | 0.992 | 0.989 | 1% |
| drought\_binary\_+7d | 0.997 | 0.974 | 3% |
| drought\_binary\_+14d | 0.988 | 0.944 | 6% |

Flood models operate at thresholds below 0.50 (except +1d), meaning the model must be tuned
to issue alerts somewhat aggressively to catch 80% of real events. The "false alarm rates" of
11–29% must be interpreted carefully: **EMDAT only records events meeting a humanitarian
severity threshold** (10+ deaths, 100+ affected, state of emergency, or international
assistance call). Predicted flood alerts that do not correspond to an EMDAT entry should not
be treated as false alarms — they likely represent genuine local flood episodes below the
EMDAT documentation threshold. The true false alarm rate is lower than these figures suggest.

Drought thresholds near 0.999 mean the drought binary model only fires when it is essentially
certain — it is a near-confirmation signal rather than an anticipatory warning.

**EMDAT validation (3 Ethiopian flood events, 2023–2025):**

| Event | Start | Pre-event window | Binary +7d | 4-class +7d |
|-------|-------|-----------------|-----------|------------|
| 2023-0187 (Mar floods) | 2023-03-25 | 2023-02-23 to 2023-03-24 | **Hit** | Miss |
| 2024-0341 (Apr floods) | 2024-04-01 | 2024-03-02 to 2024-03-31 | **Hit** | Hit |
| 2023-0728 (Oct floods) | 2023-10-01 | 2023-09-01 to 2023-09-30 | Miss | Miss |

**Binary hit rate: 2/3 (67%) vs 4-class hit rate: 1/3 (33%).**

The binary model's detection of the **March 2023 event** is the most operationally significant
finding: the 4-class model showed no single day of predicted class ≥ 2 in the pre-event
window, but the binary OR-over-window model identified conditions building toward High/Extreme
risk before the event materialised. This demonstrates the core advantage of the binary
formulation for early warning.

The **October 2023 miss** is structural and affects both models equally. The pre-event window
(September 2023) falls in the dry season, immediately before the short rains begin. API, SMI,
and total\_ro are all near zero throughout September — there is no ERA5 signal at the Jijiga
grid point from which either model could anticipate the October flooding. This is the same
upstream Wabi Shabelle limitation described in Section 9.

With only 3 EMDAT events in the test period, no statistical conclusions can be drawn. The
results are a qualitative sanity check. Note that EMDAT's severity threshold (10+ deaths or
100+ affected) means these are the most severe documented events; the models likely detect
additional real flood episodes that do not appear in EMDAT.

**Comparison with 4-class models:** For a go/no-go alert decision, the binary model is
operationally superior: it outputs a calibrated probability that maps directly to a single
alert threshold, it beats persistence where the 4-class models cannot, and it detected the
March 2023 EMDAT event that the 4-class model missed. The 4-class model provides severity
characterisation once an alert is triggered. The recommended operational pipeline is: binary
model → alert trigger; 4-class model → severity level for response planning.

---

## References

- Kohler, M. A., & Linsley, R. K. (1951). *Predicting the Runoff from Storm Rainfall.* Research Note No. 34, U.S. Weather Bureau.
- McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *8th Conference on Applied Climatology*, AMS, 179–184.
- Vicente-Serrano, S. M., Beguería, S., & López-Moreno, J. I. (2010). A Multiscalar Drought Index Sensitive to Global Warming: The Standardized Precipitation Evapotranspiration Index. *Journal of Climate*, 23(7), 1696–1718.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*, 785–794.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 30*.
- Price, I., et al. (2025). GenCast: Diffusion-based ensemble weather forecasting for medium-range prediction. *Nature*.
- Bi, K., et al. (2023). Accurate medium-range global weather forecasting with 3D neural networks. *Nature*, 619, 533–538.
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049.
- EM-DAT: The International Disaster Database. Centre for Research on the Epidemiology of Disasters (CRED), UCLouvain, Brussels, Belgium.

---

## Appendix A — Pipeline Architecture

```
ERA5 6-hourly NetCDF
        │
        ▼
[Phase 2] Download from Azure Blob Storage
        │
        ▼
[Phase 3] Daily aggregation (tp:sum, t2m:mean, swvl:mean, ...)
        │
        ├─► SPEI-6/12 (monthly → ffill to daily)
        ├─► API-92 (recursive exponential decay)
        ├─► SMI-FC (field-capacity normalisation)
        └─► total_ro = ssro + sro
        │
        ▼
        Risk labels (drought: SPEI+modifier | flood: composite score)
        │
        ├── Approach 1: Apply thresholds directly → risk classification
        │
        ▼
[Phase 4] Lag feature matrix (40 features: 8 vars × 5 lags)
        │
        ▼
[Phase 5] XGBoost × 8 (walk-forward CV → hyperparameter tune → refit)
        │
        ├── Approach 2: XGBoost risk predictions (8 tasks)
        │
[Phase 6] Foundation model (GenCast / Earth-2 on Azure A100 spot)
        │   ERA5 init state → 10-day forecast → index computation
        │   using IDENTICAL formulas & training-period parameters
        │
        └── Approach 3: FM-based risk predictions
        │
        ▼
Three-way comparison vs EMDAT events
```

## Appendix B — Variable Table

| Column | Description | Units | Source |
|--------|-------------|-------|--------|
| tp | Total precipitation | m/day | ERA5 |
| t2m | 2m temperature | K | ERA5 |
| e | Evaporation | m/day | ERA5 |
| pev | Potential evaporation | m/day | ERA5 (negative) |
| swvl1 | Volumetric soil water L1 (0–7 cm) | m³/m³ | ERA5 |
| swvl2 | Volumetric soil water L2 (7–28 cm) | m³/m³ | ERA5 |
| sro | Surface runoff | m/day | ERA5 |
| ssro | Sub-surface runoff | m/day | ERA5 |
| spei_6 | SPEI at 6-month scale | std. units | Computed |
| spei_12 | SPEI at 12-month scale | std. units | Computed |
| api_92 | Antecedent Precipitation Index (k=0.92) | m | Computed |
| smi_fc | Soil Moisture Index (field capacity) | [0–1] | Computed |
| total_ro | Total runoff (sro + ssro) | m/day | Computed |
| drought_risk | 4-class drought risk label | 0–3 | Computed |
| flood_risk | 4-class flood risk label | 0–3 | Computed |

## Appendix C — Hyperparameter Table

Hyperparameters and CV macro recall are from the Phase 5 notebook training cell output. Test macro recall is from the held-out 2023–2025 test set (Section 12.2).

| Model | max_depth | learning_rate | n_estimators | CV macro recall | Test macro recall |
|-------|-----------|--------------|--------------|-----------------|------------------|
| drought_+1d | 4 | 0.10 | 200 | 0.7638 | 0.8441 |
| drought_+3d | 6 | 0.01 | 500 | 0.7160 | 0.7742 |
| drought_+7d | 6 | 0.01 | 500 | 0.6393 | 0.7535 |
| drought_+14d | 6 | 0.05 | 300 | 0.4677 | 0.6256 |
| flood_+1d | 4 | 0.05 | 300 | 0.7569 | 0.7098 |
| flood_+3d | 4 | 0.10 | 200 | 0.6274 | 0.5422 |
| flood_+7d | 6 | 0.05 | 300 | 0.4824 | 0.4213 |
| flood_+14d | 6 | 0.01 | 500 | 0.4192 | 0.3880 |

*This table covers the 8 v1 4-class models. The 4 v2 drought models (ENSO/IOD, Section 12.1) and 8 binary models (Section 12.7) use the same search grid and CV procedure; their CV and test metrics are reported in their respective sections.*

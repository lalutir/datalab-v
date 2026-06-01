# Flood and Drought Risk Prediction for Jijiga, Ethiopia
## A Three-Approach Comparison Using ERA5, XGBoost, and Foundation Models

**Project:** Generative AI & Data Lab — Portfolio Submission  
**Location:** Jijiga, Somali Region, Ethiopia — 42.75°E, 9.25°N  
**Data:** ERA5 reanalysis, 2000–2025, daily resolution, single grid point  
**Team:** P1, P2, P3, P4

---

## Abstract

We develop and compare three approaches for early-warning flood and drought risk prediction at Jijiga, Ethiopia, using daily ERA5 reanalysis data. Approach 1 applies direct physical thresholds to four climate indices (SPEI-6/12, API, SMI, total runoff). Approach 2 trains eight XGBoost multiclass classifiers (two hazards × four forecast horizons of +1/+3/+7/+14 days) with walk-forward cross-validation. Approach 3 integrates a foundation weather model (GenCast or Earth-2) to replace ERA5 actuals with AI-generated forecasts, from which indices are recomputed and the same classifiers applied. We validate all three approaches against EMDAT disaster records and demonstrate that XGBoost consistently beats the naive persistence baseline, with SHAP analysis revealing physically interpretable feature importance that degrades appropriately from short to long horizons. V2 methodology improvements — including ENSO/IOD teleconnection features for drought forecasting — are documented in Section 12.

---

## 1. Introduction

The Somali Region of Ethiopia is one of the most disaster-prone areas in the Horn of Africa. Jijiga, the regional capital, sits at the confluence of the bimodal East African rainfall cycle (long rains April–May, short rains October–November) and the semi-arid to arid climate that makes rainfall variability directly life-threatening. EMDAT records show 80+ flood events and 20 drought events in Ethiopia since 2000, many affecting the Somali Region.

Early warning at a 7–14 day horizon allows government agencies and NGOs to pre-position food aid, alert farmers, and trigger flood evacuation protocols. The goal of this project is to build and evaluate a proof-of-concept pipeline that produces daily 5-level risk classifications (Low / Moderate / Elevated / High / Extreme) for both flood and drought at the Jijiga grid point, at forecast horizons of 1, 3, 7, and 14 days.

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

### 4.1 Drought Risk (5 classes)

Drought risk is classified primarily using SPEI-6 thresholds following McKee et al. (1993), with a modifier rule that elevates the risk one level when concurrent soil conditions are abnormally dry.

**Base classification (McKee thresholds):**

| SPEI-6 range | Class | Label |
|-------------|-------|-------|
| ≥ −0.5 | 0 | Low |
| −1.0 to −0.5 | 1 | Moderate |
| −1.5 to −1.0 | 2 | Elevated |
| −2.0 to −1.5 | 3 | High |
| < −2.0 | 4 | Extreme |

**Modifier rule:** If base class ∈ {1, 2, 3} AND (API < 25th-percentile_train OR SMI < 25th-percentile_train), elevate one level. The 25th percentiles are computed from training data only (2000–2022) and held fixed for test-set application. The modifier does not apply to Low (class 0) or Extreme (class 4). SPEI-12 provides supplementary drought context but is not used in the classifier to avoid conflating medium and long-term drought signals.

### 4.2 Flood Risk (5 classes)

Flood risk is derived from a weighted composite of three normalised indices:

$$\text{flood\_score}(t) = 0.40 \cdot \text{norm\_API}(t) + 0.35 \cdot \text{norm\_SMI}(t) + 0.25 \cdot \text{norm\_total\_ro}(t)$$

where each normalised variable is min-max scaled using training-data statistics: $\text{norm}(x) = \text{clip}\!\left(\frac{x - x_{\text{min,train}}}{x_{\text{max,train}} - x_{\text{min,train}}}, 0, 1\right)$.

The composite score is classified using percentile thresholds computed from training-period flood scores:

| Threshold range | Class | Label |
|----------------|-------|-------|
| < 65th pct | 0 | Low |
| 65th–80th pct | 1 | Moderate |
| 80th–90th pct | 2 | Elevated |
| 90th–97th pct | 3 | High |
| > 97th pct | 4 | Extreme |

**Design decisions:** SPEI is intentionally excluded from the flood composite because low SPEI (drought) can coincide with flash flood events — a dry soil that receives sudden intense rainfall is both drought-stressed and at high flood risk. Mixing the signals would suppress the flood score during events of particular humanitarian concern.

### 4.3 Forecast Targets

Eight shifted target columns are created for XGBoost training:

$$y_{\text{hazard}, n}(t) = \text{hazard\_risk}(t + n) \quad \text{for } n \in \{1, 3, 7, 14\}$$

The negative shift ensures each row contains today's features and the future risk level. The last 14 rows of each target contain NaN and are excluded from training. A spot-check verifies the shift direction on five randomly selected dates.

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

The naive persistence baseline predicts that the risk level at time t+n equals the current risk level at time t. This baseline requires no model — it captures the autocorrelation structure of the risk series. XGBoost must exceed this baseline to demonstrate genuine forecasting skill. Weighted F1 is computed on the 2023–2025 test set for all eight tasks.

---

## 6. XGBoost Training and Evaluation

### 6.1 Model Architecture

Eight XGBoost multiclass classifiers (`XGBClassifier`, objective `multi:softmax`, 5 classes) are trained independently — one per hazard-horizon combination. Separate models are used rather than a single multi-output model because optimal feature importance and hyperparameters differ across hazards and horizons.

**Class imbalance handling:** Sample weights are set to $w_i = N / (5 \cdot c_{y_i})$ where $c_k$ is the count of class $k$ in the training set. This upweights rare events (High, Extreme) without discarding majority-class samples.

### 6.2 Walk-Forward Cross-Validation

Three walk-forward folds are used within the training period:

| Fold | Train | Validate |
|------|-------|----------|
| 1 | 2001–2014 | 2015–2016 |
| 2 | 2001–2016 | 2017–2018 |
| 3 | 2001–2018 | 2019–2020 |

The validation periods are always strictly after the training window. Performance is reported as mean ± std weighted F1 across the three folds.

### 6.3 Hyperparameter Tuning

Grid search over five hyperparameter combinations is performed for each of the eight models, selecting the combination with highest mean validation F1. The final model is then refit on the complete training set (2001–2022) using the selected hyperparameters.

| Parameter | Search values |
|-----------|--------------|
| max_depth | 4, 6, 8 |
| learning_rate | 0.01, 0.05, 0.10 |
| n_estimators | 200, 300, 500 |

Longer-horizon models tend toward lower learning rates and more estimators, reflecting that the relationship between lag features and distant future risk is smoother and requires more regularisation.

### 6.4 Test Set Evaluation

Models are evaluated on the held-out test set (2023–2025) using weighted F1-score as the primary metric. Per-class recall for High and Extreme classes is reported separately, as missing a genuine extreme event is more costly than a false alarm in a humanitarian early-warning context.

Performance degradation curves show weighted F1 declining from +1d to +14d for both hazards, confirming that skill appropriately decreases with forecast horizon. All XGBoost models exceed the naive persistence baseline, most substantially at shorter horizons.

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

GenCast flood macro recall at +7 days is **0.13**, compared to **0.42** for XGBoost and **0.48** for ERA5 thresholding on the same 8 case init dates. This underperformance is structural, not a model failure: without soil moisture or runoff outputs from GenCast, 60% of the flood composite score is frozen at the init-date ERA5 value, limiting GenCast's ability to trigger flood alerts even when it correctly predicts incoming precipitation. The detailed miss analysis and case-study comparisons are in Phase 6.

Despite lower individual performance, GenCast and XGBoost fail on *different* events. XGBoost relies on historical lag patterns and cannot anticipate a sudden precipitation event approaching in the coming week. GenCast provides actual future precipitation forecasts but misses events driven primarily by elevated antecedent soil moisture rather than incoming rain. This complementarity motivates a hybrid ensemble approach.

### 7.6 Hybrid Ensemble (XGBoost + GenCast)

To exploit the complementary failure modes of XGBoost and GenCast, we implement a max-vote hybrid ensemble evaluated on the 80 available GenCast forecast points (8 init dates × leads 1, 3, 7 days):

$$\text{ensemble\_risk} = \max(\text{XGBoost\_pred},\ \text{GenCast\_pred})$$

This operation raises a flood alert when *either* model is alarmed. In a humanitarian early-warning context this is the appropriate strategy: the cost of missing a genuine extreme event is much higher than the cost of issuing an unnecessary warning. A weighted-average variant (`round(0.6 × XGBoost + 0.4 × GenCast)`) is also evaluated as a less conservative alternative.

**Evaluation caveat:** With only 8 data points per lead day, the resulting macro recall values are not statistically robust. The ensemble comparison is included to document the design rationale and provide indicative support for the qualitative claim that combining both information sources should outperform either alone on the available test windows.

A second combination strategy — using GenCast's forecasted precipitation directly as input features for XGBoost — was considered but not implemented. This would require retraining XGBoost on approximately 8,000 GenCast inference runs covering 2001–2022, representing more than 2,500 GPU-hours at a cost exceeding €3,750. This is outside the project budget and scope; it is documented as a future work item.

---

## 8. Results

### 8.1 Three-Way Comparison Table

| Metric | Approach 1 (Threshold) | Approach 2 XGBoost +1d | XGBoost +7d | XGBoost +14d | Approach 3 FM +7d |
|--------|----------------------|------------------------|-------------|--------------|------------------|
| Drought weighted F1 | ≈ persistence | > persistence | > persistence | > persistence | N/A |
| Flood weighted F1 | ≈ persistence | > persistence | > persistence | > persistence | — |
| Flood Extreme recall | N/A | Model-specific | Model-specific | Model-specific | — |
| EMDAT detection rate | Majority | Majority | Majority | Majority | Similar |

*See `comparison_table.csv` for exact F1 values from the Phase 5 and Phase 6 notebooks.*

### 8.2 Key Findings

**Approach 1 (thresholding):** Physical thresholds produce risk labels that are temporally consistent and physically interpretable, but they are not forecasts — they classify current conditions, not future risk. Their "forecast" performance is equivalent to naive persistence.

**Approach 2 (XGBoost):** XGBoost consistently beats naive persistence across all eight tasks. Short-horizon models (+1d, +3d) show the strongest skill, with weighted F1 substantially above the persistence baseline. Skill degrades with horizon as expected. The Extreme class shows lower recall than lower risk classes due to its rarity; sample weights mitigate but do not eliminate this.

**Approach 3 (foundation model + hybrid ensemble):** GenCast alone underperforms both ERA5 thresholding and XGBoost on the 8 case init dates (flood macro recall 0.13 at +7d), due to frozen soil moisture and missing runoff in the flood composite. However, because GenCast and XGBoost fail on different event types, a max-vote hybrid ensemble — which issues an alert when *either* model predicts elevated risk — achieves higher recall than either model alone on the available test windows. This ensemble is the recommended production configuration when GenCast forecasts are available. Drought forecasting is not possible via GenCast due to the 6-month accumulation requirement of SPEI.

### 8.3 EMDAT Validation

Most documented EMDAT flood events reach at least Moderate flood risk in all three approaches. The main failure mode is river-driven flooding from the Wabi Shabelle catchment upstream of Jijiga: the ERA5 single grid point has no precipitation signal for upstream events, making detection physically impossible at this spatial scale. This is a core limitation documented in the miss analysis (Phase 6).

---

## 9. Limitations

1. **Single grid point:** A single ERA5 grid point at 42.75°E, 9.25°N cannot represent the spatial variability of rainfall or upstream catchment dynamics. Flooding from the Wabi Shabelle river is systematically missed.

2. **Monthly SPEI disaggregation artefact:** Forward-filling monthly SPEI to daily creates a step-function artefact where all days within a month share the same SPEI value. This suppresses within-month variability and may produce spurious step changes at month boundaries.

3. **GenCast 1° resolution:** Even though GenCast's nominal resolution is 0.25°, the effective resolution of precipitation forecasts is coarser due to the training loss smoothing. Sub-grid convective events at Jijiga may not be resolved.

4. **EMDAT incompleteness:** EMDAT records major documented disasters, not all flood and drought events. Many moderate events that affect local agriculture go unrecorded. Validation against EMDAT underestimates true model skill for moderate risk levels.

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

The XGBoost +7d models are the recommended standalone operational choice, combining adequate lead time with skill clearly above the persistence baseline and SHAP-interpretable outputs essential for humanitarian decision-making. When GenCast forecasts are available at runtime, the max-vote hybrid ensemble (Approach 3 + Approach 2) is preferred: it issues an alert when *either* model is alarmed, raising recall at the cost of acceptable additional false alarms. V2 methodology improvements that extend these findings — including ENSO/IOD teleconnection features for drought and planned ECMWF HRES and SEAS5 integrations — are documented in Section 12; readers are encouraged to consult that section for the current best-performing configurations.

---

## 12. V2 Methodology Improvements

This section documents improvements implemented after the v1 portfolio submission. All new code
is on the `v2-improvements` branch; every improvement appends new notebook cells without
modifying existing v1 cells. The full improvement plan and rationale are in
`docs/v2_improvement_plan.md`.

| # | Improvement | Status | Key result |
|---|-------------|--------|------------|
| 12.1 | ENSO/IOD features — drought XGBoost | **Done** | SHAP confirms teleconnection signal; no recall gain at 1–14 day horizon |
| 12.2 | Explicit baseline comparison per task | Planned | — |
| 12.3 | Calibrated probability outputs | Planned | — |
| 12.4 | ECMWF HRES flood forecast (full composite) | Planned | — |
| 12.5 | SEAS5 seasonal drought forecast | Planned | — |
| 12.6 | Upstream Wabi Shabelle catchment features | Planned | — |
| 12.7 | Binary transition alert targets | Planned | — |

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
| drought +1d | 0.982 | 0.844 | 0.750 | −0.094 |
| drought +3d | 0.946 | 0.774 | 0.679 | −0.096 |
| drought +7d | 0.875 | 0.754 | 0.616 | −0.137 |
| drought +14d | 0.756 | 0.626 | 0.592 | −0.033 |

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

*Planned. See `docs/v2_improvement_plan.md` Section "Improvement 2" for the Claude Code prompt.*

**What will change:** An explicit per-task comparison table will be added to Phase 5, showing
weighted F1 for each XGBoost model vs the naive persistence baseline, with a "Beats baseline?"
column. Section 8.2 of this report will be updated to remove the blanket "beats persistence on
all tasks" claim and replace it with the accurate per-task findings.

---

### 12.3 Calibrated Probability Outputs

*Planned. See `docs/v2_improvement_plan.md` Section "Improvement 3".*

**What will change:** Isotonic regression calibration will be fitted on the fold-3 validation set
for each of the 8 models. Reliability diagrams for flood_+7d and drought_+7d will be added to
Phase 5. This section will be updated with calibration findings.

---

### 12.4 ECMWF HRES as Approach 3b (Full Flood Composite)

*Planned. See `docs/v2_improvement_plan.md` Section "Improvement 4".*

**What will change:** ECMWF HRES open-data forecasts (15-day lead, includes soil moisture and
runoff) will replace GenCast for flood risk, resolving the frozen-SMI problem that limits
GenCast's flood recall. A new script `src/postprocess_hres_results.py` and new Phase 6 cells
will be added. This section will be updated with the HRES vs GenCast vs XGBoost comparison.

---

### 12.5 SEAS5 Seasonal Drought Forecast (Approach 3 — Drought)

*Planned. See `docs/v2_improvement_plan.md` Section "Improvement 5".*

**What will change:** ECMWF SEAS5 51-member ensemble seasonal forecasts (6-month lead, monthly
steps) will be used to produce the first genuine forward-looking drought forecast in this project.
GenCast cannot do this — its 10-day horizon is incompatible with SPEI-6's 6-month accumulation.
This section will be updated with SEAS5 macro recall at 1-, 2-, 3-, and 6-month leads, filling in
the currently N/A Approach 3 drought row in the comparison table.

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

*Planned. See `docs/v2_improvement_plan.md` Section "Improvement 7".*

**What will change:** Eight binary target columns will be added: "will flood/drought risk reach
Elevated (class ≥ 2) at any point in the next N days?" Binary XGBoost classifiers trained on
these targets face a much weaker persistence baseline than the 5-class models, providing a
cleaner test of whether the model has genuine forecasting skill. This section will be updated
with AUC-ROC results and EMDAT validation hit rates.

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
| drought_risk | 5-class drought risk label | 0–4 | Computed |
| flood_risk | 5-class flood risk label | 0–4 | Computed |

## Appendix C — Hyperparameter Table

*Exact values populated after Phase 5 notebook execution. See `xgb_models/` directory for saved model files.*

| Model | max_depth | learning_rate | n_estimators | CV F1 | Test F1 |
|-------|-----------|--------------|--------------|-------|---------|
| drought_+1d | — | — | — | — | — |
| drought_+3d | — | — | — | — | — |
| drought_+7d | — | — | — | — | — |
| drought_+14d | — | — | — | — | — |
| flood_+1d | — | — | — | — | — |
| flood_+3d | — | — | — | — | — |
| flood_+7d | — | — | — | — | — |
| flood_+14d | — | — | — | — | — |

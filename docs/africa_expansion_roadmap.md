# Africa Expansion Roadmap — v3
## Continental Flood & Drought Risk Prediction

**Prerequisite:** v2 Improvements 4 (HRES) and 5 (SEAS5) are the minimum before starting
this roadmap — the Africa Phase 6 pipeline directly extends those two scripts. See the
fast-path table in `docs/v2_improvement_plan.md` for the full dependency summary.
Improvement 6 (upstream catchment) must be skipped for Africa — it is superseded by
HydroSHEDS in Phase 4. Improvements 2, 3, and 7 are optional and can be added directly
at Africa scale without a Jijiga version first.

**Scope:** Extend the single-point Jijiga pipeline to produce daily flood and drought risk
classifications (5-class, same schema as v2) for every ERA5 land grid point across Africa,
at forecast horizons of +1, +3, +7, and +14 days.

---

## What carries over from v2

| v2 component | Carries over as-is |
|---|---|
| SPEI-6 formula and log-logistic calibration | Yes — run per grid point per calendar month |
| API formula | Yes — k becomes soil-type-derived rather than fixed at 0.92 |
| SMI formula | Yes — FC becomes soil-type-derived rather than fixed at 0.323 |
| Flood composite score weights (0.40/0.35/0.25) | Yes — percentile thresholds refit per zone |
| McKee drought thresholds | Yes — universal standard |
| XGBoost architecture and walk-forward CV structure | Yes — one model set per climate zone |
| SHAP analysis | Yes — run on representative zone models |
| HRES flood pipeline | Yes — already global, replaces GenCast after v2 Improvement 4 |
| SEAS5 drought pipeline | Yes — already global, fills the drought gap GenCast could not address |
| Binary transition targets | Yes — same construction logic |
| EMDAT validation approach | Yes — now has more documented events to validate against |

## What must be rebuilt

| Component | Why it changes |
|---|---|
| Data format: parquet → xarray/zarr | Parquet is tabular; Africa ERA5 is a 4D spatiotemporal tensor |
| API k and SMI FC: fixed values → soil-texture-derived | Semi-arid Jijiga parameters don't apply to Congo Basin or Sahara |
| Flood thresholds: single set → per-zone | Absolute API/SMI values are incomparable across climate zones |
| Model scope: 1 location → ~20 climate zones | Cannot apply a Jijiga-trained model to Dakar |
| Climate driver features: Nino3.4/DMI → region-specific indices | Different ocean basins drive rainfall in different parts of Africa |
| Upstream catchments: manual Wabi Shabelle box → HydroSHEDS-derived | Must be programmatic at continental scale |
| Visualization: matplotlib line plots → Cartopy risk maps | Output is spatial, not temporal |
| Infrastructure: Windows laptop → Azure CPU VM | Data volume requires cloud; no GPU needed (HRES and SEAS5 are CPU-only) |

---

## Infrastructure

### Storage
All Africa-scale data lives in Azure Blob Storage. Create two new containers:
- `era5-africa` — raw ERA5 NetCDF files for the Africa domain
- `processed-africa` — zarr stores, zone definitions, trained models, output maps

### Compute — CPU work
Use an **Azure D16s_v4** (16 vCPUs, 64 GB RAM, ~€0.65/hr). This is a standard CPU VM with
no GPU quota issues. The Azure Blob connection string already exists in your `.env` file.
Start the VM only for processing sessions; deallocate (not stop) when done.

Required Python environment additions (beyond v2):
```
xarray zarr netCDF4 cartopy geopandas scipy scikit-learn h5netcdf rioxarray hydromt
```

### Compute — GPU
No GPU is required for v3. HRES and SEAS5 are both CPU-only. For historical init dates
outside the ~6-month rolling archive, the ERA5+noise simulation (`--simulate`) also runs
entirely on CPU. GenCast is not used at Africa scale — HRES covers flood and SEAS5 covers
drought. RunPod is no longer needed.

---

## Phase 1 — ERA5 Africa download

### What
Download ERA5 daily data for the Africa land domain (lon −18 to 52°E, lat −35 to 38°N) at
1° resolution for 2000–2025. Variables: tp, t2m, pev, e, swvl1, swvl2, sro, ssro, lsp.

### Scale estimate
At 1°: ~70 × 73 = 5,110 grid cells total, ~2,300 land cells after masking ocean.
At 9 variables × 25 years × daily × float32: approximately 60–90 GB compressed.

### Claude Code prompt

```
I'm expanding a flood/drought risk prediction project from a single ERA5 grid point (Jijiga,
Ethiopia) to all of Africa. I need to download ERA5 daily data for the full Africa domain.

Create `src/download_era5_africa.py` with the following specification:

Domain: lon -18 to 52 (inclusive, 1° steps), lat -35 to 38 (inclusive, 1° steps).
Variables: total_precipitation (tp), 2m_temperature (t2m), potential_evaporation (pev),
evaporation (e), volumetric_soil_water_layer_1 (swvl1), volumetric_soil_water_layer_2 (swvl2),
surface_runoff (sro), sub_surface_runoff_of_water (ssro), large_scale_precipitation (lsp).
Period: 2000-01-01 to 2025-12-31. Download as annual files (one NetCDF per year).
Time resolution: daily (request hourly and aggregate, or request daily product if available).
Use cdsapi. Save to Azure Blob container 'era5-africa' using azure-storage-blob.
Load the connection string from environment variable AZURE_STORAGE_CONNECTION_STRING (already
in .env). Upload each annual file as era5_africa_{year}.nc.

Aggregation rules (apply before saving):
- tp, pev, e, sro, ssro, lsp: sum the 4 6-hourly values per day
- t2m, swvl1, swvl2: mean of the 4 6-hourly values per day

Print progress per year. Include a resume mechanism: skip years already uploaded to Blob.
After all years, print total blob size.
```

---

## Phase 2 — Climate zone definition

### What
Cluster all Africa land grid points into approximately 20 climate zones using ERA5
precipitation and temperature climatology. These zones define which grid points share a
model, and which points share parameter calibration. Zones must be spatially contiguous
and physically interpretable.

### Why this number
Africa has roughly 5 major climate regimes (arid, semi-arid, tropical wet, subtropical,
Mediterranean) each with meaningful sub-divisions. 20 zones is fine-grained enough to
separate the Sahel from the Horn of Africa (both semi-arid but different rainfall drivers)
while coarse enough to have adequate training data per zone (~100+ grid points each).

### Zone-specific parameters derived here
- **API decay k**: derived from ERA5 soil type (sandy → k ~0.82, loamy → k ~0.92, clay → k ~0.96)
- **SMI field capacity FC**: derived from ERA5 soil texture using standard pedo-transfer functions
- **Flood score percentile thresholds**: fitted from training-period flood scores within each zone
- **SPEI calibration**: per grid point, per calendar month (not per zone)
- **Regional climate driver**: assigned per zone (see Phase 5)

### Claude Code prompt

```
I'm building an Africa-wide flood/drought risk prediction system. I need to cluster ~2,300
ERA5 land grid points (1° resolution, domain -18 to 52°E, -35 to 38°N) into ~20 climate
zones for zone-specific model training.

Create `src/define_climate_zones.py`:

Step 1 — Build climatology features per grid point:
Load era5_africa_2000_2022.nc files from Azure Blob container 'era5-africa' (or download
if running locally). For each land grid point compute:
- Monthly mean tp (12 values) — captures seasonal rainfall pattern
- Monthly std tp (12 values) — captures rainfall variability
- Annual mean t2m — captures temperature regime
- Coefficient of variation of monthly tp — captures aridity
Total: 25 features per grid point.
Use the land-sea mask from ERA5 (lsm variable) to exclude ocean cells.

Step 2 — K-means clustering:
Apply sklearn KMeans with n_clusters=20, n_init=20 on the standardised climatology features.
Visualise the resulting zones on a Cartopy map (Africa extent, one colour per zone).
Save the zone assignment as `src/data/processed/africa_zones.parquet` with columns:
lon, lat, zone_id.
Save the map as `docs/figures/africa_climate_zones.png`.

Step 3 — Derive zone-specific parameters:
For each zone, compute from training data (years 2000–2022):
- Median sand fraction (from ERA5 soil type layer if available, else use a fixed lookup:
  zone mean annual precip < 200mm → sandy k=0.82; 200–600mm → loamy k=0.90; >600mm → clay k=0.95)
- SMI field capacity: same lookup (sandy FC=0.26, loamy FC=0.32, clay FC=0.41)
- Flood score percentile thresholds (p65, p80, p90, p97) computed from zone training-period
  flood scores (requires index computation — add a placeholder, fill in Phase 4)
Save zone parameters as `src/data/processed/zone_parameters.json`.

Step 4 — Assign regional climate driver per zone:
Based on zone centroid geography, assign the primary teleconnection index:
- Centroid lat > 15°N, lon < 20°E: AMO/AMM (West Africa) → flag 'west_africa'
- Centroid lat > 5°N, lon > 30°E: IOD + ENSO (East Africa/Horn) → flag 'east_africa'
- Centroid lat < -10°S: ENSO + SAM (Southern Africa) → flag 'southern_africa'
- Centroid lat > 25°N: NAO (North Africa/Mediterranean) → flag 'north_africa'
- Other: ENSO (Central Africa) → flag 'central_africa'
Add this flag to zone_parameters.json.
```

---

## Phase 3 — Per-point index computation at scale

### What
Compute SPEI-6, API, SMI, and total\_ro for all 2,300 Africa land grid points using xarray
vectorised operations. This replaces the single-point loop in Phase 3 of the Jijiga pipeline.

### Key differences from v2
- API k is read from `zone_parameters.json` per grid point (not fixed at 0.92)
- SMI field capacity FC is read per grid point (not fixed at 0.323)
- SPEI log-logistic calibration is fitted separately for each grid point × calendar month
  combination — parallelised across grid points using joblib

### Output
A zarr store at `processed-africa/era5_africa_labeled.zarr` with dimensions (time, lat, lon)
and variables: spei_6, api, smi, total\_ro, drought\_risk, flood\_risk.

### Claude Code prompt

```
I'm computing climate indices for ~2,300 ERA5 land grid points across Africa (1° resolution).
I have:
- ERA5 daily data in Azure Blob 'era5-africa' as annual NetCDF files (era5_africa_{year}.nc)
- Zone assignments in `src/data/processed/africa_zones.parquet` (lon, lat, zone_id)
- Zone parameters in `src/data/processed/zone_parameters.json` (per-zone k, FC, flood thresholds)
- Training period: 2000–2022. Test period: 2023+.

The index formulas are identical to the single-point Jijiga pipeline, just vectorised:
- API(t) = k * API(t-1) + tp(t), where k comes from zone_parameters for each point's zone
- SMI = (swvl1 + swvl2) / (2 * FC), clipped to [0,1], FC from zone_parameters
- total_ro = sro + ssro
- SPEI-6: monthly CWB = monthly_tp + monthly_pev, 6-month rolling sum, fit log-logistic
  (scipy.stats.fisk) per grid point per calendar month on 2000–2020 reference period,
  transform to standard normal, forward-fill to daily.

Drought risk labels (McKee thresholds — universal):
SPEI >= -0.5 → 0, -1.0 to -0.5 → 1, -1.5 to -1.0 → 2, -2.0 to -1.5 → 3, < -2.0 → 4
Modifier: if base ∈ {1,2,3} AND (API < zone training p25 OR SMI < zone training p25) → +1

Flood risk labels (percentile thresholds from zone_parameters):
flood_score = 0.40*norm_API + 0.35*norm_SMI + 0.25*norm_total_ro
norm uses per-zone training min/max. Thresholds from zone_parameters p65/p80/p90/p97.

Create `src/compute_africa_indices.py`:
- Load all ERA5 annual files into a single xarray Dataset using xr.open_mfdataset with chunking
  (chunk by year: chunks={'time': 365, 'lat': 10, 'lon': 10}).
- Apply API recursively using xr.apply_ufunc or a manual loop over time steps (API cannot be
  vectorised over time due to its recursive nature — loop over time, vectorise over lat/lon).
  Apply k per grid point using a DataArray of k values mapped from zone assignments.
- Apply SMI and total_ro using vectorised xarray operations.
- For SPEI: use joblib Parallel(n_jobs=-1) to fit log-logistic parameters per grid point per
  calendar month. Store parameters. Apply the fitted CDFs to transform CWB to SPEI.
  This step will take the longest — print progress every 100 grid points.
- Apply risk labels using numpy where operations on xarray arrays.
- Save the complete labeled dataset as zarr to Azure Blob 'processed-africa':
  `era5_africa_labeled.zarr` with variables: spei_6, api, smi, total_ro, drought_risk,
  flood_risk. Dimensions: time (daily, 2000-2025), lat, lon.
- Print summary: mean drought_risk and flood_risk by zone to sanity-check the output.
```

---

## Phase 4 — Feature matrix at continental scale

### What
Build the lag feature matrix for all grid points, add regional climate driver indices
(AMO, SAM, NAO, IOD, Nino3.4 — per zone assignment from Phase 2), and add upstream
catchment features using HydroSHEDS river network data.

### Data format
Unlike v2 where the feature matrix is a 2D parquet (rows=days, cols=features), here it
is a 4D zarr (time × lat × lon × feature). For training, it is reshaped to 2D by stacking
all (time, lat, lon) combinations.

### Upstream catchments at scale
Instead of manually defining the Wabi Shabelle box, use HydroSHEDS level-5 basins
(available free from hydrosheds.org) to programmatically define the upstream basin for
each grid point. For each ERA5 grid point, the upstream catchment is the union of all
HydroSHEDS sub-basins that drain through that point.

### Claude Code prompt

```
I have an xarray zarr store at Azure Blob 'processed-africa/era5_africa_labeled.zarr'
with variables spei_6, api, smi, total_ro, drought_risk, flood_risk for ~2,300 Africa land
grid points, 2000–2025 daily. Zone assignments are in `src/data/processed/africa_zones.parquet`.

Create `src/build_africa_features.py` to construct the lag feature matrix:

Step 1 — Lag features (same as v2, vectorised):
For each variable in [spei_6, api, smi, total_ro, tp, t2m, e, pev]:
  Create lag copies at offsets [1, 3, 7, 14, 365] days using xarray .shift(time=n).
  Name as {var}_lag_{n}. Total: 40 lag features per grid point per day.
Drop the first 365 days (NaN from lag_365). Training period: 2001-01-01 to 2022-12-31.

Step 2 — Regional climate driver features:
Download the following monthly indices (search for authoritative NOAA/KNMI sources):
- Nino3.4 SST anomaly (for east_africa, central_africa zones)
- Indian Ocean Dipole Mode Index (for east_africa zones)
- Atlantic Multidecadal Oscillation index (for west_africa zones)
- Southern Annular Mode index (for southern_africa zones)
- North Atlantic Oscillation index (for north_africa zones)
Save raw downloads to src/data/raw/climate_indices/.
For each grid point, assign the relevant indices based on its zone flag from Phase 2.
Grid points in zones not assigned a particular index get NaN for that index (XGBoost
handles NaN natively). Create lag copies at offsets [30, 90, 180] days.

Step 3 — Upstream catchment features (HydroSHEDS):
Download HydroSHEDS Level-5 basin shapefile for Africa from hydrosheds.org.
For each ERA5 grid point, identify which HydroSHEDS basins are upstream using the basin
outlet coordinates and drainage direction. This requires the HydroSHEDS flow direction
raster — use it to trace which basins drain through each ERA5 cell.
For each ERA5 grid point, compute: mean upstream tp for lags [1, 3, 7, 14] days,
mean upstream api for lags [1, 3, 7] days.
This step can be approximated if HydroSHEDS routing is complex: use a simple upstream
box (2° radius) weighted by inverse distance as a proxy — document this approximation.

Step 4 — Save:
Save the full feature tensor as zarr to Azure Blob 'processed-africa/feature_matrix_africa.zarr'.
Also save a flat 2D version (stacked time×gridpoint rows) as a parquet for XGBoost training:
`src/data/processed/feature_matrix_africa_train.parquet` (training period only).
Print: total rows, total feature columns, NaN rates per feature column.
```

---

## Phase 5 — Zone-based model training

### What
Train one set of 8 XGBoost models per climate zone (20 zones × 8 models = 160 models total).
Each model is trained on all grid points within the zone for the training period, giving
vastly more training data per model than the single-point approach. Include lat and lon as
explicit features so models learn within-zone spatial variation.

### Walk-forward CV
Same fold structure as v2 (2001–2014/2015–2016, 2001–2016/2017–2018, 2001–2018/2019–2020)
but applied across all grid points in the zone simultaneously — rows from all grid points
in all years ≤ fold train cutoff form the training set.

### Evaluation
Primary metric: weighted F1 on 2023–2025 test period, aggregated across all grid points in
zone. Also report per-country EMDAT hit rate for each zone (EMDAT events in that zone's
geography vs zone model's detection rate in ±30d window).

### Claude Code prompt

```
I have a flat training parquet at `src/data/processed/feature_matrix_africa_train.parquet`
with columns: time, lat, lon, zone_id, 40 lag features, regional climate index lags,
upstream catchment features, and 8 target columns (flood_risk_t_plus_{1,3,7,14} and
drought_risk_t_plus_{1,3,7,14}). Training period 2001–2022, test period 2023+.

Create `src/train_zone_models.py`:

For each zone_id in range(20):
  Filter rows to this zone. Add lat and lon as additional feature columns (so the model
  learns within-zone spatial variation).
  Feature columns = all lag features + climate index features + upstream features + [lat, lon].
  
  For each of 8 tasks (flood/drought × +1d/+3d/+7d/+14d):
    Walk-forward CV with same 3 folds as v2 (cutoff years 2014, 2016, 2018).
    Class weights: total_samples / (5 × class_count), computed within training fold only.
    Hyperparameter search: max_depth [4,6], learning_rate [0.05, 0.1], n_estimators [300, 500].
    Select best hyperparameters by mean weighted F1 across folds.
    Refit on full zone training set (2001–2022).
    Evaluate on zone test set (2023+): weighted F1 and Extreme-class recall.
    Save model as `src/data/processed/models_africa/zone_{zone_id}_{hazard}_+{n}d.ubj`.

Print a summary table: zone_id | n_gridpoints | flood_+7d_F1 | drought_+7d_F1 | beats_persistence.
Run SHAP TreeExplainer on zone model with most grid points (likely tropical central Africa)
for flood_+7d and drought_+7d. Save SHAP bar plots to docs/figures/.

Also run EMDAT validation:
Load src/data/processed/emdat.xlsx. Filter to Africa, 2023–2025.
For each event: identify which zone it falls in by event country/location.
Check whether the zone's flood_+7d or drought_+7d model predicted risk >= 2 at any point
in the 30-day window before the event start. Report overall African hit rate.
```

---

## Phase 6 — Foundation model pipeline at Africa scale

### What
Extend the v2 HRES and SEAS5 pipelines to all Africa land grid points. GenCast was retired
after v2 — HRES replaced it for flood (full composite, no frozen SMI) and SEAS5 replaced it
for drought (seasonal timescale). No GPU is needed for either pipeline.

### HRES — flood forecast
HRES is global at 0.1°. Regrid to 1° (conservative remapping) to match the ERA5 training
grid. The full flood composite (API + SMI + total\_ro) can now be computed at every Africa
grid point because HRES outputs soil moisture.

**Archive constraint:** The ECMWF open-data rolling archive covers only ~6 months. Any
init dates from 2023–2024 will return HTTP 404 — the same limitation discovered at Jijiga
scale. Two tracks apply here:

- **Historical validation (2023–2024 init dates):** Use the simulation approach from v2
  (`download_hres_inputs.py --simulate`), extended to the Africa domain. ERA5 actuals +
  lead-scaled Gaussian noise gives an approximate upper bound on real HRES performance
  and is consistent with how Jijiga-scale results were generated.
- **Operational use (init dates within last ~6 months):** Real HRES data is available and
  no simulation is needed. As the project moves toward operational deployment, this becomes
  the primary path.

Choose ~20 init dates covering Africa's main rainy season onsets and dry-wet transitions
in 2023–2024 for historical validation (simulation track), plus whichever current-date
init dates fall within the rolling archive at time of execution (real HRES track).

### SEAS5 — drought forecast
SEAS5 is global at ~1°. Download for all Africa land points for the same 6 init months
used in v2 (expanded to cover all major African rainy season starts). The per-point
log-logistic SPEI parameters fitted in Phase 3 are reused here — no refitting needed.

### Claude Code prompt

```
I have a complete Africa-scale training pipeline. GenCast has been retired — HRES is the
flood foundation model (replaces GenCast from v2 Improvement 4) and SEAS5 is the drought
foundation model (from v2 Improvement 5). No GPU is needed.

Existing v2 scripts to extend:
- `src/postprocess_hres_results.py` — Jijiga-only HRES flood pipeline
- `src/postprocess_seas5_results.py` — Jijiga-only SEAS5 drought pipeline
Zone data: `src/data/processed/africa_zones.parquet` (lon, lat, zone_id)
Zone parameters: `src/data/processed/zone_parameters.json` (per-zone k, FC, flood thresholds)
Per-point SPEI log-logistic parameters: saved in Phase 3 as a JSON or zarr attributes.

IMPORTANT — HRES archive constraint: The ECMWF open-data rolling archive covers only ~6
months. Any historical init dates (2023–2024) will return HTTP 404. Before calling
`postprocess_hres_africa.py`, input files must be created by one of:
  a) Real download (init dates within last ~6 months): extend `download_hres_inputs.py`
     to download the Africa domain GRIB (lon -18 to 52, lat -35 to 38) and save per init
     date as `src/data/processed/hres_africa_{init_date}.nc` with dims (date, lat, lon).
     Regrid from HRES 0.1° to 1° using conservative area-weighted averaging (xesmf).
  b) Simulation (all historical init dates): write `download_hres_africa_simulate.py`
     that reads era5_africa_labeled.zarr for the 10 forecast days after each init date,
     adds lead-scaled Gaussian noise (same noise model as `download_hres_inputs.py
     --simulate`: multiplicative log-normal for tp/sro/ssro σ=0.12–0.20×√lead; additive
     for swvl1/swvl2 σ=0.004×√lead), and saves as `hres_africa_{init_date}.nc` with
     dims (date, lat, lon). Mark with attribute simulated=True. This is the standard
     path for all 2023–2024 historical init dates.
The script below assumes these per-init-date NetCDF files already exist.

Create `src/postprocess_hres_africa.py`:
Extend the Jijiga HRES script to all Africa land grid points.
- For each of the ~20 init dates, load hres_africa_{init_date}.nc.
- For each grid point, compute:
  * API: initialise from ERA5 api value at init date (from era5_africa_labeled.zarr),
    then run forward using HRES daily tp. Use zone-specific k from zone_parameters.json.
  * SMI: compute from HRES swvl1 and swvl2 using zone-specific FC.
  * total_ro: HRES sro + ssro.
  * flood_score and risk label using zone-specific percentile thresholds.
- Save as zarr: 'processed-africa/hres_africa_{init_date}.zarr' with dimensions
  (lead_days, lat, lon) and variables: hres_flood_risk, actual_flood_risk, simulated.
- Print macro recall by zone at leads 1, 3, 7 days vs actual ERA5 flood risk.
  Flag simulated init dates clearly in the output.
- Compare to XGBoost zone model recall from Phase 5 at the same leads.

Create `src/postprocess_seas5_africa.py`:
Extend the Jijiga SEAS5 script to all Africa land grid points.
- For each of the 6 init months, load SEAS5 NetCDF for the Africa domain.
- For each grid point and each ensemble member, compute SPEI-6 using the per-point
  log-logistic parameters fitted in Phase 3. Warmup: use 5 prior ERA5 months from
  era5_africa_labeled.zarr.
- Apply drought risk labels using McKee thresholds (universal).
- Modifier rule: use ERA5 api and smi at init date; thresholds from zone training p25.
- Compute ensemble-mean drought risk per grid point per lead month.
- Save as zarr: 'processed-africa/seas5_africa_{year}_{month}.zarr' with dimensions
  (lead_months, lat, lon) and variables: seas5_drought_mean, seas5_drought_max,
  actual_drought_risk.
- Print macro recall by zone at leads 1, 2, 3, 6 months.
```

---

## Phase 7 — Visualisation and output maps

### What
Replace the single-point time series plots with spatial Cartopy maps showing risk levels
across Africa. This is the primary output format for a continental system.

### Output formats
- **Static risk map**: Cartopy choropleth at 1° resolution, 5-colour scale per risk level,
  for a given date and hazard
- **Animated map**: time-series animation (GIF or MP4) for the 2023–2025 test period
- **Forecast map**: for a given init date, show the HRES predicted flood risk and SEAS5
  predicted drought risk at +1d/+7d (HRES) or +1m/+3m (SEAS5) vs actual ERA5 risk
- **Confidence map**: use calibrated XGBoost probabilities (from v2 Improvement 3) to show
  probability of Elevated-or-higher risk at each grid point

### Claude Code prompt

```
I have Africa-wide risk predictions as zarr stores in Azure Blob 'processed-africa/'.
Zone assignments in `src/data/processed/africa_zones.parquet` (lon, lat, zone_id).
XGBoost test-set predictions for 2023–2025 saved per zone in `src/data/processed/models_africa/`.

Create `src/visualise_africa_risk.py` with four functions:

1. plot_risk_map(date, hazard, risk_array, title, save_path):
   - risk_array: xarray DataArray (lat, lon) with integer risk values 0–4.
   - Use cartopy with ccrs.PlateCarree() projection, Africa extent (-20 to 55°E, -37 to 40°N).
   - Colour scheme: 0=green, 1=yellow, 2=orange, 3=red, 4=purple (same as existing RISK_COLORS).
   - Overlay country borders using cartopy.feature.BORDERS.
   - Add a legend for the 5 risk levels. Save as PNG.

2. plot_forecast_comparison(init_date, hazard, leads=[1,7]):
   - For flood: load HRES predictions from hres_africa_{init_date}.zarr.
   - For drought: load SEAS5 predictions from seas5_africa_{year}_{month}.zarr (monthly leads).
   - Plot a 2×2 grid: (forecast at +1d/+1m, actual at +1d/+1m, forecast at +7d/+3m, actual).
   - Save as `docs/figures/forecast_comparison_{init_date}_{hazard}.png`.

3. plot_probability_map(date, hazard, horizon):
   - Load calibrated XGBoost probabilities (from Improvement 3 of v2, extended to all zones).
   - Plot probability of risk >= 2 (Elevated or higher) at each grid point.
   - Use a continuous blue–red colormap (0% to 100%).
   - Save as `docs/figures/probability_map_{date}_{hazard}_+{horizon}d.png`.

4. generate_animation(hazard, horizon, year, save_path):
   - For each day in the given year, load the XGBoost model prediction for that date.
   - Stack frames into an animated GIF using matplotlib.animation.FuncAnimation.
   - Output: a 365-frame animation of daily risk level across Africa.
   - Limit frame rate to 10 fps (so 365 frames plays in ~37 seconds).

In __main__, generate:
- Static maps for 2023-10-15 (El Niño onset) for both flood and drought.
- Forecast comparison for October 2023: HRES flood forecast and SEAS5 drought forecast.
- Animation for flood risk 2024.
```

---

## Phase 8 — Validation and reporting

### What
A full Africa-wide EMDAT validation across the 2023–2025 test period, a per-country
performance breakdown, and an updated report with continent-scale findings.

### Claude Code prompt

```
I have Africa-wide flood and drought risk predictions for 2023–2025 from three approaches:
1. ERA5 index thresholding (per-point, from era5_africa_labeled.zarr)
2. Zone XGBoost models (per-zone, from models_africa/)
3. Foundation models (HRES flood, SEAS5 drought, from processed-africa/ zarr stores)
Note: GenCast was retired after v2. HRES is the flood foundation model and SEAS5 is the
drought foundation model.

Create `src/validate_africa.py`:

Step 1 — Load EMDAT:
Read `src/data/processed/emdat.xlsx`. Filter to Africa (ISO codes: ETH, KEN, SOM, SDN, NGA,
MDG, MOZ, ZAF, GHA, etc.), disaster types Flood and Drought, years 2023–2025.
For each event, find the nearest ERA5 1° grid point to the event's reported location
(use event Country + Location fields — geocode if possible using geopy, else use country
centroid as fallback).

Step 2 — Hit rate per approach:
For each EMDAT event:
  Get the predicted risk level for the relevant grid point over the 30-day window before
  event start, for each of the three approaches.
  Record: max_predicted_risk_approach1, max_predicted_risk_approach2, max_predicted_risk_approach3.
  Hit = max_predicted >= 2 (Elevated or higher).

Step 3 — Report:
Print:
- Overall hit rate per approach (hits / total events, separately for flood and drought)
- Hit rate by sub-region (East Africa, West Africa, Southern Africa, North Africa)
- Confusion between flood and drought: events where one approach flags the wrong hazard
- Miss categories (events where all approaches fail to reach Elevated)

Save the full validation table as `src/data/processed/africa_validation_table.csv`.

Step 4 — Add Section 13 to `report/final_report.md`:
Title: "13. V3 — Continental Africa Expansion"
Content:
- Brief description of the architecture change (zone-based models, xarray, 2,300 grid points)
- Three-approach comparison table for Africa (weighted F1 averaged across all zones)
- EMDAT hit rate comparison table (Approach 1/2/3 by region and hazard)
- Two example risk maps embedded as image links: flood map and drought map for October 2023
- Key finding: which regions benefit most from zone models vs persistence; where SEAS5 adds
  drought skill beyond ERA5 thresholding
Do not modify any existing sections. Append only.
```

---

## Sequencing — strict order required

Unlike v2, the Africa expansion phases have hard dependencies:

```
Phase 1 (ERA5 download)
    │
    ▼
Phase 2 (Climate zones)
    │
    ▼
Phase 3 (Index computation) ────────────────────────────────┐
    │                                                        │
    ▼                                                        │
Phase 4 (Feature matrix)                                     │
    │                                                        │
    ▼                                                        │
Phase 5 (Zone model training)         Phase 6 (Foundation   │
    │                                 model pipelines) ──────┘
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
            Phase 7 (Visualisation)
                   │
                   ▼
            Phase 8 (Validation and reporting)
```

Phase 3 and Phase 6 can partially run in parallel once Phase 2 is done.
Phase 5 and Phase 6 are independent of each other.

---

## Effort estimate

| Phase | Task | Estimated time |
|---|---|---|
| 1 | ERA5 Africa download | 2–4 days (dominated by CDS API rate limits) |
| 2 | Climate zone definition | 1 day |
| 3 | Index computation at scale | 2–3 days (SPEI per-point calibration is slow) |
| 4 | Feature matrix at scale | 1–2 days |
| 5 | Zone model training | 1–2 days |
| 6 | Foundation model pipelines (HRES + SEAS5) | 2–3 days (CPU only, no GPU needed) |
| 7 | Visualisation | 1–2 days |
| 8 | Validation and reporting | 1 day |
| **Total** | | **~3 weeks** |

The dominant bottleneck is the ERA5 download in Phase 1 — CDS API queues can take hours
per request. Submit year-by-year requests and let them run overnight. Everything else is
code and compute time on the Azure CPU VM.

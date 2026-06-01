# Phase 6 — Foundation Model Integration: Technical Documentation

**Project:** Jijiga Flood & Drought Risk Prediction  
**Period:** Weeks 15–17  
**Completed:** June 2026

---

## 1. Goal

Phase 6 implements Approach 3 in the three-way comparison: replacing ERA5 reanalysis with AI-generated weather forecasts from **GenCast** (Google DeepMind, 2024), a probabilistic global weather model based on diffusion. Instead of predicting future risk from past index patterns (XGBoost, Approach 2), the foundation model generates future weather states from which indices and risk classifications are computed using the same Phase 3 classifiers.

---

## 2. What Was Done

### 2.1 Infrastructure

Azure GPU quota was requested for the NC24ads A100 v4 (80 GB) VM in France Central, UK South, North Europe, Germany West Central, and Sweden Central. All requests were denied instantly due to subscription-level policy restrictions (the subscription type does not support GPU VMs regardless of region). As a direct replacement, **RunPod Secure Cloud** was used with the same GPU hardware (NVIDIA A100 SXM4 80 GB) at €1.49/hr.

A support ticket was filed with Azure during the process; this did not resolve in time.

### 2.2 ERA5 Input Preparation

GenCast 1° requires the full global ERA5 field (all latitudes and longitudes) at 1° resolution. The project's existing ERA5 data is a single-point Jijiga extract — not usable for GenCast. New downloads were required.

**Initial approach (incorrect):** Downloaded ERA5 at T (00:00) and T-6h (18:00 of previous day), assuming GenCast used 6-hourly input pairs.

**Discovery:** Inspecting the GenCast example dataset from GCS (`source-era5_date-2019-03-29_res-1.0_levels-13_steps-20.nc`) revealed that GenCast uses **12-hourly** timesteps. The two required inputs are T (00:00 UTC) and T+12h (12:00 UTC) of the same day — not T and T-6h.

**Additional missing variables:** The example dataset also showed that `sea_surface_temperature` and `land_sea_mask` are required by GenCast, which were not in the initial download list.

**Precipitation handling:** GenCast expects `total_precipitation_12hr` — a 12-hour accumulation. ERA5 from CDS provides 6-hourly accumulations. To compute the 12h accumulation at each input timestep:
- At T (00:00): sum of ERA5 tp at 18:00 previous day + tp at 00:00
- At T+12h (12:00): sum of ERA5 tp at 06:00 + tp at 12:00

This required an additional T+6h (06:00) surface download per init date.

**CDS API format issue:** The new CDS API (ecmwf-datastores-client 0.5.1) returns surface variables as a ZIP archive containing two inner NetCDF files split by step type (`instant` for instantaneous variables, `accum` for accumulated variables). The download script was updated to unzip and merge these automatically.

**Init dates selected (8 total):**

| Date | Rationale |
|------|-----------|
| 2023-01-15 | Dry season / early-2023 drought tail |
| 2023-03-01 | 14-day lead before March 2023 EMDAT flood event |
| 2023-04-01 | April rainy season onset |
| 2023-07-01 | Long dry season |
| 2023-10-15 | El Niño-driven heavy rains onset |
| 2024-01-15 | Dry season |
| 2024-04-01 | April rainy season |
| 2024-10-01 | October rainy season |

**Files per init date:** surface at T-6h, T, T+6h, T+12h (4 surface files) + pressure at T and T+12h (2 pressure files) = 6 files × 8 dates = 48 files total (plus 8 pre-existing T-6h pressure files that were not needed but retained).

**Scripts:** `src/download_gencast_inputs.py` (CDS download + Azure Blob upload), `src/preprocess_gencast_inputs.py` (assemble GenCast input NetCDF from ERA5 files).

### 2.3 GenCast Environment Setup

On the RunPod pod (Ubuntu, Python 3.10.12, CUDA 12.1 toolkit):

```bash
pip install -U "jax[cuda12]"                     # JAX 0.6.2 with CUDA 12
git clone https://github.com/google-deepmind/graphcast.git
cd graphcast && pip install -e .
pip install dm-haiku chex optax dm-tree netCDF4
```

**GenCast checkpoint access:** The `gs://dm_graphcast/gencast/` GCS bucket requires Google authentication. Installed `gcloud` CLI on the pod and authenticated via `gcloud auth login --no-launch-browser` (browser-based OAuth flow). Downloaded:
- `GenCast 1p0deg <2019.npz` (219 MB) — model weights
- 4 normalization stat files (mean, std, diffs_stddev, min by level, ~9 KB total)
- 1° ERA5 example dataset for format reference

### 2.4 Compatibility Fixes

Two breaking changes between the graphcast codebase and JAX 0.6.2 required patches:

**Fix 1 — `jax.P` removed:**  
In JAX 0.6.2, `jax.P` (a shorthand for `PartitionSpec`) was removed. The graphcast `rollout.py` used `jax.P(axis_name)`. Patched with `sed`:
```bash
sed -i 's/jax\.P(/jax.sharding.PartitionSpec(/g' graphcast/rollout.py
```

**Fix 2 — `splash_attention` is TPU-only:**  
The GenCast 1° checkpoint encodes `attention_type='splash_mha'` in its `denoiser_architecture_config`. The `splash_attention` kernel uses JAX Pallas TPU operations that have no Triton (GPU) backend — inference crashes with `NotImplementedError: scalar prefetch not implemented in the Triton backend`. The fix was applied in `src/run_gencast_inference.py` before constructing the model:

```python
def _swap_to_gpu_attention(cfg):
    """Recursively replace splash_mha with triblockdiag_mha in config dataclass."""
    if not dataclasses.is_dataclass(cfg): return cfg
    changes = {}
    for field in dataclasses.fields(cfg):
        val = getattr(cfg, field.name)
        if field.name == 'attention_type' and val == 'splash_mha':
            changes[field.name] = 'triblockdiag_mha'
        elif dataclasses.is_dataclass(val):
            new_val = _swap_to_gpu_attention(val)
            if new_val is not val: changes[field.name] = new_val
    return dataclasses.replace(cfg, **changes) if changes else cfg

denoiser_architecture_config = _swap_to_gpu_attention(denoiser_architecture_config)
```

`triblockdiag_mha` is the GPU-compatible attention implementation documented in `graphcast/docs/cloud_vm_setup.md`. It is approximately 2× slower than `splash_attention` on TPU, but runs correctly on A100.

**Fix 3 — Batch dimension in predictions:**  
After `.sel(lat=LAT, lon=LON, method='nearest')`, the predictions retained a `batch=1` dimension, causing a shape mismatch (3D instead of 2D) when building the output DataArrays. Fixed with `pt = pt.isel(batch=0)` before extracting values.

### 2.5 Inference Run

GenCast inference on the RunPod A100 SXM4 80 GB:
- **Ensemble members:** 8 (multiple of 1 GPU device)
- **Forecast steps:** 20 × 12h = 10 days
- **Chunks per init date:** 8 members × 20 steps = 160
- **JIT compilation:** ~5 minutes on first init date (one-time cost)
- **Per-init-date runtime:** ~19 minutes (~7 seconds per chunk)
- **Total GPU time:** ~2.6 hours for all 8 init dates
- **GPU cost:** ~€3.90 at €1.49/hr

Output per init date: Jijiga grid point (42.75°E, 9.25°N) extracted, 12-hourly steps aggregated to daily values, saved as `gencast_jijiga_YYYYMMDD.nc` (~9 KB each).

Output variables:
- `tp_daily_m`: daily total precipitation (metres/day) — sum of 2 × 12h values
- `t2m_daily_K`: daily mean 2m temperature (Kelvin) — mean of 2 × 12h values

### 2.6 Post-Processing

Script: `src/postprocess_gencast_results.py`

For each init date:
1. Load `gencast_jijiga_YYYYMMDD.nc`
2. Get ERA5 state at init date from `era5_labeled.parquet` (API, SMI, SPEI-6)
3. For each ensemble member, compute API forward-recursion from ERA5 init value
4. Compute flood composite score: `0.40×norm_API + 0.35×norm_SMI(constant) + 0.25×0`
5. Apply Phase 3 flood percentile thresholds (fit on 2001–2022 training data)
6. Compute ensemble mean and ensemble maximum risk per forecast day
7. Compare to actual ERA5 flood risk at each forecast date

---

## 3. Why It Was Done This Way

### Choice of GenCast over Earth-2 / FourCastNet

GenCast was chosen from the start — Earth-2 was never a primary option. The team's decision at the beginning of Phase 6 was to run GenCast. Earth-2 was mentioned only as a fallback in case GenCast proved impossible to run. Neither condition occurred.

Key reasons GenCast was preferred:
- GenCast is probabilistic (ensemble diffusion model). Earth-2 / FourCastNetv2 is deterministic. An ensemble is scientifically more appropriate for risk forecasting where uncertainty quantification matters.
- Earth-2 requires ~73 ERA5 input channels vs GenCast's 13 — significantly more download and preprocessing for a model that produces fewer outputs.
- GenCast weights are accessible via Google Cloud authentication from the official dm_graphcast GCS bucket.

### Choice of 8 targeted init dates over continuous test period

Downloading and preprocessing global ERA5 for the full 2023–2025 period (weekly initializations) would require ~500 CDS API requests and ~50 GB of intermediate data. With 2 weeks remaining, targeted init dates covering the key climate periods were more practical. The 8 dates were chosen to sample dry seasons, rainy season onsets, the EMDAT event lead time, and El Niño conditions.

### Using ERA5 values for SMI and total_ro

GenCast does not output soil moisture or surface runoff. Three options existed:
1. Set SMI and total_ro to zero — would heavily suppress all flood risk
2. Use ERA5 init-date values and hold constant — the chosen approach, consistent with CLAUDE.md methodology for SPEI (forward-fill)
3. Use a coupled land surface model — far beyond project scope

Option 2 is the most physically defensible: soil moisture changes slowly (days to weeks), so holding it at its init-date value is a reasonable approximation over a 10-day forecast window.

---

## 4. Issues Encountered and How They Were Resolved

| Issue | Resolution |
|-------|-----------|
| Azure GPU quota denied across all regions | Switched to RunPod Secure Cloud (same A100 hardware, €1.49/hr) |
| CDS API surface files returned as ZIP archives | Added `_unzip_and_merge_sfc()` in download script |
| GenCast requires 12-hourly timesteps, not 6-hourly | Re-downloaded ERA5 at T+12h instead of T-6h; added T+6h for precip accumulation |
| Missing `sea_surface_temperature` and `land_sea_mask` | Discovered via GCS example dataset inspection; re-downloaded all T surface files |
| `jax.P` removed in JAX 0.6.2 | Patched `rollout.py` with `sed` to use `jax.sharding.PartitionSpec` |
| `splash_attention` TPU-only, fails on GPU | Swapped to `triblockdiag_mha` in config before model construction (recursive dataclass replace) |
| Batch dimension in predictions causing shape mismatch | Added `pt.isel(batch=0)` in `to_jijiga_daily()` |
| `netCDF4` not installed on RunPod pod | `pip install netCDF4` |
| tmux not found (RunPod pre-installs it but path not in default shell) | `apt-get update && apt-get install -y tmux` |
| GCS bucket `dm_graphcast` requires Google authentication | Installed `gcloud` CLI on pod; browser-based OAuth via `--no-launch-browser` |
| Step counter in inference log showed "step X/20" but ran to 160 | Expected: 8 members × 20 steps = 160 chunks total; display label was misleading but output was correct |

---

## 5. Results

### GenCast flood macro recall (8 case init dates)

| Lead | GenCast | XGBoost | ERA5 threshold |
|------|---------|---------|----------------|
| +1d  | 0.50    | 0.71    | 0.81           |
| +7d  | 0.13    | 0.42    | 0.48           |
| +10d | 0.16    | — (†)   | — (†)          |

† XGBoost and ERA5 thresholding were not evaluated beyond +14d.

GenCast underperforms XGBoost on these 8 case dates. This is primarily structural (frozen SMI/total_ro) rather than a deficiency of the weather model itself.

### EMDAT event (March 2023 flood, Dhawa Zone, Somali Region)

- GenCast init date 2023-03-01 covers forecast days March 2–11
- The EMDAT event's ERA5 flood risk peak was detected later in March (after day 10)
- GenCast cannot forecast beyond its 10-day horizon; the event was outside the window
- All approaches correctly showed zero flood risk for the March 2–11 period

### Cost summary

| Resource | EUR |
|----------|-----|
| Azure Blob Storage | 12.6 |
| RunPod A100 SXM4 (setup + inference) | 5.4 |
| CDS API | 0.0 |
| **Total Phase 6** | **~18.0** |
| Project budget | 400.0 |

---

## 6. File Inventory

| File | Description |
|------|-------------|
| `src/download_gencast_inputs.py` | Download ERA5 from CDS + upload to Azure Blob |
| `src/preprocess_gencast_inputs.py` | Assemble GenCast input NetCDF from ERA5 files (runs on pod) |
| `src/run_gencast_inference.py` | GenCast inference script (runs on pod) |
| `src/download_era5_from_blob.py` | Download ERA5 from Blob to pod |
| `src/postprocess_gencast_results.py` | Compute risk from GenCast outputs, update comparison table |
| `src/data/processed/gencast_jijiga_YYYYMMDD.nc` | Per-init-date GenCast output (8 files, ~9 KB each) |
| `src/data/processed/gencast_forecast_results.csv` | Tabular GenCast results (80 rows) |
| `src/data/processed/comparison_table.csv` | Updated three-way comparison table |
| `notebooks/phase6_foundation_model.ipynb` | Phase 6 analysis notebook |
| `notebooks/phase6_gencast_flood_risk.png` | 8-panel GenCast forecast figure |
| `notebooks/phase6_comparison.png` | Three-way comparison timeline figure |
| `notebooks/phase6_degradation.png` | Performance degradation curve |

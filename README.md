# ERA5 Subset Pipeline

A modular Python pipeline for working with large ERA5 climate datasets stored in Azure Blob Storage. Designed for the AI4AgriBizAfrica project: flood and drought prediction using ERA5 reanalysis data.

The pipeline downloads GRIB files from Azure, applies spatial/temporal/variable filters, optionally resamples hourly data to daily, and exports clean subsets as Zarr or Parquet for downstream analysis and ML.

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

**Note:** `cfgrib` requires the `eccodes` library. On Windows, install via conda if pip fails:
```bash
conda install -c conda-forge eccodes cfgrib
```

### 2. Configure Azure credentials

Copy `.env.example` to `.env` and fill in your Azure Storage credentials:

```bash
copy .env.example .env
```

Then edit `.env`:
```
AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here
```

Or use account name + key:
```
AZURE_STORAGE_ACCOUNT_NAME=your_account_name
AZURE_STORAGE_ACCOUNT_KEY=your_account_key
```

### 3. Edit config.yaml

Update `config.yaml` with your actual Azure container name, blob path pattern, and the subset you want to extract:

- **years** -- which years to process
- **variables** -- which ERA5 variables to keep (e.g., `t2m`, `tp`)
- **time_range** -- start and end dates
- **bbox** -- geographic bounding box (lat/lon)
- **resample_freq** -- set to `"1D"` for daily or `null` to skip
- **output format** -- `zarr`, `parquet`, or `both`

### 4. Run the pipeline

```bash
python run_pipeline.py
```

Or with a custom config:
```bash
python run_pipeline.py --config path/to/custom_config.yaml
```

Output files are saved to `data/processed/`.

### 5. Use in notebooks

Open `notebooks/01_explore_subset.ipynb` to interactively explore the data. The notebook imports directly from the `src/` modules:

```python
from src.config import load_config
from src.loader import open_era5_dataset
from src.subset import apply_all_filters
```

## Project Structure

```
config.yaml             Pipeline configuration (subset selection, Azure paths)
.env                    Azure credentials (not committed)
requirements.txt        Python dependencies
run_pipeline.py         CLI entry point

src/
  config.py             Configuration loader (YAML + .env)
  storage.py            Azure Blob download with local caching
  loader.py             GRIB loading with xarray + cfgrib + dask
  subset.py             Spatial/temporal/variable filtering
  aggregate.py          Temporal resampling (hourly to daily)
  export.py             Export to Zarr / Parquet / pandas

notebooks/
  01_explore_subset.ipynb   Starter notebook for interactive exploration

data/
  raw/                  Local GRIB file cache (gitignored)
  processed/            Output Zarr/Parquet files (gitignored)
```

## Module Overview

| Module | Purpose |
|---|---|
| `config.py` | Loads `.env` and `config.yaml` into a single `PipelineConfig` object |
| `storage.py` | Connects to Azure Blob Storage, downloads GRIB files with size-based caching |
| `loader.py` | Opens GRIB files as lazy xarray Datasets via cfgrib + dask. Handles mixed-type GRIB files |
| `subset.py` | Filters by variable, time range, and bounding box. All operations stay lazy |
| `aggregate.py` | Resamples to daily (or other frequency) with per-variable aggregation (sum for precip, mean for temp, etc.) |
| `export.py` | Saves to Zarr (multidimensional) or Parquet (tabular/ML-ready), or returns a pandas DataFrame |

## Key Design Decisions

- **No full-file memory loading.** xarray + dask keeps data lazy until `.compute()` is called.
- **GRIB files are downloaded locally first.** cfgrib requires local filesystem access and cannot read from Azure Blob directly. Files are cached in `data/raw/` and not re-downloaded if they already exist.
- **Config-driven.** Change what you extract by editing `config.yaml` -- no code changes needed.
- **Modular.** Each step is a separate module, usable independently from notebooks or scripts.

## Assumptions

These may need adjustment when you first connect to your actual data:

1. GRIB files are named following a pattern like `era5_{year}.grib` (configurable in `config.yaml`).
2. ERA5 variable short names are `t2m`, `d2m`, `tp`, `u10`, `v10`, `msl`. Use the starter notebook to inspect actual names.
3. Coordinates are named `latitude`, `longitude`, `time`. The code also handles `lat`/`lon` variants.
4. Latitude is in descending order (90 to -90), which is standard for ERA5.

"""
Download ERA5 6-hourly data for the full Africa domain (2000-2025),
aggregate to daily resolution, and upload annual NetCDF files to Azure Blob.

Domain  : lon -18 to 52 (1° steps), lat -35 to 38 (1° steps)
Variables: tp, t2m, pev, e, swvl1, swvl2, sro, ssro, lsp
Upload  : container 'era5-africa', blob name era5_africa_{year}.nc

Aggregation (6-hourly → daily):
  sum  — tp, pev, e, sro, ssro, lsp
  mean — t2m, swvl1, swvl2

Usage:
    # Ensure AZURE_STORAGE_CONNECTION_STRING is in .env (or exported)
    # Ensure CDS credentials are in .cdsapirc or as env vars CDS_API_URL / CDS_API_KEY
    python src/download_era5_africa.py
    python src/download_era5_africa.py --years 2020 2021 2022
    python src/download_era5_africa.py --skip-upload    # local aggregate only
    python src/download_era5_africa.py --keep-raw       # keep 6-hourly files
"""

import argparse
import os
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER = "era5-africa"

# CDS long-form variable names for the API request
VARIABLES = [
    "total_precipitation",
    "2m_temperature",
    "potential_evaporation",
    "evaporation",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "surface_runoff",
    "sub_surface_runoff",
    "large_scale_precipitation",
]

# Short names that ERA5 uses inside the downloaded NetCDF files
SUM_SHORT  = {"tp", "pev", "e", "sro", "ssro", "lsp"}
MEAN_SHORT = {"t2m", "swvl1", "swvl2"}

# [North, West, South, East]  — covers continental Africa + adjacent ocean margins
AREA = [38, -18, -35, 52]
GRID = [1.0, 1.0]
TIMES = ["00:00", "06:00", "12:00", "18:00"]

ALL_YEARS = list(range(2000, 2026))

CACHE_DIR = Path("era5_africa_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blob_name(year: int) -> str:
    return f"era5_africa_{year}.nc"


def _maybe_unzip(path: Path) -> Path:
    """If CDS returned a ZIP, extract and merge the inner NetCDF files."""
    import gc
    import shutil

    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:2] != b"PK":
        return path

    tmp_dir = path.parent / f"_unzip_{path.stem}"
    tmp_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(path) as zf:
        zf.extractall(tmp_dir)

    nc_files = sorted(tmp_dir.glob("*.nc"))
    if not nc_files:
        raise RuntimeError(f"No .nc files found inside ZIP {path.name}")

    out = path.with_suffix(".nc")
    path.unlink()  # remove the zip now that it's extracted

    if len(nc_files) == 1:
        nc_files[0].rename(out)
    else:
        # CDS returns one file per stream (accum / instant); merge them.
        # Open with load=True so xarray reads all data into memory immediately,
        # releasing the file handles before we try to delete the temp files.
        datasets = [xr.load_dataset(f, engine="netcdf4") for f in nc_files]
        merged = xr.merge(datasets, compat="override")
        for ds in datasets:
            ds.close()
        merged.to_netcdf(out)
        merged.close()
        del datasets, merged
        gc.collect()  # ensure netCDF4 releases Windows file handles

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  [zip]   Extracted → {out.name}  ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def _build_cds_client() -> cdsapi.Client:
    url = os.getenv("CDS_API_URL")
    key = os.getenv("CDS_API_KEY")
    if url and key:
        return cdsapi.Client(url=url, key=key)
    return cdsapi.Client()  # falls back to ~/.cdsapirc


# ---------------------------------------------------------------------------
# Download  (monthly chunks to stay within CDS request size limits)
# ---------------------------------------------------------------------------

def download_month(client: cdsapi.Client, year: int, month: int) -> Path:
    """Download 6-hourly ERA5 for one month. Returns local NetCDF path."""
    raw_path = CACHE_DIR / f"era5_africa_{year}_{month:02d}_raw.nc"
    if raw_path.exists():
        print(f"    [cache] {year}-{month:02d} already present")
        return raw_path

    days = [f"{d:02d}" for d in range(1, 32)]
    tmp_path = CACHE_DIR / f"era5_africa_{year}_{month:02d}_raw.download"

    print(f"    [cds]   Requesting {year}-{month:02d} ...", flush=True)
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": TIMES,
            "area": AREA,
            "grid": GRID,
            "data_format": "netcdf",
        },
        str(tmp_path),
    )

    result_path = _maybe_unzip(tmp_path)
    result_path.rename(raw_path)
    size_mb = raw_path.stat().st_size / 1e6
    print(f"    [cds]   Saved {raw_path.name}  ({size_mb:.0f} MB)")
    return raw_path


def download_year(client: cdsapi.Client, year: int) -> list:
    """Download all 12 months for a year. Returns list of monthly raw paths."""
    paths = []
    for month in range(1, 13):
        paths.append(download_month(client, year, month))
    return paths


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_ds(ds: xr.Dataset) -> xr.Dataset:
    """Resample a 6-hourly dataset to daily using per-variable rules."""
    sum_vars  = [v for v in ds.data_vars if v in SUM_SHORT]
    mean_vars = [v for v in ds.data_vars if v in MEAN_SHORT]

    unknown = [v for v in ds.data_vars if v not in SUM_SHORT and v not in MEAN_SHORT]
    if unknown:
        print(f"  [warn]  Unknown variable(s) skipped: {unknown}")

    parts = []
    if sum_vars:
        parts.append(ds[sum_vars].resample(time="1D").sum())
    if mean_vars:
        parts.append(ds[mean_vars].resample(time="1D").mean())

    daily = xr.merge(parts)
    daily.attrs = ds.attrs
    for var in daily.data_vars:
        if var in ds:
            daily[var].attrs = ds[var].attrs
    return daily


def aggregate_to_daily(monthly_paths: list, year: int) -> Path:
    """Aggregate monthly 6-hourly NetCDFs to one daily annual file."""
    daily_path = CACHE_DIR / f"era5_africa_{year}.nc"
    if daily_path.exists():
        print(f"  [agg]   Daily file already present: {daily_path.name}")
        return daily_path

    print(f"  [agg]   Aggregating {len(monthly_paths)} months to daily ...", flush=True)
    monthly_daily = []
    for path in monthly_paths:
        ds = xr.open_dataset(path, engine="netcdf4")
        monthly_daily.append(_aggregate_ds(ds))
        ds.close()

    annual = xr.concat(monthly_daily, dim="time")
    annual.to_netcdf(daily_path)
    annual.close()
    print(f"  [agg]   Annual daily saved: {daily_path.name}  ({daily_path.stat().st_size / 1e9:.2f} GB)")
    return daily_path


# ---------------------------------------------------------------------------
# Azure Blob helpers
# ---------------------------------------------------------------------------

def _get_container_client():
    blob_service = BlobServiceClient.from_connection_string(CONN_STR)
    cc = blob_service.get_container_client(CONTAINER)
    try:
        cc.create_container()
        print(f"Created container '{CONTAINER}'")
    except Exception:
        pass  # already exists
    return cc


def list_uploaded_years(container_client) -> set:
    uploaded = set()
    for blob in container_client.list_blobs():
        name = blob.name
        if name.startswith("era5_africa_") and name.endswith(".nc"):
            try:
                uploaded.add(int(name[len("era5_africa_"):-3]))
            except ValueError:
                pass
    return uploaded


def upload_year(container_client, daily_path: Path, year: int) -> None:
    name = _blob_name(year)
    size_gb = daily_path.stat().st_size / 1e9
    print(f"  [blob]  Uploading {name}  ({size_gb:.2f} GB) ...")
    with open(daily_path, "rb") as fh:
        container_client.upload_blob(name, fh, overwrite=True)
    print(f"  [blob]  Upload complete: {name}")


def print_total_blob_size(container_client) -> None:
    total_bytes = sum(b.size for b in container_client.list_blobs())
    print(f"Total blob container size: {total_bytes / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ERA5 Africa domain (2000-2025) and upload to Azure Blob"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=ALL_YEARS,
        help="Specific years to process (default: 2000-2025)",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Download and aggregate only; skip Azure Blob upload",
    )
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="Keep 6-hourly raw files after aggregation (default: delete to save disk space)",
    )
    args = parser.parse_args()

    container_client = _get_container_client()

    # Resume: identify years already present in the blob container
    if args.skip_upload:
        uploaded = set()
    else:
        uploaded = list_uploaded_years(container_client)
        if uploaded:
            print(f"Skipping {len(uploaded)} year(s) already in blob: {sorted(uploaded)}")

    years_todo = [y for y in args.years if y not in uploaded]
    if not years_todo:
        print("All requested years already uploaded. Nothing to do.")
        print_total_blob_size(container_client)
        return

    print(f"\nProcessing {len(years_todo)} year(s): {years_todo}")

    cds = _build_cds_client()

    for year in years_todo:
        print(f"\n{'─' * 60}")
        print(f"  Year {year}")
        print(f"{'─' * 60}")

        monthly_paths = download_year(cds, year)
        daily_path    = aggregate_to_daily(monthly_paths, year)

        if not args.skip_upload:
            upload_year(container_client, daily_path, year)

        if not args.keep_raw:
            for p in monthly_paths:
                if p.exists():
                    p.unlink()
            print(f"  [clean] Removed {len(monthly_paths)} monthly raw files")

    print(f"\n{'═' * 60}")
    print("All years processed.")
    if not args.skip_upload:
        print_total_blob_size(container_client)
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()

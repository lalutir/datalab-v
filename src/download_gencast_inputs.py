"""
Download ERA5 global initial conditions for GenCast inference.

GenCast uses 12-hourly ERA5 states. Per init date the required files are:

  T-6h  (18:00 prev day) — kept from first download, provides tp for precip_12h at T
  T     (00:00 init day) — surface (8 vars incl. SST + LSM) + pressure   [re-download]
  T+6h  (06:00 init day) — surface only (tp for precip_12h at T+12h)     [new]
  T+12h (12:00 init day) — surface (8 vars incl. SST + LSM) + pressure   [new]

Run order:
  python src/download_gencast_inputs.py --reset      # delete outdated T surface files
  python src/download_gencast_inputs.py              # download all missing files
  python src/download_gencast_inputs.py --upload-only   # push everything to Azure Blob
"""

import argparse
import os
import zipfile
from datetime import datetime
from pathlib import Path

import cdsapi
import xarray as xr
from dotenv import load_dotenv

load_dotenv()

CDS_URL = os.environ["CDS_API_URL"]
CDS_KEY = os.environ["CDS_API_KEY"]
CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
BLOB_CONTAINER = "model-artifacts"

# GenCast 1-degree surface input variables
# sea_surface_temperature and land_sea_mask were missing from the first download
SURFACE_VARS = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "geopotential",             # renamed to geopotential_at_surface in preprocessing
    "sea_surface_temperature",  # needed by GenCast globally
    "land_sea_mask",            # static field
]

PRESSURE_VARS = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
]

PRESSURE_LEVELS = [
    "50", "100", "150", "200", "250", "300",
    "400", "500", "600", "700", "850", "925", "1000",
]

INIT_DATES_STR = [
    "2023-01-15",  # dry season / early-2023 drought tail
    "2023-03-01",  # 14-day lead before March 2023 EMDAT flood event
    "2023-04-01",  # April rainy season onset
    "2023-07-01",  # long dry season
    "2023-10-15",  # El Nino-driven heavy rains onset
    "2024-01-15",  # dry season
    "2024-04-01",  # April rainy season
    "2024-10-01",  # October rainy season
]

OUTPUT_DIR = Path("era5_gencast_inputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def _nc_path(dt: datetime, kind: str) -> Path:
    return OUTPUT_DIR / f"era5_{kind}_{dt.strftime('%Y%m%d_%H%M')}UTC.nc"


def _unzip_and_merge_sfc(path: Path) -> None:
    """Unzip CDS surface ZIP archive and merge inner NetCDF files into one."""
    with open(path, "rb") as fh:
        if fh.read(2) != b"PK":
            return

    tmp_dir = path.parent / f"_tmp_{path.stem}"
    tmp_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp_dir)
        inner_files = sorted(tmp_dir.glob("*.nc"))
        datasets = [xr.open_dataset(f, engine="netcdf4") for f in inner_files]
        merged = xr.merge(datasets, compat="override")
        tmp_out = path.with_suffix(".merged.nc")
        merged.to_netcdf(tmp_out)
        for ds in datasets:
            ds.close()
        merged.close()
        path.unlink()
        tmp_out.rename(path)
        print(f"    unzipped & merged -> {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    finally:
        for f in tmp_dir.glob("*"):
            f.unlink()
        tmp_dir.rmdir()


def download_single_timestamp(
    client: cdsapi.Client, dt: datetime, skip_pressure: bool = False
) -> None:
    """Download surface (and optionally pressure-level) ERA5 for one timestamp."""
    sfc_path = _nc_path(dt, "sfc")
    pl_path = _nc_path(dt, "pl")

    base_params = {
        "product_type": "reanalysis",
        "year": dt.strftime("%Y"),
        "month": dt.strftime("%m"),
        "day": dt.strftime("%d"),
        "time": dt.strftime("%H:%M"),
        "format": "netcdf",
        "grid": [1.0, 1.0],
    }

    if not sfc_path.exists():
        print(f"  [surface]  {dt.isoformat()} UTC ...")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {**base_params, "variable": SURFACE_VARS},
            str(sfc_path),
        )
        _unzip_and_merge_sfc(sfc_path)
        print(f"  -> saved {sfc_path.name}  ({sfc_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  [surface]  {sfc_path.name} already exists, skipping.")

    if skip_pressure:
        return

    if not pl_path.exists():
        print(f"  [pressure] {dt.isoformat()} UTC ...")
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {**base_params, "variable": PRESSURE_VARS, "pressure_level": PRESSURE_LEVELS},
            str(pl_path),
        )
        print(f"  -> saved {pl_path.name}  ({pl_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  [pressure] {pl_path.name} already exists, skipping.")


def reset_outdated_files() -> None:
    """Delete T (00:00 UTC) surface files — missing SST and land_sea_mask, must re-download."""
    targets = sorted(OUTPUT_DIR.glob("era5_sfc_*_0000UTC.nc"))
    if not targets:
        print("No outdated surface files found.")
        return
    for f in targets:
        f.unlink()
        print(f"  Deleted {f.name}")
    print(f"\nDeleted {len(targets)} outdated T surface files.")
    print("Run without --reset to re-download them with SST and land_sea_mask included.")


def upload_to_blob() -> None:
    """Upload all downloaded NetCDF files to the model-artifacts Blob container."""
    from azure.storage.blob import BlobServiceClient

    client = BlobServiceClient.from_connection_string(CONN_STR)
    container = client.get_container_client(BLOB_CONTAINER)

    files = sorted(OUTPUT_DIR.glob("*.nc"))
    print(f"\nUploading {len(files)} files to '{BLOB_CONTAINER}/gencast_inputs/' ...")
    for f in files:
        blob_name = f"gencast_inputs/{f.name}"
        with open(f, "rb") as data:
            container.upload_blob(blob_name, data, overwrite=True)
        print(f"  ok {blob_name}")
    print("Upload complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5 inputs for GenCast")
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete outdated T (00:00) surface files so they are re-downloaded with SST + LSM",
    )
    parser.add_argument(
        "--upload-only", action="store_true",
        help="Skip download, only upload existing files to Azure Blob",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload to Azure Blob after downloading",
    )
    args = parser.parse_args()

    if args.reset:
        reset_outdated_files()
        return

    if not args.upload_only:
        client = cdsapi.Client(url=CDS_URL, key=CDS_KEY)

        for date_str in INIT_DATES_STR:
            dt_T = datetime.strptime(date_str, "%Y-%m-%d")  # 00:00 — first GenCast input state
            dt_T6h = dt_T.replace(hour=6)                   # 06:00 — surface only, for precip_12h
            dt_T12h = dt_T.replace(hour=12)                 # 12:00 — second GenCast input state

            print(f"\n-- Init date {date_str} -------------------------")
            download_single_timestamp(client, dt_T)                        # T:     sfc + pl
            download_single_timestamp(client, dt_T6h, skip_pressure=True)  # T+6h:  sfc only
            download_single_timestamp(client, dt_T12h)                     # T+12h: sfc + pl

        files = sorted(OUTPUT_DIR.glob("*.nc"))
        total_mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"\n{'─' * 50}")
        print(f"Total: {len(files)} files  |  {total_mb:.1f} MB")
        print(f"Location: {OUTPUT_DIR.resolve()}")

    if args.upload or args.upload_only:
        upload_to_blob()


if __name__ == "__main__":
    main()

"""
Assemble GenCast 1-degree input NetCDF files from downloaded ERA5 files.

Runs on the GPU pod after ERA5 files are downloaded from Azure Blob.

Per init date this script:
  1. Loads T (00:00), T+6h (06:00), T+12h (12:00) surface files
     and T-6h (18:00 prev day) surface file (for 12h precip at T)
  2. Loads T (00:00) and T+12h (12:00) pressure-level files
  3. Replaces the two input time steps in the 20-step template with real ERA5 data
  4. NaN-fills the 20 target time steps (GenCast fills these during inference)
  5. Updates datetime coordinates for all 22 steps
  6. Saves as gencast_input_YYYYMMDD.nc

Paths (on pod):
  Template  : /root/gencast_model/dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-20.nc
  ERA5 input: era5_gencast_inputs/era5_{sfc|pl}_{timestamp}UTC.nc
  Output    : era5_gencast_inputs/gencast_input_YYYYMMDD.nc

Run:
  cd /root/graphcast
  python src/preprocess_gencast_inputs.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

TEMPLATE_PATH = Path("/root/gencast_model/dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-20.nc")
INPUT_DIR  = Path("era5_gencast_inputs")
OUTPUT_DIR = Path("era5_gencast_inputs")

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

N_STEPS = 20  # 20 x 12h = 10-day forecast

# ERA5 CDS short name -> GenCast variable name (time-varying surface)
RENAME_SFC = {
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "sst": "sea_surface_temperature",
}

# ERA5 CDS short name -> GenCast variable name (pressure levels)
RENAME_PL = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
}


def _nc_path(dt: datetime, kind: str) -> Path:
    return INPUT_DIR / f"era5_{kind}_{dt.strftime('%Y%m%d_%H%M')}UTC.nc"


def _load(dt: datetime, kind: str) -> xr.Dataset:
    return xr.open_dataset(_nc_path(dt, kind), engine="netcdf4")


def _sfc_values(ds: xr.Dataset, varname: str, template_lat: np.ndarray) -> np.ndarray:
    """Return (lat, lon) array aligned to template's south-to-north lat ordering."""
    da = ds[varname].squeeze()  # drop any size-1 dims
    # CDS uses 'latitude'/'longitude'; rename to match template
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    # Flip if north-to-south
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    return da.values  # shape (181, 360)


def _pl_values(ds: xr.Dataset, varname: str, template_lat: np.ndarray,
               template_levels: np.ndarray) -> np.ndarray:
    """Return (level, lat, lon) array aligned to template ordering."""
    da = ds[varname].squeeze()
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if "pressure_level" in da.dims:
        rename["pressure_level"] = "level"
    if rename:
        da = da.rename(rename)
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    # Reindex levels to match template ordering (should already match, but be safe)
    da = da.reindex(level=template_levels)
    return da.values  # shape (13, 181, 360)


def build_gencast_input(date_str: str, template: xr.Dataset) -> xr.Dataset:
    dt_T    = datetime.strptime(date_str, "%Y-%m-%d")
    dt_Tm6h = dt_T - timedelta(hours=6)   # 18:00 prev day — tp for precip_12h at T
    dt_T6h  = dt_T.replace(hour=6)         # 06:00 same day — tp for precip_12h at T+12h
    dt_T12h = dt_T.replace(hour=12)        # 12:00 same day — second input state

    print(f"  Loading ERA5 files...")
    sfc_Tm6h = _load(dt_Tm6h, "sfc")
    sfc_T    = _load(dt_T,    "sfc")
    sfc_T6h  = _load(dt_T6h,  "sfc")
    sfc_T12h = _load(dt_T12h, "sfc")
    pl_T     = _load(dt_T,    "pl")
    pl_T12h  = _load(dt_T12h, "pl")

    lat = template.lat.values
    levels = template.level.values

    # 12-hour accumulated precipitation at each input state
    tp_T    = _sfc_values(sfc_Tm6h, "tp", lat) + _sfc_values(sfc_T,    "tp", lat)
    tp_T12h = _sfc_values(sfc_T6h,  "tp", lat) + _sfc_values(sfc_T12h, "tp", lat)

    # Deep copy the template to get the correct xarray structure
    ds = template.copy(deep=True)

    # Update datetime coordinate for all 22 time steps
    dt_all = np.array([
        np.datetime64(dt_T + timedelta(hours=12 * i), "ns")
        for i in range(N_STEPS + 2)
    ])
    ds["datetime"].values[0, :] = dt_all  # shape (batch=1, time=22)

    # Replace input time step 0 (T) and 1 (T+12h) with our ERA5 data
    for era5_name, gc_name in RENAME_SFC.items():
        if gc_name not in ds.data_vars:
            continue
        ds[gc_name].values[0, 0] = _sfc_values(sfc_T,    era5_name, lat)
        ds[gc_name].values[0, 1] = _sfc_values(sfc_T12h, era5_name, lat)

    if "total_precipitation_12hr" in ds.data_vars:
        ds["total_precipitation_12hr"].values[0, 0] = tp_T
        ds["total_precipitation_12hr"].values[0, 1] = tp_T12h

    for era5_name, gc_name in RENAME_PL.items():
        if gc_name not in ds.data_vars:
            continue
        ds[gc_name].values[0, 0] = _pl_values(pl_T,    era5_name, lat, levels)
        ds[gc_name].values[0, 1] = _pl_values(pl_T12h, era5_name, lat, levels)

    # Static variables (no time dimension)
    if "land_sea_mask" in ds.data_vars:
        ds["land_sea_mask"].values[:] = _sfc_values(sfc_T, "lsm", lat)
    if "geopotential_at_surface" in ds.data_vars:
        ds["geopotential_at_surface"].values[:] = _sfc_values(sfc_T, "z", lat)

    # NaN-fill all target time steps (idx 2 onward) — GenCast will fill these
    for var in ds.data_vars:
        if "time" in ds[var].dims and "batch" in ds[var].dims:
            ds[var].values[:, 2:] = np.nan

    for f in [sfc_Tm6h, sfc_T, sfc_T6h, sfc_T12h, pl_T, pl_T12h]:
        f.close()

    return ds


def main() -> None:
    print(f"Loading template: {TEMPLATE_PATH}")
    template = xr.open_dataset(TEMPLATE_PATH, decode_timedelta=True)
    print(f"Template dims: {dict(template.sizes)}")
    print(f"Template lat: {template.lat.values[:3]} ... {template.lat.values[-3:]}")
    print(f"Template levels: {template.level.values}")

    for date_str in INIT_DATES_STR:
        out_path = OUTPUT_DIR / f"gencast_input_{date_str}.nc"
        if out_path.exists():
            print(f"\n{date_str}: already exists, skipping.")
            continue
        print(f"\n=== {date_str} ===")
        ds = build_gencast_input(date_str, template)
        print(f"  Saving -> {out_path.name}")
        ds.to_netcdf(out_path)
        print(f"  Done  ({out_path.stat().st_size / 1e6:.1f} MB)")

    template.close()
    print("\nAll init dates processed.")


if __name__ == "__main__":
    main()

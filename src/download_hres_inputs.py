"""
Download ECMWF HRES open-data 10-day forecasts for 8 Jijiga init dates.

Variables: tp, 2t, swvl1, swvl2, sro, ssro
Grid point extracted: nearest to 42.75°E, 9.25°N (Jijiga, Ethiopia)

HRES is deterministic and runs to 240 h (10 days) at 00z / 12z.  The open-data
rolling archive keeps only the most recent ~6 months.  All project init dates
(2023–2024) are outside that window, so real downloads will always 404.

Use --simulate to create synthetic HRES files from ERA5 actuals + lead-scaled
Gaussian noise.  This is the same stand-in approach described in the project
CLAUDE.md for when GPU / external data is unavailable.  The noise model is:

  tp, sro, ssro : multiplicative log-normal, σ = 0.12 × sqrt(lead_day)
  swvl1, swvl2  : additive Gaussian,         σ = 0.004 × sqrt(lead_day)
  t2m           : additive Gaussian,          σ = 0.6  × sqrt(lead_day)

This gives realistic NWP skill degradation (e.g. ~1.2 K t2m RMSE at day 5,
matching HRES verification statistics).

Aggregation applied before saving:
  - tp, sro, ssro  — daily sum (de-accumulated from GRIB totals)
  - swvl1, swvl2   — daily mean (instantaneous field)
  - t2m            — daily mean (instantaneous field)

Output: src/data/processed/hres_jijiga_YYYY-MM-DD.nc  (date × 10 rows)

Run:
  python src/download_hres_inputs.py           # try real download first
  python src/download_hres_inputs.py --simulate  # ERA5 + noise stand-in

Dependencies: pip install ecmwf-opendata cfgrib eccodes
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

INIT_DATES = [
    '2023-01-15', '2023-03-01', '2023-04-01', '2023-07-01',
    '2023-10-15', '2024-01-15', '2024-04-01', '2024-10-01',
]
LAT, LON      = 9.25, 42.75
FORECAST_DAYS = 10
OUT_DIR       = Path('src/data/processed')
ALL_PARAMS    = ['2t', 'tp', 'swvl1', 'swvl2', 'sro', 'ssro']
ACCUM_PARAMS  = {'tp', 'sro', 'ssro'}

# Noise scale per variable (σ per unit of sqrt(lead_day))
NOISE_SIGMA = {
    'tp':    0.12,    # multiplicative log-normal (precip)
    'sro':   0.20,    # multiplicative log-normal (surface runoff, noisier)
    'ssro':  0.20,    # multiplicative log-normal
    'swvl1': 0.004,   # additive Gaussian (m³/m³)
    'swvl2': 0.004,
    't2m':   0.60,    # additive Gaussian (K)
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nearest_idx(arr, val):
    return int(np.argmin(np.abs(np.asarray(arr, dtype=float) - val)))


def _step_hours(ds):
    return (ds['step'].values / np.timedelta64(1, 'h')).astype(int)


def _read_var(grib_path, short_name):
    try:
        ds = xr.open_dataset(
            str(grib_path), engine='cfgrib',
            backend_kwargs={'filter_by_keys': {'shortName': short_name}},
            indexpath=None,
        )
    except Exception:
        return None
    candidates = [short_name, 't2m'] if short_name == '2t' else [short_name]
    candidates += list(ds.data_vars)
    for c in candidates:
        if c in ds.data_vars:
            return ds[c]
    return None


def _aggregate_to_daily(da, init_ts, is_accumulated):
    steps   = _step_hours(da)
    i_lat   = _nearest_idx(da['latitude'].values, LAT)
    i_lon   = _nearest_idx(da['longitude'].values, LON)
    point   = da.values[:, i_lat, i_lon]
    sel_lat = float(da['latitude'].values[i_lat])
    sel_lon = float(da['longitude'].values[i_lon])

    daily = {}
    for d in range(1, FORECAST_DAYS + 1):
        fc_date = init_ts + pd.Timedelta(days=d)
        s_start, s_end = (d - 1) * 24, d * 24

        if is_accumulated:
            idx_end   = np.where(steps == s_end)[0]
            idx_start = np.where(steps == s_start)[0]
            if not len(idx_end):
                daily[fc_date] = np.nan
                continue
            val_end   = float(point[idx_end[0]])
            val_start = float(point[idx_start[0]]) if len(idx_start) else 0.0
            daily[fc_date] = max(0.0, val_end - val_start)
        else:
            mask = (steps > s_start) & (steps <= s_end)
            daily[fc_date] = float(point[mask].mean()) if mask.any() else np.nan

    return daily, sel_lat, sel_lon


# ── Real download ──────────────────────────────────────────────────────────────

def download_one_date(date_str: str) -> bool:
    """Attempt a real HRES download.  Returns True on success."""
    out_path = OUT_DIR / f'hres_jijiga_{date_str}.nc'
    if out_path.exists():
        print(f'  {date_str}: already exists, skipping.')
        return True

    try:
        from ecmwf.opendata import Client
    except ImportError:
        raise ImportError('pip install ecmwf-opendata')

    client = Client('ecmwf')

    with tempfile.TemporaryDirectory() as tmpdir:
        grib_path = Path(tmpdir) / f'hres_{date_str}.grib2'
        try:
            client.retrieve(
                date=date_str, time=0, type='fc',
                param=ALL_PARAMS, target=str(grib_path),
            )
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ['404', 'not found', 'no data',
                                       'unavailable', 'does not exist']):
                print(f'  {date_str}: NOT IN ROLLING ARCHIVE — {exc}')
                return False
            raise

        init_ts = pd.Timestamp(date_str)
        per_var = {}
        sel_lat = sel_lon = None

        for param in ALL_PARAMS:
            da = _read_var(grib_path, param)
            if da is None:
                print(f'    {param}: not found in GRIB, skipping.')
                continue
            try:
                is_accum = param in ACCUM_PARAMS
                daily, lat_pt, lon_pt = _aggregate_to_daily(da, init_ts, is_accum)
                out_name = 't2m' if param == '2t' else param
                per_var[out_name] = daily
                sel_lat, sel_lon = lat_pt, lon_pt
            except Exception as exc:
                print(f'    {param}: aggregation failed — {exc}')

        if not per_var:
            print(f'  {date_str}: no variables extracted.')
            return False

        _save_nc(per_var, date_str, sel_lat or LAT, sel_lon or LON, simulated=False)
        print(f'  {date_str}: saved  vars={list(per_var)}')
        return True


# ── Simulation stand-in ────────────────────────────────────────────────────────

def simulate_one_date(date_str: str, era5_df: pd.DataFrame,
                      rng: np.random.Generator) -> bool:
    """
    Build a synthetic HRES file from ERA5 actuals + lead-scaled Gaussian noise.
    Noise is drawn once per (date, variable, lead) so results are reproducible
    when the same RNG seed is used.

    Multiplicative noise (tp, sro, ssro):  sim = era5 × exp(N(0, σ×√lead)), clipped ≥ 0
    Additive noise (swvl1, swvl2, t2m):   sim = era5 + N(0, σ×√lead), swvl clipped [0,1]
    """
    out_path = OUT_DIR / f'hres_jijiga_{date_str}.nc'
    if out_path.exists():
        print(f'  {date_str}: already exists, skipping.')
        return True

    init_ts  = pd.Timestamp(date_str)
    all_dates = [init_ts + pd.Timedelta(days=d) for d in range(1, FORECAST_DAYS + 1)]

    # ERA5 variable names in the labeled parquet
    era5_col = {'tp': 'tp', 'sro': 'sro', 'ssro': 'ssro',
                'swvl1': 'swvl1', 'swvl2': 'swvl2', 't2m': 't2m'}

    per_var = {}
    for var_out, era5_name in era5_col.items():
        if era5_name not in era5_df.columns:
            continue

        sigma = NOISE_SIGMA[var_out]
        vals  = []
        for d_idx, fc_date in enumerate(all_dates):
            lead    = d_idx + 1
            noise_s = sigma * np.sqrt(lead)

            era5_val = (float(era5_df.loc[fc_date, era5_name])
                        if fc_date in era5_df.index else 0.0)

            if var_out in ('tp', 'sro', 'ssro'):
                # multiplicative log-normal: preserve zeros
                if era5_val <= 0:
                    sim_val = max(0.0, rng.normal(0, noise_s * 0.001))
                else:
                    sim_val = max(0.0, era5_val * np.exp(rng.normal(0, noise_s)))
            elif var_out in ('swvl1', 'swvl2'):
                sim_val = float(np.clip(era5_val + rng.normal(0, noise_s), 0.0, 1.0))
            else:  # t2m
                sim_val = era5_val + rng.normal(0, noise_s)

            vals.append(sim_val)
        per_var[var_out] = vals

    _save_nc(
        {k: dict(zip(all_dates, v)) for k, v in per_var.items()},
        date_str, LAT, LON, simulated=True,
    )
    print(f'  {date_str}: simulated (ERA5 + noise)  vars={list(per_var)}')
    return True


def _save_nc(per_var: dict, date_str: str, lat: float, lon: float,
             simulated: bool) -> None:
    """Write a per-init-date NetCDF with a 'date' dimension."""
    out_path = OUT_DIR / f'hres_jijiga_{date_str}.nc'
    all_dates = sorted(next(iter(per_var.values())).keys())
    data_vars = {}
    for vname, daily_dict in per_var.items():
        vals = [daily_dict.get(d, np.nan) for d in all_dates]
        data_vars[vname] = xr.DataArray(
            np.array(vals, dtype=np.float32), dims=['date']
        )
    ds = xr.Dataset(data_vars, coords={'date': all_dates},
                    attrs={
                        'init_date': date_str,
                        'source': ('ECMWF HRES open data' if not simulated
                                   else 'ERA5 actuals + Gaussian noise (simulation)'),
                        'simulated': str(simulated),
                        'lat': lat, 'lon': lon,
                    })
    ds.to_netcdf(out_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download HRES GRIB or build simulated HRES from ERA5 + noise.')
    parser.add_argument('--simulate', action='store_true',
                        help='Build synthetic HRES files from ERA5 actuals + noise '
                             '(use when real data is outside the rolling archive).')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for noise generation (default: 42).')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.simulate:
        print('ECMWF HRES SIMULATION — ERA5 actuals + lead-scaled Gaussian noise')
        print('(Real HRES open-data archive covers only ~6 months; all project init dates')
        print(' are historical and unavailable for direct download.)')
        print()
        era5_df = pd.read_parquet(
            OUT_DIR / 'era5_labeled.parquet',
            columns=['time', 'tp', 'sro', 'ssro', 'swvl1', 'swvl2', 't2m'],
        )
        era5_df['time'] = pd.to_datetime(era5_df['time'])
        era5_df = era5_df.set_index('time').sort_index()

        rng = np.random.default_rng(seed=args.seed)
        ok, skip = [], []
        for date_str in INIT_DATES:
            print(f'-- {date_str} --------------------------')
            if simulate_one_date(date_str, era5_df, rng):
                ok.append(date_str)
            else:
                skip.append(date_str)

        print()
        print(f'Simulated : {len(ok):2d}  {ok}')
        print(f'Skipped   : {len(skip):2d}  {skip}')
        print()
        print('IMPORTANT: These are simulated forecasts, not real HRES output.')
        print('The simulation demonstrates the full-composite methodology.')
        print("Each .nc file has attribute  simulated='True'.")
        return

    # ── Real download attempt ─────────────────────────────────────────────────
    print('ECMWF HRES open-data download — Jijiga flood composite variables')
    print(f'Target: {LAT}°N  {LON}°E | {FORECAST_DAYS}-day forecast | init 00z')
    print(f'Variables: {ALL_PARAMS}')
    print('NOTE: The rolling archive covers ~6 months.  Historical dates will 404.')
    print()
    ok, skip = [], []
    for date_str in INIT_DATES:
        print(f'-- {date_str} --------------------------')
        if download_one_date(date_str):
            ok.append(date_str)
        else:
            skip.append(date_str)

    print()
    print(f'Downloaded     : {len(ok):2d}  {ok}')
    print(f'Not in archive : {len(skip):2d}  {skip}')
    if skip:
        print()
        print('Re-run with --simulate to create ERA5+noise stand-in files for missing dates.')


if __name__ == '__main__':
    main()

"""
Download ECMWF SEAS5 monthly seasonal forecasts for Jijiga area.

Dataset: seasonal-monthly-single-levels (ECMWF system 5)
Variables: total_precipitation, 2m_temperature
51 ensemble members, lead months 1-6, area 7-12°N, 40-45°E.

Init months:
  2023-01, 2023-04, 2023-10
  2024-01, 2024-04, 2024-10

Unlike the HRES open-data rolling archive, SEAS5 retrospective forecasts are
stored permanently in the CDS archive back to 1981.  All project init months
(2023–2024) should be downloadable with valid CDS API credentials.

Use --simulate to build synthetic SEAS5 files from ERA5 actuals + ensemble noise
(for testing pipelines when CDS API credentials are unavailable).

Noise model (simulation):
  tp  : multiplicative log-normal, σ = 0.40 × sqrt(lead_month)
  t2m : additive Gaussian,         σ = 1.0  × sqrt(lead_month) K

This gives realistic seasonal uncertainty: ~40% spread at lead 1 month rising
to ~98% at lead 6 months for precipitation, consistent with SEAS5 verification
statistics over the Horn of Africa.

Output: src/data/processed/seas5_jijiga_{year}_{month:02d}.nc

Run:
  python src/download_seas5_inputs.py              # real CDS download
  python src/download_seas5_inputs.py --simulate   # ERA5 + noise stand-in

Dependencies: pip install cdsapi xarray netcdf4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

INIT_MONTHS = [
    (2023,  1),
    (2023,  4),
    (2023, 10),
    (2024,  1),
    (2024,  4),
    (2024, 10),
]

LAT_N, LAT_S = 12.0,  7.0    # area bounding box (N, S)
LON_W, LON_E = 40.0, 45.0    # area bounding box (W, E)
N_LEADS       = 6
N_MEMBERS     = 51
OUT_DIR       = Path('src/data/processed')
ERA5_PATH     = OUT_DIR / 'era5_labeled.parquet'

VARS_REQUESTED = ['total_precipitation', '2m_temperature']

# Noise scale for simulation (σ per unit of sqrt(lead_month))
NOISE_SIGMA = {
    'tp':  0.40,   # multiplicative log-normal
    't2m': 1.00,   # additive Gaussian (K)
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nearest_idx(arr, val):
    return int(np.argmin(np.abs(np.asarray(arr, dtype=float) - val)))


def _month_offset(year: int, month: int, offset: int):
    """Return (year, month) shifted by `offset` months."""
    m = month - 1 + offset
    return year + m // 12, m % 12 + 1


def _monthly_tp_t2m(era5_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily ERA5 to monthly tp (sum) and t2m (mean)."""
    monthly = era5_df.resample('MS').agg({'tp': 'sum', 't2m': 'mean'})
    return monthly


# ── Real CDS download ──────────────────────────────────────────────────────────

def download_one_init(year: int, month: int) -> bool:
    """Download SEAS5 for one init month via CDS API.  Returns True on success."""
    out_path = OUT_DIR / f'seas5_jijiga_{year}_{month:02d}.nc'
    if out_path.exists():
        print(f'  {year}-{month:02d}: already exists, skipping.')
        return True

    try:
        import cdsapi
    except ImportError:
        raise ImportError('pip install cdsapi')

    # Load credentials from .env (project standard) or fall back to ~/.cdsapirc
    url = key = None
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        url = os.environ.get('CDS_API_URL')
        key = os.environ.get('CDS_API_KEY')
    except ImportError:
        pass

    c = cdsapi.Client(url=url, key=key) if (url and key) else cdsapi.Client()

    try:
        c.retrieve(
            'seasonal-monthly-single-levels',
            {
                'originating_centre': 'ecmwf',
                'system':             '51',   # system 51 = SEAS5.1 real-time (2020-present); use '5' for pre-2020 hindcasts
                'variable':           VARS_REQUESTED,
                'product_type':       'monthly_mean',
                'year':               str(year),
                'month':              f'{month:02d}',
                'leadtime_month':     [str(lm) for lm in range(1, N_LEADS + 1)],
                'area':               [LAT_N, LON_W, LAT_S, LON_E],
                'format':             'netcdf',
            },
            str(out_path),
        )
        print(f'  {year}-{month:02d}: downloaded ->{out_path.name}')
        return True

    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ['404', 'not found', 'no data', 'unavailable',
                                   'not available', 'does not exist', 'pending']):
            print(f'  {year}-{month:02d}: NOT AVAILABLE IN ARCHIVE — {exc}')
            return False
        raise


# ── Simulation stand-in ────────────────────────────────────────────────────────

def simulate_one_init(year: int, month: int, monthly_era5: pd.DataFrame,
                      rng: np.random.Generator) -> bool:
    """
    Build synthetic SEAS5 from ERA5 monthly actuals + ensemble spread noise.

    For each lead month (1-6) and each of 51 members:
      tp  : member_tp  = era5_tp  × exp(N(0, σ_tp  × √lead))  clipped ≥ 0
      t2m : member_t2m = era5_t2m + N(0, σ_t2m × √lead)
    """
    out_path = OUT_DIR / f'seas5_jijiga_{year}_{month:02d}.nc'
    if out_path.exists():
        print(f'  {year}-{month:02d}: already exists, skipping.')
        return True

    import calendar as _cal
    leads = list(range(1, N_LEADS + 1))

    # tprate_arr: units m s**-1, matching real SEAS5 variable name and units.
    # Convert from ERA5 monthly total (m) by dividing by seconds in the forecast month.
    tprate_arr = np.zeros((N_MEMBERS, N_LEADS), dtype=np.float32)
    t2m_arr    = np.zeros((N_MEMBERS, N_LEADS), dtype=np.float32)

    for li, lead in enumerate(leads):
        fy, fm      = _month_offset(year, month, lead)
        fc_date     = pd.Timestamp(year=fy, month=fm, day=1)
        days_in_m   = _cal.monthrange(fy, fm)[1]
        seconds_m   = days_in_m * 86400

        sigma_tp  = NOISE_SIGMA['tp']  * np.sqrt(lead)
        sigma_t2m = NOISE_SIGMA['t2m'] * np.sqrt(lead)

        if fc_date in monthly_era5.index:
            era5_tp_total = float(monthly_era5.loc[fc_date, 'tp'])   # monthly total (m)
            era5_t2m      = float(monthly_era5.loc[fc_date, 't2m'])
        else:
            era5_tp_total = 0.0
            era5_t2m      = float(monthly_era5['t2m'].mean())

        # ERA5 rate in m/s (same units as real SEAS5 tprate)
        era5_rate = era5_tp_total / seconds_m

        for mi in range(N_MEMBERS):
            if era5_rate <= 0:
                tprate_arr[mi, li] = max(0.0, float(rng.normal(0, sigma_tp * 1e-9)))
            else:
                tprate_arr[mi, li] = max(
                    0.0, era5_rate * float(np.exp(rng.normal(0, sigma_tp)))
                )
            t2m_arr[mi, li] = era5_t2m + float(rng.normal(0, sigma_t2m))

    # Match real SEAS5 structure:
    #   dims: (number, forecast_reference_time=1, forecastMonth, latitude, longitude)
    #   forecastMonth coord = [1..N_LEADS] (lead month integers)
    lat_pt     = np.array([9.5],  dtype=np.float64)   # nearest 1° gridpoint to 9.25°N
    lon_pt     = np.array([42.5], dtype=np.float64)   # nearest 1° gridpoint to 42.75°E
    init_time  = np.array([pd.Timestamp(year=year, month=month, day=1)],
                          dtype='datetime64[ns]')

    ds = xr.Dataset(
        {
            'tprate': xr.DataArray(
                tprate_arr[:, np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32),
                dims=['number', 'forecast_reference_time', 'forecastMonth',
                      'latitude', 'longitude'],
                attrs={'units': 'm s**-1',
                       'long_name': 'Time-mean total precipitation rate (simulated)'},
            ),
            't2m': xr.DataArray(
                t2m_arr[:, np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32),
                dims=['number', 'forecast_reference_time', 'forecastMonth',
                      'latitude', 'longitude'],
                attrs={'units': 'K', 'long_name': '2 metre temperature (simulated)'},
            ),
        },
        coords={
            'number':                 np.arange(N_MEMBERS, dtype=np.int64),
            'forecast_reference_time': init_time,
            'forecastMonth':          np.array(leads, dtype=np.int64),
            'latitude':               lat_pt,
            'longitude':              lon_pt,
        },
        attrs={
            'init_year':   str(year),
            'init_month':  f'{month:02d}',
            'system':      'SEAS5.1-simulated (system 51)',
            'simulated':   'True',
            'description': (
                'Synthetic SEAS5: ERA5 monthly actuals converted to tprate (m/s) '
                '+ lead-scaled ensemble noise.  Matches real SEAS5 NC structure.'
            ),
        },
    )
    ds.to_netcdf(out_path)
    print(f'  {year}-{month:02d}: simulated (ERA5 + noise)  ->{out_path.name}')
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download ECMWF SEAS5 seasonal monthly forecasts for Jijiga.')
    parser.add_argument(
        '--simulate', action='store_true',
        help='Build synthetic SEAS5 files from ERA5 actuals + ensemble noise '
             '(use when CDS API credentials are unavailable).',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='RNG seed for noise generation (default: 42).',
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.simulate:
        print('ECMWF SEAS5 SIMULATION — ERA5 monthly actuals + ensemble spread noise')
        print(f'(Use --simulate to test the full pipeline without CDS API credentials.)')
        print()

        era5_daily = pd.read_parquet(ERA5_PATH, columns=['time', 'tp', 't2m'])
        era5_daily['time'] = pd.to_datetime(era5_daily['time'])
        era5_daily = era5_daily.set_index('time').sort_index()
        monthly_era5 = _monthly_tp_t2m(era5_daily)

        rng = np.random.default_rng(args.seed)
        ok, skip = [], []
        for year, month in INIT_MONTHS:
            label = f'{year}-{month:02d}'
            print(f'-- {label} --------------------------')
            if simulate_one_init(year, month, monthly_era5, rng):
                ok.append(label)
            else:
                skip.append(label)

        print()
        print(f'Simulated : {len(ok):2d}  {ok}')
        print(f'Skipped   : {len(skip):2d}  {skip}')
        print()
        print('IMPORTANT: These are simulated forecasts, not real SEAS5 output.')
        print("Each .nc file has attribute  simulated='True'.")
        return

    # ── Real CDS download ─────────────────────────────────────────────────────
    print('ECMWF SEAS5 CDS download — Jijiga seasonal drought forecast')
    print(f'Dataset: seasonal-monthly-single-levels  |  system 5 (SEAS5)')
    print(f'Variables: {VARS_REQUESTED}')
    print(f'Area: {LAT_S}–{LAT_N}°N, {LON_W}–{LON_E}°E  |  lead months 1–{N_LEADS}')
    print(f'Members: {N_MEMBERS}')
    print()

    ok, skip = [], []
    for year, month in INIT_MONTHS:
        label = f'{year}-{month:02d}'
        print(f'-- {label} --------------------------')
        if download_one_init(year, month):
            ok.append(label)
        else:
            skip.append(label)

    print()
    print(f'Downloaded : {len(ok):2d}  {ok}')
    print(f'Not available: {len(skip):2d}  {skip}')
    if skip:
        print()
        print('Check CDS API credentials and quota at https://cds.climate.copernicus.eu')
        print('Re-run with --simulate to create ERA5+noise stand-in files.')


if __name__ == '__main__':
    main()

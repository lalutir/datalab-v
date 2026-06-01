"""Regenerate feature_matrix_v2_drought.parquet with correct ENSO/IOD lag computation.

Key fix: lags are computed on the FULL daily spine (1998-01-01 -> 2026-12-31), not
on the already-truncated feature matrix. This means dmi_lag_180 is valid for all dates
in the feature matrix (which starts 2001-06-01), so no training rows are dropped.
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

NINO34_PATH = 'src/data/raw/nino34_monthly.csv'
DMI_PATH    = 'src/data/raw/dmi_monthly.csv'
FEAT_PATH   = 'src/data/processed/feature_matrix.parquet'
V2_OUT_PATH = 'src/data/processed/feature_matrix_v2_drought.parquet'

# ── Load monthly indices ───────────────────────────────────────────────────────
df_nino = pd.read_csv(NINO34_PATH, parse_dates=['time'])
df_dmi  = pd.read_csv(DMI_PATH,    parse_dates=['time'])

print(f'Nino3.4 monthly: {df_nino.shape}  range: {df_nino.time.min().date()} -> {df_nino.time.max().date()}')
print(f'DMI     monthly: {df_dmi.shape}   range: {df_dmi.time.min().date()} -> {df_dmi.time.max().date()}')

# ── Resample to daily on full 1998-2026 spine ──────────────────────────────────
full_daily = pd.DataFrame({'time': pd.date_range('1998-01-01', '2026-12-31', freq='D')})

def monthly_to_daily(df_monthly, col):
    tmp = full_daily.copy()
    tmp['year']  = tmp['time'].dt.year
    tmp['month'] = tmp['time'].dt.month
    df_m = df_monthly.copy()
    df_m['year']  = df_m['time'].dt.year
    df_m['month'] = df_m['time'].dt.month
    merged = tmp.merge(df_m[['year', 'month', col]], on=['year', 'month'], how='left')
    merged = merged[['time', col]].sort_values('time').reset_index(drop=True)
    merged[col] = merged[col].ffill()
    return merged

df_nino_daily = monthly_to_daily(df_nino, 'nino34')
df_dmi_daily  = monthly_to_daily(df_dmi,  'dmi')
print(f'\nnino34 daily NaNs after ffill: {df_nino_daily.nino34.isna().sum()}')
print(f'dmi    daily NaNs after ffill: {df_dmi_daily.dmi.isna().sum()}')

# ── Compute lag columns on the FULL daily spine ────────────────────────────────
ENSO_IOD_LAG_OFFSETS = [30, 90, 180]
enso_iod_lag_cols = []

df_enso_lags = full_daily.copy()
df_enso_lags = df_enso_lags.merge(df_nino_daily, on='time', how='left')
df_enso_lags = df_enso_lags.merge(df_dmi_daily,  on='time', how='left')

for col in ['nino34', 'dmi']:
    for lag in ENSO_IOD_LAG_OFFSETS:
        name = f'{col}_lag_{lag}'
        df_enso_lags[name] = df_enso_lags[col].shift(lag)
        enso_iod_lag_cols.append(name)

print(f'\nCreated {len(enso_iod_lag_cols)} lag cols on full spine: {enso_iod_lag_cols}')
valid_from = df_enso_lags.dropna(subset=['dmi_lag_180'])['time'].min().date()
print(f'dmi_lag_180 first valid date: {valid_from}  (feature matrix starts 2001-06-01 => no rows dropped)')

# ── Merge only lagged cols onto existing feature matrix ───────────────────────
lag_only = df_enso_lags[['time'] + enso_iod_lag_cols]
df_feat = pd.read_parquet(FEAT_PATH)
df_feat['time'] = pd.to_datetime(df_feat['time'])
df_feat = df_feat.merge(lag_only, on='time', how='left')

nans_after = df_feat[enso_iod_lag_cols].isna().sum()
print(f'\nNaNs after merge (should all be 0):')
print(nans_after.to_string())

# ── Validate: training period ─────────────────────────────────────────────────
train_mask = df_feat['time'].dt.year <= 2022
nans_train = df_feat.loc[train_mask, enso_iod_lag_cols].isna().sum()
print(f'\nNaN check (2001-2022 training period):')
print(nans_train.to_string())
print(f'All zero: {(nans_train == 0).all()}')

print(f'\nDate range: {df_feat.time.min().date()} -> {df_feat.time.max().date()}')
print(f'Train rows: {train_mask.sum()}  (same as v1: 7884)')
print(f'Test  rows: {(df_feat.time.dt.year >= 2023).sum()}')

# ── Save ──────────────────────────────────────────────────────────────────────
df_feat.to_parquet(V2_OUT_PATH, index=False)
print(f'\nSaved: {V2_OUT_PATH}')
print(f'Shape: {df_feat.shape}  (51 original + {len(enso_iod_lag_cols)} new = {51 + len(enso_iod_lag_cols)} cols)')
print(f'Columns: {list(df_feat.columns[-6:])}')

"""
Post-process ECMWF HRES open-data forecasts into flood risk predictions.
Approach 3b — full flood composite: API, SMI, and total_ro all vary.

This script mirrors postprocess_gencast_results.py (Approach 3a / GenCast).
The key improvement over GenCast is that HRES provides soil moisture (swvl1,
swvl2) and runoff (sro, ssro), so 100% of the flood composite score varies
over the forecast period — compared to only 40% (API-only) for GenCast.

For each init date with an available hres_jijiga_{date}.nc file:
  - API     : forward-computed from HRES tp, seeded from ERA5 api_92 at init date
  - SMI     : computed from HRES swvl1+swvl2 / (FC1+FC2); falls back to ERA5 if absent
  - total_ro: HRES sro + ssro; falls back to ERA5 init-date value if absent
  - flood_score = 0.40*norm_API + 0.35*norm_SMI + 0.25*norm_total_ro
  - Risk labels from training-period percentile thresholds (fit on 2001-2022)

Outputs:
  src/data/processed/hres_forecast_results.csv
  notebooks/phase6_hres_flood_risk.png

Prints macro recall at leads 1, 3, 7, 10 vs GenCast (gencast_forecast_results.csv).

Run: python src/postprocess_hres_results.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import recall_score

DATA_PATH = Path('src/data/processed/era5_labeled.parquet')
HRES_DIR  = Path('src/data/processed')
GM_PATH   = Path('src/data/processed/gencast_forecast_results.csv')
OUT_PATH  = Path('src/data/processed/hres_forecast_results.csv')
FIG_DIR   = Path('notebooks')

INIT_DATES = [
    '2023-01-15', '2023-03-01', '2023-04-01', '2023-07-01',
    '2023-10-15', '2024-01-15', '2024-04-01', '2024-10-01',
]
FC1 = FC2   = 0.323        # field capacity m³/m³ (loamy sand, Somali region)
API_DECAY   = 0.92
LABEL_MAP   = {0: 'None', 1: 'Moderate', 2: 'Elevated', 3: 'High', 4: 'Extreme'}


# ── Load ERA5 labeled data ────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time').sort_index()
train_mask = df.index.year <= 2022

# Training-period normalisation parameters (identical to postprocess_gencast_results.py)
norm_params = {}
for col in ['api_92', 'smi_fc', 'total_ro']:
    norm_params[col] = (float(df.loc[train_mask, col].min()),
                        float(df.loc[train_mask, col].max()))


def calc_flood_score(api_arr, smi_arr, ro_arr):
    """Vectorised flood composite score using training-period min/max."""
    mn_a, mx_a = norm_params['api_92']
    mn_s, mx_s = norm_params['smi_fc']
    mn_r, mx_r = norm_params['total_ro']
    n_api = np.clip((np.asarray(api_arr) - mn_a) / (mx_a - mn_a + 1e-12), 0, 1)
    n_smi = np.clip((np.asarray(smi_arr) - mn_s) / (mx_s - mn_s + 1e-12), 0, 1)
    n_ro  = np.clip((np.asarray(ro_arr)  - mn_r) / (mx_r - mn_r + 1e-12), 0, 1)
    return 0.40 * n_api + 0.35 * n_smi + 0.25 * n_ro


# Flood score percentile thresholds (training data)
tr_scores = calc_flood_score(
    df.loc[train_mask, 'api_92'].values,
    df.loc[train_mask, 'smi_fc'].values,
    df.loc[train_mask, 'total_ro'].values,
)
p65, p80, p90, p97 = (float(np.percentile(tr_scores, q)) for q in [65, 80, 90, 97])
print(f'Flood score thresholds (training period):')
print(f'  p65={p65:.5f}  p80={p80:.5f}  p90={p90:.5f}  p97={p97:.5f}')
print()


def flood_label(s):
    s = float(s)
    if s < p65:  return 0
    elif s < p80: return 1
    elif s < p90: return 2
    elif s < p97: return 3
    else:         return 4


def _era5_row(dt_init):
    """Fetch ERA5 row at or nearest to init date."""
    if dt_init in df.index:
        return df.loc[dt_init]
    idx = df.index.get_indexer([dt_init], method='nearest')[0]
    return df.iloc[idx]


# ── Process each HRES init date ───────────────────────────────────────────────
records        = []
component_log  = []

for date_str in INIT_DATES:
    nc_path = HRES_DIR / f'hres_jijiga_{date_str}.nc'
    if not nc_path.exists():
        print(f'  MISSING: {nc_path.name}  — run download_hres_inputs.py first')
        continue

    ds      = xr.open_dataset(nc_path)
    dt_init = pd.Timestamp(date_str)
    era5    = _era5_row(dt_init)

    api_init = float(era5['api_92'])
    smi_init = float(era5['smi_fc'])
    ro_init  = float(era5['total_ro'])

    forecast_dates = pd.to_datetime(ds['date'].values)
    n_days         = len(forecast_dates)

    # ── API from HRES tp (always use HRES if available, else freeze) ──────────
    if 'tp' in ds:
        tp_daily = np.clip(ds['tp'].values.astype(float), 0, None)
        api_source = 'HRES tp'
    else:
        tp_daily   = np.zeros(n_days)
        api_source = 'zero (tp absent)'

    api = np.empty(n_days)
    api[0] = API_DECAY * api_init + tp_daily[0]
    for i in range(1, n_days):
        api[i] = API_DECAY * api[i - 1] + tp_daily[i]

    # ── SMI from HRES swvl1 + swvl2 (or ERA5 frozen) ─────────────────────────
    if 'swvl1' in ds and 'swvl2' in ds:
        swvl1      = ds['swvl1'].values.astype(float)
        swvl2      = ds['swvl2'].values.astype(float)
        smi_daily  = np.clip((swvl1 + swvl2) / (FC1 + FC2), 0.0, 1.0)
        smi_source = 'HRES swvl1+swvl2'
    else:
        smi_daily  = np.full(n_days, smi_init)
        smi_source = f'ERA5 frozen ({smi_init:.4f})'

    # ── total_ro from HRES sro + ssro (or ERA5 frozen) ───────────────────────
    if 'sro' in ds and 'ssro' in ds:
        ro_daily   = np.clip(ds['sro'].values.astype(float)
                             + ds['ssro'].values.astype(float), 0, None)
        ro_source  = 'HRES sro+ssro'
    else:
        ro_daily   = np.full(n_days, ro_init)
        ro_source  = f'ERA5 frozen ({ro_init:.6f})'

    ds.close()

    component_log.append({
        'init_date': date_str,
        'api':       api_source,
        'smi':       smi_source,
        'ro':        ro_source,
    })

    # ── Flood composite scores and labels ─────────────────────────────────────
    scores = calc_flood_score(api, smi_daily, ro_daily)

    for d_idx, fc_date in enumerate(forecast_dates):
        fc_ts        = pd.Timestamp(fc_date)
        hres_risk    = flood_label(scores[d_idx])
        actual_flood = int(df.loc[fc_ts, 'flood_risk']) if fc_ts in df.index else np.nan
        lead         = (fc_ts - dt_init).days

        records.append({
            'init_date':         date_str,
            'forecast_date':     fc_ts,
            'lead_days':         lead,
            'hres_flood_risk':   hres_risk,
            'hres_flood_score':  round(float(scores[d_idx]), 6),
            'actual_flood_risk': actual_flood,
            'smi_varied':        smi_source.startswith('HRES'),
            'ro_varied':         ro_source.startswith('HRES'),
        })

if not records:
    print('No HRES .nc files found — nothing to process.')
    print('Run one of:')
    print('  python src/download_hres_inputs.py           # real download (needs <6m dates)')
    print('  python src/download_hres_inputs.py --simulate  # ERA5+noise stand-in')
    raise SystemExit(0)

results = pd.DataFrame(records)
n_init  = results['init_date'].nunique()
results.to_csv(OUT_PATH, index=False)

# Report if any dates used simulated data
sim_dates = []
for d in results['init_date'].unique():
    nc = HRES_DIR / f'hres_jijiga_{d}.nc'
    try:
        with xr.open_dataset(nc) as _ds:
            if _ds.attrs.get('simulated', 'False') == 'True':
                sim_dates.append(d)
    except Exception:
        pass
sim_note = f'  [{len(sim_dates)} simulated via ERA5+noise: {sim_dates}]' if sim_dates else ''

print(f'Saved HRES forecast results: {len(results)} rows '
      f'({n_init} init dates × up to 10 days){sim_note}')
print()


# ── Composite component summary ───────────────────────────────────────────────
print('Flood composite components used per init date:')
print(f'  {"Init date":<14} {"API":<22} {"SMI":<30} {"RO":<30}')
for row in component_log:
    print(f'  {row["init_date"]:<14} {row["api"]:<22} {row["smi"]:<30} {row["ro"]:<30}')
print()


# ── Evaluation metrics vs GenCast ─────────────────────────────────────────────
gm        = pd.read_csv(GM_PATH, parse_dates=['forecast_date', 'init_date'])
valid     = results.dropna(subset=['actual_flood_risk']).copy()
valid['actual_flood_risk']  = valid['actual_flood_risk'].astype(int)
valid['hres_flood_risk']    = valid['hres_flood_risk'].astype(int)

# Restrict GenCast comparison to the same init dates that HRES has data for
hres_init_dates = set(valid['init_date'].unique())
valid_gm  = gm[gm['init_date'].dt.strftime('%Y-%m-%d').isin(hres_init_dates)].dropna(
    subset=['actual_flood_risk']
).copy()
valid_gm['actual_flood_risk'] = valid_gm['actual_flood_risk'].astype(int)


def macro_recall_at_lead(df_r, lead, pred_col):
    sub = df_r[df_r['lead_days'] == lead]
    if len(sub) == 0:
        return np.nan
    return recall_score(sub['actual_flood_risk'], sub[pred_col],
                        average='macro', zero_division=0, labels=range(5))


print('Flood macro recall: HRES vs GenCast (same available init dates)')
print(f'  {"Lead":>5}  {"HRES":>10}  {"GenCast":>10}  {"D (HRES-GC)":>13}')
for lead in [1, 3, 7, 10]:
    hres_r = macro_recall_at_lead(valid,    lead, 'hres_flood_risk')
    gm_r   = macro_recall_at_lead(valid_gm, lead, 'gm_flood_rounded')
    if np.isnan(hres_r) or np.isnan(gm_r):
        delta_s = 'N/A'
    else:
        delta_s = f'{hres_r - gm_r:+.4f}'
    hres_s = f'{hres_r:.4f}' if not np.isnan(hres_r) else 'N/A'
    gm_s   = f'{gm_r:.4f}'  if not np.isnan(gm_r)   else 'N/A'
    print(f'  +{lead:2d}d   {hres_s:>10}  {gm_s:>10}  {delta_s:>13}')

print()
print(f'HRES init dates with data: {sorted(hres_init_dates)}')
print(f'Note: GenCast metrics recomputed on the same subset for a fair comparison.')


# ── Figure: HRES flood risk — all 8 init dates ────────────────────────────────
RISK_COLORS  = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
fig, axes    = plt.subplots(4, 2, figsize=(14, 12), sharey=True)
axes         = axes.flatten()

for idx, date_str in enumerate(INIT_DATES):
    ax  = axes[idx]
    sub = results[results['init_date'] == date_str].copy()

    if sub.empty:
        ax.set_title(f'Init: {date_str} (not in archive)', fontsize=9)
        ax.text(0.5, 0.5, 'No HRES data', transform=ax.transAxes,
                ha='center', va='center', color='gray', fontsize=9)
        ax.set_ylim(-0.3, 4.3)
        ax.set_yticks(range(5))
        ax.set_yticklabels([LABEL_MAP[k] for k in range(5)], fontsize=7)
        continue

    dates = sub['forecast_date'].values

    ax.step(dates, sub['hres_flood_risk'],
            where='mid', color='darkorchid', lw=1.8, label='HRES (det.)')
    ax.step(dates, sub['actual_flood_risk'].fillna(0),
            where='mid', color='black', lw=1.2, ls='--', label='Actual (ERA5)')

    ax.set_ylim(-0.3, 4.3)
    ax.set_yticks(range(5))
    ax.set_yticklabels([LABEL_MAP[k] for k in range(5)], fontsize=7)
    ax.set_title(f'Init: {date_str}', fontsize=9, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.tick_params(axis='x', labelsize=7)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

fig.suptitle(
    'ECMWF HRES 10-day flood risk forecasts — 8 init dates, Jijiga (42.75°E, 9.25°N)\n'
    'Solid = HRES deterministic (full composite: API+SMI+RO all vary)  '
    '|  Dashed = actual ERA5 flood risk',
    fontsize=9, y=1.01,
)
plt.tight_layout()
out_fig = FIG_DIR / 'phase6_hres_flood_risk.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'\nSaved figure: {out_fig}')

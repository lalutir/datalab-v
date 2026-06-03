"""
Post-process real GenCast outputs and integrate into the three-way comparison.

Reads the 8 GenCast Jijiga output files (tp_daily_m, t2m_daily_K per ensemble
member) and computes flood risk forecasts using the Phase 3 classifiers.

Approach 3 (GenCast) flood pipeline:
  - API: computed from GenCast tp (initialised from ERA5 API at init date)
  - SMI: ERA5 value at init date, held constant (GenCast has no soil moisture)
  - total_ro: 0 (conservative assumption; GenCast has no runoff output)
  - Flood score = 0.40*norm_API + 0.35*norm_SMI + 0.25*norm_total_ro
  - Risk labels from training-data flood percentiles

Approach 3 drought pipeline:
  - SPEI: ERA5 spei_6 at init date, forward-filled (6-month index can't be
           updated from a 10-day GenCast forecast)
  - Modifier: elevate one level if SPEI base class is 1 or 2 AND ERA5 SMI at
    init date is below training p25. API is excluded from the modifier (same
    reason as Phase 3: near-zero API in dry spells fires the rule spuriously)

Run:  python src/postprocess_gencast_results.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import f1_score, recall_score

DATA_PATH = Path('src/data/processed/era5_labeled.parquet')
GM_DIR    = Path('src/data/processed')
COMP_PATH = Path('src/data/processed/comparison_table.csv')
FIG_DIR   = Path('notebooks')

INIT_DATES = [
    '2023-01-15', '2023-03-01', '2023-04-01', '2023-07-01',
    '2023-10-15', '2024-01-15', '2024-04-01', '2024-10-01',
]
LABEL_MAP = {0: 'None', 1: 'Moderate', 2: 'Elevated', 3: 'High', 4: 'Extreme'}
API_DECAY = 0.92


# ── Load ERA5 labeled data ────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time').sort_index()

train_mask = df.index.year <= 2022

# Flood normalisation params (fit on training data only)
norm_params = {}
for col in ['api_92', 'smi_fc', 'total_ro']:
    norm_params[col] = (float(df.loc[train_mask, col].min()),
                        float(df.loc[train_mask, col].max()))

def calc_flood_score(api_arr, smi_val, ro_val=0.0):
    mn_a, mx_a = norm_params['api_92']
    mn_s, mx_s = norm_params['smi_fc']
    mn_r, mx_r = norm_params['total_ro']
    n_api = np.clip((np.asarray(api_arr) - mn_a) / (mx_a - mn_a + 1e-12), 0, 1)
    n_smi = np.clip((np.asarray(smi_val) - mn_s) / (mx_s - mn_s + 1e-12), 0, 1)
    n_ro  = np.clip((np.asarray(ro_val)  - mn_r) / (mx_r - mn_r + 1e-12), 0, 1)
    return 0.40 * n_api + 0.35 * n_smi + 0.25 * n_ro

# Flood score percentile thresholds from training data
def _fs(d): return calc_flood_score(d['api_92'].values, d['smi_fc'].values, d['total_ro'].values)
tr_scores = _fs(df[train_mask])
p65, p80, p90, p97 = (float(np.percentile(tr_scores, q)) for q in [65, 80, 90, 97])

def flood_label(s):
    if s < p65: return 0
    elif s < p80: return 1
    elif s < p90: return 2
    elif s < p97: return 3
    else: return 4

# Drought modifier threshold — SMI only; API excluded (spurious dry-season firing)
smi_p25 = float(df.loc[train_mask, 'smi_fc'].quantile(0.25))

def drought_label(spei, smi_val):
    """Apply McKee thresholds + SMI-only modifier rule (CLAUDE.md methodology)."""
    if spei >= -0.5:   base = 0
    elif spei >= -1.0: base = 1
    elif spei >= -1.5: base = 2
    elif spei >= -2.0: base = 3
    else:              base = 4
    # Modifier: elevate by 1 if base is 1 or 2 and SMI below training p25
    if base in (1, 2) and smi_val < smi_p25:
        base = min(base + 1, 4)
    return base


# ── Process each GenCast init date ────────────────────────────────────────────
records = []

for date_str in INIT_DATES:
    nc_path = GM_DIR / f'gencast_jijiga_{date_str}.nc'
    if not nc_path.exists():
        print(f'  MISSING: {nc_path.name}')
        continue

    ds = xr.open_dataset(nc_path)
    dt_init = pd.Timestamp(date_str)

    # ERA5 state at init date
    era5_row = df.loc[dt_init] if dt_init in df.index else df.iloc[
        df.index.get_indexer([dt_init], method='nearest')[0]]

    api_init  = float(era5_row['api_92'])
    smi_init  = float(era5_row['smi_fc'])
    spei_init = float(era5_row['spei_6'])

    # GenCast tp (days=10, members=8), clip negative numerical noise
    tp_m = np.clip(ds['tp_daily_m'].values, 0, None)   # (10, 8)
    t2m_m = ds['t2m_daily_K'].values                    # (10, 8)
    n_days, n_members = tp_m.shape
    forecast_dates = pd.to_datetime(ds['date'].values)

    # Per-member API and risk
    member_flood_risk   = np.zeros((n_members, n_days), dtype=int)
    member_drought_risk = np.zeros((n_members, n_days), dtype=int)

    for m in range(n_members):
        api = np.zeros(n_days)
        api[0] = API_DECAY * api_init + tp_m[0, m]
        for i in range(1, n_days):
            api[i] = API_DECAY * api[i - 1] + tp_m[i, m]

        for d in range(n_days):
            score = calc_flood_score(api[d], smi_init)
            member_flood_risk[m, d]   = flood_label(score)
            member_drought_risk[m, d] = drought_label(spei_init, smi_init)

    # Ensemble mean risk (rounded) and ensemble maximum
    ens_mean_flood   = member_flood_risk.mean(axis=0)
    ens_max_flood    = member_flood_risk.max(axis=0)
    ens_mean_drought = member_drought_risk.mean(axis=0)

    for d_idx, fc_date in enumerate(forecast_dates):
        fc_ts  = pd.Timestamp(fc_date)
        actual_flood   = int(df.loc[fc_ts, 'flood_risk'])   if fc_ts in df.index else np.nan
        actual_drought = int(df.loc[fc_ts, 'drought_risk']) if fc_ts in df.index else np.nan
        lead = (fc_ts - dt_init).days

        records.append({
            'init_date':          date_str,
            'forecast_date':      fc_ts,
            'lead_days':          lead,
            'gm_flood_ens_mean':  float(ens_mean_flood[d_idx]),
            'gm_flood_ens_max':   int(ens_max_flood[d_idx]),
            'gm_flood_rounded':   int(round(ens_mean_flood[d_idx])),
            'gm_drought_rounded': int(round(ens_mean_drought[d_idx])),
            'actual_flood_risk':  actual_flood,
            'actual_drought_risk':actual_drought,
        })
    ds.close()

results = pd.DataFrame(records)
results.to_csv(GM_DIR / 'gencast_forecast_results.csv', index=False)
print(f'Saved forecast results: {len(results)} rows ({len(INIT_DATES)} init dates x 10 days)')
print()


# ── Evaluation metrics (8 case dates only) ────────────────────────────────────
valid = results.dropna(subset=['actual_flood_risk'])

def macro_recall_at_lead(df_r, lead):
    sub = df_r[df_r['lead_days'] == lead]
    if len(sub) == 0:
        return np.nan
    return recall_score(sub['actual_flood_risk'], sub['gm_flood_rounded'],
                        average='macro', zero_division=0, labels=range(5))

def extreme_recall_at_lead(df_r, lead):
    sub = df_r[df_r['lead_days'] == lead]
    if len(sub) == 0 or (sub['actual_flood_risk'] == 4).sum() == 0:
        return np.nan
    mask = sub['actual_flood_risk'] == 4
    return (sub.loc[mask, 'gm_flood_rounded'] == 4).mean()

print('GenCast flood evaluation (8 case init dates):')
print(f'{"Lead":>6}  {"Macro recall":>14}  {"Extreme recall":>16}')
for lead in [1, 3, 7, 10]:
    mr = macro_recall_at_lead(valid, lead)
    er = extreme_recall_at_lead(valid, lead)
    print(f'  +{lead:2d}d   {mr:>14.4f}   {er!s:>16}')


# ── Update comparison table ───────────────────────────────────────────────────
comp = pd.read_csv(COMP_PATH, index_col=0)

# Replace simulated Approach 3 values with real GenCast results
gm_flood_1d  = macro_recall_at_lead(valid, 1)
gm_flood_7d  = macro_recall_at_lead(valid, 7)
gm_flood_10d = macro_recall_at_lead(valid, 10)

real_gm_col = {
    'Drought macro recall (+1d)':   'N/A (drought)',
    'Drought macro recall (+7d)':   'N/A (drought)',
    'Drought macro recall (+14d)':  'N/A (drought)',
    'Drought Extreme recall (+7d)': 'N/A',
    'Flood macro recall (+1d)':     f'{gm_flood_1d:.4f}' if not np.isnan(gm_flood_1d) else 'N/A',
    'Flood macro recall (+7d)':     f'{gm_flood_7d:.4f}'  if not np.isnan(gm_flood_7d)  else 'N/A',
    'Flood macro recall (+14d)':    f'{gm_flood_10d:.4f}' if not np.isnan(gm_flood_10d) else 'N/A (>10d)',
    'Flood Extreme recall (+7d)':   'N/A',
}

col_name = 'Approach 3\n(Foundation model)'
if col_name in comp.columns:
    for metric, val in real_gm_col.items():
        if metric in comp.index:
            comp.loc[metric, col_name] = val

comp.to_csv(COMP_PATH)
print()
print('Updated comparison table:')
print(comp.to_string())


# ── Figure: GenCast flood risk for all 8 init dates ──────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharey=True)
axes = axes.flatten()
RISK_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']

for idx, date_str in enumerate(INIT_DATES):
    ax = axes[idx]
    sub = results[results['init_date'] == date_str].copy()
    if sub.empty:
        continue
    dates = sub['forecast_date'].values

    # Ensemble spread (min/max shading)
    ax.fill_between(dates, sub['gm_flood_ens_mean'] - 0.5,
                    sub['gm_flood_ens_max'], alpha=0.2, color='steelblue',
                    label='Ensemble spread')
    # Ensemble mean
    ax.step(dates, sub['gm_flood_rounded'], where='mid',
            color='steelblue', lw=1.8, label='GenCast (ens. mean)')
    # Actual
    ax.step(dates, sub['actual_flood_risk'], where='mid',
            color='black', lw=1.2, ls='--', label='Actual')

    ax.set_ylim(-0.3, 4.3)
    ax.set_yticks(range(5))
    ax.set_yticklabels([LABEL_MAP[k] for k in range(5)], fontsize=7)
    ax.set_title(f'Init: {date_str}', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.tick_params(axis='x', labelsize=7)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper left')

fig.suptitle('GenCast 1° flood risk — 8 init dates (Jijiga, 2023–2024)\n'
             'Solid=ensemble mean | Dashed=actual ERA5 flood risk',
             fontsize=10)
plt.tight_layout()
out_fig = FIG_DIR / 'phase6_gencast_flood_risk.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'\nSaved figure: {out_fig}')


# ── Case study: March 2023 EMDAT flood ───────────────────────────────────────
print()
print('CASE STUDY — March 2023 EMDAT flood event (init: 2023-03-01)')
print('GenCast predicted flood risk from 14-day lead:')
case = results[results['init_date'] == '2023-03-01'][
    ['forecast_date', 'lead_days', 'gm_flood_rounded', 'gm_flood_ens_max', 'actual_flood_risk']
]
print(case.to_string(index=False))

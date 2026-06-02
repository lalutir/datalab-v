"""
Post-process ECMWF SEAS5 ensemble forecasts into monthly drought risk predictions.
Approach 3 — Drought: SEAS5 seasonal forecast -> SPEI-6 -> McKee thresholds.

This script resolves the fundamental limitation of Approaches 3a/3b (GenCast/HRES):
those models run only 10 days, far too short for SPEI-6 which requires a 6-month
climatic water balance accumulation.  SEAS5 provides monthly ensemble forecasts at
leads 1–6 months, making SPEI-6-based seasonal drought forecasting possible.

Pipeline for each init month (year, month):
  ① ERA5 warmup: 5 months of ERA5 CWB ending at the init month (months M-4 to M)
  ② SEAS5 forecast: 6 months of SEAS5 CWB (leads 1-6; months M+1 to M+6)
     - CWB = seas5_tp + clim_pev, where clim_pev is the ERA5 training-period
       climatological monthly mean pev for that calendar month (held constant
       across all 51 members; documented assumption).
  ③ Build 11-month combined series and compute 6-month rolling CWB sums
  ④ Transform to SPEI-6 using Fisk parameters re-derived from ERA5 2000-2020
  ⑤ Apply McKee (1993) drought risk thresholds
  ⑥ Apply modifier rule using ERA5 api_92 and smi_fc at the init date
  ⑦ Aggregate to ensemble-mean and ensemble-max drought risk per lead month
  ⑧ Compare to actual ERA5 drought_risk

Outputs:
  src/data/processed/seas5_drought_results.csv
  notebooks/phase6_seas5_drought.png

Prints macro recall at leads 1, 2, 3, 6 months.

Run: python src/postprocess_seas5_results.py

Assumptions documented:
  - SEAS5 pev is approximated by ERA5 training-period monthly climatological pev.
    Real SEAS5 pev is not available in the CDS seasonal-monthly-single-levels
    dataset without a separate download.  Using climatological pev introduces a
    small warm bias in CWB during anomalously wet forecast months, but the
    sensitivity is low because pev << |tp - pev| in magnitude.
  - Fisk parameters are re-derived here from era5_labeled.parquet instead of
    reading from Phase 3 notebook state, ensuring consistency and reproducibility.
  - The modifier rule uses init-date ERA5 api_92 and smi_fc because SEAS5 does
    not provide soil moisture forecasts.  This freezes the modifier state at the
    time of forecast initialisation.
"""

import calendar
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import fisk, norm
from sklearn.metrics import recall_score

DATA_PATH  = Path('src/data/processed/era5_labeled.parquet')
SEAS5_DIR  = Path('src/data/processed')
OUT_PATH   = Path('src/data/processed/seas5_drought_results.csv')
FIG_PATH   = Path('notebooks/phase6_seas5_drought.png')

INIT_MONTHS = [
    (2023,  1),
    (2023,  4),
    (2023, 10),
    (2024,  1),
    (2024,  4),
    (2024, 10),
]

API_DECAY  = 0.92
FC1 = FC2  = 0.323
LABEL_MAP  = {0: 'None', 1: 'Moderate', 2: 'Elevated', 3: 'High', 4: 'Extreme'}
RISK_COLORS = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']


# ── Load ERA5 labeled data ─────────────────────────────────────────────────────

df = pd.read_parquet(DATA_PATH)
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time').sort_index()

train_mask = df.index.year <= 2022

# Aggregate daily ERA5 to monthly
monthly_era5 = df.resample('MS').agg({'tp': 'sum', 'pev': 'sum'})
monthly_era5['cwb'] = monthly_era5['tp'] + monthly_era5['pev']

# 6-month rolling CWB accumulation (ending at each month)
monthly_era5['cwb_6m'] = monthly_era5['cwb'].rolling(6).sum()


# ── Fit Fisk distribution per calendar month (2000–2020) ──────────────────────
#
# The Phase 3 notebook computes SPEI-6 using scipy.stats.fisk fitted on the
# 2000-2020 reference period, separately for each calendar month.  We re-derive
# those parameters here from era5_labeled.parquet.

ref_mask = (monthly_era5.index.year >= 2000) & (monthly_era5.index.year <= 2020)

fisk_params = {}   # calendar_month -> (c, loc, scale)
for cal_month in range(1, 13):
    m_mask = ref_mask & (monthly_era5.index.month == cal_month)
    data   = monthly_era5.loc[m_mask, 'cwb_6m'].dropna().values
    if len(data) < 5:
        # Not enough data; use neighbouring months as fallback
        print(f'  Warning: only {len(data)} values for calendar month {cal_month}; '
              f'using global fit.')
        data = monthly_era5.loc[ref_mask, 'cwb_6m'].dropna().values
    c, loc, scale       = fisk.fit(data)
    fisk_params[cal_month] = (c, loc, scale)

print('Fisk parameters fitted per calendar month (2000-2020):')
print(f'  {"Month":<8} {"c":>10} {"loc":>12} {"scale":>12}')
for m, (c, loc, sc) in fisk_params.items():
    print(f'  {m:<8d} {c:>10.4f} {loc:>12.6f} {sc:>12.6f}')
print()


# ── Climatological monthly pev (2001–2022 training period) ────────────────────
#
# SEAS5 does not provide potential evapotranspiration.  We proxy it using the
# ERA5 training-period mean monthly pev for each calendar month.

pev_clim = {}  # calendar_month -> mean monthly pev (m)
for cal_month in range(1, 13):
    train_monthly_mask = (monthly_era5.index.year >= 2001) & \
                         (monthly_era5.index.year <= 2022) & \
                         (monthly_era5.index.month == cal_month)
    pev_clim[cal_month] = float(monthly_era5.loc[train_monthly_mask, 'pev'].mean())

print('Climatological monthly pev used for SEAS5 CWB (2001-2022 mean, m):')
print('  ' + '  '.join(f'M{m:02d}:{pev_clim[m]:.5f}' for m in range(1, 13)))
print()


# ── Training-period p25 thresholds for modifier rule ──────────────────────────

api_p25 = float(df.loc[train_mask, 'api_92'].quantile(0.25))
smi_p25 = float(df.loc[train_mask, 'smi_fc'].quantile(0.25))
print(f'Modifier rule thresholds (training p25): '
      f'api_92 < {api_p25:.4f}  OR  smi_fc < {smi_p25:.4f}')
print()


# ── Helper functions ──────────────────────────────────────────────────────────

def _month_offset(year: int, month: int, offset: int):
    m = month - 1 + offset
    return year + m // 12, m % 12 + 1


def cwb6m_to_spei(cwb6m: float, cal_month: int) -> float:
    c, loc, scale = fisk_params[cal_month]
    p = float(fisk.cdf(cwb6m, c=c, loc=loc, scale=scale))
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(norm.ppf(p))


def spei_to_base_drought_risk(spei: float) -> int:
    if spei >= -0.5:    return 0
    elif spei >= -1.0:  return 1
    elif spei >= -1.5:  return 2
    elif spei >= -2.0:  return 3
    else:               return 4


def apply_modifier(base: int, api_val: float, smi_val: float) -> int:
    """Elevate drought risk one level when antecedent conditions are abnormally dry."""
    if base in (1, 2, 3) and (api_val < api_p25 or smi_val < smi_p25):
        return base + 1
    return base


def _era5_row_nearest(dt):
    if dt in df.index:
        return df.loc[dt]
    idx = df.index.get_indexer([dt], method='nearest')[0]
    return df.iloc[idx]


def _era5_monthly_cwb(year: int, month: int) -> float:
    """Monthly CWB from ERA5 (sum tp + sum pev for that calendar month)."""
    key = pd.Timestamp(year=year, month=month, day=1)
    if key in monthly_era5.index:
        return float(monthly_era5.loc[key, 'cwb'])
    return np.nan


# ── Process each SEAS5 init month ─────────────────────────────────────────────

records = []

for year, month in INIT_MONTHS:
    label    = f'{year}-{month:02d}'
    nc_path  = SEAS5_DIR / f'seas5_jijiga_{year}_{month:02d}.nc'

    if not nc_path.exists():
        print(f'  MISSING: {nc_path.name}  — run download_seas5_inputs.py first')
        continue

    init_ts = pd.Timestamp(year=year, month=month, day=1)

    # ── ERA5 warmup: months M-4 through M (5 months) ─────────────────────────
    # For SPEI-6 at lead 1, the 6-month window is [M-4, M-3, M-2, M-1, M, M+1].
    # Five ERA5 warmup months (ending at M) supply the first 5 of those 6 months.
    warmup_cwb = []
    for offset in range(-4, 1):   # offsets: -4, -3, -2, -1, 0
        wy, wm = _month_offset(year, month, offset)
        warmup_cwb.append(_era5_monthly_cwb(wy, wm))
    warmup_cwb = np.array(warmup_cwb, dtype=float)  # shape (5,)

    # ── ERA5 api_92 and smi_fc at the last day of the init month ─────────────
    last_day_init = init_ts + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    era5_row      = _era5_row_nearest(last_day_init)
    api_init      = float(era5_row['api_92'])
    smi_init      = float(era5_row['smi_fc'])

    # ── Load SEAS5 NetCDF ─────────────────────────────────────────────────────
    ds        = xr.open_dataset(nc_path)
    simulated = str(ds.attrs.get('simulated', 'False')) == 'True'

    # Extract nearest grid point to Jijiga (9.25°N, 42.75°E)
    lats = ds['latitude'].values.astype(float)
    lons = ds['longitude'].values.astype(float)
    ilat = int(np.argmin(np.abs(lats - 9.25)))
    ilon = int(np.argmin(np.abs(lons - 42.75)))

    # Determine lead months.
    # Real SEAS5: 'forecastMonth' coord = [1,2,3,4,5,6] (integer lead)
    # Simulated:  'time' coord = datetimes of valid months
    if 'forecastMonth' in ds.coords:
        n_leads_used = min(int(ds.dims['forecastMonth']), 6)
    else:
        n_leads_used = min(int(ds.dims.get('time', 6)), 6)

    # Valid calendar months for leads 1..n_leads_used
    valid_months = [_month_offset(year, month, lead) for lead in range(1, n_leads_used + 1)]

    # Extract precipitation and convert to monthly total (m).
    # Real SEAS5: variable 'tprate', units m s**-1 (mean rate over the month)
    #   monthly_total_m = tprate * days_in_month * 86400
    # Simulated:  variable 'tp', already monthly total (m)
    if 'tprate' in ds.data_vars:
        # Shape: (number, forecast_reference_time=1, forecastMonth, lat, lon)
        raw = ds['tprate'].values[:, 0, :n_leads_used, ilat, ilon].astype(float)
        tp_seas5 = np.zeros_like(raw)
        for li, (vy, vm) in enumerate(valid_months):
            days = calendar.monthrange(vy, vm)[1]
            tp_seas5[:, li] = np.clip(raw[:, li] * days * 86400, 0, None)
    else:
        # Simulated: variable 'tp', shape (number, time, lat, lon) or (number, time, 1, 1)
        raw = ds['tp'].values
        if raw.ndim == 4:
            raw = raw[:, :n_leads_used, ilat, ilon]
        elif raw.ndim == 3:
            raw = raw[:, :n_leads_used, 0]
        tp_seas5 = np.clip(raw.astype(float), 0, None)

    n_members_actual = tp_seas5.shape[0]

    ds.close()

    # ── Per-member SPEI-6 and drought risk ────────────────────────────────────
    # member_risk: (n_members, n_leads_used)
    member_risk = np.zeros((n_members_actual, n_leads_used), dtype=int)

    for mi in range(n_members_actual):
        # SEAS5 monthly CWB for leads 1..n_leads_used
        seas5_cwb = np.zeros(n_leads_used, dtype=float)
        for li, (vy, vm) in enumerate(valid_months):
            cal_vm   = vm
            seas5_tp = tp_seas5[mi, li]
            clim_pev = pev_clim[cal_vm]
            seas5_cwb[li] = seas5_tp + clim_pev

        # Combined series: 5 ERA5 warmup + n_leads_used SEAS5 months = 5+6 = 11
        combined = np.concatenate([warmup_cwb, seas5_cwb])  # shape (5 + n_leads_used,)

        for li, (vy, vm) in enumerate(valid_months):
            # 6-month rolling sum ending at position 4+li (0-indexed)
            # combined[li : li+6] = [M+L-5 ... M+L], where L = li+1
            cwb6m = float(np.sum(combined[li : li + 6]))

            cal_vm = vm
            spei   = cwb6m_to_spei(cwb6m, cal_vm)
            base   = spei_to_base_drought_risk(spei)
            risk   = apply_modifier(base, api_init, smi_init)
            member_risk[mi, li] = risk

    # ── Ensemble aggregation ──────────────────────────────────────────────────
    ens_mean = member_risk.mean(axis=0)   # float, shape (n_leads_used,)
    ens_max  = member_risk.max(axis=0)    # int,   shape (n_leads_used,)

    # ── Actual ERA5 drought risk for the valid forecast months ────────────────
    for li, (vy, vm) in enumerate(valid_months):
        lead = li + 1
        fc_ts = pd.Timestamp(year=vy, month=vm, day=1)

        # Drought risk for the forecast month: use 15th day (SPEI constant within month)
        fc_mid  = pd.Timestamp(year=vy, month=vm, day=15)
        fc_key  = fc_mid if fc_mid in df.index else fc_ts
        actual  = (int(df.loc[fc_key, 'drought_risk'])
                   if fc_key in df.index else np.nan)

        records.append({
            'init_date':           f'{year}-{month:02d}',
            'lead_months':          lead,
            'valid_year':           vy,
            'valid_month':          vm,
            'seas5_drought_mean':   round(float(ens_mean[li]), 3),
            'seas5_drought_max':    int(ens_max[li]),
            'actual_drought_risk':  actual,
            'simulated':            simulated,
            'n_members':            n_members_actual,
        })

    sim_tag = ' [simulated]' if simulated else ''
    print(f'  {label}: processed  {n_members_actual} members x {n_leads_used} leads'
          f'{sim_tag}')
    print(f'    api_init={api_init:.4f}  smi_init={smi_init:.4f}')
    lead_str = ', '.join(
        f'L{li+1}->mean={ens_mean[li]:.2f}/max={ens_max[li]}'
        for li in range(n_leads_used)
    )
    print(f'    risk ({lead_str})')
    print()


if not records:
    print('No SEAS5 .nc files found — nothing to process.')
    print('Run one of:')
    print('  python src/download_seas5_inputs.py           # real CDS download')
    print('  python src/download_seas5_inputs.py --simulate  # ERA5+noise stand-in')
    raise SystemExit(0)

results = pd.DataFrame(records)
results.to_csv(OUT_PATH, index=False)
print(f'Saved: {OUT_PATH}  ({len(results)} rows)')
print()


# ── Macro recall at leads 1, 2, 3, 6 ──────────────────────────────────────────

valid_r = results.dropna(subset=['actual_drought_risk']).copy()
valid_r['actual_drought_risk'] = valid_r['actual_drought_risk'].astype(int)
valid_r['seas5_drought_max']   = valid_r['seas5_drought_max'].astype(int)
valid_r['seas5_rounded_mean']  = valid_r['seas5_drought_mean'].round().astype(int)


def macro_recall_at_lead(df_r, lead, pred_col):
    sub = df_r[df_r['lead_months'] == lead]
    if len(sub) == 0:
        return np.nan
    return float(recall_score(
        sub['actual_drought_risk'], sub[pred_col],
        average='macro', zero_division=0, labels=range(5),
    ))


print('SEAS5 drought macro recall (ensemble-max and rounded ensemble-mean):')
print(f'  {"Lead":>6}  {"Ens-max":>10}  {"Ens-mean (rounded)":>20}  {"N":>4}')
for lead in [1, 2, 3, 6]:
    r_max  = macro_recall_at_lead(valid_r, lead, 'seas5_drought_max')
    r_mean = macro_recall_at_lead(valid_r, lead, 'seas5_rounded_mean')
    n_pts  = len(valid_r[valid_r['lead_months'] == lead])
    max_s  = f'{r_max:.4f}' if not np.isnan(r_max) else 'N/A'
    mean_s = f'{r_mean:.4f}' if not np.isnan(r_mean) else 'N/A'
    print(f'  +{lead:2d} mo   {max_s:>10}  {mean_s:>20}  {n_pts:>4}')
print()

# Actual class distribution in the valid forecast months
print('Actual drought risk class distribution (forecast months):')
print(valid_r['actual_drought_risk'].value_counts().sort_index().to_string())
print()

n_sim = int(results['simulated'].sum())
if n_sim > 0:
    print(f'NOTE: {n_sim} of {len(results)} rows are from simulated SEAS5 data.')
    print('Results demonstrate the methodology; real SEAS5 output may differ.')
    print()


# ── Figure: 6-panel ensemble spread + mean vs actual ─────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()

RISK_COLORS_L  = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']

for idx, (year, month) in enumerate(INIT_MONTHS):
    label = f'{year}-{month:02d}'
    ax    = axes[idx]
    sub   = results[results['init_date'] == label].sort_values('lead_months')

    if sub.empty:
        ax.text(0.5, 0.5, 'No SEAS5 data', transform=ax.transAxes,
                ha='center', va='center', color='gray', fontsize=9)
        ax.set_title(f'Init: {label}', fontsize=9)
        ax.set_ylim(-0.4, 4.4)
        ax.set_yticks(range(5))
        ax.set_yticklabels([LABEL_MAP[k] for k in range(5)], fontsize=7)
        continue

    leads     = sub['lead_months'].values
    ens_mean  = sub['seas5_drought_mean'].values
    ens_max   = sub['seas5_drought_max'].values
    actual    = sub['actual_drought_risk'].values

    # Compute ensemble spread: load per-member data from the nc file
    # (only for figure; use pre-computed ens_max as upper bound)
    nc_path = SEAS5_DIR / f'seas5_jijiga_{year}_{month:02d}.nc'
    try:
        with xr.open_dataset(nc_path) as ds_fig:
            lats_f = ds_fig['latitude'].values.astype(float)
            lons_f = ds_fig['longitude'].values.astype(float)
            ilat_f = int(np.argmin(np.abs(lats_f - 9.25)))
            ilon_f = int(np.argmin(np.abs(lons_f - 42.75)))
            vm_f   = [_month_offset(year, month, lead) for lead in leads]
            n_leads_fig = len(leads)
            if 'tprate' in ds_fig.data_vars:
                raw_f = ds_fig['tprate'].values[:, 0, :n_leads_fig, ilat_f, ilon_f].astype(float)
                tp_raw_fig = np.zeros_like(raw_f)
                for li2, (vy_f2, vm_f2) in enumerate(vm_f):
                    days_f = calendar.monthrange(vy_f2, vm_f2)[1]
                    tp_raw_fig[:, li2] = np.clip(raw_f[:, li2] * days_f * 86400, 0, None)
            else:
                raw_f = ds_fig['tp'].values
                if raw_f.ndim == 4:
                    raw_f = raw_f[:, :n_leads_fig, ilat_f, ilon_f]
                elif raw_f.ndim == 3:
                    raw_f = raw_f[:, :n_leads_fig, 0]
                tp_raw_fig = np.clip(raw_f.astype(float), 0, None)
        n_mem_fig = tp_raw_fig.shape[0]
        wup_cwb = np.array([_era5_monthly_cwb(*_month_offset(year, month, o))
                            for o in range(-4, 1)], dtype=float)
        all_risks = np.zeros((n_mem_fig, len(leads)), dtype=int)
        for mi_f in range(n_mem_fig):
            seas5_cwb_f = np.array([
                tp_raw_fig[mi_f, li] + pev_clim[vm_f[li][1]]
                for li in range(len(leads))
            ])
            combined_f = np.concatenate([wup_cwb, seas5_cwb_f])
            for li, (vy_f, vm_f2) in enumerate(vm_f):
                cwb6m_f = float(np.sum(combined_f[li : li + 6]))
                spei_f  = cwb6m_to_spei(cwb6m_f, vm_f2)
                base_f  = spei_to_base_drought_risk(spei_f)
                risk_f  = apply_modifier(base_f, api_init, smi_init)
                all_risks[mi_f, li] = risk_f
        p10 = np.percentile(all_risks, 10, axis=0)
        p90 = np.percentile(all_risks, 90, axis=0)
        p25 = np.percentile(all_risks, 25, axis=0)
        p75 = np.percentile(all_risks, 75, axis=0)
        ax.fill_between(leads, p10, p90, alpha=0.15, color='royalblue', label='10-90 pct')
        ax.fill_between(leads, p25, p75, alpha=0.30, color='royalblue', label='25-75 pct')
    except Exception:
        pass   # if nc unavailable, skip spread

    # Ensemble mean and max
    ax.step(leads, ens_mean, where='mid', color='royalblue',  lw=2.0,
            label='Ens mean', zorder=3)
    ax.step(leads, ens_max,  where='mid', color='darkorange', lw=1.5, ls='--',
            label='Ens max',  zorder=3)
    # Actual ERA5 drought risk
    valid_mask  = ~pd.isna(actual)
    if valid_mask.any():
        ax.step(leads[valid_mask], actual[valid_mask].astype(float),
                where='mid', color='black', lw=1.5, ls=':', label='Actual (ERA5)',
                zorder=4)

    ax.set_ylim(-0.4, 4.4)
    ax.set_yticks(range(5))
    ax.set_yticklabels([LABEL_MAP[k] for k in range(5)], fontsize=7)
    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f'+{l}m' for l in range(1, 7)], fontsize=7)
    ax.set_title(f'Init: {label}', fontsize=9, fontweight='bold')
    ax.grid(axis='y', ls=':', alpha=0.4)
    if idx == 0:
        ax.legend(fontsize=6, loc='upper right', ncol=2)

fig.suptitle(
    'ECMWF SEAS5 seasonal drought risk forecasts — 6 init months, Jijiga (42.75°E, 9.25°N)\n'
    'Blue shading = ensemble spread (25–75 pct inner, 10–90 pct outer)  |  '
    'Blue line = ensemble mean  |  Orange dashed = ensemble max  |  '
    'Black dotted = actual ERA5 drought risk',
    fontsize=8, y=1.01,
)
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
print(f'Saved figure: {FIG_PATH}')

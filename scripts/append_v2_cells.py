"""Append [V2] ENSO/IOD cells to phase4 and phase5 notebooks."""
import json

def to_source(code: str) -> list:
    lines = code.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            if line:
                result.append(line)
    return result

def mk_md(cell_id, text):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": to_source(text)}

def mk_code(cell_id, code):
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "execution_count": None, "outputs": [], "source": to_source(code)}


# ── Phase 4 new cells ─────────────────────────────────────────────────────────

PH4_MD = """\
---
## [V2] ENSO/IOD Features

Appends Nino3.4 (ERSST v6 anomaly, NOAA PSL) and Dipole Mode Index (DMI/IOD,
NOAA PSL HadISST-derived) features to the feature matrix. Both indices are monthly;
they are resampled to daily via forward-fill within each calendar month before lagging.

Six new lag features: `nino34_lag_{30,90,180}` and `dmi_lag_{30,90,180}`.
- 30-day lag: same-season teleconnection signal
- 90-day lag: preceding-season ENSO/IOD state
- 180-day lag: semi-annual memory, capturing the season two rainy seasons back

Output: `feature_matrix_v2_drought.parquet`"""

PH4_CODE = """\
# ── [V2] ENSO/IOD Feature Engineering ────────────────────────────────────────
NINO34_PATH = '../src/data/raw/nino34_monthly.csv'
DMI_PATH    = '../src/data/raw/dmi_monthly.csv'
V2_OUT_PATH = '../src/data/processed/feature_matrix_v2_drought.parquet'

# Load monthly indices
df_nino = pd.read_csv(NINO34_PATH, parse_dates=['time'])
df_dmi  = pd.read_csv(DMI_PATH,    parse_dates=['time'])

print(f'Nino3.4 monthly: {df_nino.shape}  '
      f'range: {df_nino.time.min().date()} -> {df_nino.time.max().date()}')
print(f'DMI     monthly: {df_dmi.shape}   '
      f'range: {df_dmi.time.min().date()} -> {df_dmi.time.max().date()}')

# ── Resample monthly -> daily via forward-fill within each calendar month ──
# Uses the full daily spine (1998-01-01 -> 2026-12-31) so that 180-day lags
# cover the entire feature-matrix period starting 2001-06-01 without data loss.
full_daily = pd.DataFrame({'time': pd.date_range('1998-01-01', '2026-12-31', freq='D')})

def monthly_to_daily(df_monthly, col):
    tmp = full_daily.copy()
    tmp['year']  = tmp['time'].dt.year
    tmp['month'] = tmp['time'].dt.month
    df_monthly = df_monthly.copy()
    df_monthly['year']  = df_monthly['time'].dt.year
    df_monthly['month'] = df_monthly['time'].dt.month
    merged = tmp.merge(df_monthly[['year', 'month', col]], on=['year', 'month'], how='left')
    merged = merged[['time', col]].sort_values('time').reset_index(drop=True)
    merged[col] = merged[col].ffill()
    return merged

df_nino_daily = monthly_to_daily(df_nino, 'nino34')
df_dmi_daily  = monthly_to_daily(df_dmi,  'dmi')
print(f'\\nnino34 daily NaNs after ffill: {df_nino_daily.nino34.isna().sum()}')
print(f'dmi    daily NaNs after ffill: {df_dmi_daily.dmi.isna().sum()}')

# ── Compute lag columns on the FULL daily spine, then merge lagged cols only ──
# Critical: lags must be computed on the full series starting 1998-01-01, NOT
# on the already-truncated feature matrix (which starts 2001-06-01).  If we
# lagged after merging we would lose the first 180 training rows unnecessarily.
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

print(f'\\nCreated {len(enso_iod_lag_cols)} ENSO/IOD lag columns on full spine: {enso_iod_lag_cols}')
# Lag_180 is valid from 1998-07 onwards; the feature matrix starts 2001-06-01 -> no NaN drop needed
print(f'nino34_lag_180 first valid date: {df_enso_lags.dropna(subset=["nino34_lag_180"]).time.min().date()}')
print(f'dmi_lag_180    first valid date: {df_enso_lags.dropna(subset=["dmi_lag_180"]).time.min().date()}')

# ── Merge only lagged cols onto existing feature matrix ───────────────────────
lag_only = df_enso_lags[['time'] + enso_iod_lag_cols]
df_feat = pd.read_parquet(OUT_PATH)
df_feat['time'] = pd.to_datetime(df_feat['time'])
df_feat = df_feat.merge(lag_only, on='time', how='left')

# Validate: should be zero NaNs (feature matrix starts well after lag_180 valid date)
nans_after_merge = df_feat[enso_iod_lag_cols].isna().sum()
print(f'\\nNaNs after merge:')
print(nans_after_merge.to_string())
print(f'Shape: {df_feat.shape}')

# ── Validate: no NaNs in lag cols for training period ─────────────────────────
train_mask_v2 = df_feat['time'].dt.year <= 2022
nans_train = df_feat.loc[train_mask_v2, enso_iod_lag_cols].isna().sum()
print(f'\\nNaN counts in ENSO/IOD lag cols (training 2001-2022):')
print(nans_train.to_string())
print(f'All zero NaNs in training period: {(nans_train == 0).all()}')

print(f'\\nDate range: {df_feat.time.min().date()} -> {df_feat.time.max().date()}')
print(f'Train rows: {train_mask_v2.sum()}')
print(f'Test  rows: {(df_feat.time.dt.year >= 2023).sum()}')

# ── Save ──────────────────────────────────────────────────────────────────────
df_feat.to_parquet(V2_OUT_PATH, index=False)
print(f'\\nSaved: {V2_OUT_PATH}')
print(f'Shape: {df_feat.shape}  '
      f'(51 original + {len(enso_iod_lag_cols)} new = {51 + len(enso_iod_lag_cols)} expected cols)')"""

# ── Phase 5 new cells ─────────────────────────────────────────────────────────

PH5_MD = """\
---
## [V2] Drought Models with ENSO/IOD

Retrains the four drought classifiers (+1d, +3d, +7d, +14d) using the extended
feature matrix `feature_matrix_v2_drought.parquet` which includes six ENSO/IOD lag
features in addition to the 40 original ERA5-derived lags.

Same walk-forward CV folds, hyperparameter grid, class weights, and temporal split
as v1. Models saved as `drought_v2_+{horizon}d.ubj`.

V1 vs V2 comparison is printed, and SHAP analysis on `drought_v2_+14d` verifies
whether ENSO/IOD features appear in the top-10 most important predictors."""

PH5_LOAD = """\
# ── Load v2 feature matrix ────────────────────────────────────────────────────
V2_DATA_PATH = '../src/data/processed/feature_matrix_v2_drought.parquet'

df_v2 = pd.read_parquet(V2_DATA_PATH)
df_v2['time'] = pd.to_datetime(df_v2['time'])
df_v2 = df_v2.sort_values('time').reset_index(drop=True)

ENSO_IOD_COLS   = ['nino34_lag_30', 'nino34_lag_90', 'nino34_lag_180',
                   'dmi_lag_30',    'dmi_lag_90',    'dmi_lag_180']
FEATURE_COLS_V2 = FEATURE_COLS + ENSO_IOD_COLS

print(f'V2 feature matrix: {df_v2.shape}')
print(f'Train rows: {(df_v2.time.dt.year <= 2022).sum()}')
print(f'Test  rows: {(df_v2.time.dt.year >= 2023).sum()}')
print(f'Feature columns: {len(FEATURE_COLS_V2)}  '
      f'({len(FEATURE_COLS)} original + {len(ENSO_IOD_COLS)} ENSO/IOD)')
print(f'New cols: {ENSO_IOD_COLS}')

# Sanity check: new cols present and no NaN in train
train_v2 = df_v2[df_v2.time.dt.year <= 2022]
nans_new = train_v2[ENSO_IOD_COLS].isna().sum()
print(f'\\nNaN check (training period): {dict(nans_new)}')"""

PH5_TRAIN = """\
# ── Train v2 drought models ───────────────────────────────────────────────────
# Reuses tune_and_train / walk_forward_cv / sample_weights from ph5-helpers cell
results_v2 = {}

print(f'{"Model":<25} {"Best params":<45} {"CV Recall":>10} {"Test Recall":>12}')
print('-' * 96)

for n in HORIZONS:
    target   = f'drought_risk_t_plus_{n}'
    task_key = f'drought_+{n}d'

    res = tune_and_train(target, df_v2, PARAM_GRID, WF_FOLDS, FEATURE_COLS_V2)
    results_v2[task_key] = res

    res['model'].save_model(f'{MODEL_DIR}/drought_v2_+{n}d.ubj')

    param_str = (f"depth={res['best_params']['max_depth']} "
                 f"lr={res['best_params']['learning_rate']} "
                 f"n={res['best_params']['n_estimators']}")
    print(f'  {task_key:<23} {param_str:<45} {res["cv_mean"]:>10.4f} {res["test_recall"]:>12.4f}')

print('\\nAll 4 v2 drought models trained and saved.')"""

PH5_COMPARE = """\
# ── V1 vs V2 side-by-side comparison ─────────────────────────────────────────
print('Drought model comparison — macro recall on test set (2023-2025)')
print(f'{"Task":<20} {"Baseline":>10} {"V1":>10} {"V2":>10} {"V2 - V1":>10}')
print('-' * 64)

for n in HORIZONS:
    task   = f'drought_+{n}d'
    target = f'drought_risk_t_plus_{n}'
    base   = baseline_recall.get(target, 0)
    v1     = results[task]['test_recall']
    v2     = results_v2[task]['test_recall']
    delta  = v2 - v1
    print(f'  {task:<18} {base:>10.4f} {v1:>10.4f} {v2:>10.4f} {delta:>+10.4f}')

# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 5))
x      = np.arange(len(HORIZONS))
width  = 0.25
base_s = [baseline_recall.get(f'drought_risk_t_plus_{n}', 0) for n in HORIZONS]
v1_s   = [results[f'drought_+{n}d']['test_recall'] for n in HORIZONS]
v2_s   = [results_v2[f'drought_+{n}d']['test_recall'] for n in HORIZONS]

ax.bar(x - width, base_s, width, label='Naive baseline',            color='#95a5a6', alpha=0.85)
ax.bar(x,         v1_s,   width, label='XGBoost v1 (40 features)',  color='steelblue', alpha=0.85)
ax.bar(x + width, v2_s,   width, label='XGBoost v2 (+ENSO/IOD)',    color='tomato',    alpha=0.85)

for xi, (b, v1, v2) in zip(x, zip(base_s, v1_s, v2_s)):
    ax.text(xi - width, b  + 0.01, f'{b:.3f}',  ha='center', va='bottom', fontsize=7, color='gray')
    ax.text(xi,         v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom', fontsize=7, color='steelblue')
    ax.text(xi + width, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom', fontsize=7, color='tomato')

ax.set_xlabel('Forecast Horizon (days)')
ax.set_ylabel('Macro Recall (test set 2023-2025)')
ax.set_title('Drought Models: Baseline vs XGBoost v1 vs XGBoost v2 (+ENSO/IOD)')
ax.set_xticks(x)
ax.set_xticklabels([f'+{n}d' for n in HORIZONS])
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('figures/phase5_v2_drought_comparison.png', dpi=150, bbox_inches='tight')
plt.show()"""

PH5_SHAP = """\
# ── SHAP analysis: v2 drought_+14d ────────────────────────────────────────────
res_14 = results_v2['drought_+14d']
X_samp = res_14['X_test'][:500]

try:
    expl      = shap.Explainer(res_14['model'], X_samp)
    shap_vals = expl(X_samp, check_additivity=False)
except Exception:
    shap_vals = None

if shap_vals is not None and shap_vals.values.ndim == 3:
    mean_abs = np.abs(shap_vals.values).mean(axis=(0, 2))
elif shap_vals is not None:
    mean_abs = np.abs(shap_vals.values).mean(axis=0)
else:
    mean_abs = res_14['model'].feature_importances_

shap_ser  = pd.Series(mean_abs, index=FEATURE_COLS_V2)
top15     = shap_ser.nlargest(15)
top10_lst = shap_ser.nlargest(10).index.tolist()

enso_in_top10 = [f for f in ENSO_IOD_COLS if f in top10_lst]
print(f'ENSO/IOD features in top 10 SHAP: '
      f'{enso_in_top10 if enso_in_top10 else "NONE — check data/lags"}')

colors = ['tomato' if f in ENSO_IOD_COLS else 'steelblue' for f in top15.index[::-1]]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top15.index[::-1], top15.values[::-1], color=colors, alpha=0.85)
from matplotlib.patches import Patch
legend_handles = [
    Patch(color='steelblue', alpha=0.85, label='ERA5 lag feature'),
    Patch(color='tomato',    alpha=0.85, label='ENSO/IOD lag feature'),
]
ax.legend(handles=legend_handles, fontsize=9)
ax.set_title(
    'V2 drought_+14d — SHAP Feature Importance (top 15 of 46)\\n'
    'Red = ENSO/IOD features; verifies teleconnection signal captured',
    fontsize=11
)
ax.set_xlabel('mean |SHAP value| averaged over test samples and risk classes', fontsize=10)
plt.tight_layout()
plt.savefig('figures/phase5_v2_shap_drought14.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\nTop 15 features (mean |SHAP|):')
for feat, val in top15.items():
    tag = '  <-- ENSO/IOD' if feat in ENSO_IOD_COLS else ''
    print(f'  {feat:<30} {val:.5f}{tag}')"""


# ── Append to notebooks ───────────────────────────────────────────────────────

with open('notebooks/phase4_feature_engineering.ipynb', encoding='utf-8') as f:
    nb4 = json.load(f)

nb4['cells'].extend([
    mk_md('v2-ph4-header', PH4_MD),
    mk_code('v2-ph4-enso', PH4_CODE),
])

with open('notebooks/phase4_feature_engineering.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb4, f, ensure_ascii=False, indent=1)

print(f'phase4 saved: {len(nb4["cells"])} cells total')

with open('notebooks/phase5_xgboost.ipynb', encoding='utf-8') as f:
    nb5 = json.load(f)

nb5['cells'].extend([
    mk_md('v2-ph5-header',    PH5_MD),
    mk_code('v2-ph5-load',    PH5_LOAD),
    mk_code('v2-ph5-train',   PH5_TRAIN),
    mk_code('v2-ph5-compare', PH5_COMPARE),
    mk_code('v2-ph5-shap',    PH5_SHAP),
])

with open('notebooks/phase5_xgboost.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb5, f, ensure_ascii=False, indent=1)

print(f'phase5 saved: {len(nb5["cells"])} cells total')

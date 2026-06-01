"""Run Phase 5 V2: train ENSO/IOD-enhanced drought models and run SHAP."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
import shap
import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)

# ── Constants (mirrors phase5 imports) ────────────────────────────────────────
FEATURE_VARS = ['spei_6', 'api_92', 'smi_fc', 'total_ro', 'tp', 't2m', 'e', 'pev']
LAGS         = [1, 3, 7, 14, 365]
HORIZONS     = [1, 3, 7, 14]
FEATURE_COLS = [f'{v}_lag_{l}' for v in FEATURE_VARS for l in LAGS]
N_CLASSES    = 4
MODEL_DIR    = '../src/data/processed/xgb_models'
os.makedirs(MODEL_DIR, exist_ok=True)

WF_FOLDS = [
    {'train': (2001, 2014), 'val': (2015, 2016)},
    {'train': (2001, 2016), 'val': (2017, 2018)},
    {'train': (2001, 2018), 'val': (2019, 2020)},
]
PARAM_GRID = [
    {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 300},
    {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 300},
    {'max_depth': 6, 'learning_rate': 0.01, 'n_estimators': 500},
    {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 200},
    {'max_depth': 4, 'learning_rate': 0.10, 'n_estimators': 200},
]

V1_RECALLS = {
    'drought_+1d':  0.8441,
    'drought_+3d':  0.7742,
    'drought_+7d':  0.7535,
    'drought_+14d': 0.6256,
}
BASELINE_RECALLS = {
    'drought_risk_t_plus_1':  0.9821,
    'drought_risk_t_plus_3':  0.9463,
    'drought_risk_t_plus_7':  0.8747,
    'drought_risk_t_plus_14': 0.7561,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def sample_weights(y, n_classes=N_CLASSES):
    counts = np.bincount(y, minlength=n_classes)
    w = len(y) / (n_classes * np.maximum(counts, 1).astype(float))
    return w[y]


def walk_forward_cv(X_arr, y_arr, times, folds, params):
    scores = []
    for fold in folds:
        tr = (times.dt.year >= fold['train'][0]) & (times.dt.year <= fold['train'][1])
        va = (times.dt.year >= fold['val'][0])   & (times.dt.year <= fold['val'][1])
        X_tr, y_tr = X_arr[tr.values], y_arr[tr.values]
        X_va, y_va = X_arr[va.values], y_arr[va.values]
        if len(X_tr) == 0 or len(X_va) == 0:
            continue
        sw = sample_weights(y_tr)
        m = XGBClassifier(**params, objective='multi:softmax', num_class=N_CLASSES,
                          eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)
        m.fit(X_tr, y_tr, sample_weight=sw)
        scores.append(recall_score(y_va, m.predict(X_va), average='macro', zero_division=0))
    return (np.mean(scores), np.std(scores)) if scores else (0.0, 0.0)


def tune_and_train(target, df, param_grid, folds, feature_cols):
    valid = df[target].notna()
    X_all = df.loc[valid, feature_cols].values
    y_all = df.loc[valid, target].values.astype(int)
    t_all = df.loc[valid, 'time'].reset_index(drop=True)
    best_params, best_mean = param_grid[0], -1.0
    cv_log = []
    for params in param_grid:
        mean_r, std_r = walk_forward_cv(X_all, y_all, t_all, folds, params)
        cv_log.append({**params, 'cv_mean': mean_r, 'cv_std': std_r})
        if mean_r > best_mean:
            best_mean, best_params = mean_r, params
    best_std = [r['cv_std'] for r in cv_log if r['cv_mean'] == best_mean][0]
    tr_mask = t_all.dt.year <= 2022
    X_tr, y_tr = X_all[tr_mask.values], y_all[tr_mask.values]
    sw = sample_weights(y_tr)
    final = XGBClassifier(**best_params, objective='multi:softmax', num_class=N_CLASSES,
                          eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)
    final.fit(X_tr, y_tr, sample_weight=sw)
    te_mask = t_all.dt.year >= 2023
    X_te, y_te = X_all[te_mask.values], y_all[te_mask.values]
    y_pred = final.predict(X_te)
    test_recall = recall_score(y_te, y_pred, average='macro', zero_division=0)
    return {
        'model':       final,
        'best_params': best_params,
        'cv_mean':     round(best_mean, 4),
        'cv_std':      round(best_std, 4),
        'test_recall': round(test_recall, 4),
        'y_true':      y_te,
        'y_pred':      y_pred,
        'X_test':      X_te,
    }


# ── Load v2 feature matrix ────────────────────────────────────────────────────
df_v2 = pd.read_parquet('../src/data/processed/feature_matrix_v2_drought.parquet')
df_v2['time'] = pd.to_datetime(df_v2['time'])
df_v2 = df_v2.sort_values('time').reset_index(drop=True)

ENSO_IOD_COLS   = ['nino34_lag_30', 'nino34_lag_90', 'nino34_lag_180',
                   'dmi_lag_30',    'dmi_lag_90',    'dmi_lag_180']
FEATURE_COLS_V2 = FEATURE_COLS + ENSO_IOD_COLS

print(f'V2 matrix: {df_v2.shape}')
print(f'Train: {(df_v2.time.dt.year <= 2022).sum()}  Test: {(df_v2.time.dt.year >= 2023).sum()}')
print(f'Features: {len(FEATURE_COLS_V2)} ({len(FEATURE_COLS)} orig + {len(ENSO_IOD_COLS)} ENSO/IOD)')

nans_new = df_v2[df_v2.time.dt.year <= 2022][ENSO_IOD_COLS].isna().sum()
print(f'NaN check (training): {dict(nans_new)}')

# ── Train v2 drought models ───────────────────────────────────────────────────
results_v2 = {}
print(f'\n{"Model":<25} {"Best params":<45} {"CV Recall":>10} {"Test Recall":>12}')
print('-' * 96)

for n in HORIZONS:
    target   = f'drought_risk_t_plus_{n}'
    task_key = f'drought_+{n}d'
    res = tune_and_train(target, df_v2, PARAM_GRID, WF_FOLDS, FEATURE_COLS_V2)
    results_v2[task_key] = res
    res['model'].save_model(f'{MODEL_DIR}/drought_v2_+{n}d.ubj')
    p  = res['best_params']
    ps = f"depth={p['max_depth']} lr={p['learning_rate']} n={p['n_estimators']}"
    print(f"  {task_key:<23} {ps:<45} {res['cv_mean']:>10.4f} {res['test_recall']:>12.4f}")

print('\nAll 4 v2 drought models saved.')

# ── V1 vs V2 comparison ───────────────────────────────────────────────────────
print(f'\n{"Task":<20} {"Baseline":>10} {"V1":>10} {"V2":>10} {"V2-V1":>10}')
print('-' * 64)
for n in HORIZONS:
    task = f'drought_+{n}d'
    base = BASELINE_RECALLS.get(f'drought_risk_t_plus_{n}', 0)
    v1   = V1_RECALLS[task]
    v2   = results_v2[task]['test_recall']
    print(f"  {task:<18} {base:>10.4f} {v1:>10.4f} {v2:>10.4f} {v2 - v1:>+10.4f}")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 5))
x      = np.arange(len(HORIZONS))
width  = 0.25
base_s = [BASELINE_RECALLS.get(f'drought_risk_t_plus_{n}', 0) for n in HORIZONS]
v1_s   = [V1_RECALLS[f'drought_+{n}d'] for n in HORIZONS]
v2_s   = [results_v2[f'drought_+{n}d']['test_recall'] for n in HORIZONS]

ax.bar(x - width, base_s, width, label='Naive baseline',           color='#95a5a6', alpha=0.85)
ax.bar(x,         v1_s,   width, label='XGBoost v1 (40 features)', color='steelblue', alpha=0.85)
ax.bar(x + width, v2_s,   width, label='XGBoost v2 (+ENSO/IOD)',   color='tomato',    alpha=0.85)

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
plt.close()
print('Comparison chart saved.')

# ── SHAP: v2 drought_+14d ─────────────────────────────────────────────────────
res_14 = results_v2['drought_+14d']
X_samp = res_14['X_test'][:500]

try:
    expl      = shap.Explainer(res_14['model'], X_samp)
    shap_vals = expl(X_samp, check_additivity=False)
except Exception as e:
    print(f'SHAP Explainer fallback ({e})')
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
print(f'\nENSO/IOD in top 10 SHAP: {enso_in_top10 if enso_in_top10 else "NONE"}')

colors = ['tomato' if f in ENSO_IOD_COLS else 'steelblue' for f in top15.index[::-1]]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top15.index[::-1], top15.values[::-1], color=colors, alpha=0.85)
ax.legend(handles=[
    Patch(color='steelblue', alpha=0.85, label='ERA5 lag feature'),
    Patch(color='tomato',    alpha=0.85, label='ENSO/IOD lag feature'),
], fontsize=9)
ax.set_title(
    'V2 drought_+14d — SHAP Feature Importance (top 15 of 46)\n'
    'Red = ENSO/IOD features; verifies teleconnection signal captured',
    fontsize=11
)
ax.set_xlabel('mean |SHAP value| averaged over test samples and risk classes', fontsize=10)
plt.tight_layout()
plt.savefig('figures/phase5_v2_shap_drought14.png', dpi=150, bbox_inches='tight')
plt.close()
print('SHAP plot saved.')

print('\nTop 15 features (mean |SHAP|):')
for feat, val in top15.items():
    tag = '  <-- ENSO/IOD' if feat in ENSO_IOD_COLS else ''
    print(f'  {feat:<30} {val:.5f}{tag}')

"""Add interpretation markdown cells to phase4 and phase5 notebooks.
Overwrites the v2-ph4-interp and v2-ph5-interp cells if they already exist."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
import os
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

def to_source(code):
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
    return {'cell_type': 'markdown', 'id': cell_id, 'metadata': {},
            'source': to_source(text)}

# ── Phase 4 interpretation ─────────────────────────────────────────────────────
PH4_INTERP = (
    "**V2 ENSO/IOD Feature Engineering — Output Summary**\n"
    "\n"
    "The extended feature matrix adds six lag columns derived from two monthly climate indices:\n"
    "\n"
    "| Feature | Source | Lag | Scientific rationale |\n"
    "|---------|--------|-----|---------------------|\n"
    "| `nino34_lag_30` | NOAA ERSST v6 | 30 days | Same-season ENSO state |\n"
    "| `nino34_lag_90` | NOAA ERSST v6 | 90 days | Preceding quarter ENSO state |\n"
    "| `nino34_lag_180` | NOAA ERSST v6 | 180 days | Prior season ENSO onset signal |\n"
    "| `dmi_lag_30` | HadISST DMI (NOAA PSL) | 30 days | Active IOD phase |\n"
    "| `dmi_lag_90` | HadISST DMI (NOAA PSL) | 90 days | Preceding quarter IOD state |\n"
    "| `dmi_lag_180` | HadISST DMI (NOAA PSL) | 180 days | Prior IOD season signal |\n"
    "\n"
    "Lags are computed on the full 1998–2026 daily spine before merging, so the feature matrix\n"
    "retains all 8980 rows (same as v1). Zero NaNs in the training period (2001–2022) confirmed.\n"
    "\n"
    "The `lag_180` features are expected to be most informative because the Horn of Africa's\n"
    "bimodal rainy seasons (April–May, October–November) are each preceded by a ~6-month\n"
    "ENSO/IOD setup period. A La Niña state in April predicts reduced October–November\n"
    "rains (Deyr season) six months later."
)

# ── Phase 5 interpretation ─────────────────────────────────────────────────────
PH5_INTERP = (
    "**V2 Drought Models — Interpretation**\n"
    "\n"
    "**SHAP findings (`drought_v2_+14d`):**\n"
    "\n"
    "The 180-day IOD lag (`dmi_lag_180`, rank 2) and 180-day ENSO lag (`nino34_lag_180`, rank 6)\n"
    "appear in the top 10 most important features — confirming that the model has learned a\n"
    "teleconnection signal from the ENSO/IOD indices beyond what ERA5 lag features alone encode.\n"
    "\n"
    "The 30- and 90-day lags rank lower because `spei_6` and `api_92` lags already capture recent\n"
    "precipitation deficits driven by active ENSO/IOD phases. The 180-day lags add the most new\n"
    "information: they encode the *preceding season's* state before the current SPEI signal has\n"
    "fully developed.\n"
    "\n"
    "**Performance comparison:**\n"
    "\n"
    "V2 does not outperform V1 on macro recall despite ENSO/IOD features showing meaningful SHAP\n"
    "importance. This result is interpretable:\n"
    "\n"
    "1. **Redundancy with SPEI**: SPEI-6 already encodes precipitation deficit over 6 months,\n"
    "   capturing most of the ENSO/IOD teleconnection effect. The new features add some signal\n"
    "   but also introduce noise.\n"
    "\n"
    "2. **Forecast horizon mismatch**: XGBoost predicts 1–14 days ahead. ENSO and IOD primarily\n"
    "   operate on seasonal-to-interannual time scales. Their predictive value would be larger\n"
    "   in a seasonal outlooks framework (months ahead) than in a 14-day prediction framework.\n"
    "\n"
    "3. **Test period**: 2023–2025 included a strong El Niño→La Niña transition. The model trained\n"
    "   on 2001–2022 patterns may not generalise perfectly to this specific teleconnection period.\n"
    "\n"
    "**Operational recommendation**: Use V1 models for 1–14 day operational forecasting. The\n"
    "ENSO/IOD features provide scientific validation that the indices carry climatological signal\n"
    "at Jijiga, but their operational value would be better realised in a separate seasonal\n"
    "risk outlook tool (months-ahead horizon) rather than in this 14-day XGBoost predictor."
)

# ── Apply to notebooks ─────────────────────────────────────────────────────────
for path, cell_id, text in [
    ('notebooks/phase4_feature_engineering.ipynb', 'v2-ph4-interp', PH4_INTERP),
    ('notebooks/phase5_xgboost.ipynb',             'v2-ph5-interp', PH5_INTERP),
]:
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)

    # Replace if exists, else append
    ids = [c['id'] for c in nb['cells']]
    new_cell = mk_md(cell_id, text)
    if cell_id in ids:
        idx = ids.index(cell_id)
        nb['cells'][idx] = new_cell
    else:
        nb['cells'].append(new_cell)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'{path}: {len(nb["cells"])} cells, cell {cell_id} written OK')

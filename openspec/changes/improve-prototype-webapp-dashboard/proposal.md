## Why

The current Streamlit prototype proves the model workflow, but it still feels like a default demo page and does not make the risk warning, model components, or prototype limitations clear enough for a portfolio audience. Improving the dashboard presentation and transparency will make the webapp more attractive, easier to understand, and more scientifically honest about the hybrid XGBoost plus forecast-index demo pipeline.

## What Changes

- Redesign the prototype app into a more polished early-warning dashboard with stronger typography, clearer hierarchy, richer climate-risk colors, and less monotone styling.
- Replace the current generic result layout with prominent flood and drought risk cards that show the final risk, forecast horizon, component outputs, and concise interpretation.
- Improve model transparency by clearly distinguishing XGBoost predictions from the forecast-index/GenCast-style demo component and by avoiding wording that implies live GPU GenCast inference.
- Add a more scannable forecast timeline that shows daily flood risk, drought risk, and precipitation for the 7-day forecast period.
- Improve the model details section so users can see which variables are live forecast inputs, which values are frozen from ERA5, and which historical lag features feed XGBoost.
- Keep the app bilingual where existing text is bilingual, preserving English and Dutch labels.
- Preserve the existing model files, prediction logic, data sources, and notebook-derived methodology unless a later change explicitly targets model behavior.

## Capabilities

### New Capabilities
- `prototype-dashboard`: User-facing dashboard behavior, model transparency, and visual presentation for the Streamlit prototype app.

### Modified Capabilities

## Impact

- Affected code: `prototype_app/app.py`, and possibly small helper additions in `prototype_app/model_utils.py` if needed to expose display-friendly metadata.
- Affected documentation: `prototype_app/README.md` should be updated if terminology changes from "GenCast-style" to clearer demo wording.
- No new training data, model retraining, Azure infrastructure, or notebook reruns are required.
- No change to the persisted XGBoost model artifacts or processed parquet files is intended.

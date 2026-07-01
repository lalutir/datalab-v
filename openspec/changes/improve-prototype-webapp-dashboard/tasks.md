## 1. Dashboard Structure And Theme

- [x] 1.1 Add a scoped Streamlit CSS theme for the dashboard base, typography, cards, buttons, status badges, and responsive wrapping.
- [x] 1.2 Rework the top-level app layout into a dashboard header, status strip, controls area, results area, forecast progression area, model details area, and supporting data table.
- [x] 1.3 Update bilingual text constants for dashboard wording, prototype/demo status, component naming, risk card labels, and model transparency copy.
- [x] 1.4 Keep the sidebar controls usable while improving visual hierarchy for language, horizon, location, and model status.

## 2. Hazard Results

- [x] 2.1 Replace the current flood result block with a stronger flood hazard card showing final risk, horizon, XGBoost output, forecast-index component output, ensemble output, and interpretation.
- [x] 2.2 Replace the current drought result block with a stronger drought hazard card showing final risk, horizon, XGBoost output, and explanation that final drought risk is XGBoost-based.
- [x] 2.3 Ensure risk colors remain semantically consistent across cards, captions, timeline, and charts.
- [x] 2.4 Preserve all existing `HybridPredictor.predict_hybrid()` result keys and prediction behavior.

## 3. Forecast Progression And Data Display

- [x] 3.1 Add a compact 7-day forecast timeline or equivalent scan-first visualization for flood risk, drought risk, and precipitation.
- [x] 3.2 Restyle or replace the existing Plotly chart so it supports the new dashboard theme and remains readable.
- [x] 3.3 Keep the detailed forecast table with date, precipitation, temperature, evapotranspiration, API, flood score, flood risk, and drought risk.
- [x] 3.4 Verify the results display works for horizon +1, +3, and +7.

## 4. Model Transparency

- [x] 4.1 Rewrite the technical details section to clearly separate XGBoost inputs, forecast-index component inputs, frozen ERA5-derived values, assumptions, and limitations.
- [x] 4.2 Make primary UI wording avoid implying live GenCast GPU inference while preserving a technical note that this is a GenCast-style demo component.
- [x] 4.3 Show the XGBoost feature date and last-known ERA5 state in a clear status or model details area.
- [x] 4.4 Update `prototype_app/README.md` if user-facing terminology changes.

## 5. Verification

- [x] 5.1 Run the Streamlit app locally and confirm the default Jijiga prediction completes without errors.
- [x] 5.2 Check desktop and narrow viewport layouts for overlapping text, clipped labels, unreadable buttons, or broken charts.
- [x] 5.3 Confirm English and Dutch modes both render the updated dashboard text.
- [x] 5.4 Confirm no model files, parquet datasets, notebook outputs, or training behavior were changed.

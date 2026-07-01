## Context

The current prototype app is a Streamlit interface around `HybridPredictor`. It loads the latest ERA5-derived state, pre-trained XGBoost models, a feature matrix, and a 7-day Open-Meteo forecast. The UI currently presents the result with standard Streamlit layout, light risk cards, a Plotly chart, a weather table, and an expandable technical section.

The app is a portfolio/demo artifact, not an official warning system. It must remain scientifically consistent with the project methodology: no EM-DAT training usage, no model retraining, strict use of stored model artifacts, and clear disclosure that the "GenCast-style" component is a forecast-index simulation rather than live GenCast GPU inference.

## Goals / Non-Goals

**Goals:**
- Make the first screen feel like a polished Jijiga early-warning dashboard rather than a default Streamlit demo.
- Improve visual hierarchy so final flood and drought risk levels are immediately visible.
- Use a richer climate-risk theme that combines restrained dark dashboard styling with water, drought, and risk-status accents.
- Make the hybrid prediction easier to understand by showing component outputs and clearly naming the forecast-index demo component.
- Make the 7-day forecast progression easier to scan than the current generic chart/table combination.
- Keep the existing bilingual English/Dutch interface.
- Preserve existing prediction behavior unless a display-only helper is required.

**Non-Goals:**
- No XGBoost retraining or changes to model artifacts.
- No real GenCast inference integration, Azure GPU workflow, or new external weather provider.
- No expansion beyond the Jijiga single-grid-point proof of concept.
- No EM-DAT usage in the app prediction path.
- No replacement of Streamlit with a separate frontend framework.

## Decisions

1. Use Streamlit plus scoped CSS for the visual redesign.

   Rationale: The current app is already Streamlit-based and can be improved substantially with page-level CSS, custom markdown blocks, Streamlit containers, and Plotly styling. This avoids introducing React/Vue or a custom backend for a portfolio app.

   Alternative considered: Build a separate frontend. Rejected because it adds deployment and data plumbing complexity without improving the model demonstration.

2. Organize the app around an early-warning dashboard structure.

   The main page should have:
- A hero/header band with title, location, demo status, and disclaimer.
- A compact status strip with horizon, last ERA5 date, API, SMI, and SPEI-6.
- Prominent flood and drought risk cards after prediction.
- A forecast progression section.
- A model transparency section.
- The detailed forecast table as supporting detail, not the primary visual.

   Rationale: Users need to understand the warning first, then the evidence and limitations.

3. Rename the live forecast component in user-facing copy to avoid implying real GenCast inference.

   Use language such as "Forecast-index component" or "GenCast-style demo component" in visible UI. Technical details can explain that this component uses Open-Meteo precipitation to forward-run API and compute the flood score. The README should also reflect this terminology.

   Rationale: The project can still discuss the foundation model pipeline, but the deployed prototype should be honest about what actually runs.

4. Keep risk colors semantic, but avoid a one-note palette.

   Proposed theme:
- Dashboard base: near-black/charcoal with deep teal accents.
- Flood accent: blue/cyan.
- Drought accent: amber/sand.
- Risk levels: green, amber, red, purple/dark crimson.

   Rationale: The app should feel climate-specific and professional while preserving immediate risk-level recognition.

5. Build the forecast progression as a scan-first timeline.

   The UI should show daily risk states for flood and drought and daily precipitation in a compact visual. This can be implemented with custom HTML/CSS cells or a restyled Plotly chart. The existing detailed table should remain available below for exact values.

   Rationale: A warning dashboard benefits from quick day-by-day scanning more than from large charts alone.

6. Keep model logic stable and expose display metadata only if needed.

   `model_utils.py` may receive small constants or methods that return feature variable names, lag names, component labels, or last-known-state metadata. Prediction results should not change.

   Rationale: Display clarity should not silently alter scientific behavior.

## Risks / Trade-offs

- Styling risk: Streamlit CSS selectors can change between versions -> Mitigate by keeping CSS scoped to stable class names where possible and using custom markdown containers for key cards.
- Clarity risk: Replacing "GenCast-style" everywhere could hide the relation to the Phase 6 foundation-model work -> Mitigate by using "Forecast-index component" in headline UI and explaining "GenCast-style demo" in model details.
- Layout risk: Rich cards and timelines can become cramped on mobile -> Mitigate with responsive columns, wrapping, and viewport checks.
- Scientific risk: A prettier app may look more production-ready than intended -> Mitigate with a visible but tasteful prototype/demo badge and disclaimer.
- Scope risk: Scenario controls and maps are attractive but can expand the project -> Mitigate by keeping them optional follow-up tasks unless explicitly selected for implementation.

## Migration Plan

Implement the dashboard changes in the prototype app and run it locally with `streamlit run prototype_app/app.py`. Verify the app still loads the existing parquet files and XGBoost artifacts, runs a prediction for the default Jijiga location, and displays the same risk keys returned by `HybridPredictor`.

Rollback is straightforward: revert changes to `prototype_app/app.py`, `prototype_app/README.md`, and any display-only helper edits in `prototype_app/model_utils.py`.

## Open Questions

- Should the first implementation include a location map, or keep that for a later enhancement?
- Should the scenario mode for manual rainfall adjustment be included in this change, or treated as a separate interactive-demo change?

## Context

The app is currently a Streamlit prototype around `HybridPredictor`, intended to demonstrate Jijiga flood and drought risk prediction with XGBoost and a forecast-index demo component. The next audience is a non-technical opdrachtgever who should be able to open a public link and immediately understand what the webapp does without running anything locally.

The user supplied weather-dashboard references with a modern visual language: atmospheric image background, glass-like panels, compact controls, large headline weather/risk state, and soft rounded surfaces. The current app already has custom CSS and bilingual text, but it still shows Streamlit chrome, uses a dense dark dashboard layout, and feels more like a technical demo than a polished hosted client demo.

## Goals / Non-Goals

**Goals:**
- Make the first screen feel modern, polished, and suitable to show to a non-technical opdrachtgever.
- Remove the large white Streamlit top bar and reduce visible default Streamlit styling where feasible.
- Keep the existing Streamlit app so the project can be hosted cheaply or free through Streamlit Community Cloud.
- Use the supplied templates as visual inspiration: atmospheric hero surface, glass cards, clean status pills, and strong weather-app hierarchy.
- Keep the app bilingual and preserve the clear prototype/non-official-warning disclaimer.
- Make deployment/share instructions explicit enough that the app can be published once and then shared as a URL.
- Preserve all prediction behavior, model artifacts, and scientific methodology.

**Non-Goals:**
- No React, Tailwind, FastAPI, or separate frontend migration in this change.
- No model retraining, threshold refitting, notebook reruns, or data pipeline changes.
- No real GenCast GPU inference or Azure deployment work.
- No custom authentication or private client portal.
- No expensive hosting path.

## Decisions

1. Keep Streamlit as the delivery framework for this iteration.

   Rationale: The app is already Streamlit-based and can be deployed quickly on Streamlit Community Cloud. A React/Tailwind frontend would give more UI control, but it would also require a backend API, CORS/configuration work, packaging decisions for model artifacts, and more deployment complexity.

   Alternative considered: React + Tailwind + FastAPI. Rejected for this iteration because the immediate need is a reliable public demo link, not a full product rewrite.

2. Implement a template-inspired visual redesign using scoped CSS and custom HTML blocks.

   Rationale: Streamlit widgets can remain for reliability, while key visual areas such as the hero, status strip, risk cards, and forecast timeline can be rendered as custom HTML/CSS. This gives a more modern product feel without changing the runtime architecture.

   Alternative considered: Use only native Streamlit widgets. Rejected because native widgets alone cannot achieve the supplied weather-dashboard look.

3. Treat the first viewport as the client demo surface.

   The first viewport should contain the dashboard identity, location, selected horizon, prototype status, last known ERA5 context, and a clear action to calculate/update risk. Supporting technical details should stay lower on the page or behind an expander.

   Rationale: A non-technical opdrachtgever should understand the app before reading model details.

4. Hide Streamlit chrome and neutralize default layout artifacts.

   The CSS should hide the default top header/toolbar/menu/deploy affordances where possible, reduce top padding, and avoid the white bar seen in the screenshot.

   Rationale: Visible Streamlit chrome makes the demo feel unfinished and distracts from the project.

5. Use a balanced climate-risk palette rather than a one-color dark theme.

   Proposed visual direction:
- Atmospheric base: deep blue-green and charcoal.
- Weather light: misty blue/gray highlights.
- Flood accent: cyan/blue.
- Drought accent: warm amber.
- Risk semantics: green, amber, red, crimson/purple for Low through Extreme.

   Rationale: The app should feel like a weather/risk product, not a generic dark admin dashboard.

6. Add deployment readiness checks and documentation.

   The implementation should ensure relative paths work from the app directory, requirements are complete, and README instructions explain a Streamlit Community Cloud deployment path.

   Rationale: The final result should be shareable through a URL, not just prettier locally.

## Risks / Trade-offs

- Streamlit CSS selectors can change between versions -> Keep critical visual surfaces in custom classes and use Streamlit selector overrides only for chrome/wrapper cleanup.
- Hiding Streamlit chrome may not remove every hosted platform affordance -> Verify in local run and document any expected hosted differences.
- Rich visual styling can reduce readability on smaller screens -> Use responsive grids, stable widths, and viewport checks for desktop and mobile.
- A polished app could imply official warning status -> Keep a visible prototype badge and non-official warning copy in the first viewport.
- Free hosting can sleep, be slow, or have resource limits -> Prefer Streamlit Community Cloud for simplicity and keep model/data loading efficient with caching.

## Migration Plan

1. Update `prototype_app/app.py` presentation only: CSS, hero markup, cards, timeline styling, and first-viewport layout.
2. Keep prediction calls and result keys unchanged.
3. Update `prototype_app/README.md` with deployment steps and a client-demo sharing note.
4. Run a local syntax/import check.
5. Run the Streamlit app locally and verify default Jijiga prediction, English/Dutch mode, and +1/+3/+7 horizons if dependencies and network are available.
6. Deploy via Streamlit Community Cloud after the code is committed and pushed to GitHub.

Rollback: revert the presentation and README changes. Since no model/data behavior changes are planned, rollback does not require regenerating notebooks, parquet files, or model artifacts.

## Open Questions

- Should the app use a generated/local atmospheric background asset, or should the visual effect be CSS-only to keep deployment simpler?
- Should custom coordinates remain visible by default, or be tucked behind an advanced section for the opdrachtgever demo?
- Should the public demo expose Dutch by default because the opdrachtgever is Dutch-speaking, or keep English as the default for portfolio/report consistency?

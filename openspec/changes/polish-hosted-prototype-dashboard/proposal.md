## Why

The prototype needs to be shown to a non-technical Datalab opdrachtgever through a simple online link, so the first impression must feel polished, trustworthy, and self-explanatory. The current Streamlit app has the right model workflow, but the visible Streamlit chrome, dense dashboard blocks, and locally oriented presentation make it feel less like a client-ready webapp.

## What Changes

- Polish the existing Streamlit frontend into a modern weather-dashboard style inspired by the supplied templates: atmospheric hero panel, glass-like surfaces, softer cards, and stronger visual hierarchy.
- Hide or visually neutralize default Streamlit header/toolbar elements that create the large white top bar and make the app look unfinished.
- Rework the first viewport so a opdrachtgever immediately sees the project name, Jijiga location, selected horizon, prototype warning, and main call to calculate or update risk.
- Improve controls so language, horizon, and location settings feel simple and non-technical, while retaining the existing bilingual English/Dutch interface.
- Restyle flood and drought risk cards, the status strip, the 7-day progression, charts, and supporting table so they read as one coherent product UI.
- Add deployment-oriented documentation for sharing the app through Streamlit Community Cloud or an equivalent low-cost/free hosting path.
- Preserve existing prediction behavior, model artifacts, processed data dependencies, and methodology wording around forecast-index versus real GenCast inference.

## Capabilities

### New Capabilities
- `hosted-prototype-dashboard`: Client-facing hosted dashboard presentation, visual polish, Streamlit chrome cleanup, and deployment readiness for a non-technical opdrachtgever.

### Modified Capabilities
None.

## Impact

- Affected code: `prototype_app/app.py` for CSS, layout, copy placement, and Streamlit UI polish.
- Affected documentation: `prototype_app/README.md` for deploy/share instructions and client-demo usage notes.
- Possible supporting files: optional lightweight assets under `prototype_app/` if a local background image or static visual asset is added.
- No intended changes to `HybridPredictor`, XGBoost model files, parquet datasets, notebooks, EM-DAT usage, Azure infrastructure, or training behavior.

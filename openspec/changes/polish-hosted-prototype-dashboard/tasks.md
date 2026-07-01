## 1. Streamlit Presentation Cleanup

- [x] 1.1 Inspect current `prototype_app/app.py` layout and identify the smallest presentation-only edit surface.
- [x] 1.2 Add CSS overrides to hide or neutralize the Streamlit top header, toolbar, menu, deploy affordance, and excess top spacing.
- [x] 1.3 Confirm the CSS cleanup does not hide app content, sidebar controls, error messages, spinners, or expanders.

## 2. Modern First Viewport

- [x] 2.1 Rework the dashboard hero into a weather-app inspired first viewport with atmospheric background treatment, glass-like paneling, project title, location, horizon, and prototype warning.
- [x] 2.2 Move or restyle the status information so last ERA5 date, API, SMI/SPEI, grid point, and horizon are visible but not visually heavy.
- [x] 2.3 Simplify the pre-run state so a non-technical opdrachtgever sees what the app does and which button to press.

## 3. Controls And Result Polish

- [x] 3.1 Restyle language, horizon, location, and model-state sidebar areas so they feel consistent with the modern dashboard theme.
- [x] 3.2 Restyle flood and drought risk cards with stronger hierarchy, softer glass surfaces, semantic risk accents, component outputs, and concise explanations.
- [x] 3.3 Restyle the 7-day progression, Plotly chart, forecast table, and model details section so they remain readable and visually coherent.
- [x] 3.4 Verify English and Dutch text still fit inside controls, cards, and timeline cells on desktop and narrow widths.

## 4. Hosting Readiness

- [x] 4.1 Check app imports, data/model paths, and requirements for Streamlit Community Cloud compatibility.
- [x] 4.2 Update `prototype_app/README.md` with simple local run instructions, hosted deployment steps, and client-demo sharing guidance.
- [x] 4.3 Ensure documentation clearly says the app is a prototype demo, not an official warning system, and does not run live GenCast GPU inference.

## 5. Verification

- [x] 5.1 Run a Python syntax/import check for the prototype app.
- [x] 5.2 Run the Streamlit app locally and verify the page loads without the large white top bar.
- [x] 5.3 Run a default Jijiga prediction and confirm prediction behavior and displayed risk keys are unchanged.
- [x] 5.4 Check +1, +3, and +7 horizons, plus English and Dutch modes.
- [x] 5.5 Confirm no model files, parquet datasets, notebook outputs, threshold logic, or training behavior were changed.

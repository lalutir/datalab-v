## ADDED Requirements

### Requirement: Dashboard-first presentation
The prototype webapp SHALL present the prediction experience as an early-warning dashboard with a clear header, visible Jijiga location context, selected forecast horizon, prototype/demo status, and last-known ERA5 state.

#### Scenario: User opens the app
- **WHEN** the user opens the Streamlit app
- **THEN** the first screen shows the dashboard title, Jijiga grid-point context, forecast controls, and model status information without requiring a prediction run

#### Scenario: User changes language
- **WHEN** the user switches between English and Dutch
- **THEN** the dashboard title, controls, status labels, risk labels, and explanatory copy are shown in the selected language

### Requirement: Prominent hazard risk cards
The prototype webapp SHALL display flood and drought prediction results as prominent hazard cards after a prediction run, including final risk level, selected horizon, component outputs, and a concise interpretation.

#### Scenario: Prediction succeeds
- **WHEN** the user runs a prediction successfully
- **THEN** the app shows separate flood and drought cards with the final risk label, risk color, horizon, and human-readable risk description

#### Scenario: Flood card shows ensemble context
- **WHEN** the flood prediction result is displayed
- **THEN** the card shows the XGBoost output, forecast-index component output, and final ensemble output

#### Scenario: Drought card shows component context
- **WHEN** the drought prediction result is displayed
- **THEN** the card shows the XGBoost output and explains that the final drought risk is based on XGBoost because SPEI-6 cannot be updated from a 7-day forecast

### Requirement: Honest forecast-component terminology
The prototype webapp SHALL clearly distinguish the live demo forecast-index component from real GenCast foundation-model inference.

#### Scenario: User reads the main dashboard
- **WHEN** the app refers to the live forecast-derived component in primary UI text
- **THEN** the wording does not imply that live GenCast GPU inference is being executed

#### Scenario: User opens model details
- **WHEN** the user expands the model details section
- **THEN** the app explains that the demo component uses Open-Meteo precipitation to forward-run API and compute flood risk, while SMI, SPEI-6, and total runoff are not dynamically forecast by the app

### Requirement: Forecast progression timeline
The prototype webapp SHALL provide a scan-friendly 7-day forecast progression view that combines daily flood risk, drought risk, and precipitation.

#### Scenario: Prediction results are available
- **WHEN** forecast results have been computed
- **THEN** the app displays daily flood risk, daily drought risk, and daily precipitation for the forecast period in a compact visual timeline or equivalent scan-first visualization

#### Scenario: User needs exact values
- **WHEN** the user reviews the forecast data table
- **THEN** the app still provides exact daily values for date, precipitation, temperature, evapotranspiration, API, flood score, flood risk, and drought risk

### Requirement: Model input transparency
The prototype webapp SHALL explain which variables feed each prediction component and whether they come from live forecast data, stored ERA5 state, or historical lag features.

#### Scenario: User opens model details
- **WHEN** the model details section is visible
- **THEN** the app lists the XGBoost input variables and lags, the live forecast variable used by the forecast-index component, and the frozen ERA5-derived values used for SMI, SPEI-6, and total runoff assumptions

#### Scenario: User reviews XGBoost metadata
- **WHEN** a prediction result is displayed
- **THEN** the app shows the feature date used for the XGBoost prediction

### Requirement: Responsive climate-risk theme
The prototype webapp SHALL use a richer climate-risk visual theme that remains readable on common desktop and mobile widths.

#### Scenario: User views dashboard on desktop
- **WHEN** the app is displayed on a desktop-width viewport
- **THEN** text, cards, controls, charts, and tables do not overlap and the key risk results remain visually prominent

#### Scenario: User views dashboard on narrow viewport
- **WHEN** the app is displayed on a narrow viewport
- **THEN** major dashboard elements wrap or stack without clipped text, overlapping UI, or unreadable risk labels

### Requirement: Existing prediction behavior preserved
The prototype webapp SHALL preserve the existing model prediction behavior and data dependencies unless a separate model-change proposal is made.

#### Scenario: User runs a prediction
- **WHEN** the user runs a prediction after the dashboard redesign
- **THEN** the app still loads the existing ERA5 labeled data, feature matrix, and XGBoost model artifacts and calls the existing hybrid prediction flow

#### Scenario: User selects default location
- **WHEN** the user runs the default Jijiga prediction
- **THEN** the app still fetches Open-Meteo forecast data for the Jijiga ERA5 grid point and uses forecast precipitation in the forecast-index component

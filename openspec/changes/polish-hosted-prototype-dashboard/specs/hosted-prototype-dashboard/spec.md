## ADDED Requirements

### Requirement: Client-ready first viewport
The webapp SHALL present a polished first viewport suitable for a non-technical opdrachtgever, including the project identity, Jijiga location context, selected forecast horizon, prototype status, and a clear primary action.

#### Scenario: Client opens hosted app
- **WHEN** a user opens the hosted webapp URL
- **THEN** the first viewport shows the dashboard title, Jijiga grid-point context, selected horizon, prototype/non-official-warning status, and calculate/update action without requiring local setup knowledge

#### Scenario: User has not run a prediction
- **WHEN** no prediction result exists in the current session
- **THEN** the first viewport still communicates what the dashboard does and what action the user should take next

### Requirement: Streamlit chrome cleanup
The webapp SHALL hide or visually neutralize default Streamlit header, toolbar, menu, and deploy elements that make the app look unfinished.

#### Scenario: App renders on desktop
- **WHEN** the app is displayed in a desktop browser
- **THEN** no large white top bar or prominent default Streamlit deploy/header area appears above the dashboard content

#### Scenario: App renders on narrow viewport
- **WHEN** the app is displayed on a narrow viewport
- **THEN** hidden Streamlit chrome does not create empty top spacing or overlap the dashboard content

### Requirement: Modern weather-dashboard visual theme
The webapp SHALL use a cohesive modern weather-dashboard theme inspired by the supplied templates, with atmospheric background treatment, glass-like panels, readable typography, and climate-risk accents.

#### Scenario: User views the dashboard
- **WHEN** the dashboard loads
- **THEN** the main page uses a polished weather/risk visual style rather than a default Streamlit or generic admin-dashboard appearance

#### Scenario: User scans key content
- **WHEN** the user scans the page
- **THEN** the visual hierarchy makes the title, forecast horizon, flood risk, drought risk, and warning disclaimer easy to identify

### Requirement: Non-technical controls
The webapp SHALL keep language, horizon, and location controls understandable for non-technical users while preserving existing functionality.

#### Scenario: User changes forecast horizon
- **WHEN** the user changes the horizon control
- **THEN** the selected horizon is reflected clearly in the visible dashboard state and subsequent prediction output

#### Scenario: User changes language
- **WHEN** the user switches between English and Dutch
- **THEN** primary labels, controls, risk text, and explanatory copy are rendered in the selected language

### Requirement: Polished risk result presentation
The webapp SHALL present flood and drought results as polished, readable risk cards with final level, horizon, component context, and concise interpretation.

#### Scenario: Prediction succeeds
- **WHEN** a prediction completes successfully
- **THEN** the app shows separate flood and drought risk cards whose final risk levels are visually prominent and color-coded by risk severity

#### Scenario: User reviews component context
- **WHEN** prediction results are displayed
- **THEN** the result cards distinguish XGBoost output from the forecast-index component without implying live GenCast GPU inference

### Requirement: Hosted deployment readiness
The webapp SHALL be prepared for low-cost or free hosting so a opdrachtgever can access it through a public URL without running local commands.

#### Scenario: Developer prepares deployment
- **WHEN** the developer follows the app documentation
- **THEN** the documentation explains how to deploy the Streamlit prototype to a suitable hosted service such as Streamlit Community Cloud

#### Scenario: App runs in hosted environment
- **WHEN** the app is launched from the repository in a hosted Streamlit environment
- **THEN** it uses repository-relative paths and declared dependencies rather than local absolute paths or manual local setup

### Requirement: Scientific behavior preservation
The webapp SHALL preserve existing model behavior, model artifacts, processed data dependencies, and scientific limitations while changing only presentation and deployment readiness.

#### Scenario: User runs default Jijiga prediction
- **WHEN** the user calculates risk for the default Jijiga grid point
- **THEN** the app still uses the existing `HybridPredictor` flow, stored XGBoost artifacts, processed project data, and Open-Meteo forecast input path

#### Scenario: User reads prototype disclaimer
- **WHEN** the user views the first viewport or model details
- **THEN** the app clearly states that it is a prototype/demo and not an official warning system

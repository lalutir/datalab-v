"""
Jijiga Flood & Drought Risk Prediction - Prototype Dashboard.

Streamlit interface for the demo hybrid pipeline:
XGBoost models + a forecast-index component using Open-Meteo precipitation.

Start with: streamlit run app.py
"""

import html
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from model_utils import (  # noqa: E402
    FEATURE_VARS,
    LAGS,
    HybridPredictor,
    LABEL_MAP,
    LABEL_MAP_NL,
    risk_description_drought,
    risk_description_flood,
)
from weather_api import DEFAULT_LAT, DEFAULT_LON, fetch_forecast_safe  # noqa: E402


st.set_page_config(
    page_title="Jijiga Early Warning Dashboard",
    page_icon=None,
    layout="wide",
)


RISK_COLORS = {
    0: "#2fbf71",
    1: "#f3b23c",
    2: "#e14d3a",
    3: "#9b1c45",
}
RISK_SURFACES = {
    0: "rgba(47, 191, 113, 0.16)",
    1: "rgba(243, 178, 60, 0.18)",
    2: "rgba(225, 77, 58, 0.18)",
    3: "rgba(155, 28, 69, 0.20)",
}


TEXTS = {
    "en": {
        "title": "Jijiga Early Warning Dashboard",
        "eyebrow": "Flood and drought risk prototype",
        "subtitle": "Single ERA5 grid point near Jijiga, Ethiopia | 9.25N, 42.75E",
        "demo_badge": "Prototype demo - not an official warning system",
        "disclaimer": (
            "Predictions are indicative and intended to demonstrate the project "
            "workflow. The live app does not run real GenCast GPU inference."
        ),
        "settings": "Controls",
        "horizon_label": "Forecast horizon",
        "horizon_1": "Next day (+1d)",
        "horizon_3": "Next 3 days (+3d)",
        "horizon_7": "Next 7 days (+7d)",
        "location": "Location",
        "custom_coords": "Custom coordinates",
        "default_loc": "Default Jijiga grid point",
        "lat": "Latitude (N)",
        "lon": "Longitude (E)",
        "model_info": "Model state",
        "last_era5": "Last ERA5 date",
        "last_known": "Last known",
        "run_btn": "Calculate risk",
        "update_btn": "Update forecast",
        "fetching": "Fetching Open-Meteo forecast...",
        "computing": "Computing dashboard prediction...",
        "loading": "Loading model artifacts...",
        "fetch_error": "Error fetching weather data",
        "status_horizon": "Horizon",
        "status_location": "Grid point",
        "status_pipeline": "Pipeline",
        "status_pipeline_value": "XGBoost + forecast-index",
        "pre_run_title": "Ready to show the risk outlook",
        "pre_run_body": (
            "Select a forecast horizon and calculate the risk. The dashboard will show "
            "the flood and drought outlook, daily progression, and model context."
        ),
        "results_header": "Risk assessment",
        "results_kicker": "Forecast target: +{n} day{s}",
        "flood_risk": "Flood risk",
        "drought_risk": "Drought risk",
        "final_risk": "Final risk",
        "xgb": "XGBoost",
        "forecast_component": "Forecast-index",
        "ensemble": "Ensemble max",
        "drought_basis": "Final drought risk uses XGBoost",
        "drought_note": "SPEI-6 cannot be updated from a 7-day forecast.",
        "timeline_header": "7-day risk progression",
        "timeline_subtitle": "Daily forecast-index risk states and precipitation",
        "flood_chart_title": "Flood risk",
        "drought_chart_title": "Drought risk",
        "precip_chart_title": "Forecast precipitation",
        "weather_header": "Forecast data",
        "weather_subtitle": "Open-Meteo inputs plus computed API and risk labels",
        "col_date": "Date",
        "col_precip": "Precipitation (mm)",
        "col_temp": "Temperature (C)",
        "col_et": "Evapotranspiration (mm)",
        "col_api": "API",
        "col_flood_score": "Flood score",
        "col_flood_risk": "Flood risk",
        "col_drought_risk": "Drought risk",
        "tech_header": "Model details",
        "risk_low": "Low",
        "risk_moderate": "Moderate",
        "risk_high": "High",
        "risk_extreme": "Extreme",
        "feature_date": "XGBoost feature date",
        "live_input": "Live forecast input",
        "frozen_state": "Frozen ERA5 state",
        "footer": (
            "Datalab Portfolio Project | ERA5 + Open-Meteo demo forecast | "
            "Single-grid-point proof of concept"
        ),
    },
    "nl": {
        "title": "Jijiga Early Warning Dashboard",
        "eyebrow": "Prototype voor overstromings- en droogterisico",
        "subtitle": "Een ERA5-gridpunt nabij Jijiga, Ethiopie | 9.25N, 42.75E",
        "demo_badge": "Prototype demo - geen officieel waarschuwingssysteem",
        "disclaimer": (
            "De voorspellingen zijn indicatief en tonen de projectworkflow. "
            "De live app voert geen echte GenCast GPU-inference uit."
        ),
        "settings": "Bediening",
        "horizon_label": "Voorspelhorizon",
        "horizon_1": "Komende dag (+1d)",
        "horizon_3": "Komende 3 dagen (+3d)",
        "horizon_7": "Komende 7 dagen (+7d)",
        "location": "Locatie",
        "custom_coords": "Aangepaste coordinaten",
        "default_loc": "Standaard Jijiga-gridpunt",
        "lat": "Breedtegraad (N)",
        "lon": "Lengtegraad (E)",
        "model_info": "Modelstatus",
        "last_era5": "Laatste ERA5 datum",
        "last_known": "Laatst bekend",
        "run_btn": "Bereken risico",
        "update_btn": "Update forecast",
        "fetching": "Weersverwachting ophalen van Open-Meteo...",
        "computing": "Dashboardvoorspelling berekenen...",
        "loading": "Modelbestanden laden...",
        "fetch_error": "Fout bij ophalen weerdata",
        "status_horizon": "Horizon",
        "status_location": "Gridpunt",
        "status_pipeline": "Pipeline",
        "status_pipeline_value": "XGBoost + forecast-index",
        "pre_run_title": "Klaar om het risico-overzicht te tonen",
        "pre_run_body": (
            "Kies een voorspelhorizon en bereken het risico. Het dashboard toont daarna "
            "de overstromings- en droogte-inschatting, dagelijkse ontwikkeling en modelcontext."
        ),
        "results_header": "Risicobeoordeling",
        "results_kicker": "Forecastdoel: +{n} dag{s}",
        "flood_risk": "Overstromingsrisico",
        "drought_risk": "Droogterisico",
        "final_risk": "Eindrisico",
        "xgb": "XGBoost",
        "forecast_component": "Forecast-index",
        "ensemble": "Ensemble max",
        "drought_basis": "Eindrisico droogte gebruikt XGBoost",
        "drought_note": "SPEI-6 kan niet worden bijgewerkt met een 7-daagse forecast.",
        "timeline_header": "7-daags risicoverloop",
        "timeline_subtitle": "Dagelijkse forecast-index risico's en neerslag",
        "flood_chart_title": "Overstromingsrisico",
        "drought_chart_title": "Droogterisico",
        "precip_chart_title": "Forecast neerslag",
        "weather_header": "Forecastdata",
        "weather_subtitle": "Open-Meteo inputs plus berekende API en risicolabels",
        "col_date": "Datum",
        "col_precip": "Neerslag (mm)",
        "col_temp": "Temperatuur (C)",
        "col_et": "Evapotranspiratie (mm)",
        "col_api": "API",
        "col_flood_score": "Flood score",
        "col_flood_risk": "Overstromingsrisico",
        "col_drought_risk": "Droogterisico",
        "tech_header": "Modeldetails",
        "risk_low": "Laag",
        "risk_moderate": "Matig",
        "risk_high": "Hoog",
        "risk_extreme": "Extreem",
        "feature_date": "XGBoost featuredatum",
        "live_input": "Live forecastinput",
        "frozen_state": "Vaste ERA5-status",
        "footer": (
            "Datalab Portfolio Project | ERA5 + Open-Meteo demo forecast | "
            "Single-grid-point proof of concept"
        ),
    },
}


@st.cache_resource
def load_predictor():
    """Load the hybrid predictor once per Streamlit session."""
    return HybridPredictor()


def esc(value) -> str:
    return html.escape(str(value))


def risk_labels(t):
    return [t["risk_low"], t["risk_moderate"], t["risk_high"], t["risk_extreme"]]


def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --bg: #071013;
            --surface: rgba(12, 24, 29, 0.72);
            --surface-strong: rgba(10, 21, 26, 0.88);
            --surface-soft: rgba(255, 255, 255, 0.075);
            --border: rgba(230, 245, 244, 0.22);
            --border-strong: rgba(230, 245, 244, 0.34);
            --text: #f4fbf9;
            --muted: #b7c8c5;
            --subtle: #8fa5a1;
            --cyan: #6ed8e7;
            --blue: #78aef8;
            --amber: #f2bd63;
            --shadow: 0 30px 80px rgba(0, 0, 0, 0.36);
        }
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        .stDeployButton,
        #MainMenu,
        footer {
            visibility: hidden;
            height: 0;
        }
        header[data-testid="stHeader"] {
            display: none;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: var(--bg);
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.08), transparent 16rem),
                linear-gradient(125deg, #071013 0%, #102a32 42%, #293724 72%, #0b1214 100%);
            color: var(--text);
        }
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 2.4rem;
            max-width: 1260px;
        }
        .main .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        p {
            color: var(--muted);
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(13, 29, 34, 0.96), rgba(8, 17, 20, 0.98));
            border-right: 1px solid rgba(230, 245, 244, 0.16);
            box-shadow: 18px 0 60px rgba(0, 0, 0, 0.22);
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.8rem;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--text);
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 1rem;
            margin-top: 0.2rem;
        }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--subtle);
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(230, 245, 244, 0.10);
            margin: 1.25rem 0;
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {
            background: rgba(255, 255, 255, 0.92);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.22);
        }
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label {
            padding-top: 0.1rem;
        }
        .stButton > button {
            border: 1px solid rgba(255, 255, 255, 0.28);
            background:
                linear-gradient(135deg, rgba(110, 216, 231, 0.96), rgba(68, 152, 178, 0.96) 52%, rgba(242, 189, 99, 0.92));
            color: #061216;
            font-weight: 850;
            border-radius: 14px;
            min-height: 3rem;
            padding: 0 1.2rem;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
            transition: transform 150ms ease, box-shadow 150ms ease, filter 150ms ease;
        }
        .stButton > button:hover {
            border-color: rgba(255, 255, 255, 0.46);
            filter: brightness(1.05);
            transform: translateY(-1px);
            box-shadow: 0 20px 42px rgba(0, 0, 0, 0.34);
        }
        .dashboard-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border-strong);
            border-radius: 18px;
            min-height: 430px;
            padding: clamp(1.25rem, 3vw, 2.4rem);
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.13), rgba(255, 255, 255, 0.03)),
                linear-gradient(120deg, rgba(7, 16, 19, 0.98) 0%, rgba(14, 41, 49, 0.82) 38%, rgba(85, 78, 55, 0.72) 68%, rgba(12, 18, 21, 0.98) 100%);
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }
        .dashboard-hero:before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(160deg, rgba(129, 194, 214, 0.34), transparent 31%),
                linear-gradient(15deg, transparent 52%, rgba(239, 165, 78, 0.32) 75%, rgba(250, 218, 162, 0.24) 100%),
                linear-gradient(90deg, rgba(255, 255, 255, 0.08), transparent 24%, rgba(255, 255, 255, 0.06) 67%, transparent);
            opacity: 0.9;
            pointer-events: none;
        }
        .hero-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
            gap: clamp(1rem, 3vw, 2.2rem);
            align-items: stretch;
            min-height: 360px;
        }
        .eyebrow {
            color: #b9f5f0;
            font-size: 0.82rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            margin-bottom: 0.7rem;
        }
        .dashboard-hero h1 {
            margin: 0;
            color: var(--text);
            font-size: clamp(2.7rem, 5.4vw, 5.35rem);
            line-height: 0.98;
            max-width: 820px;
            text-wrap: balance;
        }
        .dashboard-hero p {
            color: #d9ebe8;
            margin: 0.9rem 0 0;
            font-size: 1.02rem;
            max-width: 720px;
            text-shadow: 0 1px 18px rgba(0, 0, 0, 0.28);
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1.25rem;
        }
        .badge {
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 999px;
            padding: 0.48rem 0.78rem;
            background: rgba(255, 255, 255, 0.10);
            backdrop-filter: blur(18px);
            color: var(--text);
            font-size: 0.86rem;
            font-weight: 780;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10);
        }
        .badge.warn {
            border-color: rgba(242, 189, 99, 0.54);
            color: #fff0c6;
            background: rgba(242, 189, 99, 0.15);
        }
        .hero-panel {
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 18px;
            background: rgba(8, 19, 23, 0.48);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10);
            backdrop-filter: blur(20px);
            padding: 1.1rem;
            align-self: center;
        }
        .hero-panel-title {
            color: var(--muted);
            text-transform: uppercase;
            font-weight: 820;
            letter-spacing: 0.08em;
            font-size: 0.75rem;
            margin-bottom: 0.75rem;
        }
        .hero-metric {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.7rem 0;
            border-top: 1px solid rgba(255, 255, 255, 0.12);
        }
        .hero-metric:first-of-type {
            border-top: 0;
        }
        .hero-metric span {
            color: var(--subtle);
            font-size: 0.85rem;
        }
        .hero-metric strong {
            color: var(--text);
            font-size: 0.94rem;
            text-align: right;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(120px, 1fr));
            gap: 0.7rem;
            margin: 1rem 0 1.05rem;
        }
        .status-card, .info-card, .hazard-card, .timeline-wrap {
            border: 1px solid var(--border);
            border-radius: 16px;
            background: var(--surface);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.18);
            backdrop-filter: blur(18px);
        }
        .status-card {
            padding: 0.75rem 0.85rem;
            min-height: 74px;
        }
        .status-label {
            color: var(--muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            font-weight: 800;
            letter-spacing: 0.06em;
        }
        .status-value {
            color: var(--text);
            font-size: 1rem;
            font-weight: 850;
            margin-top: 0.2rem;
        }
        .info-card {
            padding: 1.2rem;
            margin: 0.85rem 0 1rem;
            background: rgba(9, 21, 26, 0.68);
        }
        .info-card h3, .timeline-wrap h3 {
            color: var(--text);
            margin: 0;
            font-size: clamp(1.25rem, 2vw, 1.85rem);
        }
        .info-card p, .timeline-wrap p {
            color: var(--muted);
            margin: 0.4rem 0 0;
        }
        .section-kicker {
            color: #b9f5f0;
            font-weight: 850;
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.10em;
            margin-bottom: 0.35rem;
        }
        .hazard-card {
            padding: 1.25rem;
            min-height: 284px;
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.085), rgba(255, 255, 255, 0.035)),
                rgba(9, 21, 26, 0.78);
        }
        .hazard-card:before {
            content: "";
            position: absolute;
            inset: 0;
            border-top: 4px solid var(--risk-color);
            background: linear-gradient(145deg, var(--risk-bg), transparent 42%);
            opacity: 0.92;
            pointer-events: none;
        }
        .hazard-card > * {
            position: relative;
            z-index: 1;
        }
        .hazard-top {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: flex-start;
        }
        .hazard-name {
            color: var(--muted);
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.82rem;
        }
        .risk-pill {
            border-radius: 999px;
            border: 1px solid var(--risk-color);
            color: var(--risk-color);
            background: var(--risk-bg);
            padding: 0.35rem 0.65rem;
            font-weight: 850;
            white-space: nowrap;
        }
        .risk-level {
            color: var(--text);
            font-size: clamp(2.25rem, 5vw, 3.6rem);
            line-height: 1;
            font-weight: 900;
            margin-top: 0.95rem;
        }
        .risk-description {
            color: #e4f0ed;
            margin: 0.75rem 0 1rem;
            min-height: 3.4rem;
        }
        .component-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
        }
        .component-grid.two {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .component-box {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.075);
            padding: 0.65rem;
        }
        .component-box span {
            color: var(--muted);
            display: block;
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .component-box strong {
            color: var(--text);
            display: block;
            margin-top: 0.2rem;
        }
        .timeline-wrap {
            padding: 1.15rem;
            margin-top: 0.9rem;
            background: rgba(9, 21, 26, 0.70);
        }
        .timeline-grid {
            display: grid;
            grid-template-columns: repeat(7, minmax(88px, 1fr));
            gap: 0.55rem;
            margin-top: 1rem;
        }
        .day-cell {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.065);
            padding: 0.7rem;
            min-height: 134px;
        }
        .day-date {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
        }
        .risk-row {
            display: flex;
            justify-content: space-between;
            gap: 0.4rem;
            align-items: center;
            margin-top: 0.45rem;
            color: var(--text);
            font-size: 0.80rem;
        }
        .risk-row > span:last-child {
            white-space: nowrap;
        }
        .dot {
            display: inline-block;
            width: 0.72rem;
            height: 0.72rem;
            border-radius: 999px;
            background: var(--dot);
            flex: 0 0 auto;
        }
        .rain-bar {
            height: 0.5rem;
            border-radius: 999px;
            background: rgba(65, 199, 215, 0.16);
            overflow: hidden;
            margin-top: 0.55rem;
        }
        .rain-fill {
            height: 100%;
            width: var(--rain-width);
            background: linear-gradient(90deg, #6ed8e7, #78aef8);
        }
        .rain-label {
            color: #bfeef3;
            font-size: 0.78rem;
            margin-top: 0.35rem;
            font-weight: 750;
        }
        .model-panel {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(16, 28, 32, 0.70);
            padding: 0.85rem;
            margin-bottom: 0.7rem;
        }
        .model-panel h4 {
            color: var(--text);
            margin: 0 0 0.45rem;
        }
        .model-panel p, .model-panel li {
            color: #d5e2df;
        }
        .model-panel ul {
            margin-bottom: 0;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(9, 21, 26, 0.58);
        }
        .element-container:has(.dashboard-hero) {
            margin-top: 0;
        }
        @media (max-width: 900px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .hero-grid {
                grid-template-columns: 1fr;
                min-height: auto;
            }
            .status-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .timeline-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .component-grid, .component-grid.two {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 560px) {
            .status-grid, .timeline-grid {
                grid-template-columns: 1fr;
            }
            .dashboard-hero {
                padding: 1rem;
                min-height: auto;
            }
            .dashboard-hero h1 {
                font-size: clamp(2.25rem, 12vw, 3.2rem);
            }
            .hero-panel {
                padding: 0.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_card(label, value):
    return (
        '<div class="status-card">'
        f'<div class="status-label">{esc(label)}</div>'
        f'<div class="status-value">{esc(value)}</div>'
        "</div>"
    )


def render_header(t, state, horizon, lat, lon):
    st.markdown(
        f"""
        <div class="dashboard-hero">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">{esc(t["eyebrow"])}</div>
                    <h1>{esc(t["title"])}</h1>
                    <p>{esc(t["subtitle"])}</p>
                    <div class="badge-row">
                        <div class="badge warn">{esc(t["demo_badge"])}</div>
                        <div class="badge">ERA5 2000-2025</div>
                        <div class="badge">Open-Meteo 7-day forecast</div>
                    </div>
                    <p>{esc(t["disclaimer"])}</p>
                </div>
                <div class="hero-panel">
                    <div class="hero-panel-title">{esc(t["status_pipeline"])}</div>
                    <div class="hero-metric">
                        <span>{esc(t["status_horizon"])}</span>
                        <strong>+{horizon}d</strong>
                    </div>
                    <div class="hero-metric">
                        <span>{esc(t["status_location"])}</span>
                        <strong>{lat:.2f}N, {lon:.2f}E</strong>
                    </div>
                    <div class="hero-metric">
                        <span>{esc(t["last_era5"])}</span>
                        <strong>{state["date"].strftime("%Y-%m-%d")}</strong>
                    </div>
                    <div class="hero-metric">
                        <span>API / SMI</span>
                        <strong>{state["api_92"]:.3f} / {state["smi_fc"]:.3f}</strong>
                    </div>
                    <div class="hero-metric">
                        <span>SPEI-6</span>
                        <strong>{state["spei_6"]:.3f}</strong>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_strip(t, state, horizon, lat, lon):
    cards = [
        status_card(t["status_horizon"], f"+{horizon}d"),
        status_card(t["status_location"], f"{lat:.2f}N, {lon:.2f}E"),
        status_card(t["last_era5"], state["date"].strftime("%Y-%m-%d")),
        status_card("API", f"{state['api_92']:.3f}"),
        status_card("SMI", f"{state['smi_fc']:.3f}"),
        status_card("SPEI-6", f"{state['spei_6']:.3f}"),
    ]
    st.markdown(f'<div class="status-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_pre_run_card(t):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="section-kicker">{esc(t["status_pipeline"])}</div>
            <h3>{esc(t["pre_run_title"])}</h3>
            <p>{esc(t["pre_run_body"])}</p>
            <div class="badge-row">
                <div class="badge">{esc(t["run_btn"])}</div>
                <div class="badge">+1d / +3d / +7d</div>
                <div class="badge">{esc(t["status_pipeline_value"])}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def component_box(label, value):
    return (
        '<div class="component-box">'
        f"<span>{esc(label)}</span>"
        f"<strong>{esc(value)}</strong>"
        "</div>"
    )


def render_hazard_card(kind, level, label_map, description, horizon, components):
    risk_color = RISK_COLORS[level]
    risk_bg = RISK_SURFACES[level]
    boxes = "".join(component_box(label, value) for label, value in components)
    grid_class = "component-grid two" if len(components) == 2 else "component-grid"
    st.markdown(
        f"""
        <div class="hazard-card" style="--risk-color: {risk_color}; --risk-bg: {risk_bg};">
            <div class="hazard-top">
                <div class="hazard-name">{esc(kind)}</div>
                <div class="risk-pill">+{horizon}d</div>
            </div>
            <div class="risk-level">{esc(label_map[level])}</div>
            <div class="risk-description">{esc(description)}</div>
            <div class="{grid_class}">
                {boxes}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_timeline(t, forecast_df, result, label_map):
    precip_mm = forecast_df["tp"].values * 1000
    max_precip = max(float(precip_mm.max()), 1.0)
    cells = []
    for idx, row in forecast_df.reset_index(drop=True).iterrows():
        flood = result["flood_risks_all_days"][idx]
        drought = result["drought_risks_all_days"][idx]
        rain = float(precip_mm[idx])
        rain_width = min(100, (rain / max_precip) * 100)
        date_label = pd.to_datetime(row["date"]).strftime("%d %b")
        cells.append(
            '<div class="day-cell">'
            f'<div class="day-date">{esc(date_label)}</div>'
            '<div class="risk-row">'
            f'<span>{esc(t["flood_risk"])}</span>'
            f'<span><span class="dot" style="--dot: {RISK_COLORS[flood]};"></span> {esc(label_map[flood])}</span>'
            "</div>"
            '<div class="risk-row">'
            f'<span>{esc(t["drought_risk"])}</span>'
            f'<span><span class="dot" style="--dot: {RISK_COLORS[drought]};"></span> {esc(label_map[drought])}</span>'
            "</div>"
            f'<div class="rain-bar"><div class="rain-fill" style="--rain-width: {rain_width:.0f}%;"></div></div>'
            f'<div class="rain-label">{rain:.1f} mm</div>'
            "</div>"
        )

    st.markdown(
        '<div class="timeline-wrap">'
        f'<div class="section-kicker">{esc(t["timeline_header"])}</div>'
        f'<h3>{esc(t["timeline_subtitle"])}</h3>'
        f'<div class="timeline-grid">{"".join(cells)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def build_chart(t, forecast_df, result, labels):
    dates = forecast_df["date"].values
    precip_mm = forecast_df["tp"].values * 1000
    flood_risks = result["flood_risks_all_days"]
    drought_risks = result["drought_risks_all_days"]

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(t["flood_chart_title"], t["drought_chart_title"], t["precip_chart_title"]),
        vertical_spacing=0.13,
    )

    fig.add_trace(
        go.Bar(
            x=dates[: len(flood_risks)],
            y=flood_risks,
            marker_color=[RISK_COLORS[r] for r in flood_risks],
            hovertemplate="%{x|%d %b}: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=dates[: len(drought_risks)],
            y=drought_risks,
            marker_color=[RISK_COLORS[r] for r in drought_risks],
            hovertemplate="%{x|%d %b}: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=dates,
            y=precip_mm,
            marker_color="#41c7d7",
            hovertemplate="%{x|%d %b}: %{y:.1f} mm<extra></extra>",
        ),
        row=3,
        col=1,
    )
    for row in [1, 2]:
        fig.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=labels, range=[-0.3, 3.5], row=row, col=1)
    fig.update_yaxes(title_text="mm/day", row=3, col=1)
    fig.update_layout(
        height=600,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.045)",
        font=dict(color="#e4f0ed", family="Arial, sans-serif"),
        margin=dict(l=48, r=18, t=58, b=22),
        bargap=0.24,
    )
    fig.update_annotations(font=dict(color="#f4fbf9", size=14))
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.10)",
        tickfont=dict(color="#b7c8c5"),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.10)",
        tickfont=dict(color="#b7c8c5"),
    )
    return fig


def prepare_display_df(forecast_df, result, label_map):
    display_df = forecast_df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df["tp_mm"] = display_df["tp"] * 1000
    display_df["t2m_C"] = display_df["t2m"] - 273.15
    display_df["e_mm"] = display_df["e"] * -1000
    display_df["api"] = result["api_forecast"][: len(display_df)]
    display_df["flood_score"] = result["flood_scores_all_days"][: len(display_df)]
    display_df["flood_risk_label"] = [
        label_map[r] for r in result["flood_risks_all_days"][: len(display_df)]
    ]
    display_df["drought_risk_label"] = [
        label_map[r] for r in result["drought_risks_all_days"][: len(display_df)]
    ]
    return display_df


def render_model_details(t, predictor, result, horizon):
    last = result["last_known_state"]
    feature_vars = ", ".join(FEATURE_VARS)
    lags = ", ".join(f"{lag}d" for lag in LAGS)
    st.markdown(
        f"""
        <div class="model-panel">
            <h4>XGBoost</h4>
            <ul>
                <li>Uses 40 historical lag features: {esc(feature_vars)}.</li>
                <li>Lags: {esc(lags)}.</li>
                <li>{esc(t["feature_date"])}: {result["feature_date"].strftime("%Y-%m-%d")}.</li>
                <li>Separate flood and drought classifiers are used for +{horizon}d.</li>
            </ul>
        </div>
        <div class="model-panel">
            <h4>{esc(t["forecast_component"])} / GenCast-style demo</h4>
            <ul>
                <li>{esc(t["live_input"])}: Open-Meteo daily precipitation.</li>
                <li>API is forward-run with k=0.92 from {last["api_92"]:.4f}.</li>
                <li>{esc(t["frozen_state"])}: SMI={last["smi_fc"]:.4f}, SPEI-6={last["spei_6"]:.4f}.</li>
                <li>total_ro is set to 0.0 for the live demo component.</li>
                <li>Flood score = 0.40 x norm_API + 0.35 x norm_SMI + 0.25 x norm_total_ro.</li>
            </ul>
        </div>
        <div class="model-panel">
            <h4>Thresholds and limitations</h4>
            <ul>
                <li>Flood p65={predictor.flood_p65:.4f}, p80={predictor.flood_p80:.4f}, p90={predictor.flood_p90:.4f}.</li>
                <li>The app does not run live GenCast GPU inference; it demonstrates the forecast-index step.</li>
                <li>SPEI-6 cannot be updated from a 7-day forecast, so final drought risk remains XGBoost-based.</li>
                <li>Single grid point near Jijiga; upstream Wabi Shabelle river floods are outside this local signal.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plural_suffix(lang, horizon):
    if horizon <= 1:
        return ""
    return "s" if lang == "en" else "en"


def main():
    inject_css()

    with st.sidebar:
        lang = st.radio(
            "Language / Taal",
            options=["en", "nl"],
            format_func=lambda x: "English" if x == "en" else "Nederlands",
            horizontal=True,
        )
        t = TEXTS[lang]
        label_map = LABEL_MAP if lang == "en" else LABEL_MAP_NL

        st.divider()
        st.header(t["settings"])
        horizon_options = {
            t["horizon_1"]: 1,
            t["horizon_3"]: 3,
            t["horizon_7"]: 7,
        }
        horizon_label = st.selectbox(t["horizon_label"], list(horizon_options.keys()), index=2)
        horizon = horizon_options[horizon_label]

        st.divider()
        st.subheader(t["location"])
        use_custom = st.checkbox(t["custom_coords"], value=False)
        if use_custom:
            lat = st.number_input(t["lat"], value=DEFAULT_LAT, format="%.4f")
            lon = st.number_input(t["lon"], value=DEFAULT_LON, format="%.4f")
        else:
            lat = DEFAULT_LAT
            lon = DEFAULT_LON
            st.caption(f"{t['default_loc']}: {lat}N, {lon}E")

        st.divider()
        st.subheader(t["model_info"])

    with st.spinner(t["loading"]):
        predictor = load_predictor()
    state = predictor.get_last_known_state()

    with st.sidebar:
        st.caption(f"{t['last_era5']}: {state['date'].strftime('%Y-%m-%d')}")
        st.caption(f"API ({t['last_known']}): {state['api_92']:.3f}")
        st.caption(f"SMI ({t['last_known']}): {state['smi_fc']:.3f}")
        st.caption(f"SPEI-6 ({t['last_known']}): {state['spei_6']:.3f}")

    render_header(t, state, horizon, lat, lon)
    render_status_strip(t, state, horizon, lat, lon)

    button_label = t["update_btn"] if "result" in st.session_state else t["run_btn"]
    run_prediction = st.button(button_label, type="primary")

    if run_prediction:
        with st.spinner(t["fetching"]):
            forecast_df, error = fetch_forecast_safe(lat, lon, forecast_days=7)
        if error:
            st.error(f"{t['fetch_error']}: {error}")
            return

        with st.spinner(t["computing"]):
            result = predictor.predict_hybrid(forecast_df["tp"].values, horizon)

        st.session_state["result"] = result
        st.session_state["forecast_df"] = forecast_df
        st.session_state["horizon"] = horizon
        st.session_state["lat"] = lat
        st.session_state["lon"] = lon

    if "result" not in st.session_state:
        render_pre_run_card(t)
        st.caption(t["footer"])
        return

    result = st.session_state["result"]
    forecast_df = st.session_state["forecast_df"]
    horizon = st.session_state["horizon"]
    labels = risk_labels(t)

    plural = plural_suffix(lang, horizon)
    st.markdown(
        f"""
        <div class="info-card">
            <div class="section-kicker">{esc(t["results_header"])}</div>
            <h3>{esc(t["results_kicker"].format(n=horizon, s=plural))}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_flood, col_drought = st.columns(2)
    with col_flood:
        flood_level = result["ensemble_flood_risk"]
        render_hazard_card(
            t["flood_risk"],
            flood_level,
            label_map,
            risk_description_flood(flood_level, lang),
            horizon,
            [
                (t["xgb"], label_map[result["xgb_flood_risk"]]),
                (t["forecast_component"], label_map[result["gencast_flood_risk"]]),
                (t["ensemble"], label_map[flood_level]),
            ],
        )

    with col_drought:
        drought_level = result["ensemble_drought_risk"]
        render_hazard_card(
            t["drought_risk"],
            drought_level,
            label_map,
            f"{risk_description_drought(drought_level, lang)} {t['drought_note']}",
            horizon,
            [
                (t["xgb"], label_map[result["xgb_drought_risk"]]),
                (t["drought_basis"], label_map[drought_level]),
            ],
        )

    render_timeline(t, forecast_df, result, label_map)
    st.plotly_chart(build_chart(t, forecast_df, result, labels), use_container_width=True)

    st.markdown(
        f"""
        <div class="info-card">
            <div class="section-kicker">{esc(t["weather_header"])}</div>
            <h3>{esc(t["weather_subtitle"])}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    display_df = prepare_display_df(forecast_df, result, label_map)
    show_cols = {
        "date": t["col_date"],
        "tp_mm": t["col_precip"],
        "t2m_C": t["col_temp"],
        "e_mm": t["col_et"],
        "api": t["col_api"],
        "flood_score": t["col_flood_score"],
        "flood_risk_label": t["col_flood_risk"],
        "drought_risk_label": t["col_drought_risk"],
    }
    st.dataframe(
        display_df[list(show_cols.keys())].rename(columns=show_cols),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(t["tech_header"], expanded=False):
        render_model_details(t, predictor, result, horizon)

    st.caption(t["footer"])


if __name__ == "__main__":
    main()

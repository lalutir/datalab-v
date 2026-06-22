"""
Jijiga Flood & Drought Risk Prediction — Prototype Demo

Streamlit interface for the hybrid ensemble model (XGBoost + GenCast-style).
Uses the pre-trained models from phase5/phase6 notebooks with real-time
weather forecast data from Open-Meteo API.

Start with: streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from model_utils import (
    HybridPredictor,
    LABEL_MAP,
    LABEL_MAP_NL,
    HORIZONS,
    risk_description_flood,
    risk_description_drought,
)
from weather_api import fetch_forecast_safe, DEFAULT_LAT, DEFAULT_LON


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jijiga Flood & Drought Prediction",
    page_icon=None,
    layout="wide",
)

# ── Risk colors ───────────────────────────────────────────────────────────────
RISK_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c", 3: "#8e44ad"}
RISK_COLORS_LIGHT = {0: "#d5f5e3", 1: "#fdebd0", 2: "#fadbd8", 3: "#e8daef"}

# ── Translations ──────────────────────────────────────────────────────────────
TEXTS = {
    "en": {
        "title": "Jijiga Flood & Drought Risk Prediction",
        "subtitle": (
            "**Prototype / Demo** — Hybrid ensemble model for drought and flood prediction  \n"
            "*Location: Jijiga, Ethiopia (Somali region) | ERA5 grid point 9.25N, 42.75E*"
        ),
        "disclaimer": (
            "**Disclaimer:** This is a prototype/demo system developed as part of a "
            "Datalab portfolio project. It must **not** be used as an official warning "
            "system. Predictions are indicative and intended to demonstrate model functionality."
        ),
        "settings": "Settings",
        "horizon_label": "Forecast horizon",
        "horizon_1": "Next day (+1d)",
        "horizon_3": "Next 3 days (+3d)",
        "horizon_7": "Next 7 days (+7d)",
        "location": "Location",
        "custom_coords": "Custom coordinates",
        "default_loc": "Default: Jijiga",
        "lat": "Latitude (N)",
        "lon": "Longitude (E)",
        "model_info": "Model information",
        "last_era5": "Last ERA5 date",
        "last_known": "Last known",
        "run_btn": "Run prediction",
        "fetching": "Fetching weather forecast from Open-Meteo...",
        "computing": "Computing hybrid model prediction...",
        "loading": "Loading model...",
        "fetch_error": "Error fetching weather data",
        "result_header": "Prediction — next {n} day{s}",
        "flood_risk": "Flood risk",
        "drought_risk": "Drought risk",
        "xgb": "XGBoost",
        "gencast": "GenCast-style",
        "ensemble": "Ensemble (max)",
        "drought_note": "Drought is primarily determined by SPEI-6 (historical)",
        "timeline_header": "Risk progression over forecast period",
        "flood_chart_title": "Flood risk per day",
        "drought_chart_title": "Drought risk per day",
        "precip_chart_title": "Forecast precipitation (mm/day)",
        "weather_header": "Weather forecast (Open-Meteo)",
        "col_date": "Date",
        "col_precip": "Precipitation (mm)",
        "col_temp": "Temperature (C)",
        "col_et": "Evapotranspiration (mm)",
        "col_api": "API (computed)",
        "col_flood_score": "Flood score",
        "col_flood_risk": "Flood risk",
        "col_drought_risk": "Drought risk",
        "tech_header": "Technical details",
        "risk_low": "Low",
        "risk_moderate": "Moderate",
        "risk_high": "High",
        "risk_extreme": "Extreme",
        "footer": (
            "Datalab Portfolio Project — Jijiga Flood & Drought Risk Prediction | "
            "Model: Hybrid XGBoost + GenCast-style ensemble | "
            "Data: ERA5 (Copernicus) + Open-Meteo forecast"
        ),
    },
    "nl": {
        "title": "Jijiga Overstromings- & Droogterisico Voorspelling",
        "subtitle": (
            "**Prototype / Demo** — Hybride ensemble model voor droogte- en overstromingsvoorspelling  \n"
            "*Locatie: Jijiga, Ethiopie (Somali regio) | ERA5 gridpunt 9.25N, 42.75E*"
        ),
        "disclaimer": (
            "**Disclaimer:** Dit is een prototype/demo-systeem ontwikkeld als onderdeel van "
            "een Datalab portfolio-project. Het mag **niet** worden gebruikt als officieel "
            "waarschuwingssysteem. De voorspellingen zijn indicatief en bedoeld om de werking "
            "van het model te demonstreren."
        ),
        "settings": "Instellingen",
        "horizon_label": "Voorspelhorizon",
        "horizon_1": "Komende dag (+1d)",
        "horizon_3": "Komende 3 dagen (+3d)",
        "horizon_7": "Komende 7 dagen (+7d)",
        "location": "Locatie",
        "custom_coords": "Aangepaste coordinaten",
        "default_loc": "Standaard: Jijiga",
        "lat": "Breedtegraad (N)",
        "lon": "Lengtegraad (E)",
        "model_info": "Model informatie",
        "last_era5": "Laatste ERA5 datum",
        "last_known": "Laatst bekend",
        "run_btn": "Voorspelling uitvoeren",
        "fetching": "Weersverwachting ophalen van Open-Meteo...",
        "computing": "Hybride model voorspelling berekenen...",
        "loading": "Model laden...",
        "fetch_error": "Fout bij ophalen weerdata",
        "result_header": "Voorspelling — komende {n} dag{s}",
        "flood_risk": "Overstromingsrisico",
        "drought_risk": "Droogterisico",
        "xgb": "XGBoost",
        "gencast": "GenCast-stijl",
        "ensemble": "Ensemble (max)",
        "drought_note": "Droogte wordt primair bepaald door SPEI-6 (historisch)",
        "timeline_header": "Risicoverloop over de forecast-periode",
        "flood_chart_title": "Overstromingsrisico per dag",
        "drought_chart_title": "Droogterisico per dag",
        "precip_chart_title": "Forecast neerslag (mm/dag)",
        "weather_header": "Weersverwachting (Open-Meteo)",
        "col_date": "Datum",
        "col_precip": "Neerslag (mm)",
        "col_temp": "Temperatuur (C)",
        "col_et": "Evapotranspiratie (mm)",
        "col_api": "API (berekend)",
        "col_flood_score": "Flood score",
        "col_flood_risk": "Overstromingsrisico",
        "col_drought_risk": "Droogterisico",
        "tech_header": "Technische details",
        "risk_low": "Laag",
        "risk_moderate": "Matig",
        "risk_high": "Hoog",
        "risk_extreme": "Extreem",
        "footer": (
            "Datalab Portfolio Project — Jijiga Flood & Drought Risk Prediction | "
            "Model: Hybride XGBoost + GenCast-stijl ensemble | "
            "Data: ERA5 (Copernicus) + Open-Meteo forecast"
        ),
    },
}


# ── Cache the model (loads once, stays in memory) ─────────────────────────────
@st.cache_resource
def load_predictor():
    """Load the hybrid predictor (cached across reruns)."""
    return HybridPredictor()


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar: settings ─────────────────────────────────────────────────────
    with st.sidebar:
        # Language selector at the top
        lang = st.radio(
            "Language / Taal",
            options=["en", "nl"],
            format_func=lambda x: "English" if x == "en" else "Nederlands",
            horizontal=True,
        )

        t = TEXTS[lang]
        label_map = LABEL_MAP if lang == "en" else LABEL_MAP_NL
        risk_labels = [t["risk_low"], t["risk_moderate"], t["risk_high"], t["risk_extreme"]]

        st.divider()
        st.header(t["settings"])

        # Horizon selection
        horizon_options = {
            t["horizon_1"]: 1,
            t["horizon_3"]: 3,
            t["horizon_7"]: 7,
        }
        horizon_label = st.selectbox(
            t["horizon_label"],
            options=list(horizon_options.keys()),
            index=2,
        )
        horizon = horizon_options[horizon_label]

        st.divider()

        # Location
        st.subheader(t["location"])
        use_custom = st.checkbox(t["custom_coords"], value=False)
        if use_custom:
            lat = st.number_input(t["lat"], value=DEFAULT_LAT, format="%.4f")
            lon = st.number_input(t["lon"], value=DEFAULT_LON, format="%.4f")
        else:
            lat = DEFAULT_LAT
            lon = DEFAULT_LON
            st.caption(f"{t['default_loc']} ({lat}N, {lon}E)")

        st.divider()

        # Model info
        st.subheader(t["model_info"])

    # Load model
    with st.spinner(TEXTS[lang]["loading"] if "lang" in dir() else "Loading..."):
        predictor = load_predictor()

    # Update sidebar model info
    with st.sidebar:
        state = predictor.get_last_known_state()
        st.caption(f"{t['last_era5']}: {state['date'].strftime('%Y-%m-%d')}")
        st.caption(f"API ({t['last_known']}): {state['api_92']:.3f}")
        st.caption(f"SMI ({t['last_known']}): {state['smi_fc']:.3f}")
        st.caption(f"SPEI-6 ({t['last_known']}): {state['spei_6']:.3f}")

    # Re-read t in case lang changed
    t = TEXTS[lang]
    label_map = LABEL_MAP if lang == "en" else LABEL_MAP_NL
    risk_labels = [t["risk_low"], t["risk_moderate"], t["risk_high"], t["risk_extreme"]]

    # Header
    st.title(t["title"])
    st.markdown(t["subtitle"])
    st.warning(t["disclaimer"])

    # ── Main content ──────────────────────────────────────────────────────────
    run_prediction = st.button(
        t["run_btn"],
        type="primary",
        use_container_width=False,
    )

    if run_prediction:
        # Fetch weather forecast
        with st.spinner(t["fetching"]):
            forecast_df, error = fetch_forecast_safe(lat, lon, forecast_days=7)

        if error:
            st.error(f"{t['fetch_error']}: {error}")
            return

        # Run hybrid prediction
        with st.spinner(t["computing"]):
            forecast_precip = forecast_df["tp"].values
            result = predictor.predict_hybrid(forecast_precip, horizon)

        # Store results in session state
        st.session_state["result"] = result
        st.session_state["forecast_df"] = forecast_df
        st.session_state["horizon"] = horizon
        st.session_state["lang"] = lang

    # ── Display results (from session state) ──────────────────────────────────
    if "result" in st.session_state:
        result = st.session_state["result"]
        forecast_df = st.session_state["forecast_df"]
        horizon = st.session_state["horizon"]

        st.divider()
        plural = "s" if lang == "en" and horizon > 1 else ("en" if lang == "nl" and horizon > 1 else "")
        st.header(t["result_header"].format(n=horizon, s=plural))

        # Risk indicators
        col_flood, col_drought = st.columns(2)

        with col_flood:
            flood_level = result["ensemble_flood_risk"]
            flood_color = RISK_COLORS[flood_level]
            flood_bg = RISK_COLORS_LIGHT[flood_level]

            st.markdown(
                f"""
                <div style="background-color: {flood_bg}; padding: 20px; 
                            border-radius: 10px; border-left: 5px solid {flood_color};">
                    <h3 style="color: {flood_color}; margin: 0;">
                        {t["flood_risk"]}: {label_map[flood_level]}
                    </h3>
                    <p style="margin-top: 10px; color: #333;">
                        {risk_description_flood(flood_level, lang)}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                f"{t['xgb']}: {label_map[result['xgb_flood_risk']]} | "
                f"{t['gencast']}: {label_map[result['gencast_flood_risk']]} | "
                f"{t['ensemble']}: **{label_map[flood_level]}**"
            )

        with col_drought:
            drought_level = result["ensemble_drought_risk"]
            drought_color = RISK_COLORS[drought_level]
            drought_bg = RISK_COLORS_LIGHT[drought_level]

            st.markdown(
                f"""
                <div style="background-color: {drought_bg}; padding: 20px; 
                            border-radius: 10px; border-left: 5px solid {drought_color};">
                    <h3 style="color: {drought_color}; margin: 0;">
                        {t["drought_risk"]}: {label_map[drought_level]}
                    </h3>
                    <p style="margin-top: 10px; color: #333;">
                        {risk_description_drought(drought_level, lang)}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                f"{t['xgb']}: {label_map[result['xgb_drought_risk']]} | "
                f"{t['drought_note']}"
            )

        # ── Forecast timeline charts ──────────────────────────────────────────
        st.divider()
        st.subheader(t["timeline_header"])

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                t["flood_chart_title"],
                t["drought_chart_title"],
                t["precip_chart_title"],
            ),
            vertical_spacing=0.12,
        )

        dates = forecast_df["date"].values
        flood_risks = result["flood_risks_all_days"]
        drought_risks = result["drought_risks_all_days"]

        # Row 1: Flood risk bars
        flood_colors = [RISK_COLORS[r] for r in flood_risks]
        fig.add_trace(
            go.Bar(
                x=dates[:len(flood_risks)],
                y=flood_risks,
                marker_color=flood_colors,
                name=t["flood_risk"],
                hovertemplate="%{x|%d %b}: %{y}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.update_yaxes(
            tickvals=[0, 1, 2, 3],
            ticktext=risk_labels,
            range=[-0.3, 3.5],
            row=1, col=1,
        )

        # Row 2: Drought risk bars
        drought_colors = [RISK_COLORS[r] for r in drought_risks]
        fig.add_trace(
            go.Bar(
                x=dates[:len(drought_risks)],
                y=drought_risks,
                marker_color=drought_colors,
                name=t["drought_risk"],
                hovertemplate="%{x|%d %b}: %{y}<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.update_yaxes(
            tickvals=[0, 1, 2, 3],
            ticktext=risk_labels,
            range=[-0.3, 3.5],
            row=2, col=1,
        )

        # Row 3: Precipitation bars
        precip_mm = forecast_df["tp"].values * 1000  # m to mm
        fig.add_trace(
            go.Bar(
                x=dates,
                y=precip_mm,
                marker_color="steelblue",
                name=t["col_precip"],
                hovertemplate="%{x|%d %b}: %{y:.1f} mm<extra></extra>",
            ),
            row=3, col=1,
        )
        fig.update_yaxes(title_text="mm/day", row=3, col=1)

        fig.update_layout(
            height=650,
            showlegend=False,
            margin=dict(l=50, r=20, t=40, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Weather data table ────────────────────────────────────────────────
        st.divider()
        st.subheader(t["weather_header"])

        display_df = forecast_df.copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df["tp_mm"] = display_df["tp"] * 1000
        display_df["t2m_C"] = display_df["t2m"] - 273.15
        display_df["e_mm"] = display_df["e"] * -1000

        # Add computed values
        api_forecast = result["api_forecast"]
        display_df["api"] = api_forecast[:len(display_df)]
        display_df["flood_score"] = result["flood_scores_all_days"][:len(display_df)]
        display_df["flood_risk_label"] = [
            label_map[r] for r in result["flood_risks_all_days"][:len(display_df)]
        ]
        display_df["drought_risk_label"] = [
            label_map[r] for r in result["drought_risks_all_days"][:len(display_df)]
        ]

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

        # ── Technical details (expandable) ────────────────────────────────────
        with st.expander(t["tech_header"]):
            st.markdown(f"""
**Hybrid ensemble method:** `max(XGBoost, GenCast-style)`

**XGBoost component:**
- Uses 40 lag features (8 variables x 5 lags: 1, 3, 7, 14, 365 days)
- Feature date: {result['feature_date'].strftime('%Y-%m-%d')}
- Separate models for flood (+{horizon}d) and drought (+{horizon}d)

**GenCast-style component:**
- API forward-running: k=0.92, initialized at {result['last_known_state']['api_92']:.4f}
- SMI: held constant at {result['last_known_state']['smi_fc']:.4f} (not available in forecast)
- SPEI-6: held constant at {result['last_known_state']['spei_6']:.4f} (requires 6 months)
- total_ro: 0 (conservative assumption)
- Flood score = 0.40 x norm_API + 0.35 x norm_SMI + 0.25 x norm_total_ro

**Flood score thresholds (training 2001-2022):**
- p65 = {predictor.flood_p65:.4f} (Low to Moderate)
- p80 = {predictor.flood_p80:.4f} (Moderate to High)
- p90 = {predictor.flood_p90:.4f} (High to Extreme)

**Limitations:**
- SPEI-6 cannot be updated from a 7-day forecast
- SMI and total_ro are frozen at the last known ERA5 value
- lag_365 features in XGBoost refer to ERA5 historical data
- Open-Meteo forecast variables may differ slightly from ERA5
- Single grid point (Jijiga) — represents local conditions only
            """)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(t["footer"])


if __name__ == "__main__":
    main()

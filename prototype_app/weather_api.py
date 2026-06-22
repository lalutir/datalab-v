"""
Fetch weather forecast data from the Open-Meteo API.

Open-Meteo is free, requires no API key, and provides 7-day forecasts
for any location worldwide. We use it to get forecast precipitation and
temperature for the Jijiga grid point (9.25°N, 42.75°E).
"""

import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta


# Default location: Jijiga, Ethiopia (ERA5 grid point)
DEFAULT_LAT = 9.25
DEFAULT_LON = 42.75

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_forecast(
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
    forecast_days: int = 7,
) -> pd.DataFrame:
    """
    Fetch daily weather forecast from Open-Meteo API.

    Returns a DataFrame with columns:
        date, tp (m/day), t2m (K), e (m/day), pev (m/day)

    Units are converted to match ERA5 conventions used in the model:
        - precipitation: mm → meters
        - temperature: °C → Kelvin
        - evapotranspiration: mm → meters (negative, as ERA5 convention)
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "precipitation_sum",
            "temperature_2m_mean",
            "et0_fao_evapotranspiration",
        ],
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Kon geen verbinding maken met Open-Meteo API: {e}"
        )

    daily = data.get("daily", {})
    dates = pd.to_datetime(daily["time"])

    df = pd.DataFrame({
        "date": dates,
        # Convert mm to meters (ERA5 precipitation unit)
        "tp": np.array(daily["precipitation_sum"], dtype=float) / 1000.0,
        # Convert °C to Kelvin (ERA5 temperature unit)
        "t2m": np.array(daily["temperature_2m_mean"], dtype=float) + 273.15,
        # ET0 as proxy for evaporation (mm → m, negative sign as ERA5 convention)
        "e": -np.array(daily["et0_fao_evapotranspiration"], dtype=float) / 1000.0,
        # ET0 also as proxy for potential evaporation
        "pev": -np.array(daily["et0_fao_evapotranspiration"], dtype=float) / 1000.0,
    })

    # Handle any NaN values from API
    df = df.fillna(0.0)

    return df


def fetch_forecast_safe(
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
    forecast_days: int = 7,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Safe wrapper around fetch_forecast that returns (df, error_message).
    Returns (DataFrame, None) on success or (None, error_string) on failure.
    """
    try:
        df = fetch_forecast(latitude, longitude, forecast_days)
        return df, None
    except Exception as e:
        return None, str(e)

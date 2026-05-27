"""Climate indices derived from daily ERA5 data.

Computes five indices for a single-location daily time series:

  API    — Antecedent Precipitation Index (k=0.92, ~12-day soil memory)
  SPEI-6 — Standardized Precipitation-Evapotranspiration Index, 6-month scale
  SPEI-12 — Same at 12-month scale
  SMI    — Soil Moisture Index (field-capacity normalised, 0–1)
  total_ro — Total runoff (surface + sub-surface) [m/day]

All functions accept and return pandas Series/DataFrames with a DatetimeIndex
or a 'time' column.  They do NOT modify the input inplace.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_FC = 0.323  # field capacity [m³/m³] — loamy sand, Somali region


def compute_total_runoff(df: pd.DataFrame) -> pd.Series:
    """Surface runoff + sub-surface runoff [m/day].

    Uses ERA5 variables ssro (surface runoff) and sro (sub-surface runoff).
    Both are accumulated over the 6-hour step and summed to daily in the
    pipeline, so the daily values are in metres of water equivalent.
    """
    return df["ssro"] + df["sro"]


def compute_api(tp: pd.Series, decay: float = 0.92) -> pd.Series:
    """Antecedent Precipitation Index.

    API_t = k * API_{t-1} + P_t

    Args:
        tp: Daily total precipitation [m].
        decay: Decay factor k (default 0.92, ~12-day soil memory for semi-arid soils).

    Returns:
        API series aligned with the input index, named 'api_92'.
    """
    vals = tp.to_numpy(dtype=float)
    api = np.zeros(len(vals), dtype=float)
    for i in range(1, len(vals)):
        api[i] = decay * api[i - 1] + vals[i]
    return pd.Series(api, index=tp.index, name="api_92")


def compute_smi(swvl1: pd.Series, swvl2: pd.Series, fc: float = _FC) -> pd.Series:
    """Soil Moisture Index — field-capacity normalisation.

    SMI = (swvl1 + swvl2) / (FC1 + FC2), clipped to [0, 1].

    Uses a fixed physical field capacity (0.323 m³/m³ per layer) rather than
    data-driven min-max scaling, so the normalisation is stable as new years
    are added and is physically interpretable.

    Args:
        swvl1: Volumetric soil water layer 1 (0–7 cm) [m³/m³].
        swvl2: Volumetric soil water layer 2 (7–28 cm) [m³/m³].
        fc:    Field capacity [m³/m³] applied to both layers (default 0.323).

    Returns:
        SMI series in [0, 1], indexed like swvl1, named 'smi_fc'.
    """
    smi = (swvl1 + swvl2) / (fc * 2)
    return smi.clip(0.0, 1.0).rename("smi_fc")


def compute_spei(
    df: pd.DataFrame,
    scale_months: int,
    calib_end_year: int = 2020,
) -> pd.Series:
    """Monthly SPEI calibrated per calendar month on a reference period, forward-filled to daily.

    Steps:
      1. Aggregate daily CWB (tp + pev) to monthly sums.
      2. Compute rolling scale_months-month accumulations.
      3. For each calendar month, fit a 3-parameter log-logistic (Fisk) distribution
         on calibration-period data (year <= calib_end_year) only — no leakage.
      4. Apply the fitted CDF to all months to obtain SPEI values.
      5. Forward-fill monthly SPEI to the original daily index.

    In ERA5, pev uses the downward-positive convention (typically negative values).
    The climatic water balance is therefore: CWB = tp + pev = tp - |pev|.

    Args:
        df:               DataFrame with 'time', 'tp', 'pev' columns.
        scale_months:     Accumulation scale in months (6 or 12).
        calib_end_year:   Last year of the calibration/reference period (default 2020).

    Returns:
        Daily SPEI series aligned with df.index, named 'spei_{scale_months}'.
    """
    cwb_daily = df["tp"] + df["pev"]
    cwb_daily.index = pd.DatetimeIndex(df["time"])

    df_m = cwb_daily.resample("MS").sum()
    cwb_roll = df_m.rolling(scale_months, min_periods=scale_months).sum()

    spei_vals = np.full(len(cwb_roll), np.nan)

    for month in range(1, 13):
        m_mask = cwb_roll.index.month == month
        train_m = m_mask & (cwb_roll.index.year <= calib_end_year)
        x_train = cwb_roll[train_m].dropna().values
        if len(x_train) < 5:
            logger.warning(
                "SPEI-%d month %d: only %d calibration values — skipping",
                scale_months, month, len(x_train),
            )
            continue
        shift = x_train.min() - 1e-9
        x_tr = x_train - shift
        try:
            c, loc, sc = stats.fisk.fit(x_tr, floc=0)
            all_m = m_mask & cwb_roll.notna()
            x_all = cwb_roll[all_m].values - shift
            p = np.clip(stats.fisk.cdf(x_all, c=c, loc=loc, scale=sc), 1e-6, 1 - 1e-6)
            spei_vals[np.where(all_m)[0]] = stats.norm.ppf(p)
        except Exception as exc:
            logger.warning("SPEI-%d month %d fit failed: %s", scale_months, month, exc)

    spei_monthly = pd.Series(spei_vals, index=cwb_roll.index, name=f"spei_{scale_months}")

    daily_idx = pd.DatetimeIndex(df["time"])
    spei_daily = spei_monthly.reindex(daily_idx, method="ffill")
    return pd.Series(spei_daily.values, index=df.index, name=f"spei_{scale_months}")


def add_indices(
    df: pd.DataFrame,
    api_decay: float = 0.92,
    spei_calib_end_year: int = 2020,
) -> pd.DataFrame:
    """Add api_92, smi_fc, spei_6, spei_12, and total_ro to a daily single-location DataFrame.

    The DataFrame must be sorted chronologically and contain columns:
        tp, pev, ssro, sro, swvl1, swvl2

    Args:
        df:                  Daily ERA5 DataFrame (already sorted by time).
        api_decay:           API decay factor k (default 0.92).
        spei_calib_end_year: Last year of SPEI calibration period (default 2020).

    Returns:
        Copy of df with five new columns appended:
            total_ro, api_92, smi_fc, spei_6, spei_12
    """
    df = df.copy().sort_values("time").reset_index(drop=True)

    df["total_ro"] = compute_total_runoff(df).values
    df["api_92"] = compute_api(df["tp"], decay=api_decay).values
    df["smi_fc"] = compute_smi(df["swvl1"], df["swvl2"]).values
    df["spei_6"] = compute_spei(df, scale_months=6, calib_end_year=spei_calib_end_year).values
    df["spei_12"] = compute_spei(df, scale_months=12, calib_end_year=spei_calib_end_year).values

    logger.info(
        "Indices added: total_ro, api_92 (k=%.2f), smi_fc (FC=%.3f m³/m³), "
        "spei_6 + spei_12 (calib ≤ %d)",
        api_decay,
        _FC,
        spei_calib_end_year,
    )
    return df

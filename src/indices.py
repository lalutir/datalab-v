"""Climate indices derived from daily ERA5 data.

Computes four indices for a single-location daily time series:

  API  — Antecedent Precipitation Index (memory of past precipitation)
  SPEI — Standardized Precipitation-Evapotranspiration Index (drought)
  SMI  — Soil Moisture Index (relative soil wetness, 0-1)
  total_ro — Total runoff (surface + sub-surface) [m/day]

All functions accept and return pandas Series/DataFrames with a DatetimeIndex
or a 'time' column.  They do NOT modify the input inplace.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_total_runoff(df: pd.DataFrame) -> pd.Series:
    """Surface runoff + sub-surface runoff [m/day].

    Uses ERA5 variables ssro (surface runoff) and sro (sub-surface runoff).
    Both are accumulated over the 6-hour step and summed to daily in the
    pipeline, so the daily values are in metres of water equivalent.
    """
    return df["ssro"] + df["sro"]


def compute_api(tp: pd.Series, decay: float = 0.85) -> pd.Series:
    """Antecedent Precipitation Index.

    API_t = k * API_{t-1} + P_t

    Args:
        tp: Daily total precipitation [m].
        decay: Decay factor k (default 0.85).

    Returns:
        API series aligned with the input index.
    """
    vals = tp.to_numpy(dtype=float)
    api = np.zeros(len(vals), dtype=float)
    for i in range(1, len(vals)):
        api[i] = decay * api[i - 1] + vals[i]
    return pd.Series(api, index=tp.index, name="api")


def compute_smi(swvl: pd.Series) -> pd.Series:
    """Soil Moisture Index.

    SMI = (θ - θ_min) / (θ_max - θ_min)

    Scales observed soil moisture to [0, 1] over the full historical record
    (0 = driest observed day, 1 = wettest observed day).  Based on swvl1
    (volumetric soil water layer 1, 0-7 cm) [m³/m³].

    Values are clipped to [0, 1] to handle marginal numerical artefacts
    in the raw ERA5 data.
    """
    vmin = float(swvl.min())
    vmax = float(swvl.max())
    if vmax - vmin < 1e-10:
        logger.warning("swvl range near zero; SMI set to NaN")
        return pd.Series(np.full(len(swvl), np.nan), index=swvl.index, name="smi")
    smi = (swvl - vmin) / (vmax - vmin)
    return smi.clip(0.0, 1.0).rename("smi")


def compute_spei(
    tp: pd.Series,
    pev: pd.Series,
    scale: int = 30,
) -> pd.Series:
    """Standardized Precipitation-Evapotranspiration Index (SPEI).

    Computes a rolling `scale`-day water balance and transforms it to the
    standard normal distribution using a 3-parameter log-logistic (Fisk)
    CDF fit.  Positive values indicate wet conditions; negative values
    indicate dry/drought conditions.

    In ERA5, pev (potential evaporation) uses the downward-positive
    convention, so its values are typically negative (upward flux).
    The water balance is therefore:
        D = tp + pev   (pev < 0 subtracts evaporative demand)

    Args:
        tp:    Daily total precipitation [m].
        pev:   Daily potential evaporation [m], negative in ERA5.
        scale: Rolling accumulation window in days (default 30 ≈ SPEI-1).

    Returns:
        SPEI series.  The first (scale - 1) values are NaN.
    """
    # Water balance: precipitation minus potential evapotranspiration
    # pev is negative in ERA5, so tp + pev = tp - |pev|
    D = tp + pev
    D_acc = D.rolling(window=scale, min_periods=scale).sum()

    result = np.full(len(D_acc), np.nan)
    valid_mask = ~D_acc.isna().to_numpy()
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < 20:
        logger.warning("Too few valid values for SPEI fit (%d); returning NaN", len(valid_idx))
        return pd.Series(result, index=tp.index, name="spei")

    x = D_acc.iloc[valid_idx].to_numpy(dtype=float)

    # Shift data so all values are strictly positive before log-logistic fit
    shift = x.min() - 1e-6
    x_shifted = x - shift  # minimum value is now ~1e-6 > 0

    try:
        # Fit 3-parameter log-logistic (Fisk) distribution with loc fixed at 0
        # because the shift already places the data origin at 0
        c, loc, scale_p = stats.fisk.fit(x_shifted, floc=0)
        p = stats.fisk.cdf(x_shifted, c=c, loc=loc, scale=scale_p)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        result[valid_idx] = stats.norm.ppf(p)
        logger.debug("SPEI fit ok: c=%.3f scale=%.4f", c, scale_p)
    except Exception as exc:
        logger.warning("Log-logistic fit failed (%s); falling back to z-score", exc)
        mean, std = x.mean(), x.std()
        if std > 0:
            result[valid_idx] = (x - mean) / std

    return pd.Series(result, index=tp.index, name="spei")


def add_indices(
    df: pd.DataFrame,
    api_decay: float = 0.85,
    spei_scale: int = 30,
) -> pd.DataFrame:
    """Add API, SPEI, SMI and total_ro columns to a daily single-location DataFrame.

    The DataFrame must be sorted chronologically and contain columns:
        tp, pev, ssro, sro, swvl1

    Args:
        df:         Daily ERA5 DataFrame (already sorted by time).
        api_decay:  API decay factor k (default 0.85).
        spei_scale: Rolling window for SPEI accumulation in days (default 30).

    Returns:
        Copy of df with four new columns appended:
            total_ro, api, smi, spei
    """
    df = df.copy().sort_values("time").reset_index(drop=True)

    df["total_ro"] = compute_total_runoff(df).values
    df["api"] = compute_api(df["tp"], decay=api_decay).values
    df["smi"] = compute_smi(df["swvl1"]).values
    df["spei"] = compute_spei(df["tp"], df["pev"], scale=spei_scale).values

    logger.info(
        "Added indices: total_ro, api, smi, spei  (spei_scale=%d days, api_decay=%.2f)",
        spei_scale,
        api_decay,
    )
    return df

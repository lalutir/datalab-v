"""Temporal aggregation and resampling for ERA5 datasets.

Supports resampling hourly data to daily (or other frequencies) with
per-variable aggregation methods (mean, sum, min, max).
"""

import logging

import xarray as xr

from src.config import PipelineConfig

logger = logging.getLogger(__name__)

VALID_AGG_METHODS = {"mean", "sum", "min", "max"}


def resample_dataset(
    ds: xr.Dataset, config: PipelineConfig
) -> xr.Dataset:
    """Resample the dataset to the configured frequency.

    Each variable is aggregated according to its configured method
    (e.g., precipitation -> sum, temperature -> mean).

    Args:
        ds: Input dataset (must have a 'time' coordinate).
        config: Pipeline configuration with resample_freq and agg_methods.

    Returns:
        Resampled dataset. Returns the input unchanged if resample_freq is None.
    """
    freq = config.processing.resample_freq
    if freq is None:
        logger.info("Resampling skipped (resample_freq is null)")
        return ds

    if "time" not in ds.coords:
        logger.warning("No 'time' coordinate found, skipping resampling")
        return ds

    agg_methods = config.processing.agg_methods
    resampled_vars = []

    for var_name in ds.data_vars:
        method = agg_methods.get(var_name, "mean")
        if method not in VALID_AGG_METHODS:
            logger.warning(
                "Unknown aggregation method '%s' for '%s', falling back to 'mean'",
                method,
                var_name,
            )
            method = "mean"

        var_resampled = getattr(ds[var_name].resample(time=freq), method)()
        resampled_vars.append(var_resampled)
        logger.debug("Resampled '%s' with method '%s'", var_name, method)

    result = xr.merge(resampled_vars)
    logger.info("Resampled to '%s': %d variables", freq, len(result.data_vars))
    return result

"""Spatial, temporal, and variable subsetting for ERA5 datasets.

All operations return lazy xarray Datasets -- nothing is loaded into memory.
"""

import logging

import xarray as xr

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def select_variables(ds: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """Keep only the specified variables from the dataset.

    Args:
        ds: Input dataset.
        variables: List of variable names to keep.

    Returns:
        Dataset with only the requested variables.
    """
    available = list(ds.data_vars)
    missing = [v for v in variables if v not in available]
    if missing:
        logger.warning(
            "Requested variables not found in dataset: %s. Available: %s",
            missing,
            available,
        )

    keep = [v for v in variables if v in available]
    if not keep:
        raise ValueError(
            f"None of the requested variables {variables} found. "
            f"Available: {available}"
        )

    ds_sub = ds[keep]
    logger.info("Selected variables: %s", keep)
    return ds_sub


def select_time_range(
    ds: xr.Dataset, start: str, end: str
) -> xr.Dataset:
    """Select a time range from the dataset.

    Args:
        ds: Input dataset (must have a 'time' coordinate).
        start: Start date string, e.g. "2018-01-01".
        end: End date string, e.g. "2018-03-31".

    Returns:
        Dataset filtered to the time range.
    """
    if "time" not in ds.coords:
        logger.warning("No 'time' coordinate found, skipping time selection")
        return ds

    ds_sub = ds.sel(time=slice(start, end))
    logger.info("Selected time range: %s to %s", start, end)
    return ds_sub


def select_bbox(
    ds: xr.Dataset,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.Dataset:
    """Select a geographic bounding box from the dataset.

    Handles both ascending and descending latitude ordering, and detects
    whether the latitude coordinate is named 'latitude' or 'lat'.

    Args:
        ds: Input dataset.
        lat_min: Southern boundary.
        lat_max: Northern boundary.
        lon_min: Western boundary.
        lon_max: Eastern boundary.

    Returns:
        Dataset filtered to the bounding box.
    """
    # Detect coordinate names
    lat_name = _detect_coord(ds, ["latitude", "lat"])
    lon_name = _detect_coord(ds, ["longitude", "lon"])

    if lat_name is None or lon_name is None:
        logger.warning("Could not detect lat/lon coordinates, skipping bbox selection")
        return ds

    # Single-point selection when min == max (within floating-point tolerance)
    if abs(lat_max - lat_min) < 1e-6 and abs(lon_max - lon_min) < 1e-6:
        ds_sub = ds.sel(
            {lat_name: lat_min, lon_name: lon_min},
            method="nearest",
        )
        # Scalar sel drops the dimension; restore as size-1 dims so downstream
        # code (to_dataframe, concat) keeps lat/lon as regular columns.
        if lat_name not in ds_sub.dims:
            ds_sub = ds_sub.expand_dims(
                {lat_name: [float(ds_sub[lat_name].values)]}
            )
        if lon_name not in ds_sub.dims:
            ds_sub = ds_sub.expand_dims(
                {lon_name: [float(ds_sub[lon_name].values)]}
            )
        logger.info(
            "Selected point (nearest): lat=%.4f, lon=%.4f",
            float(ds_sub[lat_name].values[0]),
            float(ds_sub[lon_name].values[0]),
        )
        return ds_sub

    # ERA5 often has latitude in descending order (90 to -90)
    lat_values = ds[lat_name].values
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    lon_slice = slice(lon_min, lon_max)

    ds_sub = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    logger.info(
        "Selected bbox: lat [%.2f, %.2f], lon [%.2f, %.2f]",
        lat_min, lat_max, lon_min, lon_max,
    )
    return ds_sub


def apply_all_filters(ds: xr.Dataset, config: PipelineConfig) -> xr.Dataset:
    """Apply all configured subset filters in sequence.

    Order: variables -> time range -> bounding box.
    All operations remain lazy.

    Args:
        ds: Input lazy dataset.
        config: Pipeline configuration.

    Returns:
        Filtered lazy dataset.
    """
    sel = config.selection

    ds = select_variables(ds, sel.variables)
    ds = select_time_range(ds, sel.time_start, sel.time_end)
    ds = select_bbox(ds, sel.lat_min, sel.lat_max, sel.lon_min, sel.lon_max)

    return ds


def _detect_coord(ds: xr.Dataset, candidates: list[str]) -> str | None:
    """Find the first matching coordinate name from a list of candidates."""
    for name in candidates:
        if name in ds.coords:
            return name
    return None

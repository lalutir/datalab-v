"""ERA5 dataset loading with xarray, cfgrib, and dask.

Opens GRIB files as lazy dask-backed xarray Datasets.
Nothing is loaded into memory until .compute() is called.

NOTE on GRIB loading:
cfgrib sometimes splits a GRIB file into multiple hypercubes if it contains
mixed parameter types (e.g., surface + pressure level fields). When this
happens, open_dataset raises a DatasetBuildError. The loader handles this
by using filter_by_keys to open one variable group at a time, then merging.
"""

import logging
from pathlib import Path

import xarray as xr

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def open_era5_dataset(
    file_path: Path,
    config: PipelineConfig,
    filter_keys: dict | None = None,
) -> xr.Dataset:
    """Open a single ERA5 GRIB file as a lazy xarray Dataset.

    Args:
        file_path: Path to a local .grib file.
        config: Pipeline configuration (used for chunk sizes).
        filter_keys: Optional cfgrib filter_by_keys dict for selecting
                     specific GRIB messages. Example: {"shortName": "t2m"}

    Returns:
        Lazy dask-backed xarray Dataset.
    """
    chunks = config.processing.chunks
    backend_kwargs = {"indexpath": ""}
    if filter_keys:
        backend_kwargs["filter_by_keys"] = filter_keys

    try:
        ds = xr.open_dataset(
            file_path,
            engine="cfgrib",
            chunks=chunks,
            backend_kwargs=backend_kwargs,
        )
        logger.info(
            "Opened %s: %d variables, %d coords",
            file_path.name,
            len(ds.data_vars),
            len(ds.coords),
        )
        return ds

    except Exception as e:
        if "multiple values" in str(e).lower() or "filter" in str(e).lower():
            logger.warning(
                "GRIB contains mixed parameter types. "
                "Attempting to open variable-by-variable: %s",
                e,
            )
            return _open_grib_by_variable(file_path, config)
        raise


def _open_grib_by_variable(
    file_path: Path, config: PipelineConfig
) -> xr.Dataset:
    """Fallback: open a GRIB file one variable at a time and merge.

    Used when cfgrib cannot build a single hypercube from the file.
    """
    chunks = config.processing.chunks
    variables = config.selection.variables
    datasets = []

    for var in variables:
        try:
            ds = xr.open_dataset(
                file_path,
                engine="cfgrib",
                chunks=chunks,
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {"shortName": var},
                },
            )
            datasets.append(ds)
            logger.info("Loaded variable '%s' from %s", var, file_path.name)
        except Exception as var_err:
            logger.error(
                "Failed to load variable '%s' from %s: %s",
                var,
                file_path.name,
                var_err,
            )

    if not datasets:
        raise RuntimeError(f"Could not load any variables from {file_path}")

    merged = xr.merge(datasets)
    logger.info("Merged %d variable datasets from %s", len(datasets), file_path.name)
    return merged


def open_era5_multiple(
    file_paths: list[Path], config: PipelineConfig
) -> xr.Dataset:
    """Open multiple GRIB files and concatenate along the time dimension.

    Each file is opened individually, then concatenated. This avoids
    open_mfdataset issues with GRIB files that have inconsistent structure.

    Args:
        file_paths: List of local .grib file paths.
        config: Pipeline configuration.

    Returns:
        Concatenated lazy xarray Dataset.
    """
    datasets = [open_era5_dataset(fp, config) for fp in file_paths]

    if len(datasets) == 1:
        return datasets[0]

    combined = xr.concat(datasets, dim="time")
    logger.info("Concatenated %d files along time dimension", len(datasets))
    return combined

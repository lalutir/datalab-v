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
    """Open a single ERA5 file as a lazy xarray Dataset.

    Supports both NetCDF (.nc) and GRIB (.grib, .grib2, .grb, .grb2) files.
    For NetCDF files the default xarray engine is used. For GRIB files cfgrib
    is used. In both cases the time dimension is normalised to 'time'.

    Args:
        file_path: Path to a local .nc or .grib file.
        config: Pipeline configuration (used for chunk sizes).
        filter_keys: Optional cfgrib filter_by_keys dict (GRIB only).

    Returns:
        Lazy dask-backed xarray Dataset.
    """
    chunks = config.processing.chunks

    if file_path.suffix.lower() in {".nc", ".nc4"}:
        ds = xr.open_dataset(file_path, chunks=chunks)
        # ERA5 CDS exports use 'valid_time'; rename to 'time' for pipeline compatibility
        if "valid_time" in ds.dims and "time" not in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        logger.info(
            "Opened %s (NetCDF): %d variables, %d coords",
            file_path.name,
            len(ds.data_vars),
            len(ds.coords),
        )
        return ds

    # GRIB path
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
            "Opened %s (GRIB): %d variables, %d coords",
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
    """Open multiple ERA5 files and combine into a single dataset.

    For NetCDF files, uses open_mfdataset with combine='by_coords' which
    automatically merges files with the same time range but different variables
    (e.g. instant + accum files from the same ZIP) and concatenates files with
    different time ranges (e.g. different monthly ZIPs) along the time dimension.

    For GRIB files, opens each file individually and concatenates along time
    (original behaviour preserved).

    Args:
        file_paths: List of local .nc or .grib file paths.
        config: Pipeline configuration.

    Returns:
        Combined lazy xarray Dataset.
    """
    if not file_paths:
        raise ValueError("No files to open")

    if all(fp.suffix.lower() in {".nc", ".nc4"} for fp in file_paths):
        # Group files by parent directory: each directory is one extracted ZIP (one month).
        # Files in the same directory have different variables but the same time coords
        # (instant vs accum) -> they must be merged, not concatenated.
        # Datasets from different directories cover different months -> concatenated along time.
        groups: dict[Path, list[Path]] = {}
        for fp in file_paths:
            groups.setdefault(fp.parent, []).append(fp)

        monthly_datasets = []
        for month_dir, month_files in sorted(groups.items()):
            month_ds_list = [open_era5_dataset(fp, config) for fp in sorted(month_files)]
            if len(month_ds_list) == 1:
                merged = month_ds_list[0]
            else:
                merged = xr.merge(month_ds_list)
                logger.info(
                    "Merged %d file(s) for %s: %d variables",
                    len(month_files),
                    month_dir.name,
                    len(merged.data_vars),
                )
            monthly_datasets.append(merged)

        if len(monthly_datasets) == 1:
            return monthly_datasets[0]

        combined = xr.concat(monthly_datasets, dim="time")
        logger.info(
            "Concatenated %d monthly datasets: %d variables, time=%d",
            len(monthly_datasets),
            len(combined.data_vars),
            combined.sizes.get("time", 0),
        )
        return combined

    # GRIB path: open individually then concatenate along time
    datasets = [open_era5_dataset(fp, config) for fp in file_paths]

    if len(datasets) == 1:
        return datasets[0]

    combined = xr.concat(datasets, dim="time")
    logger.info("Concatenated %d GRIB files along time dimension", len(datasets))
    return combined

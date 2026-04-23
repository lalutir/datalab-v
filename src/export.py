"""Export ERA5 subsets to Zarr, Parquet, or pandas DataFrame.

This is where lazy data is finally materialized via .compute().
By this point the dataset should already be subsetted to a small slice,
so memory usage stays manageable.
"""

import logging
import shutil
from pathlib import Path

import pandas as pd
import xarray as xr

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def to_zarr_self(ds: xr.Dataset, output_path: Path) -> Path:
    """Save an xarray Dataset to a Zarr store.

    Good for multidimensional analysis -- preserves coordinates and dimensions.

    Args:
        ds: Dataset to save (can be lazy; dask handles chunked writes).
        output_path: Directory path for the .zarr store.

    Returns:
        Path to the created Zarr store.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rechunk to uniform sizes before writing (resampling can create irregular chunks)
    if ds.chunks:
        ds = ds.chunk("auto")

    # Remove existing Zarr store first to avoid Windows permission errors
    if output_path.exists():
        shutil.rmtree(output_path)

    ds.to_zarr(str(output_path))
    logger.info("Saved Zarr store: %s", output_path)
    return output_path


def to_parquet(ds: xr.Dataset, output_path: Path) -> Path:
    """Convert an xarray Dataset to a pandas DataFrame and save as Parquet.

    The dataset is flattened: coordinates become columns alongside variables.
    Good for ML pipelines and tabular analysis.

    Args:
        ds: Dataset to save. Will be computed (loaded into memory).
        output_path: File path for the .parquet file.

    Returns:
        Path to the created Parquet file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = ds.compute().to_dataframe().reset_index()
    df.to_parquet(str(output_path), index=False)

    logger.info("Saved Parquet: %s (%d rows, %d columns)", output_path, len(df), len(df.columns))
    return output_path


def to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Convert an xarray Dataset to a pandas DataFrame in memory.

    This calls .compute() and materializes the data. Only use this after
    the dataset has been subsetted to a manageable size.

    Args:
        ds: Lazy dataset to materialize.

    Returns:
        pandas DataFrame with coordinates as columns.
    """
    df = ds.compute().to_dataframe().reset_index()
    logger.info("Created DataFrame: %d rows, %d columns", len(df), len(df.columns))
    return df


def export_dataset(ds: xr.Dataset, config: PipelineConfig) -> dict[str, Path]:
    """Export a dataset using the configured output format.

    Args:
        ds: Dataset to export.
        config: Pipeline configuration with output settings.

    Returns:
        Dictionary mapping format name to output path.
    """
    out_dir = config.output.dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = config.output.format
    results = {}

    if fmt in ("zarr", "both"):
        zarr_path = out_dir / "era5_subset.zarr"
        to_zarr_self(ds, zarr_path)
        results["zarr"] = zarr_path

    if fmt in ("parquet", "both"):
        parquet_path = out_dir / "era5_subset.parquet"
        to_parquet(ds, parquet_path)
        results["parquet"] = parquet_path

    return results

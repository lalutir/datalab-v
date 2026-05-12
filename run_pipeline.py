"""Main entry point for the ERA5 subset pipeline.

Reads config.yaml, downloads the selected GRIB files from Azure,
applies spatial/temporal/variable filters, optionally resamples,
and exports the result to Zarr and/or Parquet.

Usage:
    python run_pipeline.py
    python run_pipeline.py --config path/to/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.storage import download_selected_blobs
from src.loader import open_era5_multiple
from src.subset import apply_all_filters
from src.aggregate import resample_dataset
from src.export import export_dataset, to_dataframe


def setup_logging() -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress verbose Azure HTTP logs
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


def main() -> None:
    setup_logging()
    logger = logging.getLogger("pipeline")

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="ERA5 subset pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: project root)",
    )
    args = parser.parse_args()

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)
    logger.info(
        "Selection: blob_files=%s, variables=%s, time=%s to %s",
        config.selection.blob_files,
        config.selection.variables,
        config.selection.time_start,
        config.selection.time_end,
    )

    # Step 1: Download GRIB files from Azure
    logger.info("Step 1: Downloading GRIB files from Azure Blob Storage...")
    local_paths = download_selected_blobs(config)
    logger.info("Downloaded %d files: %s", len(local_paths), local_paths)

    # Step 2: Open datasets lazily
    logger.info("Step 2: Opening datasets with xarray + cfgrib (lazy)...")
    ds = open_era5_multiple(local_paths, config)
    logger.info("Dataset: %s", ds)

    # Step 3: Apply subset filters
    logger.info("Step 3: Applying spatial/temporal/variable filters...")
    ds = apply_all_filters(ds, config)
    logger.info("Subset: %s", ds)

    # Step 4: Resample if configured
    logger.info("Step 4: Resampling...")
    ds = resample_dataset(ds, config)

    # Step 5: Export
    logger.info("Step 5: Exporting...")
    results = export_dataset(ds, config)
    for fmt, path in results.items():
        logger.info("Exported %s: %s", fmt, path)

    # Step 6: Preview as pandas DataFrame
    logger.info("Step 6: Preview as pandas DataFrame...")
    df = to_dataframe(ds)
    logger.info("DataFrame shape: %s", df.shape)
    print("\nPreview (first 10 rows):")
    print(df.head(10))

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()

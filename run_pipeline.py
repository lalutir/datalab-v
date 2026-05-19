"""Main entry point for the ERA5 subset pipeline.

Reads config.yaml, loops over each year in year_start..year_end,
downloads all 12 monthly ZIP files from Azure, merges them into one
yearly dataset, applies spatial/variable filters, resamples to daily,
and exports one Parquet file per year: era5_{year}.parquet.

Usage:
    python run_pipeline.py
    python run_pipeline.py --config path/to/config.yaml
"""

import argparse
import copy
import logging
from pathlib import Path

from src.config import load_config
from src.storage import download_selected_blobs
from src.loader import open_era5_multiple
from src.subset import apply_all_filters
from src.aggregate import resample_dataset
from src.export import export_dataset


def setup_logging() -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


def make_year_config(config, year: int):
    """Return a deep copy of config with blob_files and time range set for the given year."""
    yc = copy.deepcopy(config)
    yc.selection.blob_files = [f"{year}_{month:02d}" for month in range(1, 13)]
    yc.selection.time_start = f"{year}-01-01"
    yc.selection.time_end = f"{year}-12-31"
    return yc


def main() -> None:
    setup_logging()
    logger = logging.getLogger("pipeline")

    parser = argparse.ArgumentParser(description="ERA5 subset pipeline")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    logger.info("Loading configuration...")
    config = load_config(args.config)
    logger.info(
        "Year range: %d-%d | Variables: %s | Bbox: lat [%.1f, %.1f] lon [%.1f, %.1f]",
        config.selection.year_start,
        config.selection.year_end,
        config.selection.variables,
        config.selection.lat_min,
        config.selection.lat_max,
        config.selection.lon_min,
        config.selection.lon_max,
    )

    years = list(range(config.selection.year_start, config.selection.year_end + 1))
    logger.info("Processing %d year(s): %d to %d", len(years), years[0], years[-1])

    for year in years:
        logger.info("--- Year %d ---", year)
        year_config = make_year_config(config, year)

        try:
            # Step 1: Download all 12 monthly ZIPs and extract NC files
            logger.info("[%d] Step 1: Downloading 12 monthly ZIPs from Azure...", year)
            local_paths = download_selected_blobs(year_config)
            logger.info("[%d] %d file(s) ready", year, len(local_paths))

            # Step 2: Open all files lazily (instant+accum merged per month, months concatenated)
            logger.info("[%d] Step 2: Opening datasets lazily...", year)
            ds = open_era5_multiple(local_paths, year_config)

            # Step 3: Apply variable, time, and bbox filters
            logger.info("[%d] Step 3: Applying filters (variables, time, bbox)...", year)
            ds = apply_all_filters(ds, year_config)
            logger.info("[%d] Filtered shape: %s", year, dict(ds.sizes))

            # Step 4: Resample to daily
            logger.info("[%d] Step 4: Resampling to daily...", year)
            ds = resample_dataset(ds, year_config)

            # Step 5: Export as era5_{year}.parquet
            logger.info("[%d] Step 5: Exporting...", year)
            results = export_dataset(ds, year_config, label=f"era5_{year}")
            for fmt, path in results.items():
                size_mb = path.stat().st_size / 1e6
                logger.info("[%d] Exported %s: %s (%.1f MB)", year, fmt, path.name, size_mb)

        except Exception as exc:
            logger.error("[%d] Failed: %s", year, exc, exc_info=True)
            logger.info("[%d] Skipping to next year.", year)
            continue

    logger.info("All years complete. Output: %s", config.output.dir)


if __name__ == "__main__":
    main()

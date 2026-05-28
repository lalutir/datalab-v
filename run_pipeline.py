"""Main entry point for the ERA5 subset pipeline.

Reads config.yaml, loops over each year in year_start..year_end,
downloads all 12 monthly ZIP files from Azure, merges them into one
yearly dataset, applies spatial/variable filters, resamples to daily,
then concatenates all years into a single DataFrame, computes climate
indices (API, SPEI, SMI, Total Runoff), and exports one combined
Parquet file: era5_{year_start}_{year_end}.parquet.

Usage:
    python run_pipeline.py
    python run_pipeline.py --config path/to/config.yaml
"""

import argparse
import copy
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
# from src.storage import download_selected_blobs
from src.loader import open_era5_multiple
from src.subset import apply_all_filters
from src.aggregate import resample_dataset
from src.export import to_dataframe
from src.indices import add_indices


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


def make_year_config(config, year: int): # type: ignore
    """Return a deep copy of config with blob_files and time range set for the given year."""
    yc = copy.deepcopy(config) # type: ignore
    yc.selection.blob_files = [f"{year}_{month:02d}" for month in range(1, 13)] # type: ignore
    yc.selection.time_start = f"{year}-01-01" # type: ignore
    yc.selection.time_end = f"{year}-12-31" # type: ignore
    return yc # type: ignore


def main() -> None:
    setup_logging()
    logger = logging.getLogger("pipeline")

    parser = argparse.ArgumentParser(description="ERA5 subset pipeline")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    logger.info("Loading configuration...")
    config = load_config(args.config)
    logger.info(
        "Year range: %d-%d | Variables: %s | Bbox: lat [%.2f, %.2f] lon [%.2f, %.2f]",
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

    all_dfs: list[pd.DataFrame] = []

    for year in years:
        logger.info("--- Year %d ---", year)
        year_config = make_year_config(config, year)

        try:
            # # Step 1: Download all 12 monthly ZIPs and extract NC files
            # logger.info("[%d] Step 1: Downloading 12 monthly ZIPs from Azure...", year)
            # local_paths = download_selected_blobs(year_config)
            # logger.info("[%d] %d file(s) ready", year, len(local_paths))

            # Step 2: Open all files lazily (instant+accum merged per month, months concatenated)
            logger.info("[%d] Step 2: Opening datasets lazily...", year)
            raw_dir = config.raw_dir
            local_paths1 = [raw_dir / f"era5_{year}_{month:02d}" / "data_stream-oper_stepType-accum.nc" for month in range(1, 13)]
            local_paths2 = [raw_dir / f"era5_{year}_{month:02d}" / "data_stream-oper_stepType-instant.nc" for month in range(1, 13)]
            local_paths = local_paths1 + local_paths2
            local_paths.sort(key=lambda p: (p.stem.split("_")[0], p.stem.split("_")[1], "instant" in p.name))
            ds = open_era5_multiple(local_paths, year_config)

            # Step 3: Apply variable, time, and bbox filters (single-point selection)
            logger.info("[%d] Step 3: Applying filters (variables, time, point)...", year)
            ds = apply_all_filters(ds, year_config)
            logger.info("[%d] Filtered shape: %s", year, dict(ds.sizes))

            # Step 4: Resample to daily (mean for instant vars, sum for accumulated)
            logger.info("[%d] Step 4: Resampling to daily...", year)
            ds = resample_dataset(ds, year_config)

            # Step 5: Materialise to DataFrame and collect
            logger.info("[%d] Step 5: Converting to DataFrame...", year)
            df_year = to_dataframe(ds)
            logger.info("[%d] %d rows collected", year, len(df_year))
            all_dfs.append(df_year)

        except Exception as exc:
            logger.error("[%d] Failed: %s", year, exc, exc_info=True)
            logger.info("[%d] Skipping to next year.", year)
            continue

    if not all_dfs:
        logger.error("No data collected for any year. Exiting.")
        return

    # Combine all years into one DataFrame
    logger.info("Combining %d year(s) into a single DataFrame...", len(all_dfs))
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.sort_values("time").reset_index(drop=True)
    logger.info("Combined shape: %d rows × %d columns", len(df_all), len(df_all.columns))

    # Compute climate indices
    logger.info(
        "Computing climate indices: api_92 (k=0.92), smi_fc (FC=0.323 m³/m³), "
        "spei_6/spei_12 (monthly, calib ≤ 2020), total_ro..."
    )
    df_all = add_indices(df_all)
    logger.info("Indices added. Final shape: %d rows × %d columns", len(df_all), len(df_all.columns))

    # Export single combined Parquet
    out_dir = Path('src/data/processed')
    label = "era5_final"
    out_path = out_dir / f"{label}.parquet"
    df_all.to_parquet(str(out_path), index=False)
    size_mb = out_path.stat().st_size / 1e6
    logger.info(
        "Exported combined dataset: %s (%d rows, %d cols, %.1f MB)",
        out_path.name, #type: ignore
        len(df_all),
        len(df_all.columns),
        size_mb,
    )

    logger.info("Pipeline complete. Output: %s", config.output.dir)


if __name__ == "__main__":
    main()

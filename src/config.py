"""Configuration loader for the ERA5 pipeline.

Loads Azure credentials from .env and pipeline settings from config.yaml.
All other modules receive config as input -- they never read env vars directly.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

import os

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class AzureConfig:
    container_name: str
    blob_prefix: str
    blob_pattern: str
    account_name: str = ""
    account_key: str = ""
    connection_string: str = ""


@dataclass
class SelectionConfig:
    blob_files: list[str] = field(default_factory=lambda: ["2018-2019"])
    variables: list[str] = field(default_factory=lambda: ["t2m", "d2m"])
    time_start: str = "2018-01-01"
    time_end: str = "2018-03-31"
    lat_min: float = -5.0
    lat_max: float = 15.0
    lon_min: float = 25.0
    lon_max: float = 45.0


@dataclass
class ProcessingConfig:
    chunks: dict[str, int] = field(
        default_factory=lambda: {"time": 100, "latitude": 50, "longitude": 50}
    )
    resample_freq: Optional[str] = "1D"
    agg_methods: dict[str, str] = field(
        default_factory=lambda: {
            "t2m": "mean",
            "d2m": "mean",
            "tp": "sum",
            "u10": "mean",
            "v10": "mean",
            "msl": "mean",
        }
    )


@dataclass
class OutputConfig:
    dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    format: str = "both"


@dataclass
class PipelineConfig:
    azure: AzureConfig = field(default_factory=lambda: AzureConfig("", "", ""))
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Load pipeline configuration from .env and config.yaml.

    Args:
        config_path: Path to YAML config file. Defaults to project root config.yaml.

    Returns:
        Fully populated PipelineConfig instance.
    """
    # Load environment variables from .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded .env from %s", env_path)
    else:
        logger.warning("No .env file found at %s", env_path)

    # Load YAML config
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    logger.info("Loaded config from %s", path)

    # Build Azure config from YAML + env vars
    az_raw = raw.get("azure", {})
    azure = AzureConfig(
        container_name=az_raw.get("container_name", ""),
        blob_prefix=az_raw.get("blob_prefix", ""),
        blob_pattern=az_raw.get("blob_pattern", ""),
        account_name=os.getenv("AZURE_STORAGE_ACCOUNT_NAME", ""),
        account_key=os.getenv("AZURE_STORAGE_ACCOUNT_KEY", ""),
        connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING", ""),
    )

    # Build selection config
    sel_raw = raw.get("selection", {})
    tr = sel_raw.get("time_range", {})
    bb = sel_raw.get("bbox", {})
    selection = SelectionConfig(
        blob_files=sel_raw.get("blob_files", ["2018-2019"]),
        variables=sel_raw.get("variables", ["t2m", "d2m"]),
        time_start=tr.get("start", "2018-01-01"),
        time_end=tr.get("end", "2018-03-31"),
        lat_min=bb.get("lat_min", -5.0),
        lat_max=bb.get("lat_max", 15.0),
        lon_min=bb.get("lon_min", 25.0),
        lon_max=bb.get("lon_max", 45.0),
    )

    # Build processing config
    proc_raw = raw.get("processing", {})
    processing = ProcessingConfig(
        chunks=proc_raw.get("chunks", {"time": 100, "latitude": 50, "longitude": 50}),
        resample_freq=proc_raw.get("resample_freq"),
        agg_methods=proc_raw.get("agg_methods", {}),
    )

    # Build output config
    out_raw = raw.get("output", {})
    output = OutputConfig(
        dir=PROJECT_ROOT / out_raw.get("dir", "data/processed"),
        format=out_raw.get("format", "both"),
    )

    config = PipelineConfig(
        azure=azure,
        selection=selection,
        processing=processing,
        output=output,
        raw_dir=PROJECT_ROOT / "data" / "raw",
    )

    logger.debug("Config loaded: blob_files=%s, variables=%s", selection.blob_files, selection.variables)
    return config

"""Azure Blob Storage access for ERA5 GRIB files.

Handles authentication, listing blobs, and downloading files with local caching.
Other modules never interact with Azure directly.
"""

import logging
import zipfile
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContainerClient

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def get_container_client(config: PipelineConfig) -> ContainerClient:
    """Create an authenticated ContainerClient for the configured container.

    Uses connection string if available, otherwise falls back to account name + key.
    """
    az = config.azure

    if az.connection_string:
        client = BlobServiceClient.from_connection_string(az.connection_string)
        logger.info("Authenticated via connection string")
    elif az.account_name and az.account_key:
        account_url = f"https://{az.account_name}.blob.core.windows.net"
        client = BlobServiceClient(account_url=account_url, credential=az.account_key)
        logger.info("Authenticated via account name + key")
    else:
        raise ValueError(
            "No Azure credentials found. "
            "Set AZURE_STORAGE_CONNECTION_STRING or both "
            "AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY in your .env file."
        )

    return client.get_container_client(az.container_name)


def list_blobs(config: PipelineConfig) -> list[str]:
    """List all blob names under the configured prefix."""
    container = get_container_client(config)
    prefix = config.azure.blob_prefix
    blobs = [b.name for b in container.list_blobs(name_starts_with=prefix)]
    logger.info("Found %d blobs under prefix '%s'", len(blobs), prefix)
    return blobs


def resolve_blob_name(config: PipelineConfig, blob_file: str) -> str:
    """Build the expected blob name using the configured pattern.

    Args:
        config: Pipeline configuration.
        blob_file: File identifier, e.g. "2018-2019".
    """
    filename = config.azure.blob_pattern.format(year=blob_file)
    prefix = config.azure.blob_prefix
    if prefix:
        return f"{prefix}{filename}"
    return filename


def download_blob(
    config: PipelineConfig, blob_name: str, dest_dir: Path | None = None
) -> Path:
    """Download a single blob to the local raw data directory.

    Skips download if the file already exists locally with matching size.

    Args:
        config: Pipeline configuration.
        blob_name: Full blob name (prefix + filename).
        dest_dir: Local directory to save to. Defaults to config.raw_dir.

    Returns:
        Path to the downloaded local file.
    """
    dest_dir = dest_dir or config.raw_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Use just the filename portion for local storage
    local_filename = Path(blob_name).name
    local_path = dest_dir / local_filename

    container = get_container_client(config)
    blob_client = container.get_blob_client(blob_name)
    blob_props = blob_client.get_blob_properties()
    remote_size = blob_props.size

    # Skip if already cached with correct size
    if local_path.exists() and local_path.stat().st_size == remote_size:
        logger.info("Skipping download, cached: %s (%d bytes)", local_path.name, remote_size)
        return local_path

    # Resume support: if partial file exists, continue from where it left off
    existing_size = local_path.stat().st_size if local_path.exists() else 0

    if existing_size > 0 and existing_size < remote_size:
        logger.info(
            "Resuming download of %s from %.2f GB / %.2f GB ...",
            blob_name,
            existing_size / 1e9,
            remote_size / 1e9,
        )
        with open(local_path, "ab") as f:
            stream = blob_client.download_blob(offset=existing_size, length=remote_size - existing_size)
            stream.readinto(f)
    else:
        logger.info("Downloading %s (%.2f GB) ...", blob_name, remote_size / 1e9)
        with open(local_path, "wb") as f:
            stream = blob_client.download_blob()
            stream.readinto(f)

    logger.info("Download complete: %s", local_path)
    return local_path


def _unzip_blob(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract a downloaded ZIP file and return paths to the contained data files.

    Each ZIP is extracted into its own subdirectory (named after the ZIP stem)
    to prevent filename collisions when multiple ZIPs share identical internal names
    (e.g. every month ZIP contains data_stream-oper_stepType-instant.nc).
    Extraction is skipped per-member if the file already exists with matching size.

    Args:
        zip_path: Path to the local .zip file.
        dest_dir: Parent directory; each ZIP gets its own subdirectory here.

    Returns:
        List of paths to extracted NetCDF (.nc) or GRIB data files.
    """
    DATA_EXTENSIONS = {".nc", ".nc4", ".grib", ".grib2", ".grb", ".grb2"}

    # Extract into a dedicated subdirectory so different ZIPs never overwrite each other
    zip_subdir = dest_dir / zip_path.stem
    zip_subdir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        logger.info(
            "ZIP %s contains %d file(s): %s",
            zip_path.name,
            len(members),
            [m.filename for m in members],
        )

        extracted = []
        for member in members:
            out_path = zip_subdir / member.filename
            if out_path.exists() and out_path.stat().st_size == member.file_size:
                logger.info("Skipping extraction, already exists: %s", member.filename)
            else:
                zf.extract(member, zip_subdir)
                logger.info(
                    "Extracted: %s (%.2f MB)",
                    member.filename,
                    member.file_size / 1e6,
                )
            extracted.append(out_path)

    data_files = [p for p in extracted if p.suffix.lower() in DATA_EXTENSIONS]
    if not data_files:
        # Fallback: return all extracted files if no recognised extension found
        logger.warning(
            "No recognised data extensions (.nc, .grib, etc.) in %s. "
            "Treating all extracted files as data: %s",
            zip_path.name,
            [p.name for p in extracted],
        )
        data_files = extracted

    logger.info(
        "Unzip complete for %s -> %s/: %d file(s): %s",
        zip_path.name,
        zip_subdir.name,
        len(data_files),
        [p.name for p in data_files],
    )
    return data_files


def download_selected_blobs(config: PipelineConfig) -> list[Path]:
    """Download ZIP archives for all configured blob_files and extract GRIB contents.

    Each ZIP is downloaded (with resume/cache support) then extracted.
    Returns the extracted GRIB file paths, which are passed directly to the loader.

    Returns:
        List of local GRIB file paths extracted from the downloaded ZIP archives.
    """
    grib_paths = []
    for blob_file in config.selection.blob_files:
        blob_name = resolve_blob_name(config, blob_file)
        logger.info("Fetching blob: %s", blob_name)
        local_zip = download_blob(config, blob_name)
        extracted = _unzip_blob(local_zip, config.raw_dir)
        grib_paths.extend(extracted)

    logger.info(
        "Prepared %d data file(s) from %d ZIP archive(s): %s",
        len(grib_paths),
        len(config.selection.blob_files),
        [p.name for p in grib_paths],
    )
    return grib_paths
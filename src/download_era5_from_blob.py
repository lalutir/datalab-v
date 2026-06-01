"""
Download ERA5 GenCast input files from Azure Blob to the pod.

Run this on the pod BEFORE preprocess_gencast_inputs.py.

Usage:
  pip install azure-storage-blob
  export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
  cd /root/graphcast
  python src/download_era5_from_blob.py
"""

import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient

CONN_STR   = os.environ['AZURE_STORAGE_CONNECTION_STRING']
CONTAINER  = 'model-artifacts'
BLOB_PREFIX = 'gencast_inputs/'
LOCAL_DIR  = Path('/root/graphcast/era5_gencast_inputs')

LOCAL_DIR.mkdir(exist_ok=True)

client    = BlobServiceClient.from_connection_string(CONN_STR)
container = client.get_container_client(CONTAINER)

blobs = list(container.list_blobs(name_starts_with=BLOB_PREFIX))
print(f"Found {len(blobs)} files in '{BLOB_PREFIX}'")

for blob in blobs:
    local_name = Path(blob.name).name  # strip the prefix
    local_path = LOCAL_DIR / local_name
    if local_path.exists():
        print(f"  exists  {local_name}")
        continue
    print(f"  downloading  {local_name} ...", end=' ', flush=True)
    with open(local_path, 'wb') as f:
        container.download_blob(blob.name).readinto(f)
    print(f"{local_path.stat().st_size / 1e6:.1f} MB")

print(f"\nDone. Files in {LOCAL_DIR}")

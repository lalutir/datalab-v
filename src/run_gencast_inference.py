"""
GenCast 1-degree inference — runs on the GPU pod.

For each init date:
  1. Loads preprocessed gencast_input_YYYYMMDD.nc
  2. Runs 8-member ensemble forecast (20 steps x 12h = 10-day forecast)
  3. Extracts Jijiga grid point (42.75E, 9.25N)
  4. Converts 12-hourly to daily tp and t2m
  5. Saves to gencast_outputs/gencast_jijiga_YYYYMMDD.nc
  6. Uploads to Azure Blob model-artifacts/gencast_outputs/

Run:
  cd /root/graphcast
  export AZURE_STORAGE_CONNECTION_STRING="..."
  python src/run_gencast_inference.py
"""

import dataclasses
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import haiku as hk
import jax
import numpy as np
import xarray

sys.path.insert(0, '/root/graphcast')
from graphcast import rollout, xarray_jax, normalization, checkpoint, data_utils
from graphcast import gencast, nan_cleaning

CHECKPOINT_PATH = Path('/root/gencast_model/params/GenCast 1p0deg <2019.npz')
STATS_DIR       = Path('/root/gencast_model/stats')
INPUT_DIR       = Path('/root/graphcast/era5_gencast_inputs')
OUTPUT_DIR      = Path('/root/graphcast/gencast_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

CONN_STR        = os.environ.get('AZURE_STORAGE_CONNECTION_STRING', '')
BLOB_CONTAINER  = 'model-artifacts'

INIT_DATES = [
    "2023-01-15",  # dry season
    "2023-03-01",  # 14-day lead before March 2023 EMDAT flood
    "2023-04-01",  # April rainy season onset
    "2023-07-01",  # long dry season
    "2023-10-15",  # El Nino heavy rains onset
    "2024-01-15",  # dry season
    "2024-04-01",  # April rainy season
    "2024-10-01",  # October rainy season
]

LAT, LON = 9.25, 42.75       # Jijiga grid point
NUM_ENSEMBLE_MEMBERS = 8      # multiple of num devices (1 GPU here)


# ── Load checkpoint ───────────────────────────────────────────────────────────
print("Loading GenCast 1deg checkpoint...")
with open(CHECKPOINT_PATH, 'rb') as f:
    ckpt = checkpoint.load(f, gencast.CheckPoint)

params                       = ckpt.params
state                        = {}   # GenCast inference uses empty state
task_config                  = ckpt.task_config
sampler_config               = ckpt.sampler_config
noise_config                 = ckpt.noise_config
noise_encoder_config         = ckpt.noise_encoder_config
denoiser_architecture_config = ckpt.denoiser_architecture_config

# The checkpoint uses splash_attention (TPU-only). Recursively swap it to
# triblockdiag_mha so GenCast runs on GPU. This must happen before GenCast()
# is constructed because Transformer.__init__ computes the mask differently
# depending on attention_type (see sparse_transformer.py lines 509 vs 517).
def _swap_to_gpu_attention(cfg):
    if not dataclasses.is_dataclass(cfg):
        return cfg
    changes = {}
    for field in dataclasses.fields(cfg):
        val = getattr(cfg, field.name)
        if field.name == 'attention_type' and val == 'splash_mha':
            changes[field.name] = 'triblockdiag_mha'
        elif dataclasses.is_dataclass(val):
            new_val = _swap_to_gpu_attention(val)
            if new_val is not val:
                changes[field.name] = new_val
    return dataclasses.replace(cfg, **changes) if changes else cfg

denoiser_architecture_config = _swap_to_gpu_attention(denoiser_architecture_config)
print("Checkpoint loaded:", ckpt.description[:100])


# ── Load normalization stats ──────────────────────────────────────────────────
print("Loading normalization stats...")
diffs_stddev_by_level = xarray.load_dataset(STATS_DIR / 'diffs_stddev_by_level.nc').compute()
mean_by_level         = xarray.load_dataset(STATS_DIR / 'mean_by_level.nc').compute()
stddev_by_level       = xarray.load_dataset(STATS_DIR / 'stddev_by_level.nc').compute()
min_by_level          = xarray.load_dataset(STATS_DIR / 'min_by_level.nc').compute()


# ── Build predictor ───────────────────────────────────────────────────────────
def construct_wrapped_gencast():
    predictor = gencast.GenCast(
        sampler_config=sampler_config,
        task_config=task_config,
        denoiser_architecture_config=denoiser_architecture_config,
        noise_config=noise_config,
        noise_encoder_config=noise_encoder_config,
    )
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )
    predictor = nan_cleaning.NaNCleaner(
        predictor=predictor,
        reintroduce_nans=True,
        fill_value=min_by_level,
        var_to_clean='sea_surface_temperature',
    )
    return predictor


@hk.transform_with_state
def run_forward(inputs, targets_template, forcings):
    predictor = construct_wrapped_gencast()
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


run_forward_jitted = jax.jit(
    lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
)
run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")
print(f"Predictor built. Devices: {jax.local_devices()}")
print("(First inference run includes JIT compilation — expect ~5 min extra for the first date)\n")


# ── Per-date inference ────────────────────────────────────────────────────────
def run_inference(date_str: str) -> xarray.Dataset:
    input_path = INPUT_DIR / f"gencast_input_{date_str}.nc"
    print(f"  Loading {input_path.name}...")
    batch = xarray.load_dataset(input_path).compute()

    n_targets = batch.dims['time'] - 2
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        batch,
        target_lead_times=slice("12h", f"{n_targets * 12}h"),
        **dataclasses.asdict(task_config),
    )
    print(f"  inputs={dict(eval_inputs.dims)}  targets={dict(eval_targets.dims)}")

    rngs = np.stack(
        [jax.random.fold_in(jax.random.PRNGKey(0), i) for i in range(NUM_ENSEMBLE_MEMBERS)],
        axis=0,
    )

    chunks = []
    for step, chunk in enumerate(rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=run_forward_pmap,
        rngs=rngs,
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
        num_steps_per_chunk=1,
        num_samples=NUM_ENSEMBLE_MEMBERS,
        pmap_devices=jax.local_devices(),
    )):
        chunks.append(chunk)
        if (step + 1) % 5 == 0:
            print(f"    step {step + 1}/{n_targets}")

    return xarray.combine_by_coords(chunks)


def to_jijiga_daily(predictions: xarray.Dataset, date_str: str) -> xarray.Dataset:
    """Extract Jijiga point and aggregate 12-hourly steps to daily values."""
    pt = predictions.sel(lat=LAT, lon=LON, method='nearest')

    # Drop batch dimension (size 1) so values are 1-D (sample,) not (sample, batch)
    if 'batch' in pt.dims:
        pt = pt.isel(batch=0)

    keep = [v for v in ['total_precipitation_12hr', '2m_temperature'] if v in pt.data_vars]
    pt = pt[keep]

    n_steps = pt.sizes['time']
    n_days  = n_steps // 2
    dt0     = datetime.strptime(date_str, "%Y-%m-%d")

    dates, tp_rows, t2m_rows = [], [], []
    for d in range(n_days):
        i0, i1 = d * 2, d * 2 + 1
        dates.append(np.datetime64(dt0 + timedelta(days=d + 1), 'D'))

        if 'total_precipitation_12hr' in pt.data_vars:
            # sum 2 x 12h periods to get 24h accumulation (metres/day)
            tp_rows.append(
                pt['total_precipitation_12hr'].isel(time=i0).values +
                pt['total_precipitation_12hr'].isel(time=i1).values
            )
        if '2m_temperature' in pt.data_vars:
            # mean of 2 x 12-hourly snapshots (Kelvin)
            t2m_rows.append((
                pt['2m_temperature'].isel(time=i0).values +
                pt['2m_temperature'].isel(time=i1).values
            ) / 2.0)

    samples = list(range(NUM_ENSEMBLE_MEMBERS))
    ds = xarray.Dataset(coords={'date': dates, 'sample': samples})

    if tp_rows:
        ds['tp_daily_m'] = xarray.DataArray(
            np.array(tp_rows), dims=['date', 'sample'],
            attrs={'units': 'm/day', 'long_name': '24h total precipitation'}
        )
    if t2m_rows:
        ds['t2m_daily_K'] = xarray.DataArray(
            np.array(t2m_rows), dims=['date', 'sample'],
            attrs={'units': 'K', 'long_name': '2m temperature daily mean'}
        )

    return ds


def upload(path: Path) -> None:
    if not CONN_STR:
        print(f"    AZURE_STORAGE_CONNECTION_STRING not set — skipping upload for {path.name}")
        return
    from azure.storage.blob import BlobServiceClient
    client = BlobServiceClient.from_connection_string(CONN_STR)
    blob_name = f"gencast_outputs/{path.name}"
    with open(path, 'rb') as f:
        client.get_container_client(BLOB_CONTAINER).upload_blob(blob_name, f, overwrite=True)
    print(f"    Uploaded -> {blob_name}")


# ── Main loop ─────────────────────────────────────────────────────────────────
for date_str in INIT_DATES:
    out_path = OUTPUT_DIR / f"gencast_jijiga_{date_str}.nc"
    if out_path.exists():
        print(f"{date_str}: output exists, skipping.")
        continue

    print(f"\n{'='*55}")
    print(f"Init date: {date_str}")

    preds  = run_inference(date_str)
    daily  = to_jijiga_daily(preds, date_str)

    daily.to_netcdf(out_path)
    size_kb = out_path.stat().st_size / 1e3
    print(f"  Saved -> {out_path.name}  ({size_kb:.1f} KB)")
    upload(out_path)

    del preds  # free GPU memory between runs

print("\nAll init dates complete.")

# Project Overview

This project trains an osu!mania (7k) next-event baseline model from beatmap/audio-derived features, then evaluates and runs inference with the same frozen vocabulary and cache conventions.

## 1) Project Goal

- Build a reproducible training pipeline for osu!mania 7k.
- Produce stable artifacts: dataset splits, cache shards, vocab snapshot, checkpoints, metrics, and inference outputs.

## 2) Repository Structure

- `configs/`: centralized runtime configs (`train.yaml`, `train.smoke.yaml`)
- `scripts/`: pipeline stages (`split_dataset`, `build_train_cache`, `build_vocab`, `eval_minimal`, `infer_minimal`)
- `data/osu/`: raw dataset placement directory (local only)
- `artifacts/`, `artifacts_smoke/`, `outputs/`, `outputs_smoke/`: generated outputs (not source)
- `tests/`: unit/integration tests
- Root training/runtime modules: `train.py`, `train_data.py`, `train_runtime.py`, etc.

## 3) Environment Requirements

- OS: Windows (PowerShell examples below)
- Python: 3.11+ recommended
- GPU (optional but recommended): CUDA-compatible setup for `train.device: "cuda"`
- Disk/Memory: enough free space for cache shards and checkpoints

## 4) Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is not present yet, create one before migration so the new machine can install the same dependency set.

## 5) Data Preparation

1. Put training dataset files under `data/osu/`.
2. Keep generated files out of source control (`data/cache*`, `artifacts*`, `outputs*`).
3. Quick sanity check:

```powershell
Get-ChildItem data\osu
```

## 6) Run Workflows

### Smoke Validation (fast environment check)

```powershell
$env:TRAIN_CONFIG_PATH = "configs/train.smoke.yaml"
python build_manifest.py
python scripts/split_dataset.py
python scripts/build_train_cache.py
python scripts/build_vocab.py
python train.py
python scripts/eval_minimal.py
python scripts/infer_minimal.py
Remove-Item Env:TRAIN_CONFIG_PATH -ErrorAction SilentlyContinue
```

### Full Training (official baseline)

```powershell
# 1) Ensure smoke config is not used
Remove-Item Env:TRAIN_CONFIG_PATH -ErrorAction SilentlyContinue

# 2) Regenerate manifest
python build_manifest.py

# 3) Rebuild train/val split (seed=42)
python scripts/split_dataset.py

# 4) Rebuild train cache on full data
python scripts/build_train_cache.py

# 5) Fit and freeze full vocab
python scripts/build_vocab.py

# 6) Start full baseline training
python train.py
```

## 7) Outputs and Logs

Typical locations (from `configs/train.yaml`):

- Splits: `artifacts/splits/`
- Vocab snapshot: `artifacts/vocab.json`
- Training run dir: `artifacts/train/`
- Checkpoints: `artifacts/train/checkpoints/{best.pt,last.pt}`
- Metrics: `artifacts/train/metrics.json`
- Train log: `artifacts/train/train_log.jsonl`
- Eval outputs: `artifacts/metrics/`
- Infer outputs: `artifacts/infer/`

## 8) Common Issues

- `TRAIN_CONFIG_PATH` still points to smoke config:
  - Clear it before full run:
  - `Remove-Item Env:TRAIN_CONFIG_PATH -ErrorAction SilentlyContinue`
- Cache rebuild blocked:
  - Check `cache.overwrite` in config.
- Vocab rebuild blocked:
  - Check vocab overwrite/frozen settings in config.
- CUDA unavailable:
  - Set `train.device` to `cpu` or fix CUDA/PyTorch install.
- Disk/memory pressure during cache/train:
  - Free space first; cache shards and checkpoints can grow quickly.

## Publish Rules (Git Hygiene)

- Do not commit generated outputs or local caches:
  - `data/cache*`, `artifacts*`, `outputs*`, `__pycache__`, `.tmp_test_*`
- Keep `.gitignore` aligned with this rule before every commit.

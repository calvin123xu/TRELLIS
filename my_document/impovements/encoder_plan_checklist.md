# Encoder Plan Checklist (ABO First)

## Scope
- Current target: ABO encoder-only fine-tune on visible-only features.
- Toys4k encoder-only fine-tune is prepared but deferred until you confirm Toys4k aggregation is complete.

## Implemented Files
- Training freeze hook:
  - `train.py` (supports `freeze_models` in config)
- ABO config:
  - `configs/finetune/abo_encoder_only_visible_only.json`
- ABO run scripts:
  - `my_scripts/encoder_plan/abo_encoder_only_finetune.sbatch`
  - `my_scripts/encoder_plan/submit_abo_encoder_only_finetune.sh`
- Shared pretrained-export helper:
  - `my_scripts/encoder_plan/prepare_pretrained_slat_ckpts.py`
- Toys4k deferred config and scripts (already prepared):
  - `configs/finetune/toys4k_encoder_only_visible_only.json`
  - `my_scripts/encoder_plan/toys4k_encoder_only_finetune.sbatch`
  - `my_scripts/encoder_plan/submit_toys4k_encoder_only_finetune.sh`

## Workload Estimate
- Dataset sizes observed in your pipeline:
  - ABO visible-only features: ~4485 objects
  - Toys4k visible-only features: ~3229 objects
- Planned training length:
  - `max_steps=120000` (encoder-only)
- Practical runtime envelope on 1 GPU with batch_size_per_gpu=1, batch_split=4:
  - Roughly 18 to 60 hours depending on GPU class, I/O, and elastic memory behavior.

## Hardware Estimate and Safeguards
- Requested for both ABO/Toys4k encoder-only stage scripts:
  - `--gpus=1`, `--cpus-per-task=8`, `--mem=64G`
- Memory-risk mitigation configured:
  - `batch_size_per_gpu=1`
  - `batch_split=4`
  - decoder frozen (`freeze_models=["decoder"]`)
- Why this is safer:
  - Cuts activation memory versus previous batch=2 setting.
  - Keeps renderer/image resolution unchanged for result comparability.

## Pre-Run Validation Checklist (ABO)
1. Feature cache exists:
   - `datasets/ABO_full/features/dinov2_vitl14_reg_visible_only/*.npz`
2. Config exists:
   - `configs/finetune/abo_encoder_only_visible_only.json`
3. Pretrained export path writable:
   - `/speed-scratch/an_zhan/project/TRELLIS-experiments/pretrained_ckpts`
4. Training output path writable:
   - `/speed-scratch/an_zhan/project/TRELLIS-experiments/results/encoder_plan_abo/<RUN_TAG>`
5. No conflicts with running jobs:
   - Encoder plan uses independent job names and output directories.

## If OOM Still Happens
- First action: increase `batch_split` from 4 to 8.
- Second action: reduce `image_size` from 512 to 384 in finetune config (quality/metric comparability trade-off).
- Third action: move to larger-memory GPU partition if available.

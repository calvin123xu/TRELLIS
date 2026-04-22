# Toys4k Encoder Plan (Encoder-Only Fine-Tune)

## Goal
Fine-tune only the structured-latent encoder on Toys4k using visible-only aggregated features, while keeping the decoder frozen.

## Files
- Config: `configs/finetune/toys4k_encoder_only_visible_only.json`
- Pretrained converter: `my_scripts/encoder_plan/prepare_pretrained_slat_ckpts.py`
- Slurm stage: `my_scripts/encoder_plan/toys4k_encoder_only_finetune.sbatch`
- Submit wrapper: `my_scripts/encoder_plan/submit_toys4k_encoder_only_finetune.sh`

## Notes
- Decoder is frozen via `freeze_models: ["decoder"]` in config.
- Pretrained encoder/decoder `.pt` checkpoints are prepared under `pretrained_ckpts/` from official TRELLIS checkpoints.
- Training input uses `datasets/Toys4k/features/dinov2_vitl14_reg_visible_only`.
- This plan does not interfere with previously submitted jobs.

## Submit (when ready)
```bash
cd /speed-scratch/an_zhan/project/TRELLIS-experiments
chmod +x my_scripts/encoder_plan/submit_toys4k_encoder_only_finetune.sh
bash my_scripts/encoder_plan/submit_toys4k_encoder_only_finetune.sh encoder_plan_toys4k_001
```

## ZBUF Optimization Candidates (Non-Collapsing)

Root issue from diagnostics: strict nearest-only zbuf collapses support (very high zero-row fraction), so candidate configs should preserve more per-voxel support.

### Candidate A (recommended first)
- `zbuf_selection=topk`
- `zbuf_topk=3`
- `zbuf_depth_tolerance=5e-3`
- `zbuf_fallback_mode=visible_only`
- `zbuf_min_visibility_count=1.0`
- Suggested feature name: `dinov2_vitl14_reg_visible_zbuf_topk3_fb1`

### Candidate B (more conservative)
- `zbuf_selection=topk`
- `zbuf_topk=2`
- `zbuf_depth_tolerance=1e-2`
- `zbuf_fallback_mode=visible_only`
- `zbuf_min_visibility_count=0.5`
- Suggested feature name: `dinov2_vitl14_reg_visible_zbuf_topk2_fb05`

### Candidate C (soft weighting)
- `zbuf_selection=soft`
- `zbuf_soft_tau=2e-2`
- `zbuf_depth_tolerance=0` (unused by soft mode)
- `zbuf_fallback_mode=visible_only`
- `zbuf_min_visibility_count=0.5`
- Suggested feature name: `dinov2_vitl14_reg_visible_zbuf_soft_tau2e2_fb05`

### Run Template (aggregation only; no submission here)
```bash
python dataset_toolkits/aggregate_features.py \
	--output_dir datasets/Toys4k \
	--model dinov2_vitl14_reg \
	--aggregation_mode visible_only_zbuf \
	--zbuf_selection topk \
	--zbuf_topk 3 \
	--zbuf_depth_tolerance 5e-3 \
	--zbuf_fallback_mode visible_only \
	--zbuf_min_visibility_count 1.0 \
	--output_feature_name dinov2_vitl14_reg_visible_zbuf_topk3_fb1
```

### Fast Selection Rule
- Keep only candidates with `visible_zbuf_zero_frac` close to `visible_only_zero_frac` (target near 0.0).
- Among those, choose highest `cos(visible_only, candidate_zbuf)` while preserving validation metrics.

#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash my_scripts/run_eval_abo_0330000_toy.sh

ROOT="/nfs/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS"
OUT="${ROOT}/outputs/finetune_vis_abo_0330000"
DATASET="${ROOT}/datasets/Toys4k_small"

ENC_CKPT="${OUT}/ckpts/encoder_ema0.9999_step0330000.pt"
DEC_CKPT="${OUT}/ckpts/decoder_ema0.9999_step0330000.pt"
FEAT_SUBDIR="dinov2_vitl14_reg_vis_weighted"
LATENT_SUBDIR="dinov2_vitl14_reg_vis_weighted_abo_ft_step330k"
GAUSS_SUBDIR="gaussians_decoded_vis_weighted_abo_ft_step330k"

printf '\n[1/4] Checking required paths...\n'
ls -lh "${ENC_CKPT}"
ls -lh "${DEC_CKPT}"
ls -lh "${DATASET}/features"

cd "${ROOT}"

printf '\n[2/4] Encoding latents with ABO finetuned encoder (step 330k)...\n'
python dataset_toolkits/encode_latent.py \
  --output_dir "${DATASET}" \
  --feat_model dinov2_vitl14_reg \
  --feature_subdir "${FEAT_SUBDIR}" \
  --enc_ckpt_path "${ENC_CKPT}" \
  --enc_model finetune_vis_abo_0330000 \
  --model_root "${ROOT}/outputs" \
  --latent_subdir "${LATENT_SUBDIR}"

printf '\n[3/4] Building metadata and decoding gaussians...\n'
python dataset_toolkits/build_metadata.py Toys4k --output_dir "${DATASET}"

python my_scripts/decode_latents_to_gaussian.py \
  --output_dir "${DATASET}" \
  --latent_model "${LATENT_SUBDIR}" \
  --dec_ckpt_path "${DEC_CKPT}" \
  --dec_model_dir "${OUT}" \
  --save_dir "${GAUSS_SUBDIR}"

printf '\n[4/4] Evaluating rendered metrics...\n'
python my_scripts/eval_render_metrics.py \
  --output_dir "${DATASET}" \
  --gauss_dir "${DATASET}/${GAUSS_SUBDIR}" \
  --resolution 512 \
  --bg 0,0,0

printf '\nDone.\n'

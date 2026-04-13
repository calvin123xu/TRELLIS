#!/encs/bin/bash
#SBATCH --job-name=trellis_abo_agg_visw_eval_toy
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=trellis_abo_agg_visw_eval_toy_%j.out
#SBATCH --error=trellis_abo_agg_visw_eval_toy_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=x_qiaoyu@speed.encs.concordia.ca

set -euo pipefail

echo "=================================================="
echo "Job ID        : ${SLURM_JOB_ID:-N/A}"
echo "Array Job ID  : ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array Task ID : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node          : $(hostname)"
echo "Start time    : $(date)"
echo "=================================================="

# Go to project root
ROOT="/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS"
cd "${ROOT}"

# Optional: load temp cache envs if available
if [ -f /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh ]; then
    source /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh
fi

# Enable conda
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/qiaoyu/env/trellis

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

printf '\n[2/4] Encoding latents with ABO finetuned encoder (step 330k)...\n'
python dataset_toolkits/encode_latent.py \
  --output_dir "${DATASET}" \
  --feat_model dinov2_vitl14_reg \
  --feature_subdir "${FEAT_SUBDIR}" \
  --enc_ckpt_path "${ENC_CKPT}" \
  --enc_model finetune_vis_abo_0330000 \
  --model_root "${ROOT}/outputs" \
  --latent_subdir "${LATENT_SUBDIR}"

printf '\n[3/4] decoding gaussians...\n'
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

printf '\nDone. End time: %s\n' "$(date)"

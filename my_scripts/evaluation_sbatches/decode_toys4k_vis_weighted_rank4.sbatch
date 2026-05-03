#!/encs/bin/bash
# 4-way parallel Gaussian decoding from encoded latents.
#
# Submit:
#   sbatch my_scripts/decode_toys4k_vis_weighted_rank4.sbatch
#
# It launches ranks 0..3 via SLURM_ARRAY_TASK_ID and each rank decodes
# its own metadata partition.

#SBATCH --job-name=trellis_toys4k_dec_visw_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=trellis_dec_visw_r4_%A_%a.out
#SBATCH --error=trellis_dec_visw_r4_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=x_qiaoyu@speed.encs.concordia.ca

set -euo pipefail

echo "HOST: $(hostname)"
echo "START: $(date)"
nvidia-smi

# Conda
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/qiaoyu/env/trellis

# Cache
if [ -f /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh ]; then
    source /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh
fi

# Job-local temp directory on compute-node local disk
JOB_TMPDIR="/tmp/${USER}/trellis_decode_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_TMPDIR"

export TMPDIR="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"

echo "TMPDIR=${TMPDIR}"
df -h "$TMPDIR" || true

cleanup_tmpdir() {
    echo "Cleaning TMPDIR: $JOB_TMPDIR"
    rm -rf "$JOB_TMPDIR" 2>/dev/null || true
}
trap cleanup_tmpdir EXIT

# Runtime env
export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /speed-scratch/qiaoyu/speed-hpc/project/TRELLIS

echo "Python: $(which python)"
python --version

# Dataset / latent variables
OUTPUT_DIR="/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS/datasets/Toys4k_small"
LATENT_MODEL="dinov2_vitl14_reg_vis_weighted_slat_enc_swin8_B_64l8_fp16"

# Decoder checkpoint variables
DEC_MODEL_DIR="/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS/outputs/finetune_vis_abo_original_config"
DEC_CKPT_PATH="${DEC_MODEL_DIR}/ckpts/decoder_ema0.9999_step0080000.pt"

# Output Gaussian folder
SAVE_DIR="gaussians_decoded_${LATENT_MODEL}_dec_ft_step80k"

WORLD_SIZE=4
RANK="${SLURM_ARRAY_TASK_ID}"

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LATENT_MODEL=${LATENT_MODEL}"
echo "DEC_MODEL_DIR=${DEC_MODEL_DIR}"
echo "DEC_CKPT_PATH=${DEC_CKPT_PATH}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}"

python my_scripts/decode_latents_to_gaussian.py \
  --output_dir "${OUTPUT_DIR}" \
  --latent_model "${LATENT_MODEL}" \
  --dec_ckpt_path "${DEC_CKPT_PATH}" \
  --dec_model_dir "${DEC_MODEL_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --rank "${RANK}" \
  --world_size "${WORLD_SIZE}"

echo "END: $(date)"

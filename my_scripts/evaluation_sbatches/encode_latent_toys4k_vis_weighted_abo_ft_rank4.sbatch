#!/encs/bin/bash
# 4-way parallel latent encoding from vis_weighted aggregated features
# using ABO finetuned encoder checkpoint.
#
# Submit as array:
#   sbatch my_scripts/evaluation_sbatches/encode_latent_toys4k_vis_weighted_abo_ft_rank4.sbatch
#
# It launches ranks 0..3 via SLURM_ARRAY_TASK_ID and each rank encodes
# latents for its own metadata partition.

#SBATCH --job-name=trellis_toys4k_enc_visw_aboft_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=trellis_enc_visw_aboft_r4_%A_%a.out
#SBATCH --error=trellis_enc_visw_aboft_r4_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=x_qiaoyu@speed.encs.concordia.ca

set -euo pipefail

echo "HOST: $(hostname)"
echo "START: $(date)"

# Conda
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/qiaoyu/env/trellis

# Cache
if [ -f /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh ]; then
    source /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh
fi

# Job-local temp directory
JOB_TMPDIR="/tmp/${USER}/trellis_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMPDIR"
export TMPDIR="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"

cleanup_tmpdir() {
    echo "Cleaning TMPDIR: $JOB_TMPDIR"
    rm -rf "$JOB_TMPDIR" 2>/dev/null || true
}
trap cleanup_tmpdir EXIT

TRELLIS_ROOT="/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS"
OUTPUT_DIR="${TRELLIS_ROOT}/datasets/Toys4k_small"

RUN_DIR="${TRELLIS_ROOT}/outputs/finetune_vis_weighted_abo/visw_ft_20260503_191609"
CKPT_DIR="${RUN_DIR}/ckpts"
ENC_CKPT="${CKPT_DIR}/encoder_ema0.9999_step0120000.pt"
ENC_CONFIG="${RUN_DIR}/config.json"

FEAT_MODEL="dinov2_vitl14_reg"
FEATURE_SUBDIR="dinov2_vitl14_reg_vis_weighted"
LATENT_SUBDIR="dinov2_vitl14_reg_vis_weighted_abo_ft"

WORLD_SIZE=4
RANK="${SLURM_ARRAY_TASK_ID}"

cd "${TRELLIS_ROOT}"

echo "Python: $(which python)"
python --version

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "ENC_CKPT=${ENC_CKPT}"
echo "ENC_CONFIG=${ENC_CONFIG}"
echo "FEAT_MODEL=${FEAT_MODEL}"
echo "FEATURE_SUBDIR=${FEATURE_SUBDIR}"
echo "LATENT_SUBDIR=${LATENT_SUBDIR}"
echo "WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}"
echo "TMPDIR=${TMPDIR}"

test -f "${ENC_CKPT}" || { echo "Missing checkpoint: ${ENC_CKPT}"; exit 1; }
test -f "${ENC_CONFIG}" || { echo "Missing config: ${ENC_CONFIG}"; exit 1; }
test -f "${OUTPUT_DIR}/metadata.csv" || { echo "Missing metadata.csv: ${OUTPUT_DIR}/metadata.csv"; exit 1; }
test -d "${OUTPUT_DIR}/features/${FEATURE_SUBDIR}" || { echo "Missing feature dir: ${OUTPUT_DIR}/features/${FEATURE_SUBDIR}"; exit 1; }

ls -lh "${ENC_CKPT}"
ls -lh "${ENC_CONFIG}"

python dataset_toolkits/encode_latent.py \
  --output_dir "${OUTPUT_DIR}" \
  --feat_model "${FEAT_MODEL}" \
  --feature_subdir "${FEATURE_SUBDIR}" \
  --enc_ckpt_path "${ENC_CKPT}" \
  --enc_config_path "${ENC_CONFIG}" \
  --latent_subdir "${LATENT_SUBDIR}" \
  --rank "${RANK}" \
  --world_size "${WORLD_SIZE}"

echo "END: $(date)"

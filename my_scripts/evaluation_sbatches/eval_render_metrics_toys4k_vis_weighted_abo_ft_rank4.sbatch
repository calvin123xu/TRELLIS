#!/encs/bin/bash
# 4-way parallel render-metric evaluation for decoded Toys4K Gaussians
# produced from vis_weighted features with the ABO finetuned checkpoint.
#
# Submit as array:
#   sbatch my_scripts/evaluation_sbatches/eval_render_metrics_toys4k_vis_weighted_abo_ft_rank4.sbatch
#
# It launches ranks 0..3 via SLURM_ARRAY_TASK_ID and each rank evaluates
# its own partition of decoded .ply files.

#SBATCH --job-name=trellis_toys4k_eval_visw_aboft_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=trellis_eval_visw_aboft_r4_%A_%a.out
#SBATCH --error=trellis_eval_visw_aboft_r4_%A_%a.err
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

# Job-local temp directory
JOB_TMPDIR="/tmp/${USER}/trellis_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

TRELLIS_ROOT="/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS"
OUTPUT_DIR="${TRELLIS_ROOT}/datasets/Toys4k_small"
GAUSS_DIR="${OUTPUT_DIR}/gaussians_decoded/gaussians_decoded_vis_weighted_abo_ft"
METRICS_DIR="${OUTPUT_DIR}/render_metrics"
OUT_CSV="${METRICS_DIR}/render_metrics_vis_weighted_abo_ft_rank{rank:04d}_of_{num_ranks:04d}.csv"
RESOLUTION=512
BG="0,0,0"
WORLD_SIZE=4
RANK="${SLURM_ARRAY_TASK_ID}"

cd "${TRELLIS_ROOT}"

echo "Python: $(which python)"
python --version

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "GAUSS_DIR=${GAUSS_DIR}"
echo "METRICS_DIR=${METRICS_DIR}"
echo "OUT_CSV=${OUT_CSV}"
echo "RESOLUTION=${RESOLUTION}"
echo "BG=${BG}"
echo "WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}"
echo "TMPDIR=${TMPDIR}"

test -f "${OUTPUT_DIR}/metadata.csv" || { echo "Missing metadata.csv: ${OUTPUT_DIR}/metadata.csv"; exit 1; }
test -d "${OUTPUT_DIR}/renders" || { echo "Missing renders dir: ${OUTPUT_DIR}/renders"; exit 1; }
test -d "${GAUSS_DIR}" || { echo "Missing gaussian dir: ${GAUSS_DIR}"; exit 1; }
mkdir -p "${METRICS_DIR}"

python my_scripts/eval_render_metrics.py \
  --output_dir "${OUTPUT_DIR}" \
  --gauss_dir "${GAUSS_DIR}" \
  --resolution "${RESOLUTION}" \
  --bg "${BG}" \
  --out_csv "${OUT_CSV}" \
  --rank "${RANK}" \
  --num_ranks "${WORLD_SIZE}"

echo "END: $(date)"

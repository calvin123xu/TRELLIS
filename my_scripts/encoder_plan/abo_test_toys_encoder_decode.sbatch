#!/encs/bin/bash
#SBATCH --job-name=abo_test_dec
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --output=abo_test_dec_%j.out
#SBATCH --error=abo_test_dec_%j.err

set -euo pipefail

STAGE_NAME="ABO_TEST_TOYS_ENCODER_DECODE"
RUN_TAG="${RUN_TAG:-abo_test_toysenc_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
ENV_PATH="/speed-scratch/an_zhan/env/trellis"
OUTPUT_DIR="datasets/ABO_full"
FEATURE_MODEL="dinov2_vitl14_reg_visible_only"
TOYS_ENCODER_RUN="${TOYS_ENCODER_RUN:-encoder_plan_toys4k_fix_20260407_002}"
TOYS_ENCODER_CKPT="${TOYS_ENCODER_CKPT:-step0120000}"
LATENT_MODEL="${FEATURE_MODEL}_${TOYS_ENCODER_RUN}_${TOYS_ENCODER_CKPT}"
GAUSS_DIR="${OUTPUT_DIR}/gaussians_decoded_${RUN_TAG}_abo_toysenc_visible_only"
TIMING_DIR="${OUTPUT_DIR}/timing/${RUN_TAG}"
TIMING_FILE="${TIMING_DIR}/stage_timing.csv"

mkdir -p "${TIMING_DIR}"
if [[ ! -f "${TIMING_FILE}" ]]; then
  echo "run_tag,stage,job_id,array_task_id,start_utc,end_utc,duration_sec,status,host" > "${TIMING_FILE}"
fi

STAGE_START_EPOCH=$(date +%s)
STAGE_START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)

log_stage_timing() {
  local status="$1"
  local end_epoch end_iso duration array_task_id
  end_epoch=$(date +%s)
  end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  duration=$((end_epoch - STAGE_START_EPOCH))
  array_task_id="${SLURM_ARRAY_TASK_ID:-NA}"
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${RUN_TAG}" "${STAGE_NAME}" "${SLURM_JOB_ID:-NA}" "${array_task_id}" \
    "${STAGE_START_ISO}" "${end_iso}" "${duration}" "${status}" "$(hostname)" >> "${TIMING_FILE}"
  echo "[TIMING] stage=${STAGE_NAME} status=${status} duration_sec=${duration} run_tag=${RUN_TAG} job_id=${SLURM_JOB_ID:-NA} task_id=${array_task_id}"
}

trap 'rc=$?; if [[ $rc -eq 0 ]]; then log_stage_timing "OK"; else log_stage_timing "FAIL_${rc}"; fi' EXIT

echo "[STAGE] ${STAGE_NAME} started run_tag=${RUN_TAG} job_id=${SLURM_JOB_ID:-NA}"

source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
cd "${REPO_DIR}"

export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers
export SPCONV_ALGO=native

python my_scripts/decode_latents_to_gaussian_from_latent_dir.py \
  --output_dir "${OUTPUT_DIR}" \
  --latent_model "${LATENT_MODEL}" \
  --save_dir "${GAUSS_DIR}"

gauss_count=$(find "${GAUSS_DIR}" -type f -name '*.ply' | wc -l)
if [[ ${gauss_count} -eq 0 ]]; then
  echo "[ERROR] No gaussian ply produced in ${GAUSS_DIR}"
  exit 2
fi

echo "${GAUSS_DIR}" > "${TIMING_DIR}/gaussian_dir.txt"
echo "[DONE] Decoded gaussians saved to: ${GAUSS_DIR}"

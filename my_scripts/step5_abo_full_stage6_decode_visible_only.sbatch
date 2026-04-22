#!/encs/bin/bash
#SBATCH --job-name=abo_vis_decode
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=abo_vis_decode_%j.out
#SBATCH --error=abo_vis_decode_%j.err

set -euo pipefail

STAGE_NAME="STAGE6_DECODE_VISIBLE_ONLY"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
ENV_PATH="/speed-scratch/an_zhan/env/trellis"
OUTPUT_DIR="datasets/ABO_full"
LATENT_MODEL="dinov2_vitl14_reg_visible_only_slat_enc_swin8_B_64l8_fp16"
GAUSS_DIR="${OUTPUT_DIR}/gaussians_decoded_${RUN_TAG}_visible_only"
REF_BASE="${OUTPUT_DIR}_refs/${RUN_TAG}"
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

mkdir -p "${REF_BASE}"

python my_scripts/decode_latents_to_gaussian.py \
  --output_dir "${OUTPUT_DIR}" \
  --latent_model "${LATENT_MODEL}" \
  --save_dir "${GAUSS_DIR}"

cp -f "${OUTPUT_DIR}/metadata.csv" "${REF_BASE}/after_decode_metadata.csv"
printf "%s\n" "${GAUSS_DIR}" > "${REF_BASE}/gaussian_dir.txt"
find "${GAUSS_DIR}" -type f -name '*.ply' | wc -l > "${REF_BASE}/after_decode_gaussian_count.txt"
touch "${REF_BASE}/after_decode_frozen.marker"

#!/encs/bin/bash
#SBATCH --job-name=toys4k_ft_bak
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=toys4k_ft_bak_%j.out
#SBATCH --error=toys4k_ft_bak_%j.err

set -euo pipefail

STAGE_NAME="TOYS4K_BACKUP_FEATURES"
RUN_TAG="${RUN_TAG:-encoder_plan_toys4k_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
OUTPUT_DIR="datasets/Toys4k"
FEATURE_MODEL="dinov2_vitl14_reg_visible_only"
FEATURE_DIR="${OUTPUT_DIR}/features/${FEATURE_MODEL}"
BACKUP_ROOT="${OUTPUT_DIR}/features_backup"
BACKUP_DIR="${BACKUP_ROOT}/${RUN_TAG}/${FEATURE_MODEL}"
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

cd "${REPO_DIR}"

if [[ ! -d "${FEATURE_DIR}" ]]; then
  echo "[ERROR] Missing visible-only feature dir: ${FEATURE_DIR}"
  exit 2
fi

feature_count=$(find "${FEATURE_DIR}" -type f -name '*.npz' | wc -l)
if [[ ${feature_count} -eq 0 ]]; then
  echo "[ERROR] Missing or empty visible-only feature dir: ${FEATURE_DIR}"
  exit 2
fi

mkdir -p "$(dirname "${BACKUP_DIR}")"
mkdir -p "${BACKUP_DIR}"
cp -a "${FEATURE_DIR}/." "${BACKUP_DIR}/"

backup_count=$(find "${BACKUP_DIR}" -type f -name '*.npz' | wc -l)
if [[ ${backup_count} -eq 0 ]]; then
  echo "[ERROR] Backup created but no .npz files found in ${BACKUP_DIR}"
  exit 3
fi

echo "[INFO] Source features: ${feature_count}"
echo "[INFO] Backup features: ${backup_count}"
echo "[DONE] Toys4k feature backup saved to: ${BACKUP_DIR}"

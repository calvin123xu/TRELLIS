#!/encs/bin/bash
#SBATCH --job-name=abo_feat_bak
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=abo_feat_bak_%j.out
#SBATCH --error=abo_feat_bak_%j.err

set -euo pipefail

STAGE_NAME="ABO_BACKUP_VISIBLE_ONLY_FEATURES"
RUN_TAG="${RUN_TAG:-abo_test_toysenc_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
OUTPUT_DIR="datasets/ABO_full"
FEATURE_MODEL="dinov2_vitl14_reg_visible_only"
FEATURE_DIR="${OUTPUT_DIR}/features/${FEATURE_MODEL}"
BACKUP_ROOT="${OUTPUT_DIR}/features_backup_clean"
BACKUP_DIR="${BACKUP_ROOT}/${RUN_TAG}/${FEATURE_MODEL}"
LATEST_LINK="${BACKUP_ROOT}/latest_${FEATURE_MODEL}"
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
  echo "[ERROR] Empty visible-only feature dir: ${FEATURE_DIR}"
  exit 2
fi

mkdir -p "$(dirname "${BACKUP_DIR}")"
if cp -al "${FEATURE_DIR}" "${BACKUP_DIR}" 2>/dev/null; then
  echo "[INFO] Backup created via hardlinks: ${BACKUP_DIR}"
else
  mkdir -p "${BACKUP_DIR}"
  cp -a "${FEATURE_DIR}/." "${BACKUP_DIR}/"
  echo "[INFO] Backup created via full copy: ${BACKUP_DIR}"
fi

chmod -R a-w "${BACKUP_DIR}" || true
ln -sfn "${BACKUP_DIR}" "${LATEST_LINK}"

backup_count=$(find "${BACKUP_DIR}" -type f -name '*.npz' | wc -l)
if [[ ${backup_count} -eq 0 ]]; then
  echo "[ERROR] Backup contains 0 npz files: ${BACKUP_DIR}"
  exit 3
fi

echo "[INFO] Source features: ${feature_count}"
echo "[INFO] Backup features: ${backup_count}"
echo "[DONE] Clean ABO visible-only backup ready: ${BACKUP_DIR}"

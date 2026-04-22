#!/encs/bin/bash
#SBATCH --job-name=hssd_dl
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=hssd_dl_%j.out
#SBATCH --error=hssd_dl_%j.err

set -euo pipefail

STAGE_NAME="HSSD_STAGE1_DOWNLOAD"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
ENV_PATH="/speed-scratch/an_zhan/env/trellis"
OUTPUT_DIR="datasets/HSSD"
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

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

python - <<'PY'
import os
import sys
import huggingface_hub

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token:
    try:
        huggingface_hub.login(token=token, add_to_git_credential=False)
    except Exception:
        pass

try:
    who = huggingface_hub.whoami()
    print(f"[AUTH] Hugging Face authenticated as: {who.get('name', 'unknown')}")
except Exception as exc:
    print("[ERROR] Hugging Face auth required for HSSD. Set HUGGINGFACE_HUB_TOKEN or run huggingface-cli login on SPEED.")
    print(f"[ERROR] {exc}")
    sys.exit(2)
PY

mkdir -p "${OUTPUT_DIR}"
python dataset_toolkits/build_metadata.py HSSD --output_dir "${OUTPUT_DIR}"
python dataset_toolkits/download.py HSSD --output_dir "${OUTPUT_DIR}"
python dataset_toolkits/build_metadata.py HSSD --output_dir "${OUTPUT_DIR}"

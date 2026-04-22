#!/encs/bin/bash
#SBATCH --job-name=toys4k_agg_vis
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=toys4k_agg_vis_%j.out
#SBATCH --error=toys4k_agg_vis_%j.err

set -euo pipefail

STAGE_NAME="TOYS4K_STAGE6_AGG_VISIBLE_ONLY"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
ENV_PATH="/speed-scratch/an_zhan/env/trellis"
OUTPUT_DIR="datasets/Toys4k"
MODEL="dinov2_vitl14_reg"
FEATURE_MODEL="dinov2_vitl14_reg_visible_only"
FEATURE_DIR="${OUTPUT_DIR}/features/${FEATURE_MODEL}"
VIEW_CACHE_DIR="${OUTPUT_DIR}/view_features/${MODEL}"
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

if [[ ! -d "${VIEW_CACHE_DIR}" ]] || [[ $(find "${VIEW_CACHE_DIR}" -type f -name '*.npz' | wc -l) -eq 0 ]]; then
  echo "[ERROR] Missing or empty view-feature cache: ${VIEW_CACHE_DIR}"
  exit 2
fi

python dataset_toolkits/aggregate_features.py \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --aggregation_mode visible_only \
  --visibility_eps 1e-6 \
  --visibility_margin 0.0 \
  --output_feature_name "${FEATURE_MODEL}" \
  --timing_log_interval 50

feature_count=$(find "${FEATURE_DIR}" -type f -name '*.npz' | wc -l)
if [[ "${feature_count}" -eq 0 ]]; then
  echo "[ERROR] Aggregation produced 0 visible-only features in ${FEATURE_DIR}"
  exit 3
fi

python dataset_toolkits/build_metadata.py Toys4k --output_dir "${OUTPUT_DIR}"

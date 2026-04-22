#!/encs/bin/bash
#SBATCH --job-name=toys4k_agg_vzbuf
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=toys4k_agg_vzbuf_%j.out
#SBATCH --error=toys4k_agg_vzbuf_%j.err

set -euo pipefail

STAGE_NAME="TOYS4K_AGG_VISIBLE_ZBUF_FROM_CACHE"
RUN_TAG="${RUN_TAG:-toys4k_vzbuf_agg_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
ENV_PATH="/speed-scratch/an_zhan/env/trellis"
OUTPUT_DIR="datasets/Toys4k"
MODEL="dinov2_vitl14_reg"
FEATURE_MODEL="${FEATURE_MODEL:-dinov2_vitl14_reg_visible_zbuf}"
ZBUF_DEPTH_TOL="${ZBUF_DEPTH_TOL:-1e-3}"
ZBUF_SELECTION="${ZBUF_SELECTION:-nearest}"
ZBUF_TOPK="${ZBUF_TOPK:-1}"
ZBUF_SOFT_TAU="${ZBUF_SOFT_TAU:-5e-3}"
ZBUF_FALLBACK_MODE="${ZBUF_FALLBACK_MODE:-none}"
ZBUF_MIN_VISIBILITY_COUNT="${ZBUF_MIN_VISIBILITY_COUNT:-0.0}"
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

mkdir -p "${FEATURE_DIR}"
find "${FEATURE_DIR}" -type f -name '*.npz' -delete

# Reset feature flags for this model to avoid stale `true` values skipping objects.
python - <<PY
import pandas as pd

path = "${OUTPUT_DIR}/metadata.csv"
col = f"feature_${FEATURE_MODEL}"
df = pd.read_csv(path)
if col in df.columns:
  df[col] = False
else:
  df[col] = False
df.to_csv(path, index=False)
print(f"[INFO] reset {col} to False for {len(df)} rows")
PY

python dataset_toolkits/aggregate_features.py \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --aggregation_mode visible_only_zbuf \
  --zbuf_depth_tolerance "${ZBUF_DEPTH_TOL}" \
  --zbuf_selection "${ZBUF_SELECTION}" \
  --zbuf_topk "${ZBUF_TOPK}" \
  --zbuf_soft_tau "${ZBUF_SOFT_TAU}" \
  --zbuf_fallback_mode "${ZBUF_FALLBACK_MODE}" \
  --zbuf_min_visibility_count "${ZBUF_MIN_VISIBILITY_COUNT}" \
  --visibility_eps 1e-6 \
  --visibility_margin 0.0 \
  --max_failures 0 \
  --output_feature_name "${FEATURE_MODEL}" \
  --timing_log_interval 50

echo "[INFO] zbuf_config selection=${ZBUF_SELECTION} topk=${ZBUF_TOPK} depth_tol=${ZBUF_DEPTH_TOL} soft_tau=${ZBUF_SOFT_TAU} fallback=${ZBUF_FALLBACK_MODE} min_support=${ZBUF_MIN_VISIBILITY_COUNT} feature=${FEATURE_MODEL}"

feature_count=$(find "${FEATURE_DIR}" -type f -name '*.npz' | wc -l)
if [[ "${feature_count}" -eq 0 ]]; then
  echo "[ERROR] Aggregation produced 0 features in ${FEATURE_DIR}"
  exit 3
fi

expected_count=$(python - <<'PY'
import pandas as pd

df = pd.read_csv('datasets/Toys4k/metadata.csv')

def as_bool(col):
    return col.astype(str).str.lower().isin(['1', 'true', 't', 'yes', 'y'])

if 'rendered' in df.columns:
    df = df[as_bool(df['rendered'])]
if 'voxelized' in df.columns:
    df = df[as_bool(df['voxelized'])]

print(len(df))
PY
)

if [[ "${feature_count}" -lt "${expected_count}" ]]; then
  echo "[ERROR] Incomplete zbuf features: produced ${feature_count}, expected ${expected_count}"
  exit 4
fi

python dataset_toolkits/build_metadata.py Toys4k --output_dir "${OUTPUT_DIR}"
echo "[DONE] toys4k zbuf features ready: ${FEATURE_DIR}"

#!/encs/bin/bash
#SBATCH --job-name=toys4k_ft_enc
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=toys4k_ft_enc_%j.out
#SBATCH --error=toys4k_ft_enc_%j.err

set -euo pipefail

STAGE_NAME="TOYS4K_ENCODER_ONLY_FINETUNE"
RUN_TAG="${RUN_TAG:-encoder_plan_toys4k_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
ENV_PATH="/speed-scratch/an_zhan/env/trellis"
OUTPUT_DIR="datasets/Toys4k"
FEATURE_MODEL="dinov2_vitl14_reg_visible_only"
FEATURE_DIR="${OUTPUT_DIR}/features/${FEATURE_MODEL}"
CFG_PATH="configs/finetune/toys4k_encoder_only_visible_only.json"
PRETRAINED_DIR="${REPO_DIR}/pretrained_ckpts"
TRAIN_OUTPUT_DIR="${REPO_DIR}/results/encoder_plan_toys4k/${RUN_TAG}"
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

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "[ERROR] Missing config: ${CFG_PATH}"
  exit 2
fi

if [[ ! -d "${FEATURE_DIR}" ]] || [[ $(find "${FEATURE_DIR}" -type f -name '*.npz' | wc -l) -eq 0 ]]; then
  echo "[ERROR] Missing or empty visible-only feature dir: ${FEATURE_DIR}"
  exit 3
fi

python my_scripts/encoder_plan/prepare_pretrained_slat_ckpts.py --output_dir "${PRETRAINED_DIR}"

python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec('tensorboard') is None:
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorboard'])
PY

mkdir -p "${TRAIN_OUTPUT_DIR}"
python train.py \
  --config "${CFG_PATH}" \
  --output_dir "${TRAIN_OUTPUT_DIR}" \
  --data_dir "${OUTPUT_DIR}" \
  --num_nodes 1 \
  --node_rank 0 \
  --num_gpus 1 \
  --ckpt none

if ! ls "${TRAIN_OUTPUT_DIR}"/ckpts/misc_step*.pt >/dev/null 2>&1; then
  echo "[ERROR] Training ended without checkpoints in ${TRAIN_OUTPUT_DIR}/ckpts"
  exit 4
fi

echo "[DONE] encoder-only fine-tune output: ${TRAIN_OUTPUT_DIR}"

#!/encs/bin/bash
#SBATCH --job-name=abo_test_enc
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=abo_test_enc_%j.out
#SBATCH --error=abo_test_enc_%j.err

set -euo pipefail

STAGE_NAME="ABO_TEST_TOYS_ENCODER_ENCODE"
RUN_TAG="${RUN_TAG:-abo_test_toysenc_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
ENV_PATH="/speed-scratch/an_zhan/env/trellis"
OUTPUT_DIR="datasets/ABO_full"
FEATURE_MODEL="dinov2_vitl14_reg_visible_only"
FEATURE_DIR="${OUTPUT_DIR}/features/${FEATURE_MODEL}"
TOYS_ENCODER_RUN="${TOYS_ENCODER_RUN:-encoder_plan_toys4k_fix_20260407_002}"
TOYS_ENCODER_CKPT="${TOYS_ENCODER_CKPT:-step0120000}"
MODEL_ROOT="${REPO_DIR}/results/encoder_plan_toys4k"
ENCODER_RUN_DIR="${MODEL_ROOT}/${TOYS_ENCODER_RUN}"
LATENT_MODEL="${FEATURE_MODEL}_${TOYS_ENCODER_RUN}_${TOYS_ENCODER_CKPT}"
LATENT_DIR="${OUTPUT_DIR}/latents/${LATENT_MODEL}"
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

if [[ ! -d "${FEATURE_DIR}" ]] || [[ $(find "${FEATURE_DIR}" -type f -name '*.npz' | wc -l) -eq 0 ]]; then
  echo "[ERROR] Missing or empty ABO visible-only features: ${FEATURE_DIR}"
  exit 2
fi

if [[ ! -f "${ENCODER_RUN_DIR}/config.json" ]]; then
  echo "[ERROR] Missing Toys encoder run config: ${ENCODER_RUN_DIR}/config.json"
  exit 3
fi
if [[ ! -f "${ENCODER_RUN_DIR}/ckpts/encoder_${TOYS_ENCODER_CKPT}.pt" ]]; then
  echo "[ERROR] Missing Toys encoder checkpoint: ${ENCODER_RUN_DIR}/ckpts/encoder_${TOYS_ENCODER_CKPT}.pt"
  exit 4
fi

python dataset_toolkits/encode_latent.py \
  --output_dir "${OUTPUT_DIR}" \
  --feat_model "${FEATURE_MODEL}" \
  --model_root "${MODEL_ROOT}" \
  --enc_model "${TOYS_ENCODER_RUN}" \
  --ckpt "${TOYS_ENCODER_CKPT}"

latent_count=$(find "${LATENT_DIR}" -type f -name '*.npz' | wc -l)
if [[ ${latent_count} -eq 0 ]]; then
  echo "[ERROR] No latent files produced in ${LATENT_DIR}"
  exit 5
fi

echo "${LATENT_MODEL}" > "${TIMING_DIR}/latent_model.txt"
echo "[DONE] Encoded ABO latents with toys encoder into: ${LATENT_DIR}"

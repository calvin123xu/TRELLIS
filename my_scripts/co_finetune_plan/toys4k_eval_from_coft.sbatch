#!/encs/bin/bash
#SBATCH --job-name=toys4k_coft_eval
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=toys4k_coft_eval_%j.out
#SBATCH --error=toys4k_coft_eval_%j.err

set -euo pipefail

STAGE_NAME="TOYS4K_EVAL_FROM_COFINETUNE"
RUN_TAG="${RUN_TAG:-toys4k_coft_eval_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
ENV_PATH="/speed-scratch/an_zhan/env/trellis"
OUTPUT_DIR="datasets/Toys4k"

# Required env vars are provided by the submit wrapper.
VARIANT="${VARIANT:?Missing VARIANT}"
FEAT_MODEL="${FEAT_MODEL:?Missing FEAT_MODEL}"
MODEL_ROOT="${MODEL_ROOT:?Missing MODEL_ROOT}"
MODEL_NAME="${MODEL_NAME:?Missing MODEL_NAME}"
CKPT_TAG="${CKPT_TAG:-ema0.9999_step0120000}"

FEATURE_DIR="${OUTPUT_DIR}/features/${FEAT_MODEL}"
LATENT_MODEL="${FEAT_MODEL}_${MODEL_NAME}_${CKPT_TAG}"
LATENT_DIR="${OUTPUT_DIR}/latents/${LATENT_MODEL}"
GAUSS_DIR="${OUTPUT_DIR}/gaussians_decoded_${RUN_TAG}_${VARIANT}"
METRICS_CSV="${OUTPUT_DIR}/render_metrics_${RUN_TAG}_${VARIANT}.csv"
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
echo "[ARGS] variant=${VARIANT} feat_model=${FEAT_MODEL} model_root=${MODEL_ROOT} model_name=${MODEL_NAME} ckpt=${CKPT_TAG}"

source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
cd "${REPO_DIR}"

if [[ ! -d "${FEATURE_DIR}" ]] || [[ $(find "${FEATURE_DIR}" -type f -name '*.npz' | wc -l) -eq 0 ]]; then
  echo "[ERROR] Missing or empty feature dir: ${FEATURE_DIR}"
  exit 2
fi

if [[ ! -f "${MODEL_ROOT}/${MODEL_NAME}/config.json" ]]; then
  echo "[ERROR] Missing model config: ${MODEL_ROOT}/${MODEL_NAME}/config.json"
  exit 3
fi

if [[ ! -f "${MODEL_ROOT}/${MODEL_NAME}/ckpts/encoder_${CKPT_TAG}.pt" ]]; then
  echo "[ERROR] Missing encoder checkpoint: ${MODEL_ROOT}/${MODEL_NAME}/ckpts/encoder_${CKPT_TAG}.pt"
  exit 4
fi

if [[ ! -f "${MODEL_ROOT}/${MODEL_NAME}/ckpts/decoder_${CKPT_TAG}.pt" ]]; then
  echo "[ERROR] Missing decoder checkpoint: ${MODEL_ROOT}/${MODEL_NAME}/ckpts/decoder_${CKPT_TAG}.pt"
  exit 5
fi

mkdir -p "${REF_BASE}"

python dataset_toolkits/encode_latent.py \
  --output_dir "${OUTPUT_DIR}" \
  --feat_model "${FEAT_MODEL}" \
  --model_root "${MODEL_ROOT}" \
  --enc_model "${MODEL_NAME}" \
  --ckpt "${CKPT_TAG}"

python dataset_toolkits/build_metadata.py Toys4k --output_dir "${OUTPUT_DIR}"

if [[ ! -d "${LATENT_DIR}" ]] || [[ $(find "${LATENT_DIR}" -type f -name '*.npz' | wc -l) -eq 0 ]]; then
  echo "[ERROR] Missing or empty latent dir: ${LATENT_DIR}"
  exit 6
fi

python my_scripts/co_finetune_plan/decode_latents_with_coft_decoder.py \
  --output_dir "${OUTPUT_DIR}" \
  --latent_model "${LATENT_MODEL}" \
  --model_root "${MODEL_ROOT}" \
  --model_name "${MODEL_NAME}" \
  --ckpt "${CKPT_TAG}" \
  --save_dir "${GAUSS_DIR}"

gauss_count=$(find "${GAUSS_DIR}" -type f -name '*.ply' | wc -l)
if [[ "${gauss_count}" -eq 0 ]]; then
  echo "[ERROR] No decoded Gaussian .ply files found in ${GAUSS_DIR}"
  exit 7
fi

python my_scripts/eval_render_metrics.py \
  --output_dir "${OUTPUT_DIR}" \
  --gauss_dir "${GAUSS_DIR}" \
  --resolution 512 \
  --bg 0,0,0 \
  --out_csv "${METRICS_CSV}"

cp -f "${METRICS_CSV}" "${REF_BASE}/after_eval_render_metrics_${VARIANT}.csv"
cp -f "${OUTPUT_DIR}/metadata.csv" "${REF_BASE}/after_eval_metadata_${VARIANT}.csv"

python - <<PY
import pandas as pd
from pathlib import Path

metrics = pd.read_csv('${METRICS_CSV}')
out = Path('${REF_BASE}') / 'after_eval_summary_${VARIANT}.txt'
if len(metrics) == 0:
    out.write_text('N=0\n', encoding='utf-8')
else:
    txt = (
        f'N={len(metrics)}\\n'
        f'PSNR={metrics["psnr"].mean():.6f}\\n'
        f'SSIM={metrics["ssim"].mean():.6f}\\n'
        f'LPIPS={metrics["lpips"].mean():.6f}\\n'
    )
    out.write_text(txt, encoding='utf-8')
PY

echo "[DONE] variant=${VARIANT} metrics=${METRICS_CSV} gaussians=${GAUSS_DIR}"

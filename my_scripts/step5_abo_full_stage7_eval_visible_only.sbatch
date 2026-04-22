#!/encs/bin/bash
#SBATCH --job-name=abo_vis_eval
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=abo_vis_eval_%j.out
#SBATCH --error=abo_vis_eval_%j.err

set -euo pipefail

STAGE_NAME="STAGE7_EVAL_VISIBLE_ONLY"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
ENV_PATH="/speed-scratch/an_zhan/env/trellis"
OUTPUT_DIR="datasets/ABO_full"
GAUSS_DIR="${OUTPUT_DIR}/gaussians_decoded_${RUN_TAG}_visible_only"
METRICS_CSV="${OUTPUT_DIR}/render_metrics_${RUN_TAG}_visible_only.csv"
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

python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec('lpips') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lpips'])
PY

python my_scripts/eval_render_metrics.py \
  --output_dir "${OUTPUT_DIR}" \
  --gauss_dir "${GAUSS_DIR}" \
  --resolution 512 \
  --bg 0,0,0 \
  --out_csv "${METRICS_CSV}"

cp -f "${METRICS_CSV}" "${REF_BASE}/after_eval_render_metrics_visible_only.csv"
cp -f "${OUTPUT_DIR}/metadata.csv" "${REF_BASE}/after_eval_metadata.csv"

python - <<PY
import pandas as pd
from pathlib import Path
metrics = pd.read_csv('${METRICS_CSV}')
out = Path('${REF_BASE}') / 'after_eval_summary_visible_only.txt'
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

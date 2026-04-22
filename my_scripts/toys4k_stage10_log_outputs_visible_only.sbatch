#!/encs/bin/bash
#SBATCH --job-name=toys4k_log_vis
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=toys4k_log_vis_%j.out
#SBATCH --error=toys4k_log_vis_%j.err

set -euo pipefail

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
OUTPUT_DIR="datasets/Toys4k"
DATA_DIR="${REPO_DIR}/${OUTPUT_DIR}"
REF_DIR="${REPO_DIR}/datasets/Toys4k_refs/${RUN_TAG}"
LOG_DIR="${DATA_DIR}/experiment_logs"
LOG_FILE="${LOG_DIR}/${RUN_TAG}_visible_only_outputs.log"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

bash my_scripts/speed_audit_and_organize_outputs.sh --base-dir "${REPO_DIR}" --model dinov2_vitl14_reg --datasets Toys4k

{
  echo "run_tag=${RUN_TAG}"
  echo "experiment=visible_only"
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "[paths]"
  echo "data_dir=${DATA_DIR}"
  echo "feature_mean_dir=${DATA_DIR}/features/dinov2_vitl14_reg"
  echo "feature_visible_only_dir=${DATA_DIR}/features/dinov2_vitl14_reg_visible_only"
  echo "view_feature_cache_dir=${DATA_DIR}/view_features/dinov2_vitl14_reg"
  echo "latent_visible_only_dir=${DATA_DIR}/latents/dinov2_vitl14_reg_visible_only_slat_enc_swin8_B_64l8_fp16"
  echo "gaussian_visible_only_dir=${DATA_DIR}/gaussians_decoded_${RUN_TAG}_visible_only"
  echo "metrics_visible_only_csv=${DATA_DIR}/render_metrics_${RUN_TAG}_visible_only.csv"
  echo "timing_stage_csv=${DATA_DIR}/timing/${RUN_TAG}/stage_timing.csv"
  echo "ref_dir=${REF_DIR}"
  echo "runs_index_csv=${DATA_DIR}/runs/RUNS_INDEX.csv"
  echo ""
  echo "[counts]"
  if [ -d "${DATA_DIR}/features/dinov2_vitl14_reg" ]; then
    echo "count_feature_mean=$(find "${DATA_DIR}/features/dinov2_vitl14_reg" -type f | grep -c .npz || true)"
  else
    echo "count_feature_mean=0"
  fi
  if [ -d "${DATA_DIR}/features/dinov2_vitl14_reg_visible_only" ]; then
    echo "count_feature_visible_only=$(find "${DATA_DIR}/features/dinov2_vitl14_reg_visible_only" -type f | grep -c .npz || true)"
  else
    echo "count_feature_visible_only=0"
  fi
  if [ -d "${DATA_DIR}/view_features/dinov2_vitl14_reg" ]; then
    echo "count_view_feature_cache=$(find "${DATA_DIR}/view_features/dinov2_vitl14_reg" -type f | grep -c .npz || true)"
  else
    echo "count_view_feature_cache=0"
  fi
  if [ -f "${DATA_DIR}/render_metrics_${RUN_TAG}_visible_only.csv" ]; then
    echo "count_metrics_rows=$(($(wc -l < "${DATA_DIR}/render_metrics_${RUN_TAG}_visible_only.csv") - 1))"
  else
    echo "count_metrics_rows=0"
  fi
} > "${LOG_FILE}"

echo "[LOG] wrote ${LOG_FILE}"

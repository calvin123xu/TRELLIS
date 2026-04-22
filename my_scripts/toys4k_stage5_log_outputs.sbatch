#!/encs/bin/bash
#SBATCH --job-name=toys4k_log
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=toys4k_log_%j.out
#SBATCH --error=toys4k_log_%j.err

set -euo pipefail

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
OUTPUT_DIR="datasets/Toys4k"
DATA_DIR="${REPO_DIR}/${OUTPUT_DIR}"
LOG_DIR="${DATA_DIR}/experiment_logs"
LOG_FILE="${LOG_DIR}/${RUN_TAG}_bootstrap_outputs.log"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

bash my_scripts/speed_audit_and_organize_outputs.sh --base-dir "${REPO_DIR}" --model dinov2_vitl14_reg --datasets Toys4k

{
  echo "run_tag=${RUN_TAG}"
  echo "experiment=toys4k_bootstrap"
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "[paths]"
  echo "data_dir=${DATA_DIR}"
  echo "metadata_csv=${DATA_DIR}/metadata.csv"
  echo "render_dir=${DATA_DIR}/renders"
  echo "voxel_dir=${DATA_DIR}/voxels"
  echo "view_feature_cache_dir=${DATA_DIR}/view_features/dinov2_vitl14_reg"
  echo "timing_stage_csv=${DATA_DIR}/timing/${RUN_TAG}/stage_timing.csv"
  echo "runs_index_csv=${DATA_DIR}/runs/RUNS_INDEX.csv"
  echo ""
  echo "[counts]"
  if [ -d "${DATA_DIR}/renders" ]; then
    echo "count_render_dirs=$(find "${DATA_DIR}/renders" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  else
    echo "count_render_dirs=0"
  fi
  if [ -d "${DATA_DIR}/voxels" ]; then
    echo "count_voxel_ply=$(find "${DATA_DIR}/voxels" -type f -name '*.ply' | wc -l)"
  else
    echo "count_voxel_ply=0"
  fi
  if [ -d "${DATA_DIR}/view_features/dinov2_vitl14_reg" ]; then
    echo "count_view_feature_cache=$(find "${DATA_DIR}/view_features/dinov2_vitl14_reg" -type f | grep -c .npz || true)"
  else
    echo "count_view_feature_cache=0"
  fi
} > "${LOG_FILE}"

echo "[LOG] wrote ${LOG_FILE}"

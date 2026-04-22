#!/encs/bin/bash
set -euo pipefail

RUN_TAG="${1:-toys_visonly_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
VIEW_CACHE_DIR="${REPO_DIR}/datasets/Toys4k/view_features/dinov2_vitl14_reg"

cd "${REPO_DIR}"

if [[ ! -d "${VIEW_CACHE_DIR}" ]] || [[ $(find "${VIEW_CACHE_DIR}" -type f -name '*.npz' | wc -l) -eq 0 ]]; then
  echo "[ERROR] Missing or empty view-feature cache: ${VIEW_CACHE_DIR}"
  echo "        Run the Toys4k bootstrap chain first (through stage4 view-feature cache)."
  exit 2
fi

echo "RUN_TAG=${RUN_TAG}"

j6=$(sbatch --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/toys4k_stage6_aggregate_visible_only_from_cache.sbatch | awk '{print $4}')
j7=$(sbatch --dependency=afterok:${j6} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/toys4k_stage7_encode_visible_only.sbatch | awk '{print $4}')
j8=$(sbatch --dependency=afterok:${j7} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/toys4k_stage8_decode_visible_only.sbatch | awk '{print $4}')
j9=$(sbatch --dependency=afterok:${j8} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/toys4k_stage9_eval_visible_only.sbatch | awk '{print $4}')
# Always run logger to capture output map, even if a previous stage fails.
j10=$(sbatch --dependency=afterany:${j9} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/toys4k_stage10_log_outputs_visible_only.sbatch | awk '{print $4}')

echo "RUN_TAG=${RUN_TAG}"
echo "STAGE6_AGG_VISIBLE_ONLY=${j6}"
echo "STAGE7_ENCODE_VISIBLE_ONLY=${j7}"
echo "STAGE8_DECODE_VISIBLE_ONLY=${j8}"
echo "STAGE9_EVAL_VISIBLE_ONLY=${j9}"
echo "STAGE10_LOG_OUTPUTS=${j10}"

squeue -u an_zhan

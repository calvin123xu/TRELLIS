#!/encs/bin/bash
set -euo pipefail

RUN_TAG="${1:-toys_bootstrap_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
TOYS4K_ZIP="${REPO_DIR}/datasets/Toys4k/raw/toys4k_blend_files.zip"

cd "${REPO_DIR}"

if [[ ! -f "${TOYS4K_ZIP}" ]]; then
	echo "[ERROR] Missing required manual file: ${TOYS4K_ZIP}"
	echo "        Please upload toys4k_blend_files.zip before submitting the Toys4k bootstrap chain."
	exit 2
fi

echo "RUN_TAG=${RUN_TAG}"

j1=$(sbatch --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/toys4k_stage1_download.sbatch | awk '{print $4}')
j2=$(sbatch --dependency=afterok:${j1} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/toys4k_stage2_render_rank4.sbatch | awk '{print $4}')
j3=$(sbatch --dependency=afterok:${j2} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/toys4k_stage3_voxelize.sbatch | awk '{print $4}')
j4=$(sbatch --dependency=afterok:${j3} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/toys4k_stage4_view_feature_cache.sbatch | awk '{print $4}')
# Always run logger to capture output map, even if bootstrap fails.
j5=$(sbatch --dependency=afterany:${j4} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/toys4k_stage5_log_outputs.sbatch | awk '{print $4}')

echo "RUN_TAG=${RUN_TAG}"
echo "STAGE1_DOWNLOAD=${j1}"
echo "STAGE2_RENDER=${j2}"
echo "STAGE3_VOXELIZE=${j3}"
echo "STAGE4_VIEW_FEATURE_CACHE=${j4}"
echo "STAGE5_LOG_OUTPUTS=${j5}"

squeue -u an_zhan

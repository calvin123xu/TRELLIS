#!/encs/bin/bash
set -euo pipefail

RUN_TAG="${1:-hssd_bootstrap_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"

cd "${REPO_DIR}"

echo "RUN_TAG=${RUN_TAG}"

j1=$(sbatch --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/hssd_stage1_download.sbatch | awk '{print $4}')
j2=$(sbatch --dependency=afterok:${j1} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/hssd_stage2_render_rank4.sbatch | awk '{print $4}')
j3=$(sbatch --dependency=afterok:${j2} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/hssd_stage3_voxelize.sbatch | awk '{print $4}')
j4=$(sbatch --dependency=afterok:${j3} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/hssd_stage4_view_feature_cache.sbatch | awk '{print $4}')
j5=$(sbatch --dependency=afterok:${j4} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/hssd_stage5_aggregate_visible_only_from_cache.sbatch | awk '{print $4}')
# Always run logger so output map is captured on both success and failure paths.
j6=$(sbatch --dependency=afterany:${j5} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/hssd_stage6_log_outputs_visible_only.sbatch | awk '{print $4}')

echo "RUN_TAG=${RUN_TAG}"
echo "STAGE1_DOWNLOAD=${j1}"
echo "STAGE2_RENDER=${j2}"
echo "STAGE3_VOXELIZE=${j3}"
echo "STAGE4_VIEW_FEATURE_CACHE=${j4}"
echo "STAGE5_AGG_VISIBLE_ONLY=${j5}"
echo "STAGE6_LOG_OUTPUTS=${j6}"

squeue -u an_zhan

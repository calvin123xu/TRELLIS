#!/encs/bin/bash
set -euo pipefail

RUN_TAG="${1:-visonly_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"

cd "${REPO_DIR}"

echo "RUN_TAG=${RUN_TAG}"

j4a=$(sbatch --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/step4_abo_full_stage4a_view_feature_visible_only.sbatch | awk '{print $4}')
j4b=$(sbatch --dependency=afterok:${j4a} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/step4_abo_full_stage4b_aggregate_visible_only.sbatch | awk '{print $4}')
j5=$(sbatch --dependency=afterok:${j4b} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/step5_abo_full_stage5_encode_visible_only.sbatch | awk '{print $4}')
j6=$(sbatch --dependency=afterok:${j5} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/step5_abo_full_stage6_decode_visible_only.sbatch | awk '{print $4}')
j7=$(sbatch --dependency=afterok:${j6} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/step5_abo_full_stage7_eval_visible_only.sbatch | awk '{print $4}')
# Always run final output logger even if an earlier stage fails.
j8=$(sbatch --dependency=afterany:${j7} --export=ALL,RUN_TAG="${RUN_TAG}" my_scripts/step8_abo_full_log_outputs.sbatch | awk '{print $4}')

echo "RUN_TAG=${RUN_TAG}"
echo "STAGE4A_VIEW_FEATURE=${j4a}"
echo "STAGE4B_AGG_VISIBLE_ONLY=${j4b}"
echo "STAGE5_ENCODE_VISIBLE_ONLY=${j5}"
echo "STAGE6_DECODE_VISIBLE_ONLY=${j6}"
echo "STAGE7_EVAL_VISIBLE_ONLY=${j7}"
echo "STAGE8_LOG_OUTPUTS=${j8}"

squeue -u an_zhan

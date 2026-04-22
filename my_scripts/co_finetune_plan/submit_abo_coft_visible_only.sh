#!/encs/bin/bash
set -euo pipefail

RUN_TAG="${1:-abo_coft_vis_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
SCRIPT_PATH="my_scripts/co_finetune_plan/abo_coft_visible_only.sbatch"

cd "${REPO_DIR}"
echo "RUN_TAG=${RUN_TAG}"
job_id=$(sbatch --export=ALL,RUN_TAG="${RUN_TAG}" "${SCRIPT_PATH}" | awk '{print $4}')
echo "RUN_TAG=${RUN_TAG}"
echo "ABO_COFINETUNE_VISIBLE_ONLY_JOB=${job_id}"
squeue -u an_zhan

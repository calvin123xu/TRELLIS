#!/encs/bin/bash
set -euo pipefail

RUN_TAG="${1:-toys4k_mean_agg_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
SCRIPT_PATH="my_scripts/co_finetune_plan/toys4k_aggregate_mean_from_cache.sbatch"

cd "${REPO_DIR}"
echo "RUN_TAG=${RUN_TAG}"
job_id=$(sbatch --export=ALL,RUN_TAG="${RUN_TAG}" "${SCRIPT_PATH}" | awk '{print $4}')
echo "RUN_TAG=${RUN_TAG}"
echo "TOYS4K_AGG_MEAN_FROM_CACHE_JOB=${job_id}"
squeue -u an_zhan

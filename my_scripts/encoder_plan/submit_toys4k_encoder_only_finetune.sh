#!/encs/bin/bash
set -euo pipefail

RUN_TAG="${1:-encoder_plan_toys4k_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
BACKUP_SCRIPT_PATH="my_scripts/encoder_plan/toys4k_backup_features.sbatch"
TRAIN_SCRIPT_PATH="my_scripts/encoder_plan/toys4k_encoder_only_finetune.sbatch"

cd "${REPO_DIR}"

echo "RUN_TAG=${RUN_TAG}"
backup_job_id=$(sbatch --export=ALL,RUN_TAG="${RUN_TAG}" "${BACKUP_SCRIPT_PATH}" | awk '{print $4}')
train_job_id=$(sbatch --dependency=afterok:${backup_job_id} --export=ALL,RUN_TAG="${RUN_TAG}" "${TRAIN_SCRIPT_PATH}" | awk '{print $4}')

echo "RUN_TAG=${RUN_TAG}"
echo "TOYS4K_BACKUP_FEATURES_JOB=${backup_job_id}"
echo "TOYS4K_ENCODER_ONLY_FINETUNE_JOB=${train_job_id}"
echo "DEPENDENCY=afterok:${backup_job_id}"

squeue -u an_zhan

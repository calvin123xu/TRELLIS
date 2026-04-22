#!/encs/bin/bash
set -euo pipefail

RUN_TAG="${1:-abo_test_toysenc_$(date +%Y%m%d_%H%M%S)}"
WAIT_JOB_ID="${2:-}"
TOYS_ENCODER_RUN="${3:-encoder_plan_toys4k_fix_20260407_002}"
TOYS_ENCODER_CKPT="${4:-step0120000}"

REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
BACKUP_SCRIPT="my_scripts/encoder_plan/abo_backup_visible_only_features.sbatch"
ENCODE_SCRIPT="my_scripts/encoder_plan/abo_test_toys_encoder_encode.sbatch"
DECODE_SCRIPT="my_scripts/encoder_plan/abo_test_toys_encoder_decode.sbatch"
EVAL_SCRIPT="my_scripts/encoder_plan/abo_test_toys_encoder_eval.sbatch"

cd "${REPO_DIR}"

echo "RUN_TAG=${RUN_TAG}"
echo "TOYS_ENCODER_RUN=${TOYS_ENCODER_RUN}"
echo "TOYS_ENCODER_CKPT=${TOYS_ENCODER_CKPT}"

base_args=(--export=ALL,RUN_TAG="${RUN_TAG}",TOYS_ENCODER_RUN="${TOYS_ENCODER_RUN}",TOYS_ENCODER_CKPT="${TOYS_ENCODER_CKPT}")

if [[ -n "${WAIT_JOB_ID}" ]]; then
  j1=$(sbatch --dependency=afterany:${WAIT_JOB_ID} "${base_args[@]}" "${BACKUP_SCRIPT}" | awk '{print $4}')
  echo "WAITING_ON_JOB=${WAIT_JOB_ID}"
else
  j1=$(sbatch "${base_args[@]}" "${BACKUP_SCRIPT}" | awk '{print $4}')
fi

j2=$(sbatch --dependency=afterok:${j1} "${base_args[@]}" "${ENCODE_SCRIPT}" | awk '{print $4}')
j3=$(sbatch --dependency=afterok:${j2} "${base_args[@]}" "${DECODE_SCRIPT}" | awk '{print $4}')
j4=$(sbatch --dependency=afterok:${j3} "${base_args[@]}" "${EVAL_SCRIPT}" | awk '{print $4}')

echo "ABO_TEST_BACKUP_JOB=${j1}"
echo "ABO_TEST_ENCODE_JOB=${j2}"
echo "ABO_TEST_DECODE_JOB=${j3}"
echo "ABO_TEST_EVAL_JOB=${j4}"

squeue -u an_zhan

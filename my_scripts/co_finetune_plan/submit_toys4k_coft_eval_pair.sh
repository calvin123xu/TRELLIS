#!/encs/bin/bash
set -euo pipefail

RUN_ROOT_TAG="${1:-toys4k_coft_pair_$(date +%Y%m%d_%H%M%S)}"
CKPT_TAG="${CKPT_TAG:-ema0.9999_step0120000}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"

VIS_MODEL_ROOT="${REPO_DIR}/results/co_finetune_abo_visible_only"
VIS_MODEL_NAME="coft_plan_20260411_122908_abo_vis"
MEAN_MODEL_ROOT="${REPO_DIR}/results/co_finetune_abo_mean"
MEAN_MODEL_NAME="coft_plan_20260411_122908_abo_mean"

SCRIPT_PATH="my_scripts/co_finetune_plan/toys4k_eval_from_coft.sbatch"

cd "${REPO_DIR}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] Missing script: ${SCRIPT_PATH}"
  exit 2
fi

if [[ ! -f "${VIS_MODEL_ROOT}/${VIS_MODEL_NAME}/ckpts/encoder_${CKPT_TAG}.pt" ]]; then
  echo "[ERROR] Missing visible encoder ckpt for ${CKPT_TAG}"
  exit 3
fi

if [[ ! -f "${MEAN_MODEL_ROOT}/${MEAN_MODEL_NAME}/ckpts/encoder_${CKPT_TAG}.pt" ]]; then
  echo "[ERROR] Missing mean encoder ckpt for ${CKPT_TAG}"
  exit 4
fi

vis_tag="${RUN_ROOT_TAG}_vis"
mean_tag="${RUN_ROOT_TAG}_mean"

j_vis=$(sbatch \
  --export=ALL,RUN_TAG="${vis_tag}",VARIANT="vis",FEAT_MODEL="dinov2_vitl14_reg_visible_only",MODEL_ROOT="${VIS_MODEL_ROOT}",MODEL_NAME="${VIS_MODEL_NAME}",CKPT_TAG="${CKPT_TAG}" \
  "${SCRIPT_PATH}" | awk '{print $4}')

j_mean=$(sbatch \
  --export=ALL,RUN_TAG="${mean_tag}",VARIANT="mean",FEAT_MODEL="dinov2_vitl14_reg",MODEL_ROOT="${MEAN_MODEL_ROOT}",MODEL_NAME="${MEAN_MODEL_NAME}",CKPT_TAG="${CKPT_TAG}" \
  "${SCRIPT_PATH}" | awk '{print $4}')

echo "RUN_ROOT_TAG=${RUN_ROOT_TAG}"
echo "CKPT_TAG=${CKPT_TAG}"
echo "TOYS4K_COFINETUNE_VISIBLE_RUN_TAG=${vis_tag}"
echo "TOYS4K_COFINETUNE_VISIBLE_JOB=${j_vis}"
echo "TOYS4K_COFINETUNE_MEAN_RUN_TAG=${mean_tag}"
echo "TOYS4K_COFINETUNE_MEAN_JOB=${j_mean}"

squeue -u an_zhan

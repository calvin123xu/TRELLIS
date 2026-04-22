#!/encs/bin/bash
set -euo pipefail

RUN_ROOT_TAG="${1:-vzbuf_branch_$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
CKPT_TAG="${CKPT_TAG:-ema0.9999_step0120000}"

# Default to a non-collapsing zbuf variant (top-k + visible-only fallback).
FEATURE_MODEL="${FEATURE_MODEL:-dinov2_vitl14_reg_visible_zbuf_topk3_fb1}"
COFT_CFG_PATH="${COFT_CFG_PATH:-configs/finetune/abo_co_finetune_visible_zbuf_topk3_fb1.json}"
COFT_RESULT_SUBDIR="${COFT_RESULT_SUBDIR:-co_finetune_abo_visible_zbuf_topk3_fb1}"
EVAL_VARIANT="${EVAL_VARIANT:-viszbuf_topk3_fb1}"

ZBUF_DEPTH_TOL="${ZBUF_DEPTH_TOL:-5e-3}"
ZBUF_SELECTION="${ZBUF_SELECTION:-topk}"
ZBUF_TOPK="${ZBUF_TOPK:-3}"
ZBUF_SOFT_TAU="${ZBUF_SOFT_TAU:-5e-3}"
ZBUF_FALLBACK_MODE="${ZBUF_FALLBACK_MODE:-visible_only}"
ZBUF_MIN_VISIBILITY_COUNT="${ZBUF_MIN_VISIBILITY_COUNT:-1.0}"

SBATCH_EXTRA=()
if [[ -n "${SBATCH_CONSTRAINT:-}" ]]; then
  SBATCH_EXTRA+=(--constraint="${SBATCH_CONSTRAINT}")
fi

cd "${REPO_DIR}"

tag_abo_agg="${RUN_ROOT_TAG}_abo_agg"
tag_toys_agg="${RUN_ROOT_TAG}_toys_agg"
tag_abo_coft="${RUN_ROOT_TAG}_abo_coft_zbuf"
tag_toys_eval="${RUN_ROOT_TAG}_toys_eval_zbuf"

j_abo_agg=$(sbatch "${SBATCH_EXTRA[@]}" \
  --export=ALL,RUN_TAG="${tag_abo_agg}",FEATURE_MODEL="${FEATURE_MODEL}",ZBUF_DEPTH_TOL="${ZBUF_DEPTH_TOL}",ZBUF_SELECTION="${ZBUF_SELECTION}",ZBUF_TOPK="${ZBUF_TOPK}",ZBUF_SOFT_TAU="${ZBUF_SOFT_TAU}",ZBUF_FALLBACK_MODE="${ZBUF_FALLBACK_MODE}",ZBUF_MIN_VISIBILITY_COUNT="${ZBUF_MIN_VISIBILITY_COUNT}" \
  my_scripts/co_finetune_plan/abo_aggregate_visible_zbuf_from_cache.sbatch | awk '{print $4}')
j_toys_agg=$(sbatch "${SBATCH_EXTRA[@]}" \
  --export=ALL,RUN_TAG="${tag_toys_agg}",FEATURE_MODEL="${FEATURE_MODEL}",ZBUF_DEPTH_TOL="${ZBUF_DEPTH_TOL}",ZBUF_SELECTION="${ZBUF_SELECTION}",ZBUF_TOPK="${ZBUF_TOPK}",ZBUF_SOFT_TAU="${ZBUF_SOFT_TAU}",ZBUF_FALLBACK_MODE="${ZBUF_FALLBACK_MODE}",ZBUF_MIN_VISIBILITY_COUNT="${ZBUF_MIN_VISIBILITY_COUNT}" \
  my_scripts/co_finetune_plan/toys4k_aggregate_visible_zbuf_from_cache.sbatch | awk '{print $4}')
j_abo_coft=$(sbatch "${SBATCH_EXTRA[@]}" \
  --dependency=afterok:${j_abo_agg} \
  --export=ALL,RUN_TAG="${tag_abo_coft}",FEATURE_MODEL="${FEATURE_MODEL}",CFG_PATH="${COFT_CFG_PATH}",RESULT_SUBDIR="${COFT_RESULT_SUBDIR}" \
  my_scripts/co_finetune_plan/abo_coft_visible_zbuf.sbatch | awk '{print $4}')

j_toys_eval=$(sbatch "${SBATCH_EXTRA[@]}" \
  --dependency=afterok:${j_toys_agg}:${j_abo_coft} \
  --export=ALL,RUN_TAG="${tag_toys_eval}",VARIANT="${EVAL_VARIANT}",FEAT_MODEL="${FEATURE_MODEL}",MODEL_ROOT="${REPO_DIR}/results/${COFT_RESULT_SUBDIR}",MODEL_NAME="${tag_abo_coft}",CKPT_TAG="${CKPT_TAG}" \
  my_scripts/co_finetune_plan/toys4k_eval_from_coft.sbatch | awk '{print $4}')

echo "RUN_ROOT_TAG=${RUN_ROOT_TAG}"
echo "FEATURE_MODEL=${FEATURE_MODEL}"
echo "ZBUF_SELECTION=${ZBUF_SELECTION}"
echo "ZBUF_TOPK=${ZBUF_TOPK}"
echo "ZBUF_DEPTH_TOL=${ZBUF_DEPTH_TOL}"
echo "ZBUF_FALLBACK_MODE=${ZBUF_FALLBACK_MODE}"
echo "ZBUF_MIN_VISIBILITY_COUNT=${ZBUF_MIN_VISIBILITY_COUNT}"
echo "COFT_CFG_PATH=${COFT_CFG_PATH}"
echo "COFT_RESULT_SUBDIR=${COFT_RESULT_SUBDIR}"
echo "ABO_AGG_VISIBLE_ZBUF_JOB=${j_abo_agg}"
echo "TOYS4K_AGG_VISIBLE_ZBUF_JOB=${j_toys_agg}"
echo "ABO_COFINETUNE_VISIBLE_ZBUF_JOB=${j_abo_coft}"
echo "TOYS4K_EVAL_VISIBLE_ZBUF_JOB=${j_toys_eval}"
echo "ABO_COFINETUNE_RUN_TAG=${tag_abo_coft}"
echo "TOYS4K_EVAL_RUN_TAG=${tag_toys_eval}"

squeue -u an_zhan

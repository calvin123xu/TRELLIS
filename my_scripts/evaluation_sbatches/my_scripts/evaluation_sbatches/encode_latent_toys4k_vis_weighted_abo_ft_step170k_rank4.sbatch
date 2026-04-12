#!/encs/bin/bash
# 4-way parallel latent encoding from vis_weighted aggregated features
# using ABO finetuned encoder checkpoint (step 170000).
#
# Submit as array:
#   sbatch my_scripts/evaluation_sbatches/encode_latent_toys4k_vis_weighted_abo_ft_step170k_rank4.sbatch
#
# It launches ranks 0..3 via SLURM_ARRAY_TASK_ID and each rank encodes
# latents for its own metadata partition.

#SBATCH --job-name=trellis_toys4k_enc_visw_aboft170k_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=trellis_enc_visw_aboft170k_r4_%A_%a.out
#SBATCH --error=trellis_enc_visw_aboft170k_r4_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=x_qiaoyu@speed.encs.concordia.ca

set -euo pipefail

echo "HOST: $(hostname)"
echo "START: $(date)"

source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/qiaoyu/env/trellis

if [ -f /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh ]; then
    source /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh
fi

TRELLIS_ROOT="/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS"
OUTPUT_DIR="${TRELLIS_ROOT}/datasets/Toys4k_small"
CKPT_DIR="${TRELLIS_ROOT}/outputs/finetune_vis_abo/ckpts"
ENC_CKPT="${CKPT_DIR}/encoder_ema0.9999_step0170000.pt"
FEAT_MODEL="dinov2_vitl14_reg"
FEATURE_SUBDIR="dinov2_vitl14_reg_vis_weighted"
LATENT_SUBDIR="dinov2_vitl14_reg_vis_weighted_abo_ft_step170k"
ENC_MODEL="finetune_vis_abo"
MODEL_ROOT="${TRELLIS_ROOT}/outputs"
WORLD_SIZE=4
RANK="${SLURM_ARRAY_TASK_ID}"

cd "${TRELLIS_ROOT}"

echo "Python: $(which python)"
python --version

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ENC_CKPT=${ENC_CKPT}"
echo "FEAT_MODEL=${FEAT_MODEL}"
echo "FEATURE_SUBDIR=${FEATURE_SUBDIR}"
echo "LATENT_SUBDIR=${LATENT_SUBDIR}"
echo "ENC_MODEL=${ENC_MODEL}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}"

ls -lh "${ENC_CKPT}"

python dataset_toolkits/encode_latent.py \
  --output_dir "${OUTPUT_DIR}" \
  --feat_model "${FEAT_MODEL}" \
  --feature_subdir "${FEATURE_SUBDIR}" \
  --enc_ckpt_path "${ENC_CKPT}" \
  --enc_model "${ENC_MODEL}" \
  --model_root "${MODEL_ROOT}" \
  --latent_subdir "${LATENT_SUBDIR}" \
  --rank "${RANK}" \
  --world_size "${WORLD_SIZE}"

echo "END: $(date)"

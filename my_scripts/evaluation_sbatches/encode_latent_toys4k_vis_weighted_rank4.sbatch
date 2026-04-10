#!/encs/bin/bash
# 4-way parallel latent encoding from vis_weighted aggregated features
# using world_size/rank.
#
# Submit as array:
#   sbatch my_scripts/encode_latent_toys4k_vis_weighted_rank4.sbatch
#
# It launches ranks 0..3 via SLURM_ARRAY_TASK_ID and each rank encodes
# latents for its own metadata partition.

#SBATCH --job-name=trellis_toys4k_enc_visw_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=trellis_enc_visw_r4_%A_%a.out
#SBATCH --error=trellis_enc_visw_r4_%A_%a.err
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

cd /speed-scratch/qiaoyu/speed-hpc/project/TRELLIS

echo "Python: $(which python)"
python --version

OUTPUT_DIR="/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS/datasets/Toys4k_small"
FEAT_MODEL="dinov2_vitl14_reg"
FEATURE_SUBDIR="dinov2_vitl14_reg_vis_weighted"
LATENT_SUBDIR="dinov2_vitl14_reg_vis_weighted_slat_enc_swin8_B_64l8_fp16"
WORLD_SIZE=4
RANK="${SLURM_ARRAY_TASK_ID}"

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "FEAT_MODEL=${FEAT_MODEL}"
echo "FEATURE_SUBDIR=${FEATURE_SUBDIR}"
echo "LATENT_SUBDIR=${LATENT_SUBDIR}"
echo "WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}"

python dataset_toolkits/encode_latent.py \
  --output_dir "${OUTPUT_DIR}" \
  --feat_model "${FEAT_MODEL}" \
  --feature_subdir "${FEATURE_SUBDIR}" \
  --latent_subdir "${LATENT_SUBDIR}" \
  --rank "${RANK}" \
  --world_size "${WORLD_SIZE}"

echo "END: $(date)"

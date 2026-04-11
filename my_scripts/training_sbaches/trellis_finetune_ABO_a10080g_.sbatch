#!/encs/bin/bash
#SBATCH --job-name=trellis_vae_ft_ABO_visw
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --gres=gpu:nvidia_a100_7g.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=trellis_vae_ft_ABO_visw_%j.out
#SBATCH --error=trellis_vae_ft_ABO_visw_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=x_qiaoyu@speed.encs.concordia.ca

set -euo pipefail

echo "HOST: $(hostname)"
echo "START: $(date)"
nvidia-smi

# Conda
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/qiaoyu/env/trellis

# Cache
source /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh

# Runtime env
export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths
PROJECT_DIR=/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS
DATA_DIR=${PROJECT_DIR}/datasets/ABO
CONFIG=${PROJECT_DIR}/configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16_vis_weighted_finetune.json

# Replace this with your actual pretrained VAE run directory
PRETRAIN_DIR=${PROJECT_DIR}/outputs/<your_pretrained_vae_run_dir>

OUTPUT_DIR=${PROJECT_DIR}/outputs/vae_finetune_ABO_vis_weighted_full_a10080g

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "CONFIG=${CONFIG}"
echo "PRETRAIN_DIR=${PRETRAIN_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

cd "${PROJECT_DIR}"

python train.py \
  --config "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --load_dir "${PRETRAIN_DIR}" \
  --ckpt latest \
  --data_dir "${DATA_DIR}" \
  --num_gpus 1 \
  --auto_retry 0

echo "END: $(date)"

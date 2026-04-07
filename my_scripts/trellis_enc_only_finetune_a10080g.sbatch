#!/encs/bin/bash
#SBATCH --job-name=trellis_enc_ft
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --gres=gpu:nvidia_a100_7g.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=trellis_enc_ft_%j.out
#SBATCH --error=trellis_enc_ft_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=x_qiaoyu@speed.encs.concordia.ca

set -euo pipefail

echo "HOST: $(hostname)"
echo "START: $(date)"
nvidia-smi

# Conda
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/qiaoyu/env/trellis

# Cache (bash version)
source /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh

# Runtime env for stable backend/memory behavior
export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run
cd /speed-scratch/qiaoyu/speed-hpc/project/TRELLIS
python train.py \
  --config configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16_enc_only_finetune.json \
  --output_dir /speed-scratch/qiaoyu/speed-hpc/project/TRELLIS/outputs/enc_only_finetune_a10080g_firstpass \
  --data_dir /speed-scratch/qiaoyu/speed-hpc/project/TRELLIS/data \
  --ckpt none \
  --num_gpus 1 \
  --auto_retry 0

echo "END: $(date)"

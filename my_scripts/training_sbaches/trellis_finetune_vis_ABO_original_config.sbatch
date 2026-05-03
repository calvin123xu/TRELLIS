#!/encs/bin/bash
#SBATCH --job-name=trellis_vae_ft_ABO_visw
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
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

# Cache (bash version)
source /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh

# Job-local temp directory
# Put Python multiprocessing / DataLoader temp files outside shared NFS-style locations.
JOB_TMPDIR="/tmp/${USER}/trellis_${SLURM_JOB_ID}"
mkdir -p "$JOB_TMPDIR"

export TMPDIR="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"

echo "TMPDIR: $TMPDIR"
df -h "$TMPDIR" || true

cleanup_tmpdir() {
    echo "Cleaning TMPDIR: $JOB_TMPDIR"
    rm -rf "$JOB_TMPDIR" 2>/dev/null || true
}
trap cleanup_tmpdir EXIT

# Runtime env for stable backend/memory behavior
export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run
cd /speed-scratch/qiaoyu/speed-hpc/project/TRELLIS

RUN_TAG="visw_ft_$(date +%Y%m%d_%H%M%S)"
TRAIN_OUTPUT_DIR="/speed-scratch/qiaoyu/speed-hpc/project/TRELLIS/outputs/finetune_vis_weighted_abo/${RUN_TAG}"

python train.py \
  --config configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16_vis_weighted_finetune_20g.json \
  --output_dir "${TRAIN_OUTPUT_DIR}" \
  --data_dir /speed-scratch/qiaoyu/speed-hpc/project/TRELLIS/datasets/ABO \
  --num_gpus 1 \
  --ckpt none
  --auto_retry 0

echo "END: $(date)"

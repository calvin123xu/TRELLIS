#!/encs/bin/bash
#SBATCH --job-name=trellis_abo_agg_visw_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpu=1
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=trellis_abo_agg_visw_r4_%A_%a.out
#SBATCH --error=trellis_abo_agg_visw_r4_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=x_qiaoyu@speed.encs.concordia.ca

set -euo pipefail

echo "=================================================="
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Array Job ID  : ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array Task ID : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node          : $(hostname)"
echo "Start time    : $(date)"
echo "=================================================="

# Go to project root
cd /speed-scratch/qiaoyu/speed-hpc/project/TRELLIS

# Optional: load your cache/temp envs if you usually use them
if [ -f /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh ]; then
    source /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh
fi

# Enable conda
source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/qiaoyu/env/trellis

# Rank/world size setup
RANK=${SLURM_ARRAY_TASK_ID}
WORLD_SIZE=4

echo "Using RANK=${RANK}, WORLD_SIZE=${WORLD_SIZE}"

# Good practice on shared nodes
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0

# Run aggregation
python dataset_toolkits/aggregate_features.py \
  --output_dir datasets/ABO \
  --model dinov2_vitl14_reg \
  --agg vis_weighted \
  --batch_size 8 \
  --rank ${RANK} \
  --world_size ${WORLD_SIZE} \
  --output_feature_name dinov2_vitl14_reg_vis_weighted

echo "Finished rank ${RANK} at $(date)"

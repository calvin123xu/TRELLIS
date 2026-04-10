#!/encs/bin/bash
# 4-way parallel per-view DINO cache extraction using world_size/rank.
#
# Submit as array:
#   sbatch my_scripts/extract_view_features_toys4k_rank4.sbatch
#
# It launches ranks 0..3 via SLURM_ARRAY_TASK_ID and each rank extracts
# view features for its own metadata partition.

#SBATCH --job-name=trellis_toys4k_viewfeat_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=trellis_viewfeat_r4_%A_%a.out
#SBATCH --error=trellis_viewfeat_r4_%A_%a.err
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
MODEL="dinov2_vitl14_reg"
BATCH_SIZE=8
WORLD_SIZE=4
RANK="${SLURM_ARRAY_TASK_ID}"

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODEL=${MODEL}, BATCH_SIZE=${BATCH_SIZE}"
echo "WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}"

python dataset_toolkits/extract_view_features.py \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --batch_size "${BATCH_SIZE}" \
  --rank "${RANK}" \
  --world_size "${WORLD_SIZE}"

echo "END: $(date)"

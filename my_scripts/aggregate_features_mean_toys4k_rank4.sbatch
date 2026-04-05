#!/encs/bin/bash
# 4-way parallel voxel feature aggregation (mean baseline)
# using world_size/rank.
#
# Submit as array:
#   sbatch my_scripts/aggregate_features_mean_toys4k_rank4.sbatch
#
# It launches ranks 0..3 via SLURM_ARRAY_TASK_ID and each rank aggregates
# features for its own metadata partition.

#SBATCH --job-name=trellis_toys4k_agg_mean_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=trellis_agg_mean_r4_%A_%a.out
#SBATCH --error=trellis_agg_mean_r4_%A_%a.err
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
AGG="mean"
OUTPUT_FEATURE_NAME="dinov2_vitl14_reg_baseline"
BATCH_SIZE=16
WORLD_SIZE=4
RANK="${SLURM_ARRAY_TASK_ID}"

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODEL=${MODEL}, AGG=${AGG}, OUTPUT_FEATURE_NAME=${OUTPUT_FEATURE_NAME}"
echo "BATCH_SIZE=${BATCH_SIZE}, WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}"

python dataset_toolkits/aggregate_features.py \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --agg "${AGG}" \
  --output_feature_name "${OUTPUT_FEATURE_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --rank "${RANK}" \
  --world_size "${WORLD_SIZE}"

echo "END: $(date)"

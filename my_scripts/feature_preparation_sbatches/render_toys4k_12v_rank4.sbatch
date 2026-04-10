#!/encs/bin/bash
# 4-way parallel render using world_size/rank (no txt chunking).
#
# Submit as array:
#   sbatch my_scripts/render_toys4k_12v_rank4.sbatch
#
# It launches ranks 0..3 via SLURM_ARRAY_TASK_ID and each rank renders
# its own partition from metadata.
#
# IMPORTANT:
# 1) Run metadata merge ONCE before submitting this array job:
#      python dataset_toolkits/build_metadata.py Toys4k --output_dir datasets/Toys4k_small
# 2) After all array tasks complete, run build_metadata again once to merge rendered_<rank>.csv.

#SBATCH --job-name=trellis_toys4k_render_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=trellis_render_r4_%A_%a.out
#SBATCH --error=trellis_render_r4_%A_%a.err
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

echo "Checking Blender..."
/speed-scratch/qiaoyu/tools/blender-3.0.1-linux-x64/blender --version

OUTPUT_DIR="datasets/Toys4k_small"
WORLD_SIZE=4
RANK="${SLURM_ARRAY_TASK_ID}"

echo "WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}"

python dataset_toolkits/render.py Toys4k \
  --output_dir "${OUTPUT_DIR}" \
  --num_views 12 \
  --max_workers 6 \
  --rank "${RANK}" \
  --world_size "${WORLD_SIZE}" \
  --record_tag "${RANK}"

echo "END: $(date)"

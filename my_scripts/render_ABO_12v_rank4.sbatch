#!/encs/bin/bash
# 4-way parallel ABO render using world_size/rank (single sbatch job with 4 lanes).
#
# Submit:
#   sbatch my_scripts/render_ABO_12v_rank4.sbatch
#
# IMPORTANT:
# 1) Run metadata build ONCE before submitting this job:
#      python dataset_toolkits/build_metadata.py ABO --output_dir datasets/ABO
# 2) After this job completes, run build_metadata again once to merge rendered_<rank>.csv.

#SBATCH --job-name=trellis_abo_render_12v_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G  # 4 concurrent lanes × 128G/lane = 512G total requested memory.
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=trellis_abo_render_12v_r4_%j.out
#SBATCH --error=trellis_abo_render_12v_r4_%j.err
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

OUTPUT_DIR="datasets/ABO"
WORLD_SIZE=4
NUM_VIEWS=12
MAX_WORKERS=6

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "WORLD_SIZE=${WORLD_SIZE}, NUM_VIEWS=${NUM_VIEWS}, MAX_WORKERS=${MAX_WORKERS}"

pids=()
for RANK in 0 1 2 3; do
  echo "Launching rank ${RANK}/${WORLD_SIZE}..."
  python dataset_toolkits/render.py ABO \
    --output_dir "${OUTPUT_DIR}" \
    --num_views "${NUM_VIEWS}" \
    --max_workers "${MAX_WORKERS}" \
    --rank "${RANK}" \
    --world_size "${WORLD_SIZE}" \
    --record_tag "${RANK}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "END: $(date)"

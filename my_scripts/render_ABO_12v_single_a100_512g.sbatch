#!/encs/bin/bash
# Single-process ABO render job (no SLURM array):
# - Uses world_size=1, rank=0 so one job handles the full render queue.
# - Requests 1x A100 GPU and 512G memory.
#
# Usage:
#   sbatch my_scripts/render_ABO_12v_single_a100_512g.sbatch
#
# IMPORTANT:
# 1) Run metadata build ONCE before submitting:
#      python dataset_toolkits/build_metadata.py ABO --output_dir datasets/ABO
# 2) After render completes, run build_metadata once again to merge rendered csv files.

#SBATCH --job-name=trellis_abo_render_12v_single
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:nvidia_a100_7g.80gb:1
#SBATCH --time=36:00:00
#SBATCH --output=trellis_abo_render_12v_single_%j.out
#SBATCH --error=trellis_abo_render_12v_single_%j.err
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
WORLD_SIZE=1
RANK=0
MAX_WORKERS=12

echo "WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}, MAX_WORKERS=${MAX_WORKERS}"

python dataset_toolkits/render.py ABO \
  --output_dir "${OUTPUT_DIR}" \
  --num_views 12 \
  --max_workers "${MAX_WORKERS}" \
  --rank "${RANK}" \
  --world_size "${WORLD_SIZE}" \
  --record_tag "single"

echo "END: $(date)"

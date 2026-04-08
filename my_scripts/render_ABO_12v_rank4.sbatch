#!/encs/bin/bash
# 4-way parallel ABO render using world_size/rank (array-based, same pattern as Toys4k rank4).
#
# Submit as array:
#   sbatch my_scripts/render_ABO_12v_rank4.sbatch
#
# It launches ranks 0..3 via SLURM_ARRAY_TASK_ID and each rank renders
# its own partition from metadata.
#
# IMPORTANT:
# 1) Run metadata build ONCE before submitting this array job:
#      python dataset_toolkits/build_metadata.py ABO --output_dir datasets/ABO
# 2) After all array tasks complete, run build_metadata again once to merge rendered_<rank>.csv.

#SBATCH --job-name=trellis_abo_render_12v_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=trellis_abo_render_12v_r4_%A_%a.out
#SBATCH --error=trellis_abo_render_12v_r4_%A_%a.err
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
MAX_WORKERS=2
RANK="${SLURM_ARRAY_TASK_ID}"

# Headless runtime fixes for Blender on shared HPC nodes
export DISPLAY=""
export SDL_AUDIODRIVER="dummy"
export PULSE_SERVER="unix:/dev/null"
export XDG_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/xdg-runtime-${UID}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}, NUM_VIEWS=${NUM_VIEWS}, MAX_WORKERS=${MAX_WORKERS}"
echo "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}"

python dataset_toolkits/render.py ABO \
  --output_dir "${OUTPUT_DIR}" \
  --num_views "${NUM_VIEWS}" \
  --max_workers "${MAX_WORKERS}" \
  --rank "${RANK}" \
  --world_size "${WORLD_SIZE}" \
  --record_tag "${RANK}"

echo "END: $(date)"

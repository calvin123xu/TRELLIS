#!/encs/bin/bash
# 4-way parallel ABO render using world_size/rank (array-based).
# - Uses the same headless-safe runtime setup as the single-rank ABO script.
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

# Headless/HPC runtime guards:
# - Ensure runtime dir exists (avoids PulseAudio /run/user/$UID warnings on compute nodes).
# - Clear DISPLAY to avoid stale login-node X11 forwarding.
# - Disable audio backends that are irrelevant for batch rendering.
if [ -d "/run/user/${UID}" ]; then
  export XDG_RUNTIME_DIR="/run/user/${UID}"
else
  export XDG_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/xdg-runtime-${UID}"
fi
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
# Some GLib/GIO code paths probe ${XDG_RUNTIME_DIR}/gvfs and print a warning
# if it does not exist. Pre-create it to reduce noisy logs in batch jobs.
mkdir -p "${XDG_RUNTIME_DIR}/gvfs"
export DISPLAY=
export SDL_AUDIODRIVER=dummy
export PULSE_SERVER=127.0.0.1

echo "Checking Blender..."
/speed-scratch/qiaoyu/tools/blender-3.0.1-linux-x64/blender --version

OUTPUT_DIR="datasets/ABO"
WORLD_SIZE=4
RANK="${SLURM_ARRAY_TASK_ID}"
MAX_WORKERS=4

echo "WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}, MAX_WORKERS=${MAX_WORKERS}"

if command -v xvfb-run >/dev/null 2>&1; then
  echo "Using xvfb-run for Blender headless context"
  xvfb-run -a -s "-screen 0 1280x720x24" \
    python dataset_toolkits/render.py ABO \
      --output_dir "${OUTPUT_DIR}" \
      --num_views 12 \
      --max_workers "${MAX_WORKERS}" \
      --rank "${RANK}" \
      --world_size "${WORLD_SIZE}" \
      --record_tag "${RANK}"
else
  echo "xvfb-run not found; running render.py directly"
  python dataset_toolkits/render.py ABO \
    --output_dir "${OUTPUT_DIR}" \
    --num_views 12 \
    --max_workers "${MAX_WORKERS}" \
    --rank "${RANK}" \
    --world_size "${WORLD_SIZE}" \
    --record_tag "${RANK}"
fi

echo "END: $(date)"

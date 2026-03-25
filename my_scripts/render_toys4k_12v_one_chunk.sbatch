#!/encs/bin/bash
# Parallel-safe version:
# - No build_metadata call inside each job (avoid concurrent metadata writes)
# - Unique --rank per chunk file to avoid rendered_<rank>.csv collisions
#
# Usage:
#   1) Edit INSTANCES_TXT to one chunk file
#   2) sbatch my_scripts/render_toys4k_12v_one_chunk_parallel_safe.sbatch
#   3) After all chunk jobs finish, run once:
#        python dataset_toolkits/build_metadata.py Toys4k --output_dir datasets/Toys4k_small

#SBATCH --job-name=trellis_toys4k_render_chunk
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=trellis_render_chunk_%j.out
#SBATCH --error=trellis_render_chunk_%j.err
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

# ------------------------------------------------------------------
# EDIT THIS: choose one chunk txt
INSTANCES_TXT="${OUTPUT_DIR}/render_chunks/unrendered_0000.txt"
# ------------------------------------------------------------------

if [ ! -f "${INSTANCES_TXT}" ]; then
    echo "[ERROR] instances txt not found: ${INSTANCES_TXT}"
    exit 1
fi

# Derive a stable numeric rank from chunk filename suffix.
# e.g. unrendered_0007.txt -> RANK=7
base_name="$(basename "${INSTANCES_TXT}")"
if [[ "${base_name}" =~ ([0-9]+)\.txt$ ]]; then
    RANK=$((10#${BASH_REMATCH[1]}))
else
    # Fallback: deterministic-ish per job (still unique across concurrent jobs)
    RANK=${SLURM_JOB_ID}
fi

echo "Using instances file: ${INSTANCES_TXT}"
echo "Derived rank: ${RANK}"
echo "Num instances in file: $(wc -l < "${INSTANCES_TXT}")"
head -5 "${INSTANCES_TXT}" || true

# NOTE: intentionally no build_metadata call here (parallel-safe).
python dataset_toolkits/render.py Toys4k \
  --output_dir "${OUTPUT_DIR}" \
  --instances "${INSTANCES_TXT}" \
  --num_views 12 \
  --max_workers 6 \
  --rank "${RANK}" \
  --world_size 1

echo "END: $(date)"

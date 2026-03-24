#!/encs/bin/bash
#SBATCH --job-name=trellis_toys4k_render_12v
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=trellis_render_%j.out
#SBATCH --error=trellis_render_%j.err
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

echo "Checking instances..."
wc -l my_20_assets.txt
head -5 my_20_assets.txt || true


python dataset_toolkits/render.py Toys4k \
  --output_dir datasets/Toys4k_small \
  --instances my_20_assets.txt \
  --num_views 12 \
  --max_workers 6

echo "END: $(date)"

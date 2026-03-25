#!/encs/bin/bash
#SBATCH --job-name=trellis_toys4k_render_20
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1 
#SBATCH --time=08:00:00
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

python - <<'PY'
import pandas as pd
df = pd.read_csv('datasets/Toys4k_small/metadata.csv')
df = df[df['local_path'].notna()].reset_index(drop=True)
ids = df['sha256'].head(20)
with open('my_20_assets.txt', 'w') as f:
    for x in ids:
        f.write(str(x) + '\n')
print(f"wrote {len(ids)} ids to my_20_assets.txt")
print(ids.tolist()[:5])
PY

wc -l my_20_assets.txt
head -5 my_20_assets.txt || true

python dataset_toolkits/build_metadata.py Toys4k --output_dir datasets/Toys4k_small

python dataset_toolkits/render.py Toys4k \
  --output_dir datasets/Toys4k_small \
  --instances my_20_assets.txt \
  --num_views 12 \
  --max_workers 6

echo "END: $(date)"

#!/encs/bin/bash
#SBATCH --job-name=abo_viewfeat_r4
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=abo_viewfeat_r4_%A_%a.out
#SBATCH --error=abo_viewfeat_r4_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=x_qiaoyu@speed.encs.concordia.ca

set -euo pipefail

echo "HOST: $(hostname)"
echo "START: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID (rank): ${SLURM_ARRAY_TASK_ID}"

source /encs/pkg/anaconda3-2023.03/root/etc/profile.d/conda.sh
conda activate /speed-scratch/qiaoyu/env/trellis

# Optional but recommended if you normally use scratch-backed caches
if [ -f /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh ]; then
  source /speed-scratch/qiaoyu/speed-hpc/temp_cache_bash.sh
fi

cd /speed-scratch/qiaoyu/speed-hpc/project/TRELLIS

nvidia-smi

python dataset_toolkits/extract_view_features.py \
  --output_dir datasets/ABO \
  --model dinov2_vitl14_reg \
  --batch_size 8 \
  --rank "${SLURM_ARRAY_TASK_ID}" \
  --world_size 4

echo "END: $(date)"

#!/encs/bin/bash
#SBATCH --job-name=trellis_audit_outputs
#SBATCH --partition=pa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=trellis_audit_outputs_%j.out
#SBATCH --error=trellis_audit_outputs_%j.err

set -euo pipefail

REPO_DIR="/speed-scratch/an_zhan/project/TRELLIS-experiments"
DATASETS="ABO_full Toys4k"
MODEL="dinov2_vitl14_reg"

cd "${REPO_DIR}"
chmod +x my_scripts/speed_audit_and_organize_outputs.sh

my_scripts/speed_audit_and_organize_outputs.sh \
  --base-dir "${REPO_DIR}" \
  --model "${MODEL}" \
  --datasets ${DATASETS}

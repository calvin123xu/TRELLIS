#!/encs/bin/bash
#SBATCH --job-name=test_tmpdir
#SBATCH --account=weiping
#SBATCH --partition=pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=test_tmpdir_%j.out
#SBATCH --error=test_tmpdir_%j.err

set -euo pipefail

echo "===== BASIC INFO ====="
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "USER: ${USER}"
echo "SHELL: ${SHELL:-unknown}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

echo
echo "===== EXISTING TEMP ENV ====="
echo "SLURM_TMPDIR: ${SLURM_TMPDIR:-EMPTY}"
echo "TMPDIR before override: ${TMPDIR:-EMPTY}"
echo "TMP before override: ${TMP:-EMPTY}"
echo "TEMP before override: ${TEMP:-EMPTY}"

echo
echo "===== CHECK /tmp ====="
ls -ld /tmp || true
df -h /tmp || true

echo
echo "===== CHECK /speed-scratch fallback ====="
ls -ld /speed-scratch/qiaoyu || true
df -h /speed-scratch/qiaoyu || true

echo
echo "===== SELECT JOB TMPDIR ====="

if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" && -w "$SLURM_TMPDIR" ]]; then
    JOB_TMPDIR="${SLURM_TMPDIR}/trellis_test_${SLURM_JOB_ID}"
    echo "Using SLURM_TMPDIR"
elif [[ -d /tmp && -w /tmp ]]; then
    JOB_TMPDIR="/tmp/${USER}/trellis_test_${SLURM_JOB_ID}"
    echo "Using /tmp"
else
    JOB_TMPDIR="/speed-scratch/qiaoyu/tmp/trellis_test_${SLURM_JOB_ID}"
    echo "Using /speed-scratch fallback"
fi

mkdir -p "$JOB_TMPDIR"

export TMPDIR="$JOB_TMPDIR"
export TMP="$JOB_TMPDIR"
export TEMP="$JOB_TMPDIR"

echo "JOB_TMPDIR: $JOB_TMPDIR"
echo "TMPDIR after override: $TMPDIR"

echo
echo "===== WRITE TEST ====="
TEST_FILE="$JOB_TMPDIR/test_write.txt"
echo "hello from $(hostname)" > "$TEST_FILE"
cat "$TEST_FILE"
ls -lh "$JOB_TMPDIR"

echo
echo "===== PYTHON TEMPFILE TEST ====="
python - <<'PY'
import os
import tempfile
from pathlib import Path

print("Python TMPDIR env:", os.environ.get("TMPDIR"))
print("tempfile.gettempdir():", tempfile.gettempdir())

with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
    f.write("python tempfile test\n")
    name = f.name

print("Created temp file:", name)
print("Exists:", Path(name).exists())

with open(name, "r") as f:
    print("Content:", f.read().strip())

os.remove(name)
print("Deleted temp file:", not Path(name).exists())
PY

echo
echo "===== CLEANUP TEST ====="
rm -rf "$JOB_TMPDIR"
echo "Removed JOB_TMPDIR: $JOB_TMPDIR"
echo "Exists after cleanup?"
ls -ld "$JOB_TMPDIR" || echo "Cleaned successfully."

echo
echo "END: $(date)"

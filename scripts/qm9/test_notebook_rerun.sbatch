#!/usr/bin/env bash
#SBATCH --job-name=qm9_nb07_rerun
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Executes a COPY of notebooks/07_qm9_property_guidance.ipynb with RERUN = True on one
# GPU (small-n reruns), writing outputs/notebook07/07_rerun_true.ipynb. The committed
# notebook is not touched.   sbatch scripts/qm9/test_notebook_rerun.sbatch
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source scripts/env.sh
mkdir -p outputs/notebook07
python - <<'PY'
import json
nb = json.load(open("notebooks/07_qm9_property_guidance.ipynb"))
src = "".join(nb["cells"][0]["source"])
assert src.startswith("RERUN = False")
nb["cells"][0]["source"] = src.replace("RERUN = False", "RERUN = True", 1).splitlines(True)
json.dump(nb, open("notebooks/_07_rerun_true.ipynb", "w"), indent=1)
PY
cd notebooks
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=multimfm \
  --ExecutePreprocessor.timeout=3000 --output ../outputs/notebook07/07_rerun_true.ipynb _07_rerun_true.ipynb
rm -f _07_rerun_true.ipynb
echo "done: outputs/notebook07/07_rerun_true.ipynb"

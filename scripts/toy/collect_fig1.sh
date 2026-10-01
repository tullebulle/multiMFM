#!/usr/bin/env bash
# Finish the toy example after scripts/toy/steer_fig1.sh (Slurm job toy_fig1) has
# completed: check the outputs, refresh the checkpoint manifest and re-execute
# notebooks/01_toy_multimodal.ipynb on CPU.  Run from the repo root:
#   bash scripts/toy/collect_fig1.sh
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
PY="${PY:-$HOME/micromamba/envs/multimfm/bin/python}"

missing=0
for f in results/toy/fig1_panels.npz results/toy/steering_metrics.json \
         results/toy/table12_actual.json results/toy/fig1_glass.png results/toy/fig1_mfm.png \
         results/toy/probe_mfm_diagonal_10000.json results/toy/probe_mfm_consistency_10000.json; do
  [ -f "$f" ] || { echo "MISSING $f" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "run 'sbatch --partition=gpu_test --time=4:00:00 --job-name=toy_fig1 --export=ALL,STAGES=steer_fig1 scripts/toy/toy.sbatch' first" >&2; exit 1; }

"${PY}" scripts/manifest.py verify --group toy

# Re-render the panels with the current plotting code (the sampling job may have
# been launched before a plotting change).
PYTHONPATH=src "${PY}" -m toy.fig1 --redraw --out_dir results/toy

"${PY}" - <<'EOF'
import json, numpy as np
from pathlib import Path
p = Path("results/toy")
panels = np.load(p / "fig1_panels.npz")
print("panels:", {k: v.shape for k, v in panels.items()})
m = json.loads((p / "steering_metrics.json").read_text())["metrics"]
for k, v in m.items():
    print(f"  {k:<22} n={v['n']:>6}  reward={v['mean_reward']:.3f}  match={v['match']:.3f}")
EOF

cd notebooks
"${HOME}/micromamba/envs/multimfm/bin/jupyter" nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=multimfm --ExecutePreprocessor.timeout=1800 \
  01_toy_multimodal.ipynb
echo "collected: results/toy/ + notebooks/01_toy_multimodal.ipynb"

#!/bin/bash
#SBATCH --job-name=dna_nb_rerun
#SBATCH --partition=gpu_test
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Execute the DNA notebooks with RERUN = True on one GPU (written by the dna-rerun agent; moved here
# from outputs/dna/notebook_rerun/ by dna-core).
#
# DNA_RERUN=1 is read by notebooks/_dna_helpers.rerun_flag from each notebook's first cell, so the
# tracked notebooks keep `RERUN = False` and their CPU outputs; the executed copies go to OUT
# (default outputs/dna/notebook_rerun/executed, gitignored), never in place. Every path must be on a
# SHARED filesystem: /tmp is node-local on the compute nodes.
#
#   sbatch scripts/dna/notebook_rerun.sbatch
#   NOTEBOOKS="03_dna_cyclizability 05_dna_mc_scaling" sbatch scripts/dna/notebook_rerun.sbatch
#
# Single-notebook equivalent (same RERUN substitution, done in Python):
#   python notebooks/_dna_helpers.py rerun 05_dna_mc_scaling.ipynb --out_dir <dir>
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"
set +e

# Job-unique scratch for the rerun cells: notebooks/_dna_helpers.rerun_dir(fresh=True) wipes its
# target, so concurrent rerun jobs must not share it. _dna_helpers reads DNA_RERUN_OUT.
export DNA_RERUN_OUT="${DNA_RERUN_OUT:-${REPO}/outputs/dna/notebook_rerun/job_${SLURM_JOB_ID:-local_$$}}"
OUT="${OUT:-${DNA_RERUN_OUT}/executed}"
NOTEBOOKS="${NOTEBOOKS:-03_dna_cyclizability 05_dna_mc_scaling 02_dna_base_model 04_dna_exact_rewards}"
mkdir -p "${OUT}"
export DNA_RERUN=1
echo "out=${OUT} rerun_scratch=${DNA_RERUN_OUT} notebooks=${NOTEBOOKS}"
cd "${REPO}/notebooks"
FAILED=()
for nb in ${NOTEBOOKS}; do
  echo "===== ${nb} ($(date +%T))"
  ${PY} -m jupyter nbconvert --to notebook --execute --allow-errors \
      --ExecutePreprocessor.kernel_name=multimfm --ExecutePreprocessor.timeout=10800 \
      --ExecutePreprocessor.startup_timeout=300 \
      --output-dir "${OUT}" --output "${nb}_RERUN.ipynb" "${nb}.ipynb"
  rc=$?
  ${PY} - "${OUT}/${nb}_RERUN.ipynb" <<'PY'
import sys, nbformat
nb = nbformat.read(sys.argv[1], as_version=4)
bad = [(i, o.ename, (o.evalue or "")[:200]) for i, c in enumerate(nb.cells) if c.cell_type == "code"
       for o in c.get("outputs", []) if o.output_type == "error"]
print(f"[{sys.argv[1]}] cells with errors: {bad if bad else 'none'}")
sys.exit(1 if bad else 0)
PY
  if [[ $? -ne 0 || ${rc} -ne 0 ]]; then FAILED+=("${nb}"); fi
  echo "===== ${nb} done ($(date +%T))"
done
if [[ ${#FAILED[@]} -gt 0 ]]; then echo "NB_RERUN_FAILED: ${FAILED[*]}"; exit 1; fi
echo "NB_RERUN_ALL_DONE"

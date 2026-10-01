#!/usr/bin/env bash
#SBATCH --job-name=qm9_mmfm_at_ss_gamma
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Does plain multiMFM get better numbers at steer-search's guidance strength?
#
# Table 2's two rows use very different gamma (e.g. alpha 3.4 vs 11). Both were
# chosen under one rule -- lowest oracle MAE with PoseBusters validity >= 0.90 --
# and the question is whether the multiMFM row is leaving MAE on the table by
# obeying that floor. Neither sweep ever reached the other's gamma
# (multimfm_search_grid.csv tops out at 4.0 for alpha against SS's 11), so this
# is unmeasured, not merely unreported.
#
# Runs plain multiMFM at SS's gamma for all six properties, same seed and target
# seed as the stored seed-1 runs, so the molecules are comparable one to one.
#
#   sbatch scripts/qm9/multimfm_at_ss_gamma.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh

N="${N:-256}"
SEED="${SEED:-1}"
export TARGET_SEED="${TARGET_SEED:-2027}"
export PER_SAMPLE_PB=1 PB_WORKERS="${SLURM_CPUS_PER_TASK:-8}" VALUE_BATCH_SIZE=0
export SEED

# SS's selected strengths (scripts/run_steer_search.sh; lumo uses the 3-seed rerun value).
declare -A SSG=( [cv]=5 [mu]=3 [alpha]=11 [gap]=1 [homo]=1.5 [lumo]=3 )

for prop in cv mu alpha gap homo lumo; do
  out="outputs/qm9/multimfm_at_ss_gamma/${prop}_g${SSG[$prop]}_s${SEED}_n${N}"
  if [ -f "${out}/summary.json" ]; then echo "skip ${prop} (done)"; continue; fi
  echo "== ${prop} at gamma=${SSG[$prop]} (multiMFM default is the script's own)"
  OUTPUT_DIR="${out}" bash scripts/qm9/run_multimfm.sh "${prop}" "${N}" "${SSG[$prop]}" || \
    echo "FAILED ${prop}" >&2
done
echo "done"

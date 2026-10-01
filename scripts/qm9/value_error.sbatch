#!/usr/bin/env bash
#SBATCH --job-name=qm9_value_error
#SBATCH --partition=kempner
#SBATCH --account=kempner_albergo_lab
#SBATCH --constraint="a100|h100|h200"
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# QM9 value-function estimation error against the number of posterior samples: the QM9
# analogue of the DNA posterior-recovery panel, so the two panels of the steering figure
# can both be QM9 rather than one per domain.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env.sh
# The reference is itself a Monte-Carlo estimate, and the QM9 reward makes it a demanding
# one: r = -(pred - target)^2 in Bohr^3 squared, so exp(r) spans many orders of magnitude
# and the log-mean-exp needs a large pool before its own self-split spread drops below the
# errors being measured. At 32 reference samples that spread was 7.69 against a DPS error
# of 5.79 -- the floor was above the signal. Hence the large default here.
python -u scripts/qm9/value_error.py \
  --n-conditions "${NCOND:-32}" --n-ref "${NREF:-4096}" --chunk "${CHUNK:-128}" \
  --mcs "${MCS:-1,2,4,8,16,32,64,128,256,512}" \
  --out "${OUT:-results/qm9/value_error.csv}"

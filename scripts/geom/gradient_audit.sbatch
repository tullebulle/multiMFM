#!/usr/bin/env bash
#SBATCH --job-name=geom_gradaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Is the GEOM-Drugs fingerprint guidance term implemented correctly?
#
# Runs the two diagnostics of `multimfm.geom.diagnostics` back to back in one
# allocation, because the second is the one that can actually answer the question:
#
#   grad-sign  the original directional probe (§7 item 1 of the reproducibility
#              plan). Re-draws look-ahead noise between the gradient and the probes,
#              so it measures a NOISY function: it can only ever say "the gradient
#              points somewhere plausible", and its default step sizes
#              (0.1/0.3/1.0 times a gradient of RMS ~2) move coordinates by ~1 A,
#              far outside any linear regime.
#   fd-check   the deterministic replacement. Freezes the look-ahead prior noise so
#              V_t(x) = log mean_i exp(lambda r(x_1^i(x))) is an ordinary function,
#              then compares the coded gradient with a central finite difference of
#              that same function at several step sizes. Ratio -> +1 means the sign
#              AND the magnitude are right; ratio -> -1 is a sign inversion.
#
# Deliberately small (2 CPUs, 12G, 30 min) so it backfills on a contended cluster.
# On a GPU add --gres=gpu:1 and DEVICE=cuda; it then takes well under a minute.
#
#   sbatch --partition=serial_requeue --requeue scripts/geom/gradient_audit.sbatch
#   sbatch --partition=gpu_test --account=kempner_albergo_lab --gres=gpu:1 \
#          DEVICE=cuda scripts/geom/gradient_audit.sbatch
#
# The login node cannot host these reliably: the user slice is capped at 8 GB
# (systemd MemoryMax) and shared with every other process of the same user, and
# fd-check peaks near 3.7 GB, so it is SIGKILLed whenever anything else is running.
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

DEVICE="${DEVICE:-cpu}"
OUTDIR="${OUTDIR:-results/geom/diagnostics}"
NUM_MOLS="${NUM_MOLS:-4}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-4}"   # App. F.8: four posterior samples
MFM_STEPS="${MFM_STEPS:-10}"                  # App. F.8: ten flow-map inner steps
OUTER_STEPS="${OUTER_STEPS:-100}"             # App. F.8: 100-step outer ODE
T_PROBE="${T_PROBE:-0.5}"
LAMBDA="${LAMBDA:-0.01}"                      # the as-described rerun's lambda
SEED="${SEED:-0}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-2}}"

mkdir -p "${OUTDIR}"

echo "=== fd-check (deterministic; this is the verdict) ==="
"${PY}" -m multimfm.geom.diagnostics fd-check \
  --device "${DEVICE}" --num-mols "${NUM_MOLS}" --t "${T_PROBE}" \
  --outer-steps "${OUTER_STEPS}" --posterior-samples "${POSTERIOR_SAMPLES}" \
  --mfm-steps "${MFM_STEPS}" --reward-lambda "${LAMBDA}" \
  --value-batch-size 1 --threads "${THREADS}" --seed "${SEED}" \
  --rel-steps ${REL_STEPS:-1e-8 1e-7 1e-6 1e-5 1e-4} \
  --out "${OUTDIR}/fd_check_t${T_PROBE}_lam${LAMBDA}_${DEVICE}.json"

echo
echo "=== grad-sign (the original probe, for the record) ==="
"${PY}" -m multimfm.geom.diagnostics grad-sign \
  --device "${DEVICE}" --num-mols "${NUM_MOLS}" --t "${T_PROBE}" \
  --outer-steps "${OUTER_STEPS}" --posterior-samples "${POSTERIOR_SAMPLES}" \
  --mfm-steps "${MFM_STEPS}" --probe-samples 16 \
  --reward-lambda "${LAMBDA}" --steps 0.001 0.01 0.1 0.3 1.0 --seed "${SEED}" \
  --out "${OUTDIR}/grad_sign_${DEVICE}.json"

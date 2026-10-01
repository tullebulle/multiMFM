#!/usr/bin/env bash
#SBATCH --job-name=geom_gradsign
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Does the estimated value gradient actually ascend the guide's own reward?
# (`multimfm.geom.diagnostics grad-sign`; see docs/provenance/geom/sign_audit.md.)
#
# This is the experiment that separates "implementation defect" from "property of the method"
# for the GEOM-Drugs fingerprint result: every held-out (lambda, mu) tried lowers both the
# counted FP2 Tanimoto AND the head-predicted one, i.e. guidance fails to improve the very
# quantity it optimises. The hand audit found no sign inversion, so what remains to be
# measured is whether the 4-sample log-mean-exp differentiated through a 10-step flow map
# points uphill at all.
#
# GPU (fast, minutes) -- needs a free GPU on a contended account:
#   sbatch --partition=kempner_requeue --requeue --constraint="cc8.0|cc9.0" \
#          --account=kempner_albergo_lab --gres=gpu:1 scripts/geom/grad_sign.sbatch
#   sbatch --partition=gpu_test --account=kempner_albergo_lab --gres=gpu:1 \
#          scripts/geom/grad_sign.sbatch
#
# CPU fallback:
#   DEVICE=cpu NUM_MOLS=4 PROBE_SAMPLES=8 MFM_STEPS=6 OUTER_STEPS=20 \
#       sbatch --partition=serial_requeue --requeue --time=02:00:00 scripts/geom/grad_sign.sbatch
#
# CORRECTION (2026-09-29): the comment that used to sit here -- "the login node cannot run
# it, it gets SIGKILLed, since each checkpoint carries a 1.14M-entry all_smiles list" -- is
# wrong. Loading the teacher, the student and the head on the login node takes 2.2 s and
# peaks at 1.28 GB. The actual limit is that the login node puts the whole *user* in a
# systemd slice with MemoryMax=8G and CPUQuotaPerSecUSec=1s, shared with every other
# process that user is running; with several agents active, 5.5-7.3 GB of the 8 GB was
# already taken and this command (peak 3.65 GB) was SIGKILLed. So it is contention, not the
# checkpoint. `multimfm.geom.diagnostics fd-check --stages reward soft` needs neither the
# teacher nor the student (2.05 GB peak, ~2 min) and does run on the login node.
#
# READ THIS BEFORE QUOTING grad-sign's VERDICT. Its default probe steps (0.1/0.3/1.0) are
# absolute, against a measured gradient RMS of ~2.2, so even the smallest displaces every
# coordinate by ~0.4 A -- and `diagnostics fp2-sensitivity` shows FP2 is fully randomised by
# 0.2 A. It also re-draws the look-ahead noise between the gradient and the probes, so the
# function it probes is not the one the gradient was taken of. Use
# `diagnostics fd-check` (deterministic, auto-scaled steps) for the sign question, or
# `scripts/geom/gradient_audit.sbatch`, which runs both. See docs/geom_guidance_diagnosis.md.
#
# Writes results/geom/grad_sign.json (path overridable with OUT).
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/geom/_header.sh"

DEVICE="${DEVICE:-cuda}"
NUM_MOLS="${NUM_MOLS:-8}"
PROBE_SAMPLES="${PROBE_SAMPLES:-16}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-4}"   # App. F.8: four posterior samples
MFM_STEPS="${MFM_STEPS:-10}"                  # App. F.8: ten flow-map inner steps
OUTER_STEPS="${OUTER_STEPS:-100}"             # App. F.8: 100-step outer ODE
T_PROBE="${T_PROBE:-0.5}"
LAMBDA="${LAMBDA:-0.02}"
STEPS="${STEPS:-0.1 0.3 1.0}"
SEED="${SEED:-0}"
OUT="${OUT:-results/geom/grad_sign.json}"

echo "device=${DEVICE} n=${NUM_MOLS} t=${T_PROBE} lambda=${LAMBDA} probes=${STEPS}"
"${PY}" -m multimfm.geom.diagnostics grad-sign \
  --device "${DEVICE}" --num-mols "${NUM_MOLS}" --t "${T_PROBE}" \
  --outer-steps "${OUTER_STEPS}" --posterior-samples "${POSTERIOR_SAMPLES}" \
  --mfm-steps "${MFM_STEPS}" --probe-samples "${PROBE_SAMPLES}" \
  --reward-lambda "${LAMBDA}" --steps ${STEPS} --seed "${SEED}" --out "${OUT}"
echo "wrote ${OUT}"

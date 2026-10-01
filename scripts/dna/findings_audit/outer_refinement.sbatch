#!/bin/bash
#SBATCH --job-name=dna_fa_outer
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
# kempner_requeue holds 24 RTX PRO 6000 Blackwell nodes (sm_120); torch 2.5.1+cu124 has no sm_120
# kernels, so a job landing there dies ~40 s in. Restrict to the A100/H100/H200 pool.
#SBATCH --constraint="a100|a100-mig|h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# The DNA analogue of the QM9 DPS `sample_steps` sweep, which is the cross-domain constraint on
# item (c). The two experiments refine *different* things:
#
#   QM9 DPS  --sample-steps 128 -> 4096 refines the OUTER guided trajectory. Guidance enters as
#            `mu * grad * dt` (`glass_guidance.production_step_with_optional_guidance`) with
#            `--guidance-max-rms 0` by default, so the guidance integral really is dt-invariant and
#            the limit is the exact solution of the guided ODE.
#   DNA      `glass_euler32` refines the INNER posterior that estimates the value gradient, leaving
#            the outer trajectory at `nfe_traj 64`. Its limit is the exact value gradient, a
#            different limit, and the arm that was run is degenerate (see docs/dna_findings_audit.md).
#
# So the QM9 result does not support the DNA one, and vice versa. This job supplies the missing
# matched arm: refine the DNA *outer* trajectory at a fixed inner posterior (GLASS-4, the Table 1
# estimator) and at a fixed guidance rule, and see whether DNA degrades the way QM9 does. The
# guidance coefficient here is `dt * min(guidance_frac sigma_t^2, coeff_cap) * clip(grad, grad_clip)`,
# which carries the same dt and is therefore invariant in the same sense.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

PROBE="scripts/dna/findings_audit/posterior_accuracy_probe.py"
OUT="results/dna/findings_audit"

for nfe in 64 128 256 512; do
  ${PY} -u "${PROBE}" --mode sample --n_samples "${N_SAMPLES:-32}" --batch_size 8 \
    --targets 1.0 -1.0 --nfe_traj "${nfe}" \
    --configs glass_euler4 glass_euler4@norm \
    --out_dir "${OUT}/sample_outer_nfe${nfe}"
done

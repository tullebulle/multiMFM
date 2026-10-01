#!/bin/bash
#SBATCH --job-name=dna_fa_postacc
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
# kempner_requeue holds 24 RTX PRO 6000 Blackwell nodes (sm_120); torch 2.5.1+cu124 has no sm_120
# kernels, so a job landing there dies ~40 s in. Restrict to the A100/H100/H200 pool.
#SBATCH --constraint="a100|a100-mig|h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Audit of the discretisation-error finding (plan S7 item 9 / docs/dmfm_glass_parity.md S2):
# does a *well-integrated* posterior really fail to guide, or was `glass_euler32` (a) integrated
# to the singular data time t=1 and (b) run at a guidance strength calibrated for a gradient
# 20-5000x larger?  See docs/dna_findings_audit.md.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

PROBE="scripts/dna/findings_audit/posterior_accuracy_probe.py"
OUT="results/dna/findings_audit"

# 1. Gradient agreement, against the Table-18 reference integrator (RK4-25 to 0.999), with the
#    Euler ladder to 1.0 included so the degeneracy of a fine Euler grid at t=1 is visible.
${PY} -u "${PROBE}" --mode grad \
  --n_probes 4 --probe_times 0.5 0.7 --targets 1.0 -1.0 \
  --reference_steps 25 --reference_solver rk4 --reference_end_time 0.999 \
  --configs glass_euler4 glass_euler8 glass_euler16 glass_euler32 glass_euler100 \
            glass_euler4_e999 glass_euler32_e999 glass_euler100_e999 \
            glass_rk4_8_e999 glass_rk4_25_e999 \
  --out_dir "${OUT}/grad_posterior_accuracy"

# 2. End-to-end guided sampling, matched noise / matched MC pool, n=64 both targets, exactly the
#    protocol of results/dna/parity/sample_L50_n64 so the numbers are comparable to that table.
${PY} -u "${PROBE}" --mode sample --n_samples "${N_SAMPLES:-64}" --batch_size 8 --targets 1.0 -1.0 \
  --configs glass_euler4 \
            glass_euler32 glass_euler32@norm \
            glass_euler32_e999 glass_euler32_e999@norm \
            glass_rk4_25_e999 glass_rk4_25_e999@norm \
  --out_dir "${OUT}/sample_posterior_accuracy"

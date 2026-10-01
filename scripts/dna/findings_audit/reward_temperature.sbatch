#!/bin/bash
#SBATCH --job-name=dna_fa_rewardT
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
# Item (d): is the undisclosed reward temperature only a disclosure gap?
# Eq (17)/(48) state r = -(f - y*)^2; the code exponentiates
# reward_scale * -0.5 ((f - y*)/reward_sigma)^2, i.e. -11.111 (f - y*)^2 at the Table 1 defaults
# (reward_sigma 0.15, reward_scale 0.5) and -22.222 at reward_scale 1.0 (the June Tables 4-6 sets).
# reward_scale 0.045 = 2 * 0.15^2 makes the coefficient exactly 1, i.e. the literal equation.
#
# Two configurations per temperature: the stock clip-down rule (where the coefficient multiplies the
# step directly, because GLASS-4's gradient is mostly below grad_clip) and the scale-free rule
# (where only the softmax-weight shape can matter). Same ids, same noise, same MC pool throughout.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

PROBE="scripts/dna/findings_audit/posterior_accuracy_probe.py"
OUT="results/dna/findings_audit"

for rs in 0.045 0.5 1.0; do
  tag="${rs/./p}"
  ${PY} -u "${PROBE}" --mode sample --n_samples "${N_SAMPLES:-32}" --batch_size 8 \
    --targets 1.0 -1.0 --reward_scale "${rs}" \
    --configs glass_euler4 glass_euler4@norm \
    --out_dir "${OUT}/sample_reward_scale_${tag}"
done

# The ESS diagnostic on GPU as well, so the CPU numbers have a device-independent check.
${PY} -u scripts/dna/findings_audit/reward_temperature.py \
  --n_probes 8 --probe_times 0.5 0.7 0.9 --mc 8 64 --n_calib 256 \
  --out_dir "${OUT}/reward_temperature_gpu"

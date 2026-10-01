#!/bin/bash
#SBATCH --job-name=dna_fa_junedose
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
# kempner_requeue holds 24 RTX PRO 6000 Blackwell nodes (sm_120); torch 2.5.1+cu124 has no sm_120
# kernels, so a job landing there dies ~40 s in. Restrict to the A100/H100/H200 pool.
#SBATCH --constraint="a100|a100-mig|h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Audit of the Tables 4-6 attribution (plan S7 item 7): the regenerated diversity / Evo 2 sets
# were sampled with a *much weaker* guidance protocol than the published June sets, so the move
# from `split65k` to parent-disjoint is not the only candidate cause of the lost clustering.
#
# Published June protocol (results/dna/june_split65k_recovered/samples_target_*.args.json, the
# producer being scripts/sample_yeast_split_dmfm_c0.py + DMFMSteeringRunner, ported verbatim as
# dmfm.experiments.sample_c0_guidance_dmfm whose *defaults* are exactly those args):
#   mc 16, nfe_value 1 (one-step student posterior), nfe_sample 96 to t_max 1.0,
#   guidance window [0.01, 0.95] (~91 guided steps), reward_scale 1.0, guidance_frac 8,
#   coeff_cap 10, grad_clip 10 (stock clip-down), batch 10, seed 0 (y*=+1) / 1 (y*=-1).
# Regenerated protocol (results/dna/diversity/regen, results/dna/dmfm_demos/diversity):
#   mc 8, nfe_value 4, nfe_sample 64 to t_max 0.95, window [0.50, 0.95] (~30 guided steps),
#   reward_scale 0.5, paired noise.
#
# This job re-runs the *published* protocol on the parent-disjoint model, changing only the model
# and the guide (which is all S7 item 7 is about). ARM=ablate runs two one-knob-back controls.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

N="${N_SAMPLES:-1000}"
OUT="${OUT_ROOT:-outputs/dna/findings_audit}"
ARM="${ARM:-june}"

run () {  # run <subdir> <target> <seed> [extra flags...]
  local sub="$1" target="$2" seed="$3"; shift 3
  local dir="${OUT}/${sub}"
  if [ -f "${dir}/sample_scores.csv" ] && \
     [ "$(( $(wc -l < "${dir}/sample_scores.csv") - 1 ))" -eq "${N}" ]; then
    echo "skip ${sub} (already complete)"; return 0
  fi
  echo "=== ${sub}: target=${target} seed=${seed} extra='$*'"
  ${PY} -u -m dmfm.experiments.sample_c0_guidance_dmfm \
    --out_dir "${dir}" --n_samples "${N}" --target_c0 "${target}" --seed "${seed}" "$@"
}

if [ "${ARM}" = "june" ]; then
  # Published June protocol = every module default; only --target_c0/--seed/--out_dir given.
  run june_dose_m1p0 -1.0 1
  run june_dose_1p0   1.0 0
else
  # One knob back toward the regenerated protocol, y*=-1 only (the contested set).
  run window050_m1p0 -1.0 1 --guide_t_start 0.50
  run rscale05_m1p0  -1.0 1 --reward_scale 0.5
fi

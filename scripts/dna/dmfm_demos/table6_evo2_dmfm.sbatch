#!/bin/bash
#SBATCH --job-name=dna_t6_evo2_dmfm
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_albergo_lab
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Table 6 (Evo 2 plausibility) recomputed on the **dMFM**-guided n = 1,000 C0 sets
# (REPRODUCIBILITY_PLAN.md S7 item 8). Table 6 scores *generated sequences*, so scoring the
# GLASS sets describes GLASS output; this scores the dMFM sets that Table 1 / Fig 3 now cite
# (`--posterior dmfm --grad_normalize --nfe_value 4`, Slurm 49352242).
#
# Identical to `scripts/dna/regen/score_evo2.sbatch MODE=c0` in every argument except
# `--run_glob` and `--out_dir`, so the dMFM and GLASS columns are comparable. The real-data
# reference rows do not depend on the generator and are recomputed here as a cross-check.
#
#   sbatch scripts/dna/dmfm_demos/table6_evo2_dmfm.sbatch
#   ARM=glass_norm sbatch scripts/dna/dmfm_demos/table6_evo2_dmfm.sbatch   # matched control
set -euo pipefail
REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
[[ -d src/dmfm ]] || { echo "Run from the MultiMFM repo root (src/dmfm not found)" >&2; exit 2; }
mkdir -p logs

ARM="${ARM:-dmfm4_norm}"
SET_ROOT="${SET_ROOT:-outputs/dna/table1_${ARM}_n1000}"
RUN_GLOB_P1="${RUN_GLOB_P1:-${SET_ROOT}/target_1p0/merged/sample_scores.csv}"
RUN_GLOB_M1="${RUN_GLOB_M1:-${SET_ROOT}/target_m1p0/merged/sample_scores.csv}"
OUT_DIR="${OUT_DIR:-outputs/dna/dmfm_demos/table6/${ARM}}"
for f in "${RUN_GLOB_P1}" "${RUN_GLOB_M1}"; do
  [ -f "${f}" ] || { echo "missing ${f}" >&2; exit 2; }
done

# Environment: the separate evo2 micromamba env (scripts/dna/regen/evo2_environment.yml).
EVO2_PREFIX="${EVO2_PREFIX:-${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}/envs/${EVO2_ENV:-evo2}}"
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate "${EVO2_PREFIX}"
else
  export PATH="${EVO2_PREFIX}/bin:${PATH}"
fi
[ "$(command -v python)" = "${EVO2_PREFIX}/bin/python" ] || { echo "evo2 env not active: $(command -v python)" >&2; exit 2; }
if command -v module >/dev/null 2>&1; then
  module load gcc/12.2.0-fasrc01 cuda/12.9.1-fasrc01 2>/dev/null || true
fi
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HOME="${HF_HOME:-${REPO_ROOT}/cache/xdg/huggingface}"
python "${REPO_ROOT}/scripts/dna/regen/evo2_postinstall.py"

mkdir -p "${OUT_DIR}"
{
  printf '{\n'
  printf '  "host": "%s",\n' "$(hostname)"
  printf '  "slurm_job_id": "%s",\n' "${SLURM_JOB_ID:-none}"
  printf '  "partition": "%s",\n' "${SLURM_JOB_PARTITION:-unknown}"
  printf '  "account": "%s",\n' "${SLURM_JOB_ACCOUNT:-unknown}"
  printf '  "gpu": "%s",\n' "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd'; ' - || echo unknown)"
  printf '  "arm": "%s",\n' "${ARM}"
  printf '  "sets": ["%s", "%s"],\n' "${RUN_GLOB_M1}" "${RUN_GLOB_P1}"
  printf '  "model_name": "%s",\n' "${MODEL_NAME:-evo2_7b_base}"
  printf '  "date": "%s"\n' "$(date -Is)"
  printf '}\n'
} > "${OUT_DIR}/run_environment.json"
cat "${OUT_DIR}/run_environment.json"

python -u -m dmfm.experiments.score_yeast_c0_evo2 \
  --run_glob "${RUN_GLOB_P1}" \
  --run_glob "${RUN_GLOB_M1}" \
  --train_pt "${TRAIN_PT:-data/dna/yeast_parent_disjoint/yeast_parent_L50.pt}" \
  --split_pt "${SPLIT_PT:-data/dna/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt}" \
  --train_split "${TRAIN_SPLIT:-a_train}" \
  --val_split "${VAL_SPLIT:-a_val}" \
  --test_split "${TEST_SPLIT:-test}" \
  --out_dir "${OUT_DIR}" \
  --targets -1 1 \
  --baseline_n "${BASELINE_N:-1000}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --bootstrap "${BOOTSTRAP:-2000}" \
  --model_name "${MODEL_NAME:-evo2_7b_base}" \
  --device "${DEVICE:-cuda:0}"

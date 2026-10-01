#!/bin/bash
#SBATCH --job-name=dna_evo2
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Evo2-7B plausibility scoring for the DNA experiments, in the separate Evo2 environment
# (scripts/dna/evo2_environment.yml). Submit from the repo root:
#
#   # Table 6: C0-steered sets (target +1 / -1) + real-data baselines (1,000 each)
#   MODE=c0 sbatch scripts/dna/score_evo2.sbatch
#
#   # Table 13 / App. F.1 Evo2 column: one job per reward x length (8 jobs in the paper)
#   for R in motif conjunction; do for L in 50 100 200 400; do
#     MODE=reward SAMPLE_DIR=outputs/dna/rebuttal_reward_attainment_${R}_20260731/L${L} \
#       sbatch scripts/dna/score_evo2.sbatch; done; done
#
#   # Posterior-diversity Evo2 check (rebuttal; not in the paper tables; original --time 12:00:00)
#   MODE=posterior FUTURES_CSV=<.../posterior_futures.csv> sbatch --time=12:00:00 scripts/dna/score_evo2.sbatch
#
# Arguments reproduce the original invocations exactly:
#   c0        June-10 Table 6 run (Codex session log, 2026-06-10T20:08Z; outputs purged):
#             --targets -1 1 --baseline_n 1000 --batch_size 1 --bootstrap 2000
#             --model_name evo2_7b_base --device cuda:0   (seed 0, RC-averaged, no BOS: defaults)
#             NOTE: that run scored the older split65k sample sets
#             (workdir/yeast_split_dmfm_c0_samples_target_{p1,m1}_2026-06-02) against
#             data/yeast_mid50.pt with split keys train/val/test. Those inputs are purged; the
#             defaults below score the parent-disjoint n=1000 sets (Tables 1/4/5 pipeline) with
#             split keys a_train/a_val/test instead. Set TRAIN_PT/SPLIT_PT/TRAIN_SPLIT=train/
#             VAL_SPLIT=val/TEST_SPLIT=test and RUN_GLOB_* to reproduce the old configuration.
#   reward    slurm/score_parent_reward_attainment_evo2.sbatch (jobs 35387510-35387523, 2026-07-26):
#             --model_name evo2_7b_base --batch_size 1 --bootstrap 2000 --seed 20260731
#   posterior slurm/score_parent_posterior_evo2.sbatch (job 35387531, 2026-07-26):
#             --model_name evo2_7b_base --batch_size 4 --sources_per_cell 16 --bootstrap 2000
#             --seed 20260730
set -euo pipefail

REPO_ROOT="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
[[ -d src/dmfm ]] || { echo "Run from the MultiMFM repo root (src/dmfm not found in ${REPO_ROOT})" >&2; exit 2; }
mkdir -p logs

MODE="${MODE:?Set MODE to c0, reward or posterior}"

# Environment: micromamba env built from scripts/dna/evo2_environment.yml.
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${EVO2_ENV:-evo2}"
if command -v module >/dev/null 2>&1; then
  module load gcc/12.2.0-fasrc01 cuda/12.9.1-fasrc01 2>/dev/null || true   # as in the original runs
fi
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
# Evo2 weights (~14 GB) cache: git-ignored and inside the repo by default; override freely.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/cache/xdg/huggingface}"

MODEL_NAME="${MODEL_NAME:-evo2_7b_base}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
DEVICE="${DEVICE:-cuda:0}"
echo "host=$(hostname) mode=${MODE} model=${MODEL_NAME} python=$(command -v python) HF_HOME=${HF_HOME}"

case "${MODE}" in
  c0)
    RUN_GLOB_P1="${RUN_GLOB_P1:-results/dna/c0_guidance/n1000/target_1p0/sample_scores.csv}"
    RUN_GLOB_M1="${RUN_GLOB_M1:-results/dna/c0_guidance/n1000/target_m1p0/sample_scores.csv}"
    TRAIN_PT="${TRAIN_PT:-data/dna/yeast_parent_disjoint/yeast_parent_L50.pt}"
    SPLIT_PT="${SPLIT_PT:-data/dna/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt}"
    OUT_DIR="${OUT_DIR:-outputs/dna/yeast_parent_c0_evo2}"
    python -u -m dmfm.experiments.score_yeast_c0_evo2 \
      --run_glob "${RUN_GLOB_P1}" \
      --run_glob "${RUN_GLOB_M1}" \
      --train_pt "${TRAIN_PT}" \
      --split_pt "${SPLIT_PT}" \
      --train_split "${TRAIN_SPLIT:-a_train}" \
      --val_split "${VAL_SPLIT:-a_val}" \
      --test_split "${TEST_SPLIT:-test}" \
      --out_dir "${OUT_DIR}" \
      --targets -1 1 \
      --baseline_n "${BASELINE_N:-1000}" \
      --batch_size "${BATCH_SIZE:-1}" \
      --bootstrap "${BOOTSTRAP}" \
      --model_name "${MODEL_NAME}" \
      --device "${DEVICE}"
    ;;
  reward)
    SAMPLE_DIR="${SAMPLE_DIR:?Set SAMPLE_DIR to a length-specific reward-attainment directory}"
    python -u -m dmfm.experiments.score_parent_reward_attainment_evo2 \
      --sample_dir "${SAMPLE_DIR}" \
      --out_dir "${OUT_DIR:-${SAMPLE_DIR}/evo2}" \
      --model_name "${MODEL_NAME}" \
      --batch_size "${BATCH_SIZE:-1}" \
      --bootstrap "${BOOTSTRAP}" \
      --seed "${SEED:-20260731}" \
      --device "${DEVICE}"
    ;;
  posterior)
    FUTURES_CSV="${FUTURES_CSV:-outputs/dna/parent_posterior_diversity_mapselected_flowmap1_glass100_20260726/posterior_futures.csv}"
    [[ -f "${FUTURES_CSV}" ]] || { echo "Missing futures CSV: ${FUTURES_CSV}" >&2; exit 2; }
    python -u -m dmfm.experiments.score_parent_posterior_evo2 \
      --futures_csv "${FUTURES_CSV}" \
      --out_dir "${OUT_DIR:-outputs/dna/parent_posterior_diversity_mapselected_flowmap1_glass100_evo2_20260726}" \
      --model_name "${MODEL_NAME}" \
      --batch_size "${BATCH_SIZE:-4}" \
      --sources_per_cell 16 \
      --bootstrap "${BOOTSTRAP}" \
      --seed "${SEED:-20260730}" \
      --device "${DEVICE}"
    ;;
  *)
    echo "Unknown MODE=${MODE}; expected c0, reward or posterior" >&2
    exit 2
    ;;
esac

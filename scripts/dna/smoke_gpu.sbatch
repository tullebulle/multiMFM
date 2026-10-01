#!/bin/bash
#SBATCH --job-name=dna_smoke
#SBATCH --partition=gpu_test
#SBATCH --account=kempner_albergo_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# GPU smoke test of the ported DNA code (minutes, not a paper run). PARTS selects what runs
# (default: all; split across two jobs to stay inside the 1 h gpu_test limit):
#
#   sample     Table 1 / Fig 3: C0-guided sampling at L=50, target +1, 8 pairs; independent-oracle
#              scoring; check that the (deterministic) unguided samples and scores equal the stored
#              n=1000 run (results/dna/c0_guidance/n1000/target_1p0).
#   train      20 steps each of base DFM, dMFM (diagonal + ESD) and 4-step student (from the shipped
#              diagonal student), and a 2-epoch C0 regressor.
#   gradmc     Fig 5 / Tables 14-15 and Table 18 numerical check against stored gradients
#              (tests/dna/check_gradient_mc_reference.py), then the scale-free rescore/summary on the
#              tiny Fig 5 output.
#   reward     Table 13: exact motif reward attainment, L=50, 4 pairs.
#   posterior  Table 17: posterior diversity, L=50, 2 sources x 4 futures.
#   (Fig 4 has its own smoke test, scripts/dna/fig4_smoke.sbatch.)
#
#   PARTS="sample train reward posterior" sbatch scripts/dna/smoke_gpu.sbatch
#   PARTS=gradmc sbatch scripts/dna/smoke_gpu.sbatch
# Slurm runs a copy of this file from its spool dir, so locate the repo from the submit dir.
_H="${MULTIMFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/dna/_header.sh"
[ -f "${_H}" ] || { echo "submit from the MultiMFM repo root (or set MULTIMFM_ROOT)" >&2; exit 1; }
source "${_H}"

PARTS="${PARTS:-sample train gradmc reward posterior}"
OUT="${OUT:-outputs/dna/smoke_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}"
mkdir -p "${OUT}"
echo "smoke output: ${OUT}; parts: ${PARTS}"
${PY} -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
has() { [[ " ${PARTS} " == *" $1 "* ]]; }
set +e   # keep going after a failing step; failures are listed at the end
FAILED=()
run() { "$@"; local rc=$?; if [[ ${rc} -ne 0 ]]; then echo "!!!!! FAILED (rc=${rc}): $*" | cut -c1-300; FAILED+=("${4:-$3}"); fi; }
tick() { echo "===== $1 ($(date +%T))"; }

if has sample; then
  tick "sample: C0-guided sampling (L=50, target +1, n=8)"
  run ${PY} -u -m dmfm.experiments.sample_c0_guidance \
    --ckpt checkpoints/dna/base/L50/best.pt \
    --c0_ckpt checkpoints/dna/c0/guide/best_state.pt \
    --output_dir "${OUT}/c0_target1p0" \
    --seed 0 --n_samples 8 --batch_size 8 --target_c0 1.0 \
    --reward_sigma 0.15 --reward_scale 0.5 --mc 8 --mc_chunk 8 \
    --nfe_traj 64 --nfe_value 4 --t_max 0.95 \
    --guide_t_start 0.50 --guide_t_end 0.95 --guidance_frac 8.0 \
    --coeff_cap 10.0 --grad_clip 10.0
  tick "sample: oracle scoring"
  run ${PY} -u -m dmfm.experiments.score_c0_oracle \
    --oracle_ckpt checkpoints/dna/c0/oracle/best_state.pt \
    --oracle_model_type park_cnn \
    --run_glob "${OUT}/c0_target1p0/sample_scores.csv" \
    --out_dir "${OUT}/c0_target1p0/oracle" \
    --batch_size 512 --bootstrap 200 --seed 0 \
    --reverse_complement_average --device cuda:0
  tick "sample: compare with the stored n=1000 run (unguided must match)"
  run ${PY} -u tests/dna/check_smoke_against_reference.py \
    --run_dir "${OUT}/c0_target1p0" \
    --reference "${REFERENCE:-results/dna/c0_guidance/n1000/target_1p0}"
fi

if has train; then
  tick "train: base DFM, 20 steps"
  run ${PY} -u -m dmfm.experiments.train_base_dfm \
    --run_name smoke_base --output_dir "${OUT}/train_base" \
    --yeast_data_pt data/dna/yeast_parent_disjoint/yeast_parent_L50.pt \
    --yeast_split_pt data/dna/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt \
    --yeast_split a_train --yeast_val_split a_val --max_train_seqs 0 \
    --steps 20 --batch_size 64 --lr 5e-5 --weight_decay 0.01 --grad_clip 1.0 \
    --hidden_dim 192 --transformer_heads 6 --transformer_ff_mult 4 --dit_blocks 4 \
    --dropout 0.0 --gaussian_beta_schedule linear \
    --save_every 0 --val_every 20 --val_batches 2 --eval_samples 0 \
    --print_every 5 --num_workers 2

  dmfm_common=(
    --no-wandb --dataset_type yeast
    --yeast_data_pt data/dna/yeast_parent_disjoint/yeast_parent_L50.pt
    --yeast_split_pt data/dna/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt
    --yeast_train_split a_train --yeast_val_split a_val
    --mode gaussian --gaussian_beta_schedule linear
    --gaussian_mfm_diag_distill --gaussian_adaptive_loss
    --gaussian_adaptive_loss_c 0.01 --gaussian_adaptive_loss_r 0.5
    --mfm_teacher_ckpt checkpoints/dna/base/L50/best.pt
    --mfm_teacher_ckpt_hparams checkpoints/dna/base/L50/args.json
    --mfm_t_cond_warmup_steps 0 --mfm_t_cond_power 1.0 --mfm_t_cond_0_rate 0.0
    --mfm_consistency --mfm_consistency_type esd_teacher
    --mfm_diag_loss_weight 1.0 --mfm_lsd_loss_weight 1.0
    --mfm_diag_adaptive_p 0.5 --mfm_lsd_adaptive_p 1.0
    --mfm_fixed_probe_batch_size 8 --mfm_fixed_probe_seed 12345
    --model dit --hidden_dim 192 --transformer_heads 6 --transformer_ff_mult 4
    --dit_blocks 4 --dropout 0.0 --optimizer adam --weight_decay 0.0
    --grad_clip 1.0 --batch_size 8 --num_workers 2
    --limit_val_batches 1 --val_check_interval 10 --max_steps 20
    --print_freq 5 --log_every_n_steps 5 --no_tqdm
  )
  tick "train: dMFM (diagonal + ESD), 20 steps"
  run ${PY} -u -m dmfm.experiments.train_dmfm \
    --run_name smoke_dmfm --output_dir "${OUT}/train_dmfm" "${dmfm_common[@]}" \
    --mfm_consistency_warmup_steps 5 --mfm_consistency_anneal_end_step 1 \
    --mfm_val_gaps 0.25 --mfm_checkpoint_metric val_loss --lr 1e-4

  tick "train: 4-step student from the shipped L50 diagonal dMFM, 20 steps"
  run ${PY} -u -m dmfm.experiments.train_dmfm \
    --run_name smoke_dmfm4 --output_dir "${OUT}/train_dmfm4" "${dmfm_common[@]}" \
    --init_ckpt "checkpoints/dna/dmfm/L50/epoch=95-step=49000.ckpt" \
    --mfm_consistency_warmup_steps 0 --mfm_consistency_gap_mode long \
    --mfm_consistency_min_gap 0.0 --mfm_consistency_max_gap 0.25 \
    --mfm_val_gaps 0.0625 0.125 0.25 --mfm_checkpoint_metric val_mfm_probe_esd_gap_0.25 --lr 2e-5

  tick "train: C0 guide regressor, 2 epochs"
  run ${PY} -u -m dmfm.experiments.train_c0_regressor \
    --data_pt data/dna/yeast_parent_disjoint/yeast_parent_L50.pt \
    --split_pt data/dna/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt \
    --model_type park_cnn --epochs 2 --patience 20 --batch_size 1024 --lr 0.001 \
    --weight_decay 0.0001 --reverse_complement_augmentation --reverse_complement_average \
    --device cuda:0 --role guide --train_split a_train --validation_split a_val \
    --test_split test --seed 0 --out_dir "${OUT}/c0_guide"
fi

if has gradmc; then
  tick "gradmc: Fig 5 + Table 18 numerical check against stored gradients"
  run ${PY} -u tests/dna/check_gradient_mc_reference.py --out_dir "${OUT}/gradmc_check"
  tick "gradmc: scale-free rescore + summary of the tiny Fig 5 run"
  for rd in "${OUT}"/gradmc_check/fig5_motif_L50_end1.0; do
    run ${PY} -u -m dmfm.experiments.rescore_gradient_pairs_scale_free --input-dir "${rd}" --bootstrap 200 --seed 20260802
    run ${PY} -u -m dmfm.experiments.summarize_gradient_accuracy_metrics --run_dir "${rd}"
  done
fi

if has reward; then
  tick "reward: Table 13 motif, L=50, 4 pairs"
  run ${PY} -u -m dmfm.experiments.sample_reward_attainment \
    --lengths 50 --reward motif --output_dir "${OUT}/reward_motif" --seed 20260731 \
    --n_samples 4 --batch_size 4 --nfe_traj 64 --t_end 0.999 \
    --guide_t_start 0.50 --guide_t_end 0.95 --mc 8 --mc_chunk 4 --guidance_frac 8.0 \
    --grad_clip 10.0 --coeff_cap 10.0 --calibration_samples 64 --nfe_calibration 64 \
    --calibration_batch_size 32 --target_percentile 0.90 --reward_beta 1.0 --reward_scale 1.0 \
    --motif TTTTTC --motif2 AAAATT --motif_tau 0.10 --conjunction_tau 0.10 \
    --bootstrap 100 --device cuda:0
fi

if has posterior; then
  tick "posterior: Table 17, L=50, 2 sources x 4 futures"
  run ${PY} -u -m dmfm.experiments.eval_posterior_diversity \
    --lengths 50 --times 0.0 0.5 0.9 --n_sources 2 --n_futures 4 --source_batch 2 \
    --dmfm_steps 1 --dmfm_sampler flow_map --glass_steps 100 --bootstrap 100 \
    --seed 20260730 --out_dir "${OUT}/posterior_diversity"
fi


if [[ ${#FAILED[@]} -gt 0 ]]; then
  tick "SMOKE_FAILED: ${FAILED[*]} (output ${OUT})"
  exit 1
fi
tick "SMOKE_ALL_DONE ${OUT}"

# DNA experiments: original runs, arguments and checkpoints

Research date 2026-09-29. Sources (read-only):
`/n/netscratch/kozinsky_lab/Lab/uunneberg/repos/dirichlet-flow-matching` (below: "DFM repo"; all
`workdir/`, `data/`, `scripts/`, `slurm/` paths are relative to it), the reconstruction
`/n/netscratch/kozinsky_lab/Lab/uunneberg/scratch_dfm_recon_20260925` ("recon"), the paper text
(`dMFM (40).pdf`) and the tex in `repos/dMFM/`. Every paper-relevant file named here is also in
`/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/dirichlet-flow-matching/`.

All original runs used env `seq_h100` (Python 3.9, torch 2.1, pytorch-lightning 2.1) with
`PYTHONPATH=<DFM repo>` and library commit `187fe7b` (recorded as `commit` in every dMFM args.json),
except: Evo2 scoring (`.envs/evo2`, Python 3.12) and the 2026-09-25 recon runs (recon library,
`PYTHONPATH=<recon>:<recon>/pydeps`, env `seq_h100`).

Status legend: VERIFIED = numbers in the paper match the stored result file; UNCERTAIN = stated
with the reason.

---

## 0. Summary of checkpoints actually loaded by paper runs

| Checkpoint (DFM repo `workdir/`) | Used by | Verified |
|---|---|---|
| `yeast_parent_a_L{50,100,200,400}_dfm_dit_h192_b4/best.pt` | teacher of every dMFM; base model for Table 1 (L50), Tables 14–15/Fig 5, Table 17 GLASS reference, Table 13 teacher/calibration, Table 18 GLASS reference | yes (args/metadata of each run) |
| `yeast_parent_a_L50_dmfm_diagonal_h192_b4/epoch=95-step=49000.ckpt` | Table 13, Table 17, init of 4-step L50 | yes |
| `yeast_parent_a_L100_dmfm_diagonal_h192_b4/epoch=80-step=39000.ckpt` | Table 13, Table 17, init of 4-step L100 | yes |
| `yeast_parent_a_L200_dmfm_diagonal_h192_b4/epoch=51-step=45000.ckpt` | Table 13, Table 17, init of 4-step L200 | yes |
| `yeast_parent_a_L400_dmfm_diagonly_finetune_20260726/epoch=33-step=45750-v1.ckpt` | Table 13, Table 17, init of 4-step L400 | yes |
| `yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=29-step=15000.ckpt` | Table 18 | yes |
| `yeast_parent_a_L100_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=306-step=149000.ckpt` | Table 18 | yes |
| `yeast_parent_a_L200_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=39-step=34000.ckpt` | Table 18 | yes |
| `yeast_parent_a_L400_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=96-step=129000.ckpt` | Table 18 | yes |
| `yeast_parent_c0_guide_2026-07-26/best_state.pt` | Table 1 guidance (n=32 and n=1000) | yes |
| `yeast_parent_c0_oracle_2026-07-26/best_state.pt` | Table 1 oracle scoring; CPU re-score check in Tables 4–5 recompute | yes |

Facts about these checkpoints that matter for porting:

- Base `best.pt` is **not** the 200k-step model. It is the best a_val loss checkpoint:
  L50 step 50,000 (val loss 0.7776), L100 55,000 (0.7830), L200 45,000 (0.7960),
  L400 45,000 (0.7984). `last.pt` is step 200,000. Keys: `step, model_state_dict,
  optimizer_state_dict, args, model_cfg, metrics, rng_state, cuda_rng_state_all`.
  DiT parameters: 2,912,836 (= Table 7 "DiT backbone").
- dMFM student parameter count 3,420,484 (x_embed 768, x_cond_embed 768, x_cond_adaLN 74,112,
  5 time embedders x 86,400, DiT 2,912,836) = Table 7 exactly.
- L400 finetune: `epoch=33-step=45750.ckpt` and `epoch=33-step=45750-v1.ckpt` have **bitwise
  identical state_dicts** (max |diff| = 0); the files differ only in Lightning metadata
  (sha256 86b109e1… vs 28163b80…). The run was launched twice with identical args; `-v1` is from
  the second launch (`lightning_logs/version_1`) and is the one referenced everywhere.
- 4-step students: only one checkpoint per run dir survives, and it is exactly the best
  checkpoint under the run's own `ModelCheckpoint` monitor `val_mfm_probe_esd_gap_0.25`
  (recomputed from `val_<step>.csv`: L50 step 15,000 = 7.88e-4; L100 149,000 = 8.64e-3;
  L200 34,000 = 7.47e-3; L400 129,000 = 9.83e-3). `last.ckpt` symlinks in the L50 and L200 dirs
  dangle. All four are the checkpoints hard-coded in the Table 18 launcher and recorded in each
  Table 18 `metadata.json`.
- WARNING for the port: `scripts/ablate_dmfm_one_step_gradient_mc.py` (mtime 2026-07-27 22:41)
  now has `DEFAULT_DMFMS` = the 4-step students, but when Table 13 ran (2026-07-26 20:17)
  `sample_parent_reward_attainment.py` imported `DEFAULT_DMFMS` from it and got the
  **diagonal/selected** students (confirmed by each Table 13 `metadata.json`). A faithful Table 13
  rerun must pass the selected students explicitly.

---

## 1. Data split (Tables 9–10; all experiments)

- Script: `data/prepare_yeast_parent_splits.py`. Launcher `slurm/prepare_yeast_parent_disjoint_data.sbatch`
  (serial_requeue, 4 CPU, 32G, 1 h).
- Jobs: 35116368 (2026-07-25 18:21; build used by the base models, trained from 18:3x), rerun
  35241754 / 35241776 (2026-07-26 08:21; the files on disk now). The script was edited at
  2026-07-26 08:18 to add `c0` labels to the L50 file (`payload["c0"]`, needed by the C0 regressors).
  The printed split JSON (all index counts, parent lists, `source_sha256`) is identical across all
  runs. UNCERTAIN (low risk): byte identity of the Jul-25 `seqs` tensors cannot be checked (overwritten),
  but the procedure is deterministic (seed 0).
- Output `data/yeast_parent_disjoint/`: `yeast_parent_L{50,100,200,400}.pt`
  (`seqs, parent_id, window_start, seq_len, stride, parent_length, source` [+ `c0` for L50]),
  `yeast_parent_L*_split_seed0.pt` (`a_train_idx, a_val_idx, b_train_idx, b_val_idx, test_idx`,
  `parent_ids`, …), `parent_split_seed0.json` (`source: data/yeast_sequences.txt`,
  `source_sha256: aabfb6c51cb19e99a492f1a86d93fd658360181b927ba31489f432f810aadbc2`, stride 7,
  parent_length 1044, 576 parents).
- Table 10 counts VERIFIED against `parent_split_seed0.json` (a_train 32,890/31,050/27,830/21,390;
  a_val 4,147/3,915/3,509/2,697; test 8,294/7,830/7,018/5,394; windows per parent 143/135/121/93).

```bash
python -u data/prepare_yeast_parent_splits.py --input data/yeast_sequences.txt \
  --out_dir data/yeast_parent_disjoint --seed 0 --lengths 50 100 200 400
```

---

## 2. Base DFM generators (Table 9 top half)

- Script `scripts/train_dna_dfm_dit.py` (mtime 2026-07-25 18:11, before the runs: surviving file = used file).
- Launchers `slurm/train_yeast_parent_L{50,100,200,400}_dfm.sbatch`: partition **seas_gpu**, `-A albergo_lab`,
  1 GPU, 8 CPU, 64G, 24 h. Jobs: L50 35140730, L100 35140732, L200 35140729, L400 35140734
  (2026-07-25, ~2 h each; earlier 35115687–90 / 35116374–90 were 2-step smoke tests into
  `workdir/_smoke_*`).
- Result dirs `workdir/yeast_parent_a_L{L}_dfm_dit_h192_b4/{args.json,best.pt,last.pt}`.
  args.json has `args` + `model_cfg` (model_cfg: hidden 192, blocks 4, heads 6, ff 4.0,
  dit_cond_dim 192, double_temb true, scale_by_sigma false, preserve_denoiser false,
  time_delta_param true, softcap 50, use_flash_attn false, self_condition_ratio 0).
- Table 9 metrics location: `best.pt["metrics"]` and the Slurm logs (`slurm/logs/slurm-yeast_L*_dfm-<job>.out`,
  one `val_loss`/`val_token_acc` line every 5,000 steps). Best a_val: L50 0.7776/0.6425,
  L100 0.7830/0.6418, L200 0.7960/0.6355, L400 0.7984/0.6360 (loss/token acc).

```bash
# L50 (L100/200/400: change L and --batch_size 256/128/64)
python -u scripts/train_dna_dfm_dit.py --run_name yeast_parent_a_L50_dfm_dit_h192_b4 \
  --output_dir workdir/yeast_parent_a_L50_dfm_dit_h192_b4 \
  --yeast_data_pt data/yeast_parent_disjoint/yeast_parent_L50.pt \
  --yeast_split_pt data/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt \
  --yeast_split a_train --yeast_val_split a_val --max_train_seqs 0 \
  --steps 200000 --batch_size 512 --lr 5e-5 --weight_decay 0.01 --grad_clip 1.0 \
  --hidden_dim 192 --transformer_heads 6 --transformer_ff_mult 4 --dit_blocks 4 --dropout 0.0 \
  --gaussian_beta_schedule linear --save_every 5000 --val_every 5000 --val_batches 0 \
  --eval_samples 0 --num_workers 4
# script defaults also recorded in args.json: seed 0, softcap 50.0, use_flash_attn false,
# self_condition_ratio 0.0, use_adaptive false, gen_steps 64, gen_batch 256
```

---

## 3. C0 guide and oracle regressors (Table 11, Fig 3 caption, Table 1)

- Script `scripts/train_parent_c0_regressor.py` (mtime 2026-07-26 08:18; job ran 08:19–08:23: surviving = used).
- Launcher `slurm/train_yeast_parent_c0_regressors.sbatch` (kempner_h100, 1 GPU, 8 CPU, 48G, 1 h).
  Job 35241778 (2026-07-26), both roles sequentially.
- Result dirs `workdir/yeast_parent_c0_{guide,oracle}_2026-07-26/` with `best_state.pt`,
  `metadata.json`, `history.csv`, `{test,validation}_predictions.csv`.
- **Table 11 numbers are in `metadata.json["test_metrics"]`** — VERIFIED: guide MAE 0.15877,
  RMSE 0.23065, Pearson 0.87741, Spearman 0.88383; oracle 0.16198 / 0.23469 / 0.87242 / 0.87636
  (n = 8,294, 58 test parents).

```bash
common="--data_pt data/yeast_parent_disjoint/yeast_parent_L50.pt \
  --split_pt data/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt --model_type park_cnn \
  --epochs 150 --patience 20 --batch_size 1024 --lr 0.001 --weight_decay 0.0001 \
  --reverse_complement_augmentation --reverse_complement_average --device cuda:0"
python -u scripts/train_parent_c0_regressor.py $common --role guide \
  --train_split a_train --validation_split a_val --test_split test --seed 0 \
  --out_dir workdir/yeast_parent_c0_guide_2026-07-26
python -u scripts/train_parent_c0_regressor.py $common --role oracle \
  --train_split b_train --validation_split b_val --test_split test --seed 1 \
  --out_dir workdir/yeast_parent_c0_oracle_2026-07-26
```

---

## 4. dMFM students (Table 9 bottom half, Table 7)

### 4a. Main diagonal+ESD runs (L50, L100, L200 selected checkpoints come from here)

- Script `scripts/train_dna.py` + `utils/parsing.py` + `utils/mfm_diag_distill.py` + library @187fe7b.
  NOTE: the surviving `train_dna.py` (2026-07-26 22:39), `utils/parsing.py` (22:39) and
  `utils/mfm_diag_distill.py` (2026-07-27 07:08) are all newer than these runs (09:32). The newer
  versions add `--init_ckpt` and the gap-bound fix (§4c); neither affects these runs
  (`gap_mode random`, no init_ckpt). The recon's older `mfm_diag_distill.py` equals the version these
  runs used.
- Launcher `slurm/train_yeast_parent_dmfm_diagonal.sbatch` with `LENGTH=L`, all other env defaults
  (kempner_h100, 1 GPU, 8 CPU, 64G, 24 h). Jobs: L50 35247690, L100 35247691, L200 35247693,
  L400 35247701 (2026-07-26 09:32–10:14, i.e. **~42 min for 50k steps**).
- `ModelCheckpoint(monitor=val_loss, save_top_k=5, save_last=True)`; final checkpoint choice was made
  afterwards by posterior-map accuracy on a_val (§4d).

```bash
# LENGTH=50 sbatch slurm/train_yeast_parent_dmfm_diagonal.sbatch expands to
# (L100: batch 64; L200: batch 32; L400: batch 16; --mfm_fixed_probe_batch_size = batch)
python -u scripts/train_dna.py --run_name yeast_parent_a_L50_dmfm_diagonal_h192_b4 \
  --output_dir workdir/yeast_parent_a_L50_dmfm_diagonal_h192_b4 --no-wandb --dataset_type yeast \
  --yeast_data_pt data/yeast_parent_disjoint/yeast_parent_L50.pt \
  --yeast_split_pt data/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt \
  --yeast_train_split a_train --yeast_val_split a_val --mode gaussian --gaussian_beta_schedule linear \
  --gaussian_mfm_diag_distill --gaussian_adaptive_loss --gaussian_adaptive_loss_c 0.01 \
  --gaussian_adaptive_loss_r 0.5 \
  --mfm_teacher_ckpt workdir/yeast_parent_a_L50_dfm_dit_h192_b4/best.pt \
  --mfm_teacher_ckpt_hparams workdir/yeast_parent_a_L50_dfm_dit_h192_b4/args.json \
  --mfm_t_cond_warmup_steps 0 --mfm_t_cond_power 1.0 --mfm_t_cond_0_rate 0.0 \
  --mfm_consistency --mfm_consistency_type esd_teacher --mfm_consistency_warmup_steps 1000 \
  --mfm_consistency_anneal_end_step 1 --mfm_consistency_step_offset 0 \
  --mfm_diag_loss_weight 1.0 --mfm_lsd_loss_weight 1.0 --mfm_diag_adaptive_p 0.5 --mfm_lsd_adaptive_p 1.0 \
  --mfm_val_gaps 0.01 0.05 0.1 0.25 0.5 0.75 --mfm_fixed_probe_batch_size 64 --mfm_fixed_probe_seed 12345 \
  --mfm_checkpoint_metric val_loss --model dit --hidden_dim 192 --transformer_heads 6 \
  --transformer_ff_mult 4 --dit_blocks 4 --dropout 0.0 --optimizer adam --weight_decay 0.0 --lr 1e-4 \
  --grad_clip 1.0 --batch_size 64 --num_workers 4 --limit_val_batches 1 --val_check_interval 1000 \
  --max_steps 50000 --print_freq 200 --log_every_n_steps 50 --no_tqdm
# args.json also records: mfm_consistency_gap_mode random (min/max gap 0.75/1.0 are unused defaults),
# mfm_preserve_t_cond_0 true, mfm_encoder_depth null (=2), dfm_vfm_loss true, commit 187fe7b.
```

### 4b. L400 diagonal-only finetune (source of the L400 selected checkpoint)

- No Slurm script; launched **interactively** (Lightning "srun available but not used" warning;
  host holygpu7c26105) on 2026-07-26 16:11 and again at 16:12 with identical args (two
  `lightning_logs/version_*`, `-v1` files from the second launch). Resumes the full training state
  (`--ckpt`, not `--init_ckpt`) of `yeast_parent_a_L400_dmfm_diagonal_h192_b4/epoch=33-step=45000.ckpt`
  and trains 1,000 more steps (45,000 → 46,000) **without consistency loss, batch 8**.
- `workdir/parent_dmfm_diagonly_finetune_20260726/` is an empty directory (probably intended as a log dir).
  Analogous finetunes/continuations exist for L50/L100/L200 (`*_diagonly_finetune_20260726`,
  `*_diagonly_finetune_retry_20260726`, `*_diagonly_continue{10k,30m}_20260726`; launchers
  `slurm/train_yeast_parent_L100_dmfm_diag_continue*.sbatch`) but were **not** selected.

```bash
# reconstructed from workdir/yeast_parent_a_L400_dmfm_diagonly_finetune_20260726/args.json
# (every other flag identical to the L400 command of 4a)
python -u scripts/train_dna.py --run_name yeast_parent_a_L400_dmfm_diagonly_finetune_20260726 \
  --output_dir workdir/yeast_parent_a_L400_dmfm_diagonly_finetune_20260726 \
  --ckpt workdir/yeast_parent_a_L400_dmfm_diagonal_h192_b4/epoch=33-step=45000.ckpt \
  ...L400 data/teacher flags... --batch_size 8 --max_steps 46000 \
  --mfm_checkpoint_metric val_mfm_probe_diag_mse --mfm_lsd_loss_weight 0.0 \
  --mfm_consistency_type lsd --mfm_consistency_warmup_steps 2500000 \
  --mfm_consistency_anneal_end_step 25000 --mfm_fixed_probe_batch_size 128 \
  --mfm_val_gaps 0.01 0.1 0.5 --num_workers 0 --print_freq 100 --val_check_interval 250
# i.e. no --mfm_consistency flag (mfm_consistency=false); lr 1e-4 unchanged.
# DNAModule.on_fit_start resets the restored optimizer LR to args.lr.
```

### 4c. Four-step students (Table 18)

- Launcher `slurm/train_yeast_parent_dmfm_4step_continue.sbatch` with `LENGTH=L` (kempner_h100, 1 GPU,
  8 CPU, 64G, 16 h). Jobs (the "corrected" runs that produced the surviving dirs): L50 35458568,
  L100 35458569, L200 35458567, L400 35458570 (2026-07-27 07:12–11:05; all reached max_steps 150,000).
- Requires the July `utils/mfm_diag_distill.py` (2026-07-27 07:08). Its fix replaces
  `float(x or default)` so that `--mfm_consistency_min_gap 0.0` is honoured. The earlier runs
  35409636–39 (2026-07-26 22:47, output `*_upto025_20260731`, not surviving) hit the bug and sampled
  gaps in [0.25, 0.75] (logged mean gap 0.4996 vs 0.1248 after the fix). Jobs 35409308–11
  (a `fixed_gap=0.25` variant) were cancelled after ~2,400 steps.
- Init weights (weights only, fresh optimizer) = the four selected diagonal students.

```bash
# LENGTH=50 sbatch slurm/train_yeast_parent_dmfm_4step_continue.sbatch expands to
# (L100 batch 64, L200 32, L400 16; init = selected student of that length)
python -u scripts/train_dna.py --run_name yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727 \
  --output_dir workdir/yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727 \
  --init_ckpt workdir/yeast_parent_a_L50_dmfm_diagonal_h192_b4/epoch=95-step=49000.ckpt \
  --no-wandb --dataset_type yeast --yeast_data_pt data/yeast_parent_disjoint/yeast_parent_L50.pt \
  --yeast_split_pt data/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt \
  --yeast_train_split a_train --yeast_val_split a_val --mode gaussian --gaussian_beta_schedule linear \
  --gaussian_mfm_diag_distill --gaussian_adaptive_loss --gaussian_adaptive_loss_c 0.01 --gaussian_adaptive_loss_r 0.5 \
  --mfm_teacher_ckpt workdir/yeast_parent_a_L50_dfm_dit_h192_b4/best.pt \
  --mfm_teacher_ckpt_hparams workdir/yeast_parent_a_L50_dfm_dit_h192_b4/args.json \
  --mfm_t_cond_warmup_steps 0 --mfm_t_cond_power 1.0 --mfm_t_cond_0_rate 0.0 \
  --mfm_consistency --mfm_consistency_type esd_teacher --mfm_consistency_warmup_steps 0 \
  --mfm_consistency_gap_mode long --mfm_consistency_min_gap 0.0 --mfm_consistency_max_gap 0.25 \
  --mfm_diag_loss_weight 1.0 --mfm_lsd_loss_weight 1.0 --mfm_diag_adaptive_p 0.5 --mfm_lsd_adaptive_p 1.0 \
  --mfm_val_gaps 0.0625 0.125 0.25 --mfm_fixed_probe_batch_size 64 --mfm_fixed_probe_seed 12345 \
  --mfm_checkpoint_metric val_mfm_probe_esd_gap_0.25 --model dit --hidden_dim 192 --transformer_heads 6 \
  --transformer_ff_mult 4 --dit_blocks 4 --dropout 0.0 --optimizer adam --weight_decay 0.0 --lr 2e-5 \
  --grad_clip 1.0 --batch_size 64 --num_workers 4 --limit_val_batches 1 --val_check_interval 1000 \
  --max_steps 150000 --print_freq 200 --log_every_n_steps 50 --no_tqdm
```

### 4d. Selection of the diagonal-student checkpoints (Table 9 "Selected step")

- Script `scripts/evaluate_dmfm_glass_accuracy_curve.py`; result dirs
  `workdir/rebuttal_dmfm_L{L}_glass_accuracy_val_selection_20260726/` (`run_metadata.json`,
  `posterior_map_accuracy_summary.csv`). Criterion (docs/what_rebuttal_posterior_map_accuracy.md):
  64 fixed a_val sequences, seed 20260730, gaps {0.1, 0.5, 0.75}, target = teacher posterior map by
  32-step Heun GLASS, lowest mean relative L2. Selected: L50 49,000 (0.0455), L100 39,000 (0.0535),
  L200 45,000 (0.0479), L400 finetune 45,750-v1 (0.0528). Slurm job: UNCERTAIN (no matching log found;
  possibly interactive).

---

## 5. Table 1 and Fig 3 (cyclizability guidance with independent oracle), n=32 and n=1000

- Generator: `scripts/sample_yeast_c0_guidance_hist.py`. The July original is purged; the recon copy
  (`<recon>/scripts/sample_yeast_c0_guidance_hist.py`, = DNA-MFM@187fe7b) reproduces the unguided
  branch of the surviving pilot bit for bit (recon `compare.py`, `out_selfrep_*`). Scorer
  `scripts/score_yeast_c0_oracle.py` (surviving; edited 2026-07-26 08:32 after the in-job crash below).
- **Sampler used: base DFM `L50 best.pt` + GLASS posterior value gradients (MC=8, 4-step Euler GLASS
  on the base model); no dMFM checkpoint is loaded.** Reward
  `reward_scale * (-0.5 ((f_guide(x1) - y*) / reward_sigma)^2)` = -11.1 (f - y*)^2.
- n = 32 pilot (**these are the Table 1 numbers**, VERIFIED to 3 decimals):
  `workdir/yeast_parent_glass_pilot_target{1p0,m1p0}_g8.0_seed0/` (`args.json`, `sample_scores.csv`,
  `sequences_*.fa`, `summary.json`, `histogram.png`, `cdf.png`, `oracle/oracle_{summary,per_sample}.csv`).
  Launcher `slurm/sample_yeast_parent_c0_glass_pilot.sbatch` with `TARGET_C0=1.0` / `-1.0`
  (kempner_h100, 1 GPU, 8 CPU, 64G, 1 h); jobs 35242279 (+1) and 35242278 (-1), 2026-07-26 08:30.
  Oracle provenance: the in-job oracle step crashed (`numpy()` on a grad tensor) and standalone
  oracle jobs 35242533/35242534 were killed; the stored `oracle/` CSVs (08:36) come from an unlogged
  rerun. The recon's CPU re-score (`<recon>/oracle_check/`) matches them within 4e-4 per sample.
- n = 1000 rerun: `workdir/yeast_parent_glass_n1000_target{1p0,m1p0}_g8.0_seed0/`, recon library,
  `<recon>/run_n1000.sbatch` (copy in `workdir/rebuttal_parent_c0_diversity_n1000_20260925/generate_n1000.sbatch`),
  kempner_h100, 1 GPU, 8 CPU, 64G, 3 h; jobs 48651189 (+1), 48651190 (-1), 2026-09-25; generation +
  oracle in the same logged job. Results: +1 unguided MAE 1.174, guided 0.148, paired 1.025 [0.999, 1.052]
  (98.3% improved); -1: 0.833 / 0.119 / 0.713 [0.687, 0.740] (97.1%). Only `--n_samples` differs from the pilot.

```bash
python -u scripts/sample_yeast_c0_guidance_hist.py \
  --ckpt workdir/yeast_parent_a_L50_dfm_dit_h192_b4/best.pt \
  --c0_ckpt workdir/yeast_parent_c0_guide_2026-07-26/best_state.pt \
  --output_dir workdir/yeast_parent_glass_pilot_target1p0_g8.0_seed0 \
  --seed 0 --n_samples 32 --batch_size 8 --target_c0 1.0 --reward_sigma 0.15 --reward_scale 0.5 \
  --mc 8 --mc_chunk 8 --nfe_traj 64 --nfe_value 4 --t_max 0.95 --guide_t_start 0.50 --guide_t_end 0.95 \
  --guidance_frac 8.0 --coeff_cap 10.0 --grad_clip 10.0
# (defaults recorded in args.json: flow_temp 1.0, linear beta, hist_bins 50, taper false,
#  use_flash_attn false, target_c0_second null). n=1000: --n_samples 1000, same --batch_size 8.
python -u scripts/score_yeast_c0_oracle.py \
  --oracle_ckpt workdir/yeast_parent_c0_oracle_2026-07-26/best_state.pt --oracle_model_type park_cnn \
  --run_glob workdir/yeast_parent_glass_pilot_target1p0_g8.0_seed0/sample_scores.csv \
  --out_dir workdir/yeast_parent_glass_pilot_target1p0_g8.0_seed0/oracle \
  --batch_size 512 --bootstrap 2000 --seed 0 --reverse_complement_average --device cuda:0
```

- Fig 3 is the schematic `repos/dMFM/figures/dMFM_cyclizability_schematic_final_1.png`; its caption
  numbers (guide r 0.88 / MAE 0.159, oracle r 0.87 / MAE 0.162) are the Table 11 metadata. No
  code producer (artwork).

---

## 6. Tables 4–5 (diversity) and Table 6 (Evo2 NLL)

- **The paper's Tables 4, 5, 6 do not come from the parent-disjoint models.** Their numbers exist only
  in the tex (`repos/dMFM/main.tex` l.~760–800, present since the first Overleaf import 2026-09-03).
  Table 4's set names (`train_sample`, `split_train_sample`, `split_val_sample`, `split_test_sample`)
  and Table 6 come from the older "clean-split" pipeline: the Evo2 scorer
  `scripts/__pycache__/score_yeast_c0_evo2.cpython-39.pyc` has defaults
  `--train_pt data/yeast_mid50.pt`, `--split_pt data/yeast_mid50_split_seed0_train65000_val8000.pt`,
  `--run_glob workdir/yeast_split_dmfm_c0_samples_target_*/sample_scores.csv`,
  `--out_dir workdir/yeast_split_dmfm_c0_evo2`; the samples came from `scripts/sample_yeast_split_dmfm_c0.py`
  (defaults: split65k teacher `workdir/yeast_mid50_dfm_dit_split65k_h192_b4_lr5e-5_2026-05-12_16-47-35/last.pt`,
  dMFM `workdir/yeast_mid50_dmfm_split65k_esd_resume_b64_lr1e4_to1000000_2026-05-12_16-47-35/last.ckpt`,
  guide `dinko/C0free_torch.pt`, n 1000, mc 16, nfe_sample 96, nfe_value 1, guide [0.01, 0.95],
  guidance_frac 8, reward_scale 1.0). All of these models, data files and the result dirs
  (`workdir/yeast_c0_diversity_pm1*`, empty) are purged. Not reproducible as published.
- Parent-disjoint recomputation of Tables 4–5 (2026-09-25, CPU): analyzer
  `workdir/rebuttal_parent_c0_diversity_20260925/analyze_parent_c0_diversity.py` (+ `vendor/dna_mfm_187fe7b/`:
  `scripts/analyze_yeast_c0_diversity.py`, `utils/yeast_splits.py`), env multimfm.
  n=32 on the pilot: `workdir/rebuttal_parent_c0_diversity_20260925/`; n=1000:
  `workdir/rebuttal_parent_c0_diversity_n1000_20260925/` (`tables.md`, `table6_baselines.csv`,
  `table7_generated.csv`, …). The n=1000 numbers differ from the paper (e.g. target -1 guided pairwise
  0.7287 vs paper 0.7010; test_sample pairwise 0.7381 vs 0.7374); only `full_sample` reproduces the
  paper's `train_sample` row (0.52188 / 0.7376993 / 0.74).
- No parent-disjoint Evo2 rescoring of the C0 sets exists (Table 6 lost).

```bash
python -W ignore::FutureWarning workdir/rebuttal_parent_c0_diversity_20260925/analyze_parent_c0_diversity.py \
  --run_dirs workdir/yeast_parent_glass_n1000_targetm1p0_g8.0_seed0 workdir/yeast_parent_glass_n1000_target1p0_g8.0_seed0 \
  --out_dir workdir/rebuttal_parent_c0_diversity_n1000_20260925
# defaults: --baseline_n 1000 --seed 0 --max_pairwise_n 3000 --k_nearest_train 5 --pair_chunk 128
#           --size_matched_draws 1000; data data/yeast_parent_disjoint/yeast_parent_L50{,_split_seed0}.pt;
#           oracle workdir/yeast_parent_c0_oracle_2026-07-26/best_state.pt (CPU re-score check only)
```

---

## 7. Table 13 (exact motif / conjunction rewards) and its F.1 companions

- Script `scripts/sample_parent_reward_attainment.py` (mtime 2026-07-26 19:54, before the runs), which
  imports `DEFAULT_DMFMS, load_dmfm, one_step_dmfm` from `scripts/ablate_dmfm_one_step_gradient_mc.py`
  (later modified, see §0 warning). Guidance posterior = **one direct dMFM map F(0,1,ε | t, x_t)**
  of the selected diagonal students, MC=8; teacher = base best.pt (sampling trajectory and 1,024-sample
  calibration of the target score, 90th percentile).
- Launcher `slurm/sample_parent_reward_attainment.sbatch` with `LENGTH`, `REWARD`; defaults
  `OUT_ROOT=workdir/rebuttal_reward_attainment_${REWARD}_20260731`, `SEED=20260731` (kempner_h100, 1 GPU,
  8 CPU, 64G, 2 h). Jobs (2026-07-26 ~20:1x): motif L50/100/200/400 = 35383601/35383624/35383648/35383653;
  conjunction = 35383663/35383669/35383675/35383680. (The "20260731" in the dir names is a tag, not the date.)
- Result dirs `workdir/rebuttal_reward_attainment_{motif,conjunction}_20260731/L{L}/`
  (`metadata.json` with `dmfm_checkpoint`, `teacher_checkpoint`, `generation` args;
  `reward_attainment_summary.csv`, `reward_attainment_samples.csv`, `heldout_test_reference.csv`,
  `unconditional_score_calibration.npy`, `evo2/`). The top-level `*_summary_all_lengths.csv` holds only the
  last length written; use the per-length files. Table 13 VERIFIED (all 8 cells, 3 decimals, CIs);
  93–98% improved pairs, diversity drop 0.008–0.020 also match.
- Evo2 (F.1 "NLL improves in every setting"): `scripts/score_parent_reward_attainment_evo2.py`,
  `slurm/score_parent_reward_attainment_evo2.sbatch` (`SAMPLE_DIR=<run>/L{L}`, `PYTHON=.envs/evo2/bin/python`,
  `module load gcc/12.2.0-fasrc01 cuda/12.9.1-fasrc01`, kempner_h100, 96G, 2 h). Successful jobs
  35387510/512/514/515 (motif L50..400), 35387517/518/520/523 (conjunction); the first batch
  35383646–82 failed (Evo2 import).

```bash
python -u scripts/sample_parent_reward_attainment.py --lengths 50 --reward motif \
  --output_dir workdir/rebuttal_reward_attainment_motif_20260731 --seed 20260731 --n_samples 100 \
  --batch_size 8 --nfe_traj 64 --t_end 0.999 --guide_t_start 0.50 --guide_t_end 0.95 --mc 8 --mc_chunk 4 \
  --guidance_frac 8.0 --grad_clip 10.0 --coeff_cap 10.0 --calibration_samples 1024 --nfe_calibration 64 \
  --calibration_batch_size 32 --target_percentile 0.90 --reward_beta 1.0 --reward_scale 1.0 \
  --motif TTTTTC --motif2 AAAATT --motif_tau 0.10 --conjunction_tau 0.10 --bootstrap 2000 --device cuda:0
# faithful rerun today must add (defaults changed after the run):
#   --dmfm_ckpt 50=workdir/yeast_parent_a_L50_dmfm_diagonal_h192_b4/epoch=95-step=49000.ckpt  (etc. per L)
.envs/evo2/bin/python -u scripts/score_parent_reward_attainment_evo2.py \
  --sample_dir workdir/rebuttal_reward_attainment_motif_20260731/L50 \
  --out_dir workdir/rebuttal_reward_attainment_motif_20260731/L50/evo2 \
  --model_name evo2_7b_base --batch_size 1 --bootstrap 2000 --seed 20260731 --device cuda:0
```

---

## 8. Fig 5, Tables 14–15 (scale-free finite-MC gradient accuracy)

- **Estimator: GLASS on the base model (`L* best.pt`), not dMFM.** Finite-N GLASS (8-step Euler) vs an
  independent GLASS-2048 reference; 32 probe states per length, obtained by integrating the base flow
  from 32 seeded Gaussian draws to t = 0.5 (32 Euler steps) — model-generated states, not held-out
  data; 8 independent noise pools per state; N ∈ {1, …, 256}.
- Script `scripts/ablate_glass_gradient_mc.py`. The surviving file (mtime 2026-07-26 21:49)
  is newer than the runs (10:02–16:37). It has options absent from the runs' `run_metadata.json`
  (`--glass_end_time` default 0.999, `--glass_solver`, `reward_objective`). RESOLVED 2026-09-29: the
  runs integrated GLASS to t = 1.0 (as `glass_integrate_diff` in the recon sampler does). The port
  accepts end time 1.0 for Euler; recomputing the stored L=50 GLASS-2048 reference gradients (probes
  0-1, motif from `rebuttal_gradient_accuracy_motif`, gc from `..._gc_pairmetrics`, original seeds)
  gives 0.3-0.6 % relative error with 1.0 vs 1.2-1.3 % with 0.999 on every probe, against a
  0.2-0.4 % TF32-precision noise floor (`tests/dna/check_gradient_mc_reference.py`, jobs 49264548,
  49272897). Calibration statistics agree to <5e-4. The older `rebuttal_gradient_accuracy_gc` run
  used a still earlier version (arg `gc_calibration_samples`, no `reward`).
- Launcher `slurm/ablate_yeast_glass_gradient_mc.sbatch` (`REWARD`, `OUTPUT_DIR`; kempner_h100, 1 GPU,
  8 CPU, 64G, 8 h). Jobs: GC 35338216 (`OUTPUT_DIR=workdir/rebuttal_gradient_accuracy_gc_pairmetrics`,
  2026-07-26 16:37), motif 35258314, conjunction 35258315 (11:1x). (35243681 → `rebuttal_gradient_accuracy_gc`,
  an earlier GC run without stored tensors; 35253092/94 cancelled.)
- Post-processing that produced the paper numbers: `scripts/rescore_gradient_pairs_scale_free.py --input-dir <run>`
  (defaults `--bootstrap 10000 --seed 20260802`), run 2026-08-02 14:46–14:47 (no log; interactive) →
  `<run>/gradient_scale_free_summary_all_lengths.csv` and per-length `gradient_scale_free_{summary,per_state}.csv`.
  Tables 14–15 VERIFIED against these (`global_relative_l1` / `mean_cosine`), e.g. GC L50 N=8 1.92/.24.
- `scripts/summarize_gradient_accuracy_metrics.py` (→ `gradient_accuracy_relative_directional_summary.csv`,
  2026-07-26 15:00/18:06) and `scripts/plot_gradient_accuracy_required_mc.py --gc_dir <gc_pairmetrics>`
  (→ `workdir/rebuttal_gradient_accuracy_figures/`, required-MC-vs-length plots) are earlier
  diagnostics. **No surviving script plots Fig 5** (relative L1 / cosine vs N, three panels); the data
  are the three `gradient_scale_free_summary_all_lengths.csv`.

```bash
python -u scripts/ablate_glass_gradient_mc.py --output_dir workdir/rebuttal_gradient_accuracy_gc_pairmetrics \
  --reward gc --lengths 50 100 200 400 --t_eval 0.50 --nfe_probe 32 --nfe_calibration 64 \
  --calibration_samples 1024 --sample_batch_size 32 --nfe_value 8 --n_probes 32 --n_repeats 8 \
  --reference_mc 2048 --mc_values 1 2 4 8 16 32 64 128 256 --mc_chunk 16 --z_target 1.0 \
  --reward_beta 1.0 --reward_scale 1.0 --thresholds 0.05 0.10 0.20 --bootstrap 2000
# run_metadata also records seed 20260726, motif TTTTTC, motif2 AAAATT, motif_tau 0.1,
# conjunction_tau 0.1, target_percentile 0.9, store_gradient_tensors true.
# motif / conjunction: --reward motif|conjunction --output_dir workdir/rebuttal_gradient_accuracy_{motif,conjunction}
python scripts/rescore_gradient_pairs_scale_free.py --input-dir workdir/rebuttal_gradient_accuracy_gc_pairmetrics
```

---

## 9. Table 17 (posterior diversity, dMFM vs GLASS)

- Script `scripts/eval_parent_posterior_diversity.py` (hard-coded `DEFAULT_STUDENTS` = the four selected
  diagonal students; teacher `workdir/yeast_parent_a_L{L}_dfm_dit_h192_b4`). GLASS reference uses
  `glass_integrate_diff` imported from `sample_yeast_c0_guidance_hist.py` (Euler, 0 → 1).
  UNCERTAIN: script mtime 2026-07-26 19:51 is ~20 min after the run (19:30–19:32).
- Launcher `slurm/eval_parent_posterior_diversity.sbatch` with
  `OUT_DIR=workdir/parent_posterior_diversity_mapselected_flowmap1_glass100_20260726` (other env defaults:
  `DMFM_STEPS=1 DMFM_SAMPLER=flow_map GLASS_STEPS=100`); kempner_h100, 1 GPU, 8 CPU, 80G, 4 h. Job 35378081.
- Result dir: `run_config.csv` (records the four student checkpoints), `posterior_diversity_summary.csv`,
  `posterior_condition_metrics.csv`, `posterior_futures.csv` (11 MB), `posterior_diversity_vs_time.png`.
  Table 17 VERIFIED (`pairwise_hamming_frac_mean`, dmfm/glass rows, t ∈ {0, .25, .5, .75}).
- Companion Evo2: `slurm/score_parent_posterior_evo2.sbatch` job 35387531 (final; 35384036 earlier) →
  `workdir/parent_posterior_diversity_mapselected_flowmap1_glass100_evo2_20260726/`.

```bash
python -u scripts/eval_parent_posterior_diversity.py --lengths 50 100 200 400 \
  --times 0.0 0.25 0.5 0.75 0.9 0.99 1.0 --n_sources 32 --n_futures 16 --source_batch 4 \
  --dmfm_steps 1 --dmfm_sampler flow_map --glass_steps 100 --bootstrap 2000 --seed 20260730 \
  --out_dir workdir/parent_posterior_diversity_mapselected_flowmap1_glass100_20260726
.envs/evo2/bin/python -u scripts/score_parent_posterior_evo2.py \
  --futures_csv workdir/parent_posterior_diversity_mapselected_flowmap1_glass100_20260726/posterior_futures.csv \
  --out_dir workdir/parent_posterior_diversity_mapselected_flowmap1_glass100_evo2_20260726 \
  --model_name evo2_7b_base --batch_size 4 --sources_per_cell 16 --bootstrap 2000 --seed 20260730 --device cuda:0
```

---

## 10. Table 18 (raw gradient MAE, 4-step dMFM vs GLASS-128)

- Script `scripts/ablate_dmfm_one_step_gradient_mc.py` (mtime 2026-07-27 22:41; the first array
  35642202 was cancelled at 22:40:56 and resubmitted as 35642548, so the surviving version is the one
  used). MAE table from `scripts/rescore_gradient_pairs_mae.py` (mtime 22:38; run 2026-07-27 23:33, no
  log) → `<run>/<reward>/L{L}/gradient_mae_summary.csv` (`mean_mae`) and
  `<run>/<reward>/gradient_mae_summary_all_lengths.csv`.
- Launcher `slurm/evaluate_dmfm4_glass_mae_precision.sbatch` (array 0–11%6: reward x length;
  kempner_h100, 1 GPU, 8 CPU, 64G, 1 h). Job 35642548_0..11 (2026-07-27 22:4x–23:18).
- Result dir `workdir/rebuttal_dmfm4_glass128_mae_precision_20260727/{gc,motif,conjunction}/`.
  Table 18 VERIFIED except two cells that differ by 1e-4: GC L100 MC=2 stored 0.123347 (paper 0.1234) and
  motif L50 MC=1 stored 0.218746 (paper 0.2188). The paper copied `docs/rebuttal_gradient_acc.md`,
  which already has these values.
- Probe states: 32 held-out `test` sequences (`probe_split test`, `probe_data_dir data/yeast_parent_disjoint`),
  noised to t_eval = 0.5 (paper says t = 0.3; known discrepancy).
- Numerical conditioning (found 2026-09-29): the GLASS-128 RK4 reference gradients (integrated to
  0.999, where the drift is stiff) are very sensitive to matmul precision. For motif L50 probe 0,
  recomputing with the stored states and noise gives norm 0.0125 with TF32 and 0.0168 without
  (stored 0.0187, cosine >= 0.992 in all cases); probe 1 agrees within 1.5 %. The port reproduces
  the probe states, noise pools and calibration of the run exactly and the 4-step dMFM estimates
  within 2 %. Reruns of Table 18 on other GPUs can therefore move individual cells by a few
  percent even with identical code and seeds.
- `workdir/rebuttal_dmfm4_glass128_mae_lowmc_allrewards_20260727/` (8 probes x 2 repeats, seed 20260726,
  `slurm/evaluate_dmfm4_glass_mae_lowmc.sbatch`) is a smaller pilot, not the table.

```bash
# array task: REWARD in {gc,motif,conjunction}, LENGTH in {50,100,200,400}; MC_CHUNK 16 (8 for L400)
python -u scripts/ablate_dmfm_one_step_gradient_mc.py \
  --output_dir workdir/rebuttal_dmfm4_glass128_mae_precision_20260727/gc --lengths 50 \
  --dmfm_ckpt 50=workdir/yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=29-step=15000.ckpt \
  --reward gc --reward_objective target --seed 20260802 --t_eval 0.50 --nfe_calibration 32 \
  --calibration_samples 128 --sample_batch_size 16 --n_probes 32 --n_repeats 8 --reference_mc 128 \
  --reference_sampler glass --reference_glass_steps 50 --reference_glass_solver rk4 --glass_end_time 0.999 \
  --dmfm_sampler flow_map --dmfm_steps 4 --dmfm_end_time 0.999 --mc_values 1 2 4 8 --nested_mc_pools \
  --mc_chunk 16 --z_target 1.0 --reward_beta 1.0 --reward_scale 1.0 --thresholds 0.05 0.10 0.20 --bootstrap 500
# run_metadata also records nfe_probe 32, probe_split test, target_percentile 0.9, motif_tau/conjunction_tau 0.1,
# store_gradient_tensors true; teacher = default base best.pt.
python scripts/rescore_gradient_pairs_mae.py --input-dir workdir/rebuttal_dmfm4_glass128_mae_precision_20260727/gc
# (defaults --bootstrap 10000 --seed 20260802; run once per reward dir; no log, invocation inferred from argparse)
```

---

## 11. Fig 4 (left: value error vs N; right: NFE benchmark)

- Right: `scripts/benchmark_yeast_split_dmfm_steering.py` (mtime 2026-07-26 15:30). Its non-dMFM
  baselines (beam, best-of-N, FK, MCTS) are a hard-coded CSV string `OLD_NON_DMFM_BASELINES`
  (lines 28–48, all `mae_sem = 0`, `n_repeats = 3`); no code for them exists in the repo. dMFM defaults: split65k
  teacher/student (purged), `dinko/C0free_torch.pt` (purged), `target_c0 0.30`, `n_outputs 100`, `repeats 3`,
  `mc_list 1 2 4 8 16`, `nfe_sample 96`, `nfe_value 1`, guide [0.01, 0.95], `guidance_frac 8`,
  `reward_scale 1.0`. No output dir survives (`figures/split_clean/` is empty).
- UNCERTAIN / likely correction to plan §7.3: the paper's `figures/nfe_benchmark.pdf` has
  CreationDate 2026-05-07 (matplotlib 3.10.8). The split65k models are dated 2026-05-12, so the
  published panel predates them. It was probably made with the earlier May `yeast_mid50` models
  (random 95/5 window split), not split65k.
- Left: `figures/MC_value.pdf`, CreationDate 2026-05-04 (matplotlib 3.9.4). No producer or data survives.
- Both panels are not reproducible from surviving artifacts; neither uses parent-disjoint models.

---

## 12. Fig 6, Table 16 (base-model marginals), Fig 7 (tilt histograms), Fig 8 (guided sample)

- Fig 6 PDFs (`base.pdf`, `gen_vs_real.pdf`, `marginals.pdf`, `cycl_dist.pdf`): CreationDate 2026-05-03
  (matplotlib 3.9.4) — **May mid50 models, not the parent-disjoint generators** of Table 9. No producer
  script or data survives. Table 16 (Pearson r of the same statistics) has the same origin; lost.
- Fig 7 (`histogram4.png` = none, `histogram2.png` = moderate, `histogram.png` = strong; 1600x900,
  matplotlib 3.9.4): the file name and layout match the `histogram.png` output of
  `scripts/sample_yeast_c0_guidance_hist.py`. UNCERTAIN: they were most likely made by the 2026-05-02/03
  runs `workdir/yeast_c0_guidance_hist_2026-05-0*` (all empty now) with the old mid50 model and
  `dinko/C0free_torch.pt`. Guidance strengths unknown.
- Fig 8 (`steering.png`, Apple ICC/XMP metadata: exported on a Mac): producer unknown.

---

## 13. Paper-vs-record discrepancies found here (beyond the brief's list)

1. **Table 1 / Fig 3 / §3.1 attribute the cyclizability results to dMFM, but the sampler was the base
   DFM with GLASS posterior value gradients** (MC=8, 4-step GLASS); no dMFM checkpoint is loaded
   (same for the n=1000 rerun). `docs/rebuttal_cyc.md` says "GLASS posterior guidance (MC=8)".
2. **Fig 5 caption and §3 text ("finite-N MC samples from dMFM") vs record:** Tables 14–15/Fig 5 are
   GLASS finite-MC vs GLASS-2048 on the base model. The paper contradicts itself: App. F.6 says Table 15
   "compares GLASS Monte Carlo error … without measuring dMFM approximation error". The "32 held-out
   conditioning states" are base-flow samples at t = 0.5, not held-out data.
3. **Tables 4, 5, 6 are from the old split65k/"clean-split" dMFM + `dinko/C0free_torch.pt` pipeline,**
   but App. B describes them as parent-disjoint L=50 dMFM samples. The parent-disjoint n=1000
   recompute gives different numbers (Tables 4–5 only; no Evo2 recompute for Table 6).
4. **Fig 6 / Table 16 (and Fig 7, Fig 4 left) predate the parent-disjoint models** (figure files dated
   May 2026), although App. F.4 presents Fig 6 as validation of "our generative backbone".
5. **Table 9 "Training steps 200,000" for the base generators:** the checkpoints actually used
   (`best.pt`) are from steps 50k / 55k / 45k / 45k (best a_val loss); training ran to 200k.
6. **Table 9 L400 dMFM (step 45,750, batch 16):** the selected checkpoint is from a separate 750-step
   diagonal-only continuation at batch 8 with the consistency loss off (§4b). It is not the batch-16 ESD run.
7. **Table 7** (tex label `tab:hyperparameters_microft`) describes the May "microft" direct-map
   finetunes, not the parent-disjoint dMFM. Besides the known 0.35/10 weights: "Gap range [0.75, 1.0]"
   (parent-disjoint runs used `gap_mode random`, full sorted-uniform gaps after 1,000 diagonal steps;
   4-step students used (0, 0.25]) and "GLASS NFE 8" (the dMFM target uses the exact teacher
   posterior velocity; no GLASS integration in training) do not match.
   In the tex version, "Learning rate 2e-6, training steps 200, batch 16" do not match either.
8. **App. E.1 "trained the dMFM for 10 hours on one H100":** the 50k-step runs took ~42 min each
   (09:32 → 10:14). The 4-step continuations took ~2 h 10 min – 4 h. No DNA dMFM run took 10 h.
9. **Eq. (17) reward** r = -(f - y*)^2 vs code r = -0.5 * 0.5 * (f - y*)^2 / 0.15^2 ≈ -11.1 (f - y*)^2
   (a scale factor only).
10. **Table 18** two cells off by 1e-4 (0.1234 vs stored 0.12335; 0.2188 vs 0.21875); probably
    rounding when the doc table was written.
11. **Plan §7.3 / §2** says Fig 4 used split65k models; the PDF date suggests earlier mid50 models (§11).

## 14. Other notes for the port

- `utils/mfm_diag_distill.py`: the July fix is required only for the 4-step students (gap_mode long with
  min_gap 0.0); the diagonal students were trained with the older code, which behaves the same for gap_mode random.
- Evo2 env remnants survive in `<DFM repo>/.envs/evo2/lib/python3.12/site-packages`: `evo2-0.5.5`,
  `torch-2.7.1+cu128`, `vtx-1.0.8` (vortex), `triton-3.3.1`, `huggingface_hub-1.18.0`, `biopython-1.87`,
  `pandas-3.0.3`, `numpy-2.4.6`, `einops-0.8.2`; Transformer Engine was not installed (evo2 logs "falling back
  to bf16"); jobs loaded `gcc/12.2.0-fasrc01 cuda/12.9.1-fasrc01`, model `evo2_7b_base`, `HF_HOME=<repo>/.cache/huggingface`.

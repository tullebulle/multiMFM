# Parent-disjoint Tables 6/7 (yeast C0 diversity) at n = 1,000

Created 2026-09-25. Same analysis as `workdir/rebuttal_parent_c0_diversity_20260925/`
(n = 32), re-run against new n = 1,000 generated sets. Nothing outside this directory and the two
new run directories was modified; the n = 32 pilot directories were not touched.

## Generation (slurm jobs 48651189 target +1.0, 48651190 target -1.0)

`scripts/sample_yeast_c0_guidance_hist.py` is the reconstruction at
`/n/netscratch/kozinsky_lab/Lab/uunneberg/scratch_dfm_recon_20260925/scripts/`, whose UNGUIDED
branch reproduces the surviving July ground truth bit-for-bit. Submitted with
`generate_n1000.sbatch` (copied here), one job per target, partition `kempner_h100`, account
`kempner_albergo_lab`, 1 H100, env `micromamba/envs/seq_h100` (python 3.9, torch 2.1.0) with
`PYTHONPATH` = the scratch dir plus its `pydeps/`. Flags are exactly the n = 32 ground-truth flags
with `--n_samples 1000`; `--batch_size 8`, `--seed 0`, `--guidance_frac 8.0`, `--mc 8`,
`--nfe_traj 64`, `--nfe_value 4`, `--t_max 0.95`, `--guide_t_start 0.50`, `--guide_t_end 0.95`,
`--reward_sigma 0.15`, `--reward_scale 0.5`, `--coeff_cap 10.0`, `--grad_clip 10.0` unchanged.
`--batch_size 8` was kept so the RNG stream matches the pilot. Generation took 286 s per target.

Outputs:
- `workdir/yeast_parent_glass_n1000_target1p0_g8.0_seed0/`
- `workdir/yeast_parent_glass_n1000_targetm1p0_g8.0_seed0/`

RNG-stream check: the first 32 unguided sequences of each n = 1,000 run are identical to all 32
unguided sequences of the corresponding n = 32 pilot. The guided branch is not bit-reproducible
(nondeterministic CUDA backward kernels; see the n = 32 README) and agrees with the pilot on
3/32 (+1) and 8/32 (-1) sequences, which is the expected run-to-run behaviour.

## Oracle scoring (in the same jobs, on GPU)

```bash
scripts/score_yeast_c0_oracle.py \
  --oracle_ckpt workdir/yeast_parent_c0_oracle_2026-07-26/best_state.pt \
  --oracle_model_type park_cnn --run_glob <OUT>/sample_scores.csv \
  --out_dir <OUT>/oracle --batch_size 512 --bootstrap 2000 --seed 0 \
  --reverse_complement_average
```

Unlike the n = 32 case, the oracle CSVs here were produced in the logged job, so their provenance
is not in question. The analyzer's independent CPU re-score agrees with them to 3.1e-4 per sample.

## Diversity analysis

```bash
cd /n/netscratch/kozinsky_lab/Lab/uunneberg/repos/dirichlet-flow-matching
LD_LIBRARY_PATH=/n/home04/uunneberg/micromamba/envs/multimfm/lib \
  /n/home04/uunneberg/micromamba/envs/multimfm/bin/python -W ignore::FutureWarning \
  workdir/rebuttal_parent_c0_diversity_20260925/analyze_parent_c0_diversity.py \
  --run_dirs workdir/yeast_parent_glass_n1000_targetm1p0_g8.0_seed0 \
             workdir/yeast_parent_glass_n1000_target1p0_g8.0_seed0 \
  --out_dir workdir/rebuttal_parent_c0_diversity_n1000_20260925 \
  > workdir/rebuttal_parent_c0_diversity_n1000_20260925/run_log.txt 2>&1
```

The script itself is unchanged and still lives in the n = 32 directory, so it keeps using the same
vendored DNA-MFM@187fe7b metric functions. All other defaults are unchanged
(`--baseline_n 1000 --seed 0 --max_pairwise_n 3000 --k_nearest_train 5 --pair_chunk 128
--size_matched_draws 1000`).

## Size matching at n = 1,000

The generated sets now have n = 1,000, the same size as the `test_sample` baseline, so the plain
"pairwise / held-out" and "nearest gen / held-out" ratios against `test_sample` are already
size-matched and are the natural comparison to quote. The n-matched null column is retained for
continuity and is now nearly redundant with it: it draws 1,000 random 1,000-window subsets of the
8,294 held-out test windows, and its mean (pairwise 0.73856, nearest-gen 0.52248) differs from the
single `test_sample` draw (0.73812, 0.52192) by under 0.001. The set-size artifact that made the
n = 32 "nearest gen / held-out" ratios unusable (1.13 to 1.17) is therefore gone. The null's value
at n = 1,000 is that it supplies a sampling distribution, so the percentile ranks are meaningful:
its central 95% interval for pairwise is [0.73739, 0.73975].

## Caveats

- The two unguided sets are byte-identical (same seed, sequences, guide scores and oracle scores);
  the two unguided rows are the same 1,000 sequences. The two guided sets share 0 sequences.
- Validation carried over from the n = 32 run: the `full_sample` baseline still reproduces the old
  Table 6 `train_sample` row (nearest 0.52188, pairwise 0.7376993, median 0.74).
- Token order A,C,G,T = 0,1,2,3 is re-confirmed empirically (best mapping by di- and tri-nucleotide
  L1 and by the guide/oracle round trip, Pearson 0.985).

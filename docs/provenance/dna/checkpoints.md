# DNA (dMFM) checkpoints and data in this repo

Everything below is tracked in `checkpoints/MANIFEST.json` (group `dna`) and fetched with

```bash
python scripts/manifest.py verify --group dna              # presence + sha256
python scripts/manifest.py verify --group dna --fetch-missing
```

The recorded `source` of each file is the read-only holylabs backup
`/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/dirichlet-flow-matching/...`,
which is a byte copy (verified against its `SHA256SUMS.txt`) of the netscratch
`dirichlet-flow-matching/workdir` and `data` trees. netscratch is purge-prone;
holylabs is not. Total: 438 MB checkpoints + 368 MB data.

Selection of *which* checkpoint per run was verified against the args/metadata of
the result runs; see `original_runs.md` (section 0 for the summary table, sections 2-4
for the per-run detail) and the notes below.

## What each file is

| Repo path | Original `workdir/` path | Role |
|---|---|---|
| `checkpoints/dna/base/L{50,100,200,400}/best.pt` | `yeast_parent_a_L{L}_dfm_dit_h192_b4/best.pt` | base DFM DiT (Table 9 top half); teacher of every dMFM; sampler for Table 1/Fig 3, Tables 13-15, 17-18 references |
| `checkpoints/dna/base/L*/args.json` | same dir | `{"args": …, "model_cfg": …}`; `model_cfg` is what `DiTSequenceModel` is built from |
| `checkpoints/dna/dmfm/L{50,100,200}/epoch=*.ckpt` | `yeast_parent_a_L{L}_dmfm_diagonal_h192_b4/` | selected diagonal+ESD dMFM (Table 9 bottom half, Table 13, Table 17, init of the 4-step students) |
| `checkpoints/dna/dmfm/L400/epoch=33-step=45750-v1.ckpt` | `yeast_parent_a_L400_dmfm_diagonly_finetune_20260726/` | **different run dir**: 750-step diagonal-only finetune (batch 8) of the L400 diagonal+ESD run |
| `checkpoints/dna/dmfm/L*/args.json` | same dirs | Lightning trainer args of that run (replay with `scripts/dna/train_dmfm.sbatch`) |
| `checkpoints/dna/dmfm_lineage/L400_diagonal/{epoch=33-step=45000.ckpt,args.json}` | `yeast_parent_a_L400_dmfm_diagonal_h192_b4/` | resume point of the L400 diagonal-only finetune (`scripts/dna/train_dmfm_l400_finetune.sbatch` default `CKPT`); not loaded by any result run; added 2026-09-29, sha256 matches the backup's `SHA256SUMS.txt` |
| `checkpoints/dna/dmfm_4step/L*/epoch=*.ckpt` | `yeast_parent_a_L{L}_dmfm_4step_esd_gap_upto025_corrected_20260727/` | 4-step ESD students, Table 18 only |
| `checkpoints/dna/c0/guide/best_state.pt` | `yeast_parent_c0_guide_2026-07-26/` | C0 guide (park_cnn) trained on parent set a; the reward model in Table 1/Fig 3 and Tables 4-5 |
| `checkpoints/dna/c0/oracle/best_state.pt` | `yeast_parent_c0_oracle_2026-07-26/` | independent C0 oracle trained on the disjoint parent set b; scores Table 1 |
| `checkpoints/dna/c0/*/metadata.json`, `history.csv` | same dirs | training args + `test_metrics` (**this is where Table 11 comes from**) and the per-epoch curve |
| `data/dna/yeast_parent_disjoint/yeast_parent_L*.pt` | `data/yeast_parent_disjoint/` | integer-encoded windows (`seqs`, `parent_id`, `window_start`, …; L50 also has `c0`) |
| `data/dna/yeast_parent_disjoint/yeast_parent_L*_split_seed0.pt` | same | parent-disjoint split indices `a_train/a_val/b_train/b_val/test` (Table 10) |
| `data/dna/yeast_parent_disjoint/parent_split_seed0.json` | same | split summary, incl. `source_sha256` of the raw loop-seq table |

## Verification notes (see `original_runs.md` for the evidence)

- **Base `best.pt` is not the final model.** It is the best `a_val` checkpoint: L50 step 50,000
  (val loss 0.7776), L100 55,000 (0.7830), L200 45,000 (0.7960), L400 45,000 (0.7984). `last.pt`
  (step 200,000, not shipped) is what Table 9's "200,000 steps" describes. Discrepancy, not resolved here.
- **L400 dMFM.** `epoch=33-step=45750.ckpt` and `…-v1.ckpt` have bitwise-identical weights
  (max |diff| = 0); they differ only in Lightning metadata. `-v1` is the file every downstream run
  referenced, so it is the one shipped.
- **4-step students (Table 18).** Exactly one checkpoint per run directory survives, and in each
  case it is both the file hard-coded in the Table 18 launcher / recorded in every Table 18
  `metadata.json`, and the best checkpoint under that run's own monitor
  `val_mfm_probe_esd_gap_0.25`: L50 `epoch=29-step=15000`, L100 `epoch=306-step=149000`,
  L200 `epoch=39-step=34000`, L400 `epoch=96-step=129000`.
- **C0 regressors.** `tests/dna/test_dmfm_port.py::test_c0_regressor_metrics_match_training_metadata`
  recomputes the held-out (parent-disjoint `test`) MAE / RMSE / Pearson from the shipped weights
  and data with reverse-complement averaging, and matches the stored `test_metrics` (Table 11) to
  ~1e-5: guide MAE 0.158771 vs 0.158772, r 0.877414 vs 0.877412; oracle MAE 0.161985 vs 0.161985,
  r 0.872416 vs 0.872417. This is the numerical-faithfulness check for the port.
- **The shipped dMFM / 4-step weights train with the original objective.** Loading the L50
  `epoch=95-step=49000.ckpt` (and the 4-step `epoch=29-step=15000.ckpt`) into the original
  DNA-MFM@187fe7b `DNAModule` and into `dmfm.lightning.dna_module.DNAModule` gives bitwise
  identical losses, gradients and validation probe metrics
  (`tests/dna/check_against_original_library.py`).
- **Not slimmed.** The `.ckpt` files keep their optimizer state (~40 MB each, ~19 MB of weights).
  They are left byte-identical to the originals on purpose: their sha256 then matches both the
  holylabs `SHA256SUMS.txt` and the hashes quoted in `original_runs.md`, and `--init_ckpt` /
  `--ckpt` resumes (the 4-step students resume from the diagonal ones) keep working. Total is only
  438 MB; slimming would save ~170 MB at the cost of that chain of custody.

## Raw data (not redistributed)

`data/yeast_sequences.txt` (the loop-seq table of Basu et al. 2021, "Measuring DNA mechanics on the
genome scale", Nature 589:462-467) is gone and is not shipped here. See `README.md` in this
directory for how to re-download it and verify it against
`source_sha256 = aabfb6c51cb19e99a492f1a86d93fd658360181b927ba31489f432f810aadbc2`, then rebuild the
windows with `scripts/dna/prepare_data.sbatch`. The derived `yeast_parent_*` files in
`data/dna/yeast_parent_disjoint/` are the ones the paper runs used, so re-downloading is only needed
to rebuild the dataset from scratch.

## Manifest entries

| Repo path | MB | sha256 (first 16) | source (relative to the holylabs backup's `dirichlet-flow-matching/`) | note |
|---|---|---|---|---|
| `checkpoints/dna/base/L100/args.json` | 0.0 | `4ef934bf9c0858d2…` | `workdir/yeast_parent_a_L100_dfm_dit_h192_b4/args.json` | base DFM DiT h192 b4, L=100, best a_val checkpoint (Table 9; teacher of every dMFM; sampler/GLASS reference in Table 1, Fig 5, Tables 13-15, 17, 18) |
| `checkpoints/dna/base/L100/best.pt` | 35.0 | `32db4ea4e13e3480…` | `workdir/yeast_parent_a_L100_dfm_dit_h192_b4/best.pt` | base DFM DiT h192 b4, L=100, best a_val checkpoint (Table 9; teacher of every dMFM; sampler/GLASS reference in Table 1, Fig 5, Tables 13-15, 17, 18) |
| `checkpoints/dna/base/L200/args.json` | 0.0 | `548f034959e3fd24…` | `workdir/yeast_parent_a_L200_dfm_dit_h192_b4/args.json` | base DFM DiT h192 b4, L=200, best a_val checkpoint (Table 9; teacher of every dMFM; sampler/GLASS reference in Table 1, Fig 5, Tables 13-15, 17, 18) |
| `checkpoints/dna/base/L200/best.pt` | 35.0 | `26970cbc3978104c…` | `workdir/yeast_parent_a_L200_dfm_dit_h192_b4/best.pt` | base DFM DiT h192 b4, L=200, best a_val checkpoint (Table 9; teacher of every dMFM; sampler/GLASS reference in Table 1, Fig 5, Tables 13-15, 17, 18) |
| `checkpoints/dna/base/L400/args.json` | 0.0 | `e0c15ba241428271…` | `workdir/yeast_parent_a_L400_dfm_dit_h192_b4/args.json` | base DFM DiT h192 b4, L=400, best a_val checkpoint (Table 9; teacher of every dMFM; sampler/GLASS reference in Table 1, Fig 5, Tables 13-15, 17, 18) |
| `checkpoints/dna/base/L400/best.pt` | 35.0 | `9630a7a65a01944c…` | `workdir/yeast_parent_a_L400_dfm_dit_h192_b4/best.pt` | base DFM DiT h192 b4, L=400, best a_val checkpoint (Table 9; teacher of every dMFM; sampler/GLASS reference in Table 1, Fig 5, Tables 13-15, 17, 18) |
| `checkpoints/dna/base/L50/args.json` | 0.0 | `f06ec829d95ec3a5…` | `workdir/yeast_parent_a_L50_dfm_dit_h192_b4/args.json` | base DFM DiT h192 b4, L=50, best a_val checkpoint (Table 9; teacher of every dMFM; sampler/GLASS reference in Table 1, Fig 5, Tables 13-15, 17, 18) |
| `checkpoints/dna/base/L50/best.pt` | 35.0 | `e0338928f346ea4b…` | `workdir/yeast_parent_a_L50_dfm_dit_h192_b4/best.pt` | base DFM DiT h192 b4, L=50, best a_val checkpoint (Table 9; teacher of every dMFM; sampler/GLASS reference in Table 1, Fig 5, Tables 13-15, 17, 18) |
| `checkpoints/dna/c0/guide/best_state.pt` | 0.4 | `3403aaef91dc8e21…` | `workdir/yeast_parent_c0_guide_2026-07-26/best_state.pt` | C0 guide regressor (park_cnn) yeast_parent_c0_guide_2026-07-26 (Tables 1, 4-5, 11) |
| `checkpoints/dna/c0/guide/history.csv` | 0.0 | `e9f23073d49357e9…` | `workdir/yeast_parent_c0_guide_2026-07-26/history.csv` | C0 guide regressor (park_cnn) yeast_parent_c0_guide_2026-07-26 (Tables 1, 4-5, 11) |
| `checkpoints/dna/c0/guide/metadata.json` | 0.0 | `3f65f574bc3ed72e…` | `workdir/yeast_parent_c0_guide_2026-07-26/metadata.json` | C0 guide regressor (park_cnn) yeast_parent_c0_guide_2026-07-26 (Tables 1, 4-5, 11) |
| `checkpoints/dna/c0/oracle/best_state.pt` | 0.4 | `cbbb8a04a67ed1ba…` | `workdir/yeast_parent_c0_oracle_2026-07-26/best_state.pt` | C0 oracle regressor (park_cnn) yeast_parent_c0_oracle_2026-07-26 (Tables 1, 4-5, 11) |
| `checkpoints/dna/c0/oracle/history.csv` | 0.0 | `c06df798aeb68342…` | `workdir/yeast_parent_c0_oracle_2026-07-26/history.csv` | C0 oracle regressor (park_cnn) yeast_parent_c0_oracle_2026-07-26 (Tables 1, 4-5, 11) |
| `checkpoints/dna/c0/oracle/metadata.json` | 0.0 | `ffa406f952b45118…` | `workdir/yeast_parent_c0_oracle_2026-07-26/metadata.json` | C0 oracle regressor (park_cnn) yeast_parent_c0_oracle_2026-07-26 (Tables 1, 4-5, 11) |
| `checkpoints/dna/dmfm/L100/args.json` | 0.0 | `ae79ba0c72b7c573…` | `workdir/yeast_parent_a_L100_dmfm_diagonal_h192_b4/args.json` | selected dMFM (diagonal+ESD) L=100, run yeast_parent_a_L100_dmfm_diagonal_h192_b4 (Table 9 dMFM rows, Table 13, Table 17; init of the 4-step student) |
| `checkpoints/dna/dmfm/L100/epoch=80-step=39000.ckpt` | 39.8 | `d712081075ccc874…` | `workdir/yeast_parent_a_L100_dmfm_diagonal_h192_b4/epoch=80-step=39000.ckpt` | selected dMFM (diagonal+ESD) L=100, run yeast_parent_a_L100_dmfm_diagonal_h192_b4 (Table 9 dMFM rows, Table 13, Table 17; init of the 4-step student) |
| `checkpoints/dna/dmfm/L200/args.json` | 0.0 | `1a28be22cfe5fa81…` | `workdir/yeast_parent_a_L200_dmfm_diagonal_h192_b4/args.json` | selected dMFM (diagonal+ESD) L=200, run yeast_parent_a_L200_dmfm_diagonal_h192_b4 (Table 9 dMFM rows, Table 13, Table 17; init of the 4-step student) |
| `checkpoints/dna/dmfm/L200/epoch=51-step=45000.ckpt` | 39.8 | `a530de2c1e0b79dd…` | `workdir/yeast_parent_a_L200_dmfm_diagonal_h192_b4/epoch=51-step=45000.ckpt` | selected dMFM (diagonal+ESD) L=200, run yeast_parent_a_L200_dmfm_diagonal_h192_b4 (Table 9 dMFM rows, Table 13, Table 17; init of the 4-step student) |
| `checkpoints/dna/dmfm/L400/args.json` | 0.0 | `c9901303b97f925f…` | `workdir/yeast_parent_a_L400_dmfm_diagonly_finetune_20260726/args.json` | selected dMFM L=400: diagonal-only finetune of the L400 diagonal+ESD run, run yeast_parent_a_L400_dmfm_diagonly_finetune_20260726 (Table 9 dMFM rows, Table 13, Table 17; init of the 4-step student) |
| `checkpoints/dna/dmfm/L400/epoch=33-step=45750-v1.ckpt` | 39.8 | `28163b80bb7f2801…` | `workdir/yeast_parent_a_L400_dmfm_diagonly_finetune_20260726/epoch=33-step=45750-v1.ckpt` | selected dMFM L=400: diagonal-only finetune of the L400 diagonal+ESD run, run yeast_parent_a_L400_dmfm_diagonly_finetune_20260726 (Table 9 dMFM rows, Table 13, Table 17; init of the 4-step student) |
| `checkpoints/dna/dmfm/L50/args.json` | 0.0 | `64d34fd9ebf48848…` | `workdir/yeast_parent_a_L50_dmfm_diagonal_h192_b4/args.json` | selected dMFM (diagonal+ESD) L=50, run yeast_parent_a_L50_dmfm_diagonal_h192_b4 (Table 9 dMFM rows, Table 13, Table 17; init of the 4-step student) |
| `checkpoints/dna/dmfm/L50/epoch=95-step=49000.ckpt` | 39.8 | `bd86c8a6db490f01…` | `workdir/yeast_parent_a_L50_dmfm_diagonal_h192_b4/epoch=95-step=49000.ckpt` | selected dMFM (diagonal+ESD) L=50, run yeast_parent_a_L50_dmfm_diagonal_h192_b4 (Table 9 dMFM rows, Table 13, Table 17; init of the 4-step student) |
| `checkpoints/dna/dmfm_4step/L100/args.json` | 0.0 | `cefcae669629cae9…` | `workdir/yeast_parent_a_L100_dmfm_4step_esd_gap_upto025_corrected_20260727/args.json` | training args for 4-step dMFM L=100 (yeast_parent_a_L100_dmfm_4step_esd_gap_upto025_corrected_20260727) |
| `checkpoints/dna/dmfm_4step/L100/epoch=306-step=149000.ckpt` | 39.8 | `9f87269c1677724e…` | `workdir/yeast_parent_a_L100_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=306-step=149000.ckpt` | 4-step dMFM student L=100, run yeast_parent_a_L100_dmfm_4step_esd_gap_upto025_corrected_20260727 (Table 18) |
| `checkpoints/dna/dmfm_4step/L200/args.json` | 0.0 | `5fe8e3b9e4ef3b2f…` | `workdir/yeast_parent_a_L200_dmfm_4step_esd_gap_upto025_corrected_20260727/args.json` | training args for 4-step dMFM L=200 (yeast_parent_a_L200_dmfm_4step_esd_gap_upto025_corrected_20260727) |
| `checkpoints/dna/dmfm_4step/L200/epoch=39-step=34000.ckpt` | 39.8 | `3a15b7ef1da8722a…` | `workdir/yeast_parent_a_L200_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=39-step=34000.ckpt` | 4-step dMFM student L=200, run yeast_parent_a_L200_dmfm_4step_esd_gap_upto025_corrected_20260727 (Table 18) |
| `checkpoints/dna/dmfm_4step/L400/args.json` | 0.0 | `ba3e73c0b78ac013…` | `workdir/yeast_parent_a_L400_dmfm_4step_esd_gap_upto025_corrected_20260727/args.json` | training args for 4-step dMFM L=400 (yeast_parent_a_L400_dmfm_4step_esd_gap_upto025_corrected_20260727) |
| `checkpoints/dna/dmfm_4step/L400/epoch=96-step=129000.ckpt` | 39.8 | `0cede715d71bb943…` | `workdir/yeast_parent_a_L400_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=96-step=129000.ckpt` | 4-step dMFM student L=400, run yeast_parent_a_L400_dmfm_4step_esd_gap_upto025_corrected_20260727 (Table 18) |
| `checkpoints/dna/dmfm_4step/L50/args.json` | 0.0 | `f2745f5daada9568…` | `workdir/yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727/args.json` | training args for 4-step dMFM L=50 (yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727) |
| `checkpoints/dna/dmfm_4step/L50/epoch=29-step=15000.ckpt` | 39.8 | `998b7065f41235c4…` | `workdir/yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727/epoch=29-step=15000.ckpt` | 4-step dMFM student L=50, run yeast_parent_a_L50_dmfm_4step_esd_gap_upto025_corrected_20260727 (Table 18) |
| `data/dna/yeast_parent_disjoint/parent_split_seed0.json` | 0.0 | `4789e01590d6cd29…` | `data/yeast_parent_disjoint/parent_split_seed0.json` | parent-disjoint yeast windows/splits (Table 10; all DNA experiments) |
| `data/dna/yeast_parent_disjoint/yeast_parent_L100.pt` | 63.5 | `4ca72b27178007ef…` | `data/yeast_parent_disjoint/yeast_parent_L100.pt` | parent-disjoint yeast windows/splits (Table 10; all DNA experiments) |
| `data/dna/yeast_parent_disjoint/yeast_parent_L100_split_seed0.pt` | 0.6 | `5813b856a50b1d74…` | `data/yeast_parent_disjoint/yeast_parent_L100_split_seed0.pt` | parent-disjoint yeast windows/splits (Table 10; all DNA experiments) |
| `data/dna/yeast_parent_disjoint/yeast_parent_L200.pt` | 112.6 | `7594676b29ad587a…` | `data/yeast_parent_disjoint/yeast_parent_L200.pt` | parent-disjoint yeast windows/splits (Table 10; all DNA experiments) |
| `data/dna/yeast_parent_disjoint/yeast_parent_L200_split_seed0.pt` | 0.6 | `b50356b9cf8938cc…` | `data/yeast_parent_disjoint/yeast_parent_L200_split_seed0.pt` | parent-disjoint yeast windows/splits (Table 10; all DNA experiments) |
| `data/dna/yeast_parent_disjoint/yeast_parent_L400.pt` | 172.3 | `fd84ca90524fdc5c…` | `data/yeast_parent_disjoint/yeast_parent_L400.pt` | parent-disjoint yeast windows/splits (Table 10; all DNA experiments) |
| `data/dna/yeast_parent_disjoint/yeast_parent_L400_split_seed0.pt` | 0.4 | `92c9eb99f81e85a3…` | `data/yeast_parent_disjoint/yeast_parent_L400_split_seed0.pt` | parent-disjoint yeast windows/splits (Table 10; all DNA experiments) |
| `data/dna/yeast_parent_disjoint/yeast_parent_L50.pt` | 34.6 | `7d9f61d1e3489b78…` | `data/yeast_parent_disjoint/yeast_parent_L50.pt` | parent-disjoint yeast windows/splits (Table 10; all DNA experiments) |
| `data/dna/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt` | 0.7 | `4fd6daee8b4777c4…` | `data/yeast_parent_disjoint/yeast_parent_L50_split_seed0.pt` | parent-disjoint yeast windows/splits (Table 10; all DNA experiments) |

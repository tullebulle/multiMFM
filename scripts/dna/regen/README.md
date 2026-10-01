# `scripts/dna/regen` — regenerating the lost DNA results

Authors' decision 9 (2026-09-29): regenerate the four DNA items whose outputs were lost in the
netscratch purge. Their checkpoints survive; the published numbers came from May/June 2026
`split65k` models and data that do not, so **the regenerated numbers differ from the submitted
PDF by construction**. `notebooks/09_dna_regenerated.ipynb` shows both side by side.

| paper item | producer | wrapper | verdict |
|---|---|---|---|
| Fig 6 + Table 16 | `dmfm.regen.base_marginals` (new implementation of the authors' June recipe, `results/dna/original_scripts/make_yeast_split_sample_figures.py`) | `fig6_table16_marginals.sbatch` | regenerated / Table 16 approximated |
| Fig 7 | `dmfm.experiments.sample_c0_guidance` (original script, recovered) | `fig7_guidance_sweep.sbatch` | regenerated |
| Fig 8 | `dmfm.regen.guidance_trace` (**new implementation**) | `fig8_guidance_trace.sbatch` | approximated |
| Table 6 | `dmfm.experiments.score_yeast_c0_evo2` (recovered from bytecode) | `score_evo2.sbatch` | regenerated |
| Tables 4, 5, App. B.1 | `dmfm.experiments.analyze_c0_diversity` (ported analyser) | `scripts/dna/tables4_5_c0_diversity.sh` | regenerated |

Tables 4, 5 and 6 and the App. B.1 prose all characterise the same generated sample sets, and all
three were printed from the purged `split65k` pipeline while the text describes the parent-disjoint
models. Authors' decision (2026-09-29): regenerate all of them together on the parent-disjoint
n = 1,000 C0 sets in `results/dna/c0_guidance/n1000/`, so the three tables stay internally
consistent, and report the weaker clustering result that follows (App. B.1's target_c0 = −1
pairwise/held-out ratio moves 0.9507 → 0.9872). The regeneration is written to
`results/dna/diversity/regen/`; the split65k record in `results/dna/june_split65k_recovered/` and
the round-1 collection in `results/dna/diversity/n1000/` are left untouched.

`evo2_environment.yml` and `score_evo2.sbatch` are copies of the files of the same name in
`scripts/dna/` (see the header of each); the only content change is the `einops` pin, which pip
cannot resolve at the recorded 0.8.2 because `vtx` 1.0.8 pins 0.8.1.

## Run everything

```bash
sbatch scripts/dna/regen/fig6_table16_marginals.sbatch          # Fig 6 + Table 16 (~10 min)
sbatch scripts/dna/regen/fig7_guidance_sweep.sbatch             # Fig 7, 6-task array
sbatch scripts/dna/regen/fig8_guidance_trace.sbatch             # Fig 8
MODE=c0 OUT_DIR=outputs/dna/regen/table6 \
  sbatch scripts/dna/regen/score_evo2.sbatch                    # Table 6, separate evo2 env

# Tables 4, 5 and the App. B.1 clustering statistic (CPU only, ~20 min, no Slurm needed)
bash scripts/dna/tables4_5_c0_diversity.sh \
  results/dna/c0_guidance/n1000/target_m1p0 \
  results/dna/c0_guidance/n1000/target_1p0 \
  outputs/dna/regen/diversity

bash scripts/dna/regen/collect.sh                               # one collection command
```

### Running Fig 6 on CPU instead

`kempner_requeue` can be saturated for hours. Fig 6 / Table 16 runs perfectly well on CPU, so
there is a twin wrapper for `serial_requeue` (which denies `kempner_albergo_lab`, hence
`--account=albergo_lab`):

```bash
sbatch scripts/dna/regen/fig6_table16_marginals_cpu.sbatch     # 20 shards x 4 threads, ~6 min each
```

Sizing note for whoever plans the next CPU run: **the L = 50 base DFM is 2.9M parameters**, not the
~33M that is easy to assume from the other models in this project. Measured here: 4.9 ms per
(sample, integration step) on 4 threads, so n = 10,000 at `nfe=128` is ~105 min in a single
process and ~6 min per shard at 20 shards. Note also that `gaussian_denoiser_flow_step` evaluates
the network twice per step, so the true cost is 2x the nominal NFE.

It writes to its own sample directory `outputs/dna/regen/fig6_cpu`, so CPU and GPU runs are never
mixed, and `collect.sh` auto-detects whichever is complete (it reads the shard count off the
filenames and prefers the GPU directory). Every shard records its device, torch version, Slurm
account/partition/node and decode-margin statistics.

**Why CPU is safe here, demonstrated rather than asserted.** The initial noise is drawn by a
seeded *CPU* generator and only then moved to the sampling device, so CPU and GPU runs share
identical randomness and differ only in float arithmetic. Sampling is deterministic given that
noise, and the decoded token is `argmax` of the final state — so each shard reports the top-1
minus top-2 margin of every token. The flow is integrated to `t_max = 1.0`, which drives the
state onto a simplex vertex, giving a margin of ~1.0 against a float32 noise floor of ~1e-7:
no token's decode is close enough to be flipped by a change of device. Fig 6 and Table 16 are
in any case distributional summaries over thousands of samples.

### Which partition each item runs on

**One route per item. Do not submit duplicate routes.** On 2026-09-29 the authors instructed that
we stop requesting GPUs we are not going to use — across all agents about 27 GPU requests were
outstanding, many of them duplicate routes for work already moved to CPU — and 17 job groups were
cancelled, including this area's `kempner_requeue` copies of Fig 6, Fig 7, Fig 8 and Table 6
(49273577, 49276291, 49273114, 49277302). No shards existed for any of them, so nothing was lost.
The surviving route for each item is:

| item | partition | account |
|---|---|---|
| Fig 6 + Table 16 | `serial_requeue` (CPU, 20 shards) | `albergo_lab` |
| Fig 7 | `gpu_requeue` | `albergo_lab` |
| Fig 8 | `gpu_requeue` | `albergo_lab` |
| Table 6 | `gpu_requeue` | `albergo_lab` |

**Fig 6 has no GPU fallback any more**: the CPU array is its only route, so it must be allowed to
finish rather than cancelled in favour of a GPU copy. (An earlier version of this file said to
cancel the losing route only once the winner had produced output; that ordering no longer applies,
because there is no second route to lose.)

The commands below record how each was submitted and are what to use if one has to be *replaced*
after a failure — not run alongside an existing copy. If a job disappears unexpectedly, ask the
coordinator before resubmitting: it may have been cancelled deliberately.

`kempner_requeue` can stay saturated for hours, which is why the GPU work sits on the general
`gpu_requeue` partition instead.

```bash
# Fig 8
OUT_DIR=results/dna/regen/fig8_gr sbatch --partition=gpu_requeue --account=albergo_lab \
  --constraint="a100|h100|h200|rtx6000pro|rtxa6000" --job-name=dna_regen_fig8_gr \
  scripts/dna/regen/fig8_guidance_trace.sbatch

# Fig 7 (own output tree via FIG7_BASE, so the two sweeps never collide)
FIG7_BASE=outputs/dna/regen/fig7_gr sbatch --partition=gpu_requeue --account=albergo_lab \
  --constraint="a100|h100|h200|rtx6000pro|rtxa6000" --array=0-5%2 --job-name=dna_regen_fig7_gr \
  scripts/dna/regen/fig7_guidance_sweep.sbatch

# Table 6 (Evo 2 7B -- constrain the card, see below)
MODE=c0 OUT_DIR=outputs/dna/regen/table6_gr sbatch --partition=gpu_requeue --account=albergo_lab \
  --constraint="a100|h100|h200" --job-name=dna_regen_evo2_gr scripts/dna/regen/score_evo2.sbatch
```

Two things to know:

* **Account.** `gpu_requeue` rejects `kempner_albergo_lab` ("Invalid account or account/partition
  combination"); use `--account=albergo_lab`. `kempner_requeue` is the opposite way round.
* **Constrain the card for Evo 2.** `gpu_requeue` is a mixed pool that includes MIG slices as small
  as `nvidia_a100_1g.5gb` (5 GB) and `a100_3g.20gb` (20 GB). Evo 2 7B in bf16 is ~14 GB of weights
  (the checkpoint is 13 GB) before activations, so a small card fails at load time, an hour into the
  queue, rather than at submission. The features are `a100`, `a100-mig`, `h100`, `h200`,
  `rtx6000pro`, `rtxa6000`; constraining to `a100|h100|h200` keeps every card at >= 40 GB and
  excludes `a100-mig`. For Fig 7 and Fig 8 the model is 2.9M parameters, so only the MIG slices need
  excluding.

`collect.sh` scans every `outputs/dna/regen/fig6*`, `fig7*` and `table6*` tree and every
`results/dna/regen/fig8*` directory, takes the first complete one, and records which tree each
result came from (`fig7/index.csv` has a `source_tree` column; `table6/SOURCE.txt` names its
origin; `fig6_table16/metadata.json` lists each shard's device, card name, torch version and
Slurm account/partition/node). That multi-tree scan is kept because it costs nothing and makes a
replacement run after a failure drop straight in — not because duplicate routes should be running.

**Fig 7, Fig 8 and Table 6 do not move to CPU.**

* **Fig 7.** The queued settings are the recovered launcher's own defaults (MC = 128, 128
  trajectory steps, 12-step value rollout), which is the *faithful* configuration, and the sweep
  costs ~3.6e8 sequence evaluations (~1,000 CPU core-hours). `sample_c0_guidance` has no
  sample-range flag, so a strength cannot be sharded and a preempted `serial_requeue` task would
  restart from zero. `ESTIMATOR=table1` (MC = 8, 64 trajectory steps, 4-step rollout — the
  estimator the paper's own DNA guidance results used) is ~90x cheaper and *would* finish on CPU
  in about an hour. **It is not the default and should not be used to save queue time:** Fig 7 is
  already a reconstruction in one respect, since the published guidance strengths were never
  recorded, and swapping the estimator would add a second deviation for no scientific gain on a
  qualitative appendix figure. Use it only if the authors ask for the speed, and say so in the
  caption if you do.
* **Fig 8.** A single-trajectory illustration; cheap on GPU. On CPU the MC = 128 value gradient
  with autograd over 128 steps is both slow and memory-hungry (a login-node benchmark was killed
  by the watchdog), and no published number depends on it.
* **Table 6.** Needs Evo 2 7B in bf16, which is slow and numerically different on CPU, while its
  NLL values are published-facing.

`collect.sh` reduces the sharded / per-strength outputs into `results/dna/regen/`, rewrites
`results/dna/regen/PROVENANCE.md` and re-executes `notebooks/09_dna_regenerated.ipynb`. It is
idempotent and tolerates unfinished items (they show as `pending`), so it can be run at any time.

Every job records itself in `outputs/dna/regen/jobs.tsv`, which `collect.sh` turns into the job
table of `PROVENANCE.md`. Every job skips work whose output already exists, so a preempted
`kempner_requeue` task can simply be resubmitted.

## Smoke tests (gpu_test)

```bash
N=512 NUM_SHARDS=1 sbatch --array=0 --partition=gpu_test --time=0:20:00 \
  scripts/dna/regen/fig6_table16_marginals.sbatch
N_SAMPLES=64 sbatch --array=0,5 --partition=gpu_test --time=0:30:00 \
  scripts/dna/regen/fig7_guidance_sweep.sbatch
MC=8 NFE_TRAJ=32 NFE_VALUE=4 OUT_DIR=outputs/dna/regen/fig8_smoke \
  sbatch --partition=gpu_test --time=0:30:00 scripts/dna/regen/fig8_guidance_trace.sbatch
```

CPU checks of the same code paths (no GPU, ~3 min):

```bash
python -m dmfm.regen.base_marginals --n 64 --num_shards 1 --nfe 8 --device cpu \
  --sample_dir /tmp/f6 --out_dir /tmp/f6_out --reduce
python -m dmfm.regen.guidance_trace --sample_ids 0 --mc 4 --mc_chunk 4 --nfe_traj 4 \
  --nfe_value 2 --device cpu --out_dir /tmp/f8
```

## The Evo 2 environment (Table 6)

```bash
micromamba create -f scripts/dna/regen/evo2_environment.yml       # Python 3.12, torch 2.7.1+cu128
~/micromamba/envs/evo2/bin/python scripts/dna/regen/evo2_postinstall.py
export HF_HOME=$PWD/cache/xdg/huggingface                          # git-ignored, 13 GB
python -c "from huggingface_hub import snapshot_download; snapshot_download(
    'arcinstitute/evo2_7b_base', revision='074097e9dc788e8bfe045d6495b9f6153a7c6bfc')"
```

`arcinstitute/evo2_7b_base` at that revision is the exact model the paper runs used (it is also
the current HEAD of that repo). Evo 2 7B in bf16 needs a GPU with >= 40 GB; the original runs
used one H100 with `module load gcc/12.2.0-fasrc01 cuda/12.9.1-fasrc01`, no Transformer Engine
and no flash-attn, which is what `score_evo2.sbatch` reproduces.

Two things had to be rebuilt differently from the recorded pins, both documented in place:

* `einops` is pinned to 0.8.1, not the recorded 0.8.2: `vtx` 1.0.8 requires `einops==0.8.1`, and
  pip (unlike `uv`, which built the original env) refuses to violate that pin.
* **FlashAttention cannot be installed on this cluster.** The published
  `flash_attn ... cp312 ... torch2.7 ... cu12` wheels and the `flash_attn_2_cuda` binaries
  bundled inside `vortex/ops/depr_attn/` are built against GLIBC 2.32; the nodes run GLIBC 2.28
  (Rocky 8), so both fail with `version GLIBC_2.32 not found`. Building from source needs `nvcc`
  and hours on a GPU node. `evo2_postinstall.py` therefore enables the no-flash path that `vtx`
  documents for Evo 2 7B (guard `vortex/ops/__init__.py`'s bare `import flash_attn_2_cuda`, set
  `use_flash_attn: False` in `configs/evo2-7b-8k.yml`) and is run automatically by
  `score_evo2.sbatch`. FlashAttention is exact attention, so this changes only floating-point
  reduction order in the five attention layers, and it matches the recovered record of the
  original runs ("no Transformer Engine and no flash-attn").

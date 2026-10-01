# `dmfm` API cheat sheet (for notebooks, Fig 4 benchmarks, regenerated results)

**Status / breaking changes:** none so far (only additions since the first version). `dmfm.api` is the stable surface; its function
signatures will not change (only additions). The experiment modules
(`dmfm.experiments.*`) keep their original CLI and internals; import from them only if
`dmfm.api` lacks what you need, and tell dna-core so it can be added here.
Last updated: 2026-09-29.

Everything below runs in the `multimfm` env (`source scripts/env.sh`, or the kernel
`multimfm`). Artifacts: `python scripts/manifest.py verify --group dna`. All paths are
repo-relative via `dmfm.paths` (override the root with `MULTIMFM_ROOT`).
Checks of every snippet: `python tests/dna/test_api.py` (CPU, ~2 min).

## Conventions

- Time runs 0 (noise) -> 1 (data); `x_t = t * onehot(x1) + (1 - t) * eps` (linear beta).
- States: float `[B, L, 4]`, alphabet order **A, C, G, T**. Tokens: `LongTensor[B, L]`;
  decode states with `argmax(-1)`.
- Posterior sampler: `(eps [B,L,4], x_cond [B,L,4], t_cond [B]) -> endpoint [B,L,4]`,
  a sample of `x_1 | x_t = x_cond`. Both the GLASS and the dMFM samplers are differentiable
  w.r.t. `x_cond`.
- Loaders return frozen `eval()` models. `device=None` picks CUDA if available.
- Lengths: `paths.LENGTHS = (50, 100, 200, 400)`. The C0 guide/oracle are L=50 only.

## 1. Loading checkpoints

```python
from dmfm import api, paths

base, cfg = api.load_base(50, device)            # base DFM DiT, checkpoints/dna/base/L50/best.pt
                                                 # cfg = checkpoint model_cfg (seq_len, beta schedule, ...)
dmfm  = api.load_dmfm(50, device)                # diagonal+ESD dMFM student (Tables 9, 13, 17; steering)
dmfm4 = api.load_dmfm(50, device, kind="dmfm4")  # 4-step ESD student (Table 18 only)
guide  = api.load_c0("guide", device)            # C0 park_cnn guide (reward model; parent set a)
oracle = api.load_c0("oracle", device)           # C0 park_cnn oracle (independent; parent set b)

# explicit files (e.g. a retrained model): args.json must sit next to a dMFM ckpt
base, cfg = api.load_base(ckpt="outputs/dna/.../best.pt")
dmfm = api.load_dmfm(ckpt="outputs/dna/.../epoch=..ckpt")
```

| what | file (`paths.` helper) |
|---|---|
| base DFM | `checkpoints/dna/base/L{L}/best.pt` + `args.json` (`paths.base_ckpt(L)`, `paths.base_args(L)`) |
| dMFM | `checkpoints/dna/dmfm/L{L}/epoch=...ckpt` + `args.json` (`paths.dmfm_ckpt(L)`) |
| 4-step dMFM | `checkpoints/dna/dmfm_4step/L{L}/epoch=...ckpt` (`paths.dmfm4_ckpt(L)`) |
| C0 guide / oracle | `checkpoints/dna/c0/{guide,oracle}/best_state.pt` + `metadata.json` (`paths.c0_guide()`, `paths.c0_oracle()`) |

Lower level (same objects): `dmfm.experiments.ablate_glass_gradient_mc.load_model(path, device)`,
`dmfm.utils.model_loading.load_teacher / load_student / load_args_json`,
`dmfm.regressors.load_c0_regressor(path, "park_cnn", device)`.
The dMFM student is an MFM `BaseModel`: `student(s, t, x, t_cond, x_cond)` is the flow map
`X_{s,t}(x | t_cond, x_cond)`, `student.v(...)` its velocity, `student.Psi_st(...)` its
denoiser. The base model: `base.psi_st(x, s, s)` (denoiser), `base.v(s, s, x)` (drift),
`base.sde_sigma_sq(t) = 2 (1/t - 1)`.

## 2. Data (parent-disjoint split)

```python
seqs = api.load_split_seqs(50, "test")        # LongTensor[8294, 50]; splits: a_train a_val b_train b_val test | full
idx  = api.split_indices(50, "a_val")         # row indices into the window table
tab  = api.load_windows(50)                   # {'seqs','parent_id','window_start', 'c0' (L50 only), ...}
x, c0 = api.load_c0_labels("test")            # 58 held-out parents, 8,294 windows + measured C0
```

Split summary / Table 10: `data/dna/yeast_parent_disjoint/parent_split_seed0.json`.

## 3. Unguided sampling (base DFM)

```python
x0 = api.paired_initial_noise(range(8), 50, seed=0, device=device)   # Table 1 noise convention (seed + i)
x  = api.sample_unguided(base, cfg, x0, nfe=64, t_max=0.95)          # Table 1 unguided branch
seqs = api.tokens_to_strings(x)                                     # argmax decode
```
Fresh noise: `torch.randn(n, L, 4)`; calibration/probe code uses `nfe=..., t_max=0.999`.
The unguided branch is deterministic: ids 0..7 reproduce the stored n=1000 run
(`results/dna/c0_guidance/n1000/target_1p0/sample_scores.csv`) token for token.

## 4. C0-guided sampling (Table 1 / Fig 3)

Exact Table 1 code path (GLASS value gradients on the base DFM, MC=8, 4-step Euler GLASS,
guidance on t in [0.5, 0.95], guidance_frac 8, cap 10, clip 10, reward
`0.5 * -0.5 ((guide(x1) - y*) / 0.15)^2`):

```python
rows = api.c0_guided_pairs(base, cfg, guide, range(0, 8), target_c0=1.0)   # one batch of 8
# each row: sample_idx, unguided, guided (guide scores of the hard sequences), *_abs_to_target,
#           seq_unguided, seq_guided        -- same as sample_c0_guidance's sample_scores.csv
api.TABLE1_SETTINGS   # the settings; override with keywords, e.g. mc=16, guidance_frac=4.0
```
Full runs: `TARGET_C0=1.0 sbatch scripts/dna/table1_c0_guidance.sbatch` (n=1000, batches of 8).

Generic guided sampling with any value estimator (e.g. dMFM-based steering for Fig 4):
```python
logr = api.c0_log_reward_fn(guide, target_c0=1.0)                 # reward_sigma=0.15, reward_scale=0.5
post = api.dmfm_posterior_fn(dmfm, sampler="flow_map", n_steps=1) # or api.glass_posterior_fn(base, n_steps=4)
pool = api.paired_eps_pool(range(8), 50, mc=8, seed=0, device=device)   # frozen per-sample MC noise
vg = lambda x, t: api.value_and_grad(post, logr, x, t, pool, mc_chunk=8)
x_guided, x_unguided = api.guided_sample(base, cfg, x0, vg, return_unguided=True)  # Table 1 defaults
```
(`vg` may draw fresh noise per call instead of a frozen pool; the Table 1 sampler uses a
frozen per-sample pool, the old Fig 4 script fresh `torch.randn` per step.)

### 4b. Table 1 with the dMFM (what the paper now reports)

The printed Table 1 (and its n=1000 rerun) used GLASS value gradients on the *base DFM*; no dMFM was
loaded. Authors' decision 2026-09-29: Table 1 reports the dMFM run instead. Its sampler is
`dmfm.experiments.sample_c0_guidance_dmfm --paired --posterior dmfm` (identical protocol, only the
posterior sampler changes: 4 composed dMFM flow-map steps instead of 4 GLASS Euler steps);
`--paired --posterior glass` reproduces the printed-Table-1 code path exactly. From Python:

```python
from dmfm.experiments import sample_c0_guidance_dmfm as t1d
args = t1d.parse_args(["--paired", "--posterior", "dmfm", "--n_samples", "16", "--batch_size", "8", "--mc", "8",
                       "--mc_chunk", "8", "--nfe_sample", "64", "--nfe_value", "4", "--t_max", "0.95",
                       "--guide_t_start", "0.5", "--reward_scale", "0.5", "--target_c0", "1.0"])
rows = t1d.paired_batch(args, range(8), base, cfg, dmfm, guide, device)   # same row schema as c0_guided_pairs
```
Full run, both arms under the same scale-free rule (the only like-for-like comparison):

```bash
GRAD_NORMALIZE=1 POSTERIOR=dmfm  NFE_VALUE=4 sbatch --array=0-9%4 scripts/dna/table1_c0_guidance_dmfm.sbatch
GRAD_NORMALIZE=1 POSTERIOR=glass            sbatch --array=0-9%4 scripts/dna/table1_c0_guidance_dmfm.sbatch
bash scripts/dna/collect_table1_dmfm.sh <dmfm jobid> <glass jobid>
```
-> `results/dna/c0_guidance/{dmfm_n1000,glass_matched_n1000}/` and
`results/dna/c0_guidance/table1_dmfm_n1000_summary.json` (dMFM cells under `n1000`, the matched GLASS
arm under `matched_glass_arm`, the per-sample paired difference under `dmfm_vs_glass_matched`).

Which model produced which DNA item (from the stored run arguments):

| item | finite-N / guided side | reference | dMFM loaded? |
|---|---|---|---|
| Table 1 as submitted, Fig 7 | GLASS posterior on base DFM (4 Euler, MC 8) | - | no |
| Table 1 (new) | dMFM L50 flow map (4 steps, MC 8) | - | yes |
| Fig 5, Tables 14-15 | GLASS on base DFM (8 Euler, N=1..256) | GLASS-2048 | no |
| Table 13 | dMFM diagonal students, one-step flow map | exact reward | yes |
| Table 17 | dMFM flow map (1 step) | GLASS 100 Euler | yes |
| Table 18 | 4-step dMFM (end 0.999), N=1..8 | GLASS-128, RK4 50 steps | yes |

## 5. Posterior samples, value, value gradient

```python
t = 0.5
x_cond = api.forward_noise(api.load_split_seqs(50, "test", max_n=4).to(device), t, cfg)  # or a trajectory state
t_cond = torch.full((4,), t, device=device)
eps = torch.randn(4, 50, 4, device=device)

x1_dmfm  = api.dmfm_posterior(dmfm, eps, x_cond, t_cond, sampler="flow_map", n_steps=1)  # one-step map
x1_glass = api.glass_posterior(base, eps, x_cond, t_cond, n_steps=100)                 # GLASS reference
# dMFM samplers: "flow_map" (compose n_steps maps on linspace(0, end_time)), "one_step",
#                "diagonal_rk4" (end_time<1), "diagonal_euler".
# GLASS: end_time=1.0/solver="euler" (Tables 1, 14-15, 17, Fig 5); Table 18 used end_time=0.999, solver="rk4".

eps_pool = torch.randn(4, 16, 50, 4, device=device)   # [B, MC, L, 4]
V, dV = api.value_and_grad(api.dmfm_posterior_fn(dmfm), logr, x_cond, t, eps_pool, mc_chunk=16)
V_ref, dV_ref = api.value_and_grad(api.glass_posterior_fn(base, n_steps=128), logr, x_cond, t, eps_pool)
V_only, _ = api.value_and_grad(post, logr, x_cond, t, eps_pool, need_grad=False)
```
`V = log mean_j exp(logr(posterior(eps_j)))` (so `logr` must already include the reward
scale); the gradient is exact autodiff through the sampler (two-pass chunked estimator,
numerically identical to `estimate_v_and_grad` / `estimate_value_gradient` in the
experiment modules; checked in `tests/dna/test_api.py`).

Direct entry points in the experiment modules (what `dmfm.api.value_and_grad` reproduces):
`dmfm.experiments.ablate_glass_gradient_mc.estimate_value_gradient(base, x, t_eval=, eps_pool=, ..., nfe_value=,
glass_end_time=, glass_solver=)` (GLASS side) and its dMFM-side twin
`dmfm.experiments.ablate_dmfm_one_step_gradient_mc.estimate_value_gradient(student, x, t_eval=, eps_pool=, ...,
dmfm_sampler="flow_map"|"one_step"|"diagonal_rk4", dmfm_steps=, dmfm_end_time=)`, which `dmfm.scalefree`
reuses. Both take the reward keywords below and return `(V [B], grad [B, L, 4])`.

Other rewards (GC content, motif, conjunction; Fig 5 / Tables 13-15, 18):
`dmfm.experiments.ablate_glass_gradient_mc.target_reward(endpoint, score_center=..., score_std=...,
z_target=..., beta=..., reward="gc"|"motif"|"conjunction", motif=motif_tensor("TATAAA", dev),
motif2=..., motif_tau=..., conjunction_tau=..., objective="target"|"maximize")`
-> wrap as `logr = lambda e: reward_scale * target_reward(e, ...)`.

## 6. Oracle scoring

```python
y = api.oracle_score(oracle, seqs)   # list[str] | LongTensor[N,50] | [N,50,4]; RC-averaged by default
mae = np.abs(y - target).mean()      # Table 1 metric: oracle MAE to target (unguided vs guided)
```
Same numerics as `dmfm.experiments.score_c0_oracle --reverse_complement_average`
(which also writes bootstrap CIs; use it on a `sample_scores.csv`). Table 11 (guide/oracle
test MAE/Pearson on the 58 held-out parents) = `api.oracle_score(model, api.load_c0_labels("test")[0])`
vs the labels.

## 7. Known issues in the experiment modules (for dependents)

- Probe states differ between the gradient experiments: `ablate_glass_gradient_mc` (Fig 5, Tables
  14-15) uses *model samples* (`seeded_randn` + `flow_to_time` to `t_eval`), while
  `ablate_dmfm_one_step_gradient_mc` (Table 18) forward-noises held-out `test` windows.
- `rescore_gradient_pairs_scale_free` builds its parser inside `main()` (no `parse_args`); use
  `main([...])`.
- Small reruns of Table 18 (2 states) should be compared per condition, not by the aggregate: e.g.
  gc L=50 states 0-1 have MC=8 MAE ~0.017 vs the 32-state 0.1049.

- **Comparing posterior samplers needs `grad_normalize=True`.** `api.guided_sample`'s `grad_clip`
  only clips *down*, and Table 1's `guidance_frac 8 / coeff_cap 10 / grad_clip 10` were tuned to the
  GLASS-4 gradient norm (~1-3, clip binds on 1-5% of guided steps). A dMFM flow-map posterior's
  gradient there has norm 5-125 and binds on 6-52% of steps, so swapping the posterior under the
  stock rule multiplies the effective drive 5-12x. `grad_normalize=True` rescales the gradient to
  exactly `grad_clip`, which makes the step scale-free; apply it to **both** arms or neither
  (`GRAD_NORMALIZE=1` in `scripts/dna/table1_c0_guidance_dmfm.sbatch`). Default stays `False`, so
  stored results are unaffected. See `docs/dmfm_glass_parity.md`.
- **Pair the student with its jump.** `checkpoints/dna/dmfm/L50` is ESD-distilled on random gaps and
  is a **one**-step map (Tables 13/17, Fig 4); `checkpoints/dna/dmfm_4step/L50` is distilled on gaps
  [0, 0.25] and selected on `val_mfm_probe_esd_gap_0.25`, for **4** steps (Table 18). Use
  `paths.dmfm_ckpt_for_steps(50, n_steps)`; `sample_c0_guidance_dmfm` enforces it unless
  `--allow_gap_mismatch`. Composing the diagonal student 4 times is the worst of nine samplers
  measured (cosine 0.171 vs GLASS-4's 0.430) and was the defect in withdrawn job 49277977.
- `dmfm.utils.flow_utils.gaussian_denoiser_flow_step` now evaluates the network **once** per step
  (the original called it twice; outputs bit-identical; `legacy_double_eval=True` restores the old
  path). Count one NFE per base-flow step; the paper's NFE counts already do.

- Stored `results/dna/scale_free_mc/*/run_metadata.json` (Fig 5 / Tables 14-15) predate
  `--glass_end_time`: use `.get("glass_end_time", 1.0)`; those runs integrated GLASS to t=1.0
  (a GPU rerun of their first probes matches the stored gradients better with 1.0 than 0.999).
- Table 18's numbers are the MAE files of `dmfm.experiments.rescore_gradient_pairs_mae` (ported
  2026-09-29, reproduces the stored CSVs exactly); `scripts/dna/table18_dmfm4_gradient_mc.sbatch`
  now runs it at the end of each task.

- `experiments/eval_posterior_diversity.py` and `experiments/sample_reward_attainment.py` loaded
  data from `data/yeast_parent_disjoint/...` (the old repo layout) instead of
  `data/dna/yeast_parent_disjoint/...`; fixed 2026-09-29 (now via `dmfm.paths`).
- `eval_posterior_diversity.py` wrote `run_config.csv` with `DEFAULT_STUDENTS[L][0]` (first
  character of a path); fixed 2026-09-29.
- `experiments/benchmark_nfe_steering.py` (owned by dna-fig4): if `--student_args` is set to an
  empty value it falls back to the *teacher's* `args.json` (`cli.student_args or teacher_args`),
  which builds the wrong student; the default (`paths.dmfm_args(50)`) is fine. Its C0 guide is
  the parent-disjoint park_cnn guide, not the purged split65k Keras guide.

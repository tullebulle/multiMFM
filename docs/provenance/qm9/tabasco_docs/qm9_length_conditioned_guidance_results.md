# QM9 length-conditioned alpha guidance results

Documented on 2026-06-02. Main runs were completed on 2026-06-01.

## Summary

We evaluated GLASS value-gradient guidance for QM9 alpha targeting with the
TABASCO QM9 DFM checkpoint. For each generated sample, we first sample a heavy
atom length from the model's learned length prior, then sample an alpha target
from the empirical QM9 alpha distribution conditioned on that heavy atom length.

The best scaled `n=100` result is the aggressive guided setting:

- Guide-regressor MAE: `8.720 -> 2.213`
- Guide-regressor median AE: `7.707 -> 1.508`
- Oracle-regressor MAE: `8.439 -> 2.647`
- Oracle-regressor median AE: `7.720 -> 2.139`
- PB valid: `0.96 -> 0.87`

The more conservative setting gives a better PB tradeoff:

- Guide-regressor MAE: `8.720 -> 2.886`
- Oracle-regressor MAE: `8.439 -> 3.558`
- PB valid: `0.96 -> 0.95`

Later `n=1000` PB-valid runs are the main large-sample reference for "strong
GLASS guidance":

| property | output | mu | reward scale | guide MAE | oracle MAE | PB valid |
|---|---|---:|---:|---:|---:|---:|
| alpha | `n1000_mu045_s128` | `0.45` | `0.3` | `9.30 -> 1.33` | `9.10 -> 1.81` | `0.987 -> 0.936` |
| cv | `propc_cv_mu070_s128_n1000` | `0.70` | `1.209` | `3.48 -> 0.521` | `3.49 -> 0.670` | `0.987 -> 0.944` |
| mu | `propc_mu_mu035_s128_n1000` | `0.35` | `8.638` | `1.67 -> 0.648` | `1.70 -> 0.827` | `0.987 -> 0.883` |
| gap | `propc_gap_mu015_s128_n1000` | `0.15` | `8925` | `0.0726 -> 0.0223` | `0.0704 -> 0.0272` | `0.987 -> 0.873` |
| homo | `propc_homo_mu035_s128_n1000` | `0.35` | `41500` | `0.0283 -> 0.0077` | `0.0272 -> 0.0105` | `0.987 -> 0.866` |
| lumo | `propc_lumo_mu025_s128_n1000` | `0.25` | `9088` | `0.0557 -> 0.0144` | `0.0557 -> 0.0183` | `0.987 -> 0.910` |

Energy-unit note: `gap`, `homo`, and `lumo` are raw Hartree values from the
TFG QM9 dataset/regressors, not eV. Convert those rows to meV with
`1 Ha = 27211.386 meV`; for example, the GLASS guided `gap` oracle MAE
`0.0272 Ha` is about `740 meV`.

These are length-conditioned targets, not the fixed alpha-target-90 stress test.
For MFM guidance work, this table is the production GLASS scorecard to
reproduce; smaller fixed-target runs are diagnostics.

## Current MFM comparison

On 2026-06-11, the length-conditioned script was extended to support
`--value-sampler mfm`. A direct GPU `n=100` alpha comparison with PB enabled
shows that the current diagonal MFM moves the property substantially but does
not match GLASS.

Settings: seed `8`, target seed `2026`, sample steps `128`, guide-every `1`,
soft atom guidance, target histogram `alpha_train_flow_bins100`.

| value sampler | value steps | mu | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|---:|---:|
| GLASS | 4 | 0.45 | `8.59 -> 1.38` | `8.30 -> 1.80` | `0.96 -> 0.93` |
| MFM diagonal `sens2000` | 8 | 0.45 | `8.59 -> 2.15` | `8.30 -> 3.02` | `0.96 -> 0.85` |
| MFM diagonal `sens2000` | 8 | 0.35 | `8.59 -> 2.56` | `8.30 -> 3.39` | `0.96 -> 0.88` |
| MFM diagonal `sens2000` | 16 | 0.45 | `8.59 -> 2.87` | `8.30 -> 4.09` | `0.96 -> 0.82` |

Conclusion: current MFM is useful but not GLASS-level on this scorecard.

The current MFM `n=1000` alpha run confirms the gap:

| value sampler | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|
| GLASS, `n1000_mu045_s128` | `9.30 -> 1.33` | `9.10 -> 1.81` | `0.987 -> 0.936` |
| MFM diagonal `sens2000`, `mfm_len_alpha_n1000_s128_mu045_diag8_sens2000_pb` | `9.29 -> 2.49` | `9.10 -> 3.34` | `0.987 -> 0.824` |

A forced diagonal-rollout training branch improved the `n=100` MFM score but
did not close the gap:

| MFM checkpoint | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|
| `sens2000` | `8.59 -> 2.15` | `8.30 -> 3.02` | `0.96 -> 0.85` |
| forced rollout step 200 | `8.59 -> 1.99` | `8.30 -> 2.71` | `0.96 -> 0.88` |
| forced rollout step 600 | `8.59 -> 1.99` | `8.30 -> 2.77` | `0.96 -> 0.87` |

Longer forced-rollout training is queued as Slurm job `21601396`.

MC-sample sweep for forced rollout step 200:

| value samples | guide MAE | oracle MAE | PB valid |
|---:|---:|---:|---:|
| 4 | `8.59 -> 1.99` | `8.30 -> 2.71` | `0.96 -> 0.88` |
| 8 | `8.59 -> 2.02` | `8.30 -> 2.95` | `0.96 -> 0.80` |
| 16 | `8.59 -> 1.98` | `8.30 -> 2.62` | `0.96 -> 0.92` |

Use `value_samples=16` for serious MFM evaluation. `4` is only a smoke-test
setting for the MFM guidance map.

Long forced-rollout status as of 2026-06-11:

| MFM checkpoint | value samples | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|---:|
| short forced rollout step 200 | 16 | `8.59 -> 1.98` | `8.30 -> 2.62` | `0.96 -> 0.92` |
| long spread-2 step 1000 | 16 | `8.59 -> 2.11` | `8.30 -> 2.64` | `0.96 -> 0.87` |

The long spread-2 checkpoint did not improve the actual scorecard, so we should
select forced-rollout checkpoints by this guidance/PB gate rather than by
training loss. Pending scorecards: spread-2 steps `4000/6000/7000` and spread-4
steps `3000/5000/6000`, all at `value_samples=16`.

Updated forced-rollout checkpoint selection:

| MFM checkpoint | value samples | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|---:|
| spread-2 step 2000 | 16 | `8.59 -> 1.58` | `8.30 -> 2.16` | `0.96 -> 0.88` |
| spread-2 step 2000 | 32 | `8.59 -> 1.91` | `8.30 -> 2.64` | `0.96 -> 0.93` |
| spread-2 step 4000 | 16 | `8.59 -> 1.87` | `8.30 -> 2.41` | `0.96 -> 0.88` |
| spread-4 step 3000 | 16 | `8.59 -> 1.82` | `8.30 -> 2.44` | `0.96 -> 0.91` |
| spread-4 step 3000 | 32 | `8.59 -> 1.80` | `8.30 -> 2.51` | `0.96 -> 0.93` |

Simple guidance clipping on spread-2 step 2000 was not successful: clip
`coord=15, atom=8` gave guide/oracle/PB `2.32 / 2.99 / 0.91`, and stronger
clip `coord=10, atom=5` was worse. The current balanced MFM leader is
spread-4 step 3000 with `value_samples=32`.

## Important implementation detail

The important fix was making TFG-Flow property guidance differentiable through
atom types. Hard `argmax(atomics)` is still used for final property evaluation,
but during guidance the TFG regressor receives soft atom embeddings:

```text
probs = softmax(atomics / soft_atom_temperature)
h = probs @ regressor.atom_emb.weight
```

Without this, `--guide-atomics` has no useful atom-type gradient because the
reward path is cut by `argmax`.

Relevant files:

- `src/glass_guided_sampling_experiment.py`
- `src/qm9_length_conditioned_guidance_experiment.py`
- `slurm/qm9_length_conditioned_guidance.sbatch`

## Model and data setup

Checkpoint:

```text
outputs/qm9_dfm_long/qm9_dfm_trainflow_long100k_20260531/checkpoints/model_step_100000.pt
```

Property regressors:

```text
TFG-Flow/storage/guide_clf_ckpt.zip
TFG-Flow/storage/oracle_clf_ckpt.zip
```

Length-conditioned target distribution:

```text
property_name = alpha
partition = train_flow
hist_bins = 100
hist_cache = cache/qm9_property_histograms/alpha_train_flow_bins100.json
target_seed = 2026
```

The sampled target batch for the `n=100` runs had:

```text
target_mean = 75.3240
target_std = 7.7923
```

## Shared guidance hyperparameters

Unless noted otherwise:

```text
seed = 8
num_samples = 100 for scaled runs, 10 for tuning runs
sample_steps = 64
value_samples = 4
value_glass_steps = 4
reward = target_property
reward_function = -(alpha_prediction - alpha_target)^2
reward_scale = 0.3
guide_atomics = true
soft_atom_temperature = 0.25
guide_every = 2
guide_min_t = 0.05
guide_max_t = 0.95
posebusters_config = src/tabasco/utils/posebusters_no_strain.yaml
sanitize = true
```

Unguided and guided samples use the same initial noise and target values for a
given seed. The guided sampler uses the production TABASCO sampling step plus
the additive guidance drift:

```text
u_star = u + mu * grad_x V_t(x)
```

## Scaled n=100 results

These are the GPU Slurm runs on `seas_gpu`, node `holygpu8a18102`, with CUDA
available. Both ran through `slurm/qm9_length_conditioned_guidance.sbatch`.

| n | mu | run | guide MAE | guide med AE | oracle MAE | oracle med AE | PB valid | conversion | unique total |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.3 | unguided | 8.720 | 7.707 | 8.439 | 7.720 | 0.96 | 0.98 | 0.92 |
| 100 | 0.3 | guided | 2.886 | 2.186 | 3.558 | 3.007 | 0.95 | 0.99 | 0.97 |
| 100 | 0.5 | unguided | 8.720 | 7.707 | 8.439 | 7.720 | 0.96 | 0.98 | 0.92 |
| 100 | 0.5 | guided | 2.213 | 1.508 | 2.647 | 2.139 | 0.87 | 0.99 | 0.99 |

PB component rates for the guided `mu=0.3` run:

```text
pb_mol_pred_loaded = 0.99
pb_sanitization = 0.99
pb_inchi_convertible = 0.99
pb_all_atoms_connected = 0.99
pb_bond_lengths = 0.96
pb_bond_angles = 0.99
pb_internal_steric_clash = 0.98
pb_aromatic_ring_flatness = 0.99
pb_double_bond_flatness = 0.99
```

PB component rates for the guided `mu=0.5` run:

```text
pb_mol_pred_loaded = 0.99
pb_sanitization = 0.99
pb_inchi_convertible = 0.99
pb_all_atoms_connected = 0.99
pb_bond_lengths = 0.91
pb_bond_angles = 0.93
pb_internal_steric_clash = 0.98
pb_aromatic_ring_flatness = 0.99
pb_double_bond_flatness = 0.99
```

Output directories:

```text
outputs/qm9_length_conditioned_guidance/alpha_n100_seed8_mu030_lam03_ge2_t095_atomsoft025_pb
outputs/qm9_length_conditioned_guidance/alpha_n100_seed8_mu050_lam03_ge2_t095_atomsoft025_pb
```

Slurm jobs:

```text
17934176  mu=0.3
17934177  mu=0.5
```

## n=10 tuning results

The `n=10` runs were useful for selecting the guidance window and strength.
The same initial samples and targets were used for each row in this table.

| n | mu | guide_every | guide_max_t | run | guide MAE | guide med AE | oracle MAE | oracle med AE | PB valid |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 10 | 0.3 | 2 | 0.95 | unguided | 4.488 | 2.568 | 4.294 | 3.402 | 1.00 |
| 10 | 0.3 | 2 | 0.95 | guided | 1.530 | 0.859 | 1.778 | 1.206 | 0.90 |
| 10 | 0.5 | 2 | 0.95 | unguided | 4.488 | 2.568 | 4.294 | 3.402 | 1.00 |
| 10 | 0.5 | 2 | 0.95 | guided | 1.388 | 0.918 | 2.196 | 1.752 | 0.90 |

The aggressive `mu=0.5` setting gives the best guide-regressor targeting, while
`mu=0.3` is better on the independent oracle regressor in the `n=10` tuning run.
At `n=100`, `mu=0.5` is best on both guide and oracle MAE, but with a larger PB
validity drop.

## Reproduction commands

Conservative `n=100`:

```bash
sbatch --parsable \
  --export=ALL,RUN_NAME=alpha_n100_seed8_mu030_lam03_ge2_t095_atomsoft025_pb,NUM_SAMPLES=100,SEED=8,TARGET_SEED=2026,MU=0.3,REWARD_SCALE=0.3,GUIDE_EVERY=2,GUIDE_MAX_T=0.95,SOFT_ATOM_TEMPERATURE=0.25,COMPUTE_PB_FLAG=--compute-pb,GUIDE_ATOMICS_FLAG=--guide-atomics \
  slurm/qm9_length_conditioned_guidance.sbatch
```

Aggressive `n=100`:

```bash
sbatch --parsable \
  --export=ALL,RUN_NAME=alpha_n100_seed8_mu050_lam03_ge2_t095_atomsoft025_pb,NUM_SAMPLES=100,SEED=8,TARGET_SEED=2026,MU=0.5,REWARD_SCALE=0.3,GUIDE_EVERY=2,GUIDE_MAX_T=0.95,SOFT_ATOM_TEMPERATURE=0.25,COMPUTE_PB_FLAG=--compute-pb,GUIDE_ATOMICS_FLAG=--guide-atomics \
  slurm/qm9_length_conditioned_guidance.sbatch
```

## Current interpretation

Soft atom-type guidance is the main reason the property errors improved. The
guidance can now alter the atom-type endpoint distribution, not only the
coordinates.

The `mu=0.5` setting is the strongest property-targeting setting so far. Its PB
validity drop is mostly from bond length and bond angle checks, not steric
clash. The `mu=0.3` setting is the safer default if PB validity matters more
than squeezing out the lowest property MAE.

## 2026-06-12 MFM diagonal-4 update

The historical strong GLASS scorecard used four posterior value steps. Matching
that step count with the diagonal MFM is critical.

Alpha length-conditioned gate, `n=100`, `sample_steps=128`, `guide_every=1`,
`mu=0.45`, PB enabled:

| value sampler | value samples | value steps | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|---:|---:|
| GLASS | 4 | 4 | `1.381` | `1.803` | `0.93` |
| MFM spread-4 step3000 | 32 | 4 | `1.224` | `1.738` | `0.99` |
| MFM spread-2 step2000 | 32 | 4 | `1.418` | `1.787` | `0.97` |
| MFM spr8-anchor step4000 | 32 | 4 | `1.560` | `2.047` | `0.96` |

Current best MFM row:

```text
outputs/qm9_length_conditioned_guidance/mfm_len_alpha_n100_s128_mu045_vs32_diag4_forcedspr4_step3000_pb_slurm
```

This row is the first MFM value sampler result that reaches the GLASS-level
target/PB tradeoff on this scorecard. The next required gate is the same setup
at `n=1000`, followed by all-property validation if the result holds.

Large alpha validation:

| value sampler | n | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|---:|
| GLASS reference | 1000 | `1.33` | `1.81` | `0.936` |
| MFM spread-4 step3000 diagonal-4 | 1000 | `1.450` | `2.042` | `0.946` |
| MFM train4 step5000 diagonal-4 | 1000 | `1.514` | `2.021` | `0.952` |
| MFM coord-res step2000 diagonal-4 | 1000 | `1.510` | `2.060` | `0.954` |

Current production checkpoint for alpha remains:

```text
outputs/mfm_diag_rollout8_forced_20260611_long_spr4/stage4_20260611-171509/student_step_3000.pt
```

It does not fully beat GLASS at `n=1000`, but it is the strongest MFM diagonal-4
checkpoint so far on the large guide-MAE gate.

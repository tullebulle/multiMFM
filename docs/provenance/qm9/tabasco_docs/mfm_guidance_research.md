# MFM Guidance Research Log

Goal: make the TABASCO MFM mimic GLASS posterior guidance with an 8-step
integrator, so strong GLASS-style property guidance can be reproduced through
MFM rather than by running GLASS rollouts.

## Current Best Checkpoints

- Off-diagonal/posterior-fidelity baseline:
  `outputs/mfm_student_rollout_stage4_residual_gated_big_grid_mid2_spr075_s4_b16_lr1e6/stage4_20260609-090016/student_step_500.pt`
- Best 8-step diagonal posterior-fidelity checkpoint so far:
  `outputs/mfm_diagroll8_from_current_20260609-182834/stage4_20260609-182848/student_step_2000.pt`

## Leaderboard

All rows use matched posterior noise where applicable.

| checkpoint | sampler | posterior summary | alpha grad cosine | alpha guidance |
|---|---:|---|---|---|
| offdiag baseline | jump-4 | t=0.3/0.6/0.9 spread `0.981/0.958/1.015`, meanRMSD `0.047/0.060/0.036`, pairRMSD `0.236/0.162/0.045`, atomTV `0.0168/0.0184/0.0043` | jump-4 coord `0.129`, atom `0.120` | atom-guided `mu=0.1`: oracle MAE `12.07 -> 10.01` |
| offdiag baseline | diag-8 | t=0.3/0.6/0.9 spread `0.91/0.84/0.91`, meanRMSD `0.123/0.071/0.039`, pairRMSD `0.335/0.180/0.045`, atomTV `0.032/0.029/0.000` | coord `0.169`, atom `0.092` | not best |
| offdiag baseline | diag-16 | t=0.3/0.6/0.9 spread `0.95/0.91/1.13`, meanRMSD `0.105/0.071/0.038`, pairRMSD `0.294/0.158/0.044`, atomTV `0.030/0.026/0.000` | coord `0.277`, atom `0.209` | `mu=0.1`: oracle MAE `12.07 -> 11.13` |
| diagroll8 step2000 | diag-8 | t=0.3/0.6/0.9 spread `0.97/0.91/1.04`, meanRMSD `0.060/0.062/0.037`, pairRMSD `0.224/0.141/0.043`, atomTV `0.020/0.019/0.001` | coord `0.179`, atom `0.131` | `mu=0.1`: oracle MAE `12.07 -> 11.52` |
| diagroll8 step2000 | diag-16 | not fully re-evaluated | coord `0.251`, atom `0.191` | `mu=0.1`: oracle MAE `12.07 -> 11.73` |

## Working Diagnosis

1. More diagonal integration helps gradient alignment, so the problem is still
   productively framed as an MFM integrator problem.
2. Diagonal rollout training improved endpoint/posterior fidelity, especially at
   low and mid conditioning times.
3. The alpha guidance stress test improved only modestly, which means endpoint
   fidelity is not sufficient. The posterior map needs better condition
   sensitivity/Jacobian fidelity.
4. The next objective must remain property-agnostic. We should train the MFM
   posterior map to match GLASS condition sensitivity for random conditioning
   perturbations, not train a property-specific value function.

## Next Experiment

Train an 8-step diagonal MFM with a condition-JVP finite-difference objective:

1. sample a real endpoint and build `x_cond` at `t_cond`;
2. sample common posterior prior noise;
3. perturb `x_cond` along random masked coordinate/atom directions;
4. run GLASS-64 from the same prior noise for base and perturbed conditions;
5. run diagonal MFM-8 from the same prior noise for base and perturbed conditions;
6. match:
   - base endpoint to GLASS endpoint;
   - endpoint delta `(post(x_cond + eps d) - post(x_cond)) / eps` to the GLASS
     endpoint delta.

This targets the derivative needed for arbitrary downstream property guidance
while keeping the model and training MFM-centered.

## 2026-06-10 Diagnostics

Velocity-grid check:
- Baseline and diagonal-rollout step2000 have similar local diagonal velocity
  relative error over the on-distribution `(s, t_cond)` grid.
- Typical coordinate rel-error remains about `0.05-0.17` except near `s=0.95`,
  where it rises to about `0.21-0.29`.
- Therefore the diagonal-rollout gain came mostly from better accumulated
  posterior fidelity, not a dramatic local velocity-grid improvement.

Factor diagnostic on diagonal-rollout step2000, diagonal-8:
`outputs/mfm_guidance_factor_diagnostic/diagroll8_step2000_alpha_diag8_seed1_b8_vs4_ge8`
- actual alpha value-gradient cosine:
  - coord `0.1850`
  - atom `0.0914`
- same terminal cotangent through MFM Jacobian:
  - coord `0.6436`
  - atom `0.4313`
- same GLASS Jacobian with MFM terminal cotangent:
  - coord `0.3578`
  - atom `0.3022`
- direct terminal cotangent:
  - coord `0.2149`
  - atom `0.3300`

Interpretation:
- The original pure terminal-cotangent failure has improved with better endpoint
  fidelity, but actual alpha guidance is still low.
- Both endpoint precision/cotangent stability and condition-Jacobian fidelity
  need improvement.

Random condition-JVP objective:
- Added `src/mfm_random_condition_jvp_train.py`.
- Added Slurm wrapper `slurm/mfm_random_condition_jvp.sbatch`.
- This trains the 8-step diagonal MFM endpoint map to match GLASS endpoint
  finite differences under random conditioning perturbations.
- 10-step smoke completed.
- 100-step small run:
  `outputs/mfm_random_condition_jvp_small/randjvp_20260610-093746/student_step_100.pt`
  - training was stable with robust adaptive JVP loss;
  - alpha diag-8 gradient moved only slightly:
    - coord `0.1787 -> 0.1819`
    - atom `0.1311 -> 0.1273`
  - posterior fidelity remained acceptable.

Decision:
- Random condition-JVP is safe enough to scale, but not proven strong.
- Run it as one branch, while also running a precision diagonal-rollout
  continuation to reduce endpoint/cotangent error.

Jobs launched:
- `20890794`: precision diagonal-rollout continuation
  - init:
    `outputs/mfm_diagroll8_from_current_20260609-182834/stage4_20260609-182848/student_step_2000.pt`
  - objective: Stage-4 diagonal rollout, 8 steps, GLASS-64 target
  - key changes vs previous diagroll8:
    - LR `3e-7`
    - `diag_weight=1.0`
    - `coord_loss_weight=6.0`
    - `rollout_final_weight=6.0`
    - keep `rollout_velocity_weight=1.0`, `spread_weight=0.25`
  - intent: improve endpoint/cotangent precision without changing sampler mode.
- `20890793`: random condition-JVP branch
  - init:
    `outputs/mfm_diagroll8_from_current_20260609-182834/stage4_20260609-182848/student_step_2000.pt`
  - objective: random conditioning finite-difference sensitivity, diagonal MFM-8 vs GLASS-64
  - LR `3e-7`, steps `1000`, batch `4`
  - intent: improve property-agnostic condition Jacobian for guidance.

Additional local experiments while Slurm was pending:
- Random terminal-adjoint objective:
  - added `src/mfm_random_adjoint_train.py`;
  - property-agnostic VJP matching objective:
    train `(dF_MFM/dx_cond)^T c` to match `(dF_GLASS/dx_cond)^T c` for random
    terminal cotangents `c`;
  - 5-step smoke succeeded;
  - 50-step small run:
    `outputs/mfm_random_adjoint_small/randadj_20260610-094800/student_step_50.pt`
    - alpha diag-8 gradient:
      - coord `0.1780`
      - atom `0.1248`
    - posterior fidelity stayed similar to the base diagonal-8 checkpoint.
  - conclusion: white-noise terminal adjoints are trainable and safe but do not
    move alpha guidance quickly; likely too generic or too weakly aligned with
    property-regressor terminal cotangents.
- Corrected direct MFM value-gradient finetune:
  - patched `src/mfm_value_gradient_finetune.py` to support
    `--match-posterior-noise` and `--mfm-diagonal`;
  - this fixes the old independent posterior-noise issue and targets the actual
    diagonal-8 integrator;
  - 50-step fixed-trajectory alpha run:
    `outputs/mfm_value_gradient_finetune_matched_diag_small/valuegrad_20260610-095217`
    - step 25 alpha diag-8 gradient:
      - coord `0.1792`
      - atom `0.1317`
    - step 50 alpha diag-8 gradient:
      - coord `0.1765`
      - atom `0.1258`
  - conclusion: corrected direct alpha value-gradient training is executable,
    but the short fixed-trajectory test did not transfer to the held diagnostic.
    Do not scale this blindly without changing the loss/regularization.

Guidance-log comparison:
- Off-diagonal jump-4 still gives better actual alpha guidance than the improved
  diagonal-8 checkpoint, despite lower matched-noise gradient cosine.
- Diagonal-8 has comparable gradient RMS, so the weak guidance is not simply a
  small-gradient issue.
- In the matched diagnostic, diagonal-8 MFM posterior reward/value estimates are
  biased relative to GLASS by a few reward units:
  - GLASS posterior reward mean about `-198.51`
  - MFM posterior reward mean about `-195.58`
  - GLASS value mean about `-32.89`
  - MFM value mean about `-30.06`
- This means the MFM posterior can be close in RMSD while still producing a
  different reward-weighted posterior ensemble. Strong GLASS guidance requires
  matching this reward-weighted ensemble, not just mean RMSD.

Existing off-diagonal checkpoint in 8-step mode:
- The current best off-diagonal checkpoint was trained for 4-step rollout, but
  can be sampled with 8 off-diagonal jumps.
- Posterior fidelity:
  `outputs/mfm_student_eval/baseline_jump8_n16_k64`
  - t=0.3: spread `0.98`, meanRMSD `0.066`, pairRMSD `0.222`,
    atomTV `0.021`, atomMatch `0.970`
  - t=0.6: spread `0.96`, meanRMSD `0.061`, pairRMSD `0.140`,
    atomTV `0.019`, atomMatch `0.976`
  - t=0.9: spread `1.09`, meanRMSD `0.037`, pairRMSD `0.043`,
    atomTV `0.000`, atomMatch `1.000`
- Alpha matched-noise gradient:
  `outputs/mfm_guidance_gradient_diagnostic/baseline_alpha_jump8_seed1_b16_vs4`
  - coord `0.1998`
  - atom `0.1186`
- Actual alpha guidance:
  - jump-8, `mu=0.05`:
    `outputs/glass_guided_sampling_experiment/mfm_jump8_alpha_target90_seed1_mu005_lam03_steps64_atom`
    - guide MAE `12.04 -> 12.05`
    - oracle MAE `12.07 -> 12.18`
  - jump-8, `mu=0.1`:
    `outputs/glass_guided_sampling_experiment/mfm_jump8_alpha_target90_seed1_mu010_lam03_steps64_atom`
    - guide MAE `12.04 -> 11.47`
    - oracle MAE `12.07 -> 11.59`
  - jump-8, `mu=0.2`:
    `outputs/glass_guided_sampling_experiment/mfm_jump8_alpha_target90_seed1_mu020_lam03_steps64_atom`
    - guide MAE `12.04 -> 11.58`
    - oracle MAE `12.07 -> 11.77`
- Interpretation:
  - off-diagonal jump-8 already has better posterior fidelity than diagonal-8;
  - but actual alpha guidance is still much weaker than the old off-diagonal
    jump-4 value-sampler run (`oracle MAE -> 10.01`);
  - `mu=0.05` is too weak/wrong and `mu=0.2` is worse than `mu=0.1`, so the
    jump-8 problem is not solved by scalar guidance-strength tuning;
  - current checkpoint was trained for 4-step off-diagonal rollout, so the next
    high-value MFM branch is direct 8-step off-diagonal rollout training.

Job launched:
- `20898158`: off-diagonal jump-8 rollout training
  - init:
    `outputs/mfm_student_rollout_stage4_residual_gated_big_grid_mid2_spr075_s4_b16_lr1e6/stage4_20260609-090016/student_step_500.pt`
  - objective: Stage-4 GLASS rollout distillation, off-diagonal MFM jumps
  - `rollout_steps=8`, GLASS teacher steps `64`
  - LR `5e-7`, batch `8`, `diag_weight=0.25`
  - `rollout_final_weight=4.0`, `rollout_velocity_weight=1.0`,
    `rollout_spread_weight=0.25`, `rollout_spread_samples=4`
  - intent: optimize the 8-step off-diagonal integrator directly, since jump-8
    already looks like the best 8-step posterior-fidelity mode.

## 2026-06-10 continued: off-diagonal jump-8 and generated-condition branch

Completed jobs:
- `20890793`: random condition-JVP diagonal branch.
  - Step 250 held-out alpha diag-8 gradient:
    `outputs/mfm_guidance_gradient_diagnostic/randjvp250_alpha_diag8_seed1_b16_vs4`
    - coord `0.1725`
    - atom `0.1252`
  - Step 250 posterior fidelity stayed acceptable:
    `outputs/mfm_student_eval/randjvp250_diag8_n16_k64`
    - t=0.3: spread `0.97`, meanRMSD `0.061`, pairRMSD `0.217`,
      atomTV `0.021`, atomMatch `0.970`
    - t=0.6: spread `0.92`, meanRMSD `0.061`, pairRMSD `0.146`,
      atomTV `0.022`, atomMatch `0.974`
    - t=0.9: spread `1.03`, meanRMSD `0.037`, pairRMSD `0.042`,
      atomTV `0.000`, atomMatch `1.000`
  - Decision: random condition-JVP did not improve the alpha guidance
    derivative; deprioritize this objective.
- `20890794`: diagonal precision continuation.
  - Step 1000 alpha diag-8 gradient:
    `outputs/mfm_guidance_gradient_diagnostic/diagprec_step1000_alpha_diag8_seed1_b16_vs4`
    - coord `0.1908`
    - atom `0.1208`
  - Decision: endpoint precision alone did not fix alpha guidance.
- `20898158`: off-diagonal jump-8 rollout training.
  - Output:
    `outputs/mfm_jump8_rollout_20260610-100051/stage4_20260610-100640`
  - Alpha matched-noise gradient:
    - baseline jump-8: coord `0.1998`, atom `0.1186`
    - step 500:
      `outputs/mfm_guidance_gradient_diagnostic/jump8roll_step500_alpha_jump8_seed1_b16_vs4`
      - coord `0.2111`, atom `0.1285`
    - step 1000:
      `outputs/mfm_guidance_gradient_diagnostic/jump8roll_step1000_alpha_jump8_seed1_b16_vs4`
      - coord `0.2142`, atom `0.1455`
    - step 1500:
      `outputs/mfm_guidance_gradient_diagnostic/jump8roll_step1500_alpha_jump8_seed1_b16_vs4`
      - coord `0.1963`, atom `0.1431`
    - step 2000:
      `outputs/mfm_guidance_gradient_diagnostic/jump8roll_step2000_alpha_jump8_seed1_b16_vs4`
      - coord `0.2027`, atom `0.1503`
  - Posterior fidelity improved monotonically in pair RMSD on real-endpoint
    conditions:
    - step 1000:
      `outputs/mfm_student_eval/jump8roll_step1000_jump8_n16_k64`
      - t=0.3: spread `0.99`, meanRMSD `0.059`, pairRMSD `0.211`,
        atomTV `0.020`, atomMatch `0.972`
      - t=0.6: spread `0.97`, meanRMSD `0.060`, pairRMSD `0.139`,
        atomTV `0.019`, atomMatch `0.976`
      - t=0.9: spread `1.10`, meanRMSD `0.035`, pairRMSD `0.042`,
        atomTV `0.000`, atomMatch `1.000`
    - step 2000:
      `outputs/mfm_student_eval/jump8roll_step2000_jump8_n16_k64`
      - t=0.3: spread `0.99`, meanRMSD `0.055`, pairRMSD `0.205`,
        atomTV `0.018`, atomMatch `0.973`
      - t=0.6: spread `0.97`, meanRMSD `0.059`, pairRMSD `0.137`,
        atomTV `0.019`, atomMatch `0.976`
      - t=0.9: spread `1.11`, meanRMSD `0.034`, pairRMSD `0.041`,
        atomTV `0.000`, atomMatch `1.000`

Actual alpha guidance with off-diagonal jump-8:
- Strong GLASS reference:
  `outputs/glass_guided_sampling_experiment/qm9_alpha_target90_seed1_mu300_lam03_steps64_prod`
  - guide MAE `12.04 -> 8.57`
  - oracle MAE `12.07 -> 8.69`
  - property mean `78.90 -> 82.42`
- Old best off-diagonal jump-4 MFM:
  `outputs/glass_guided_sampling_experiment/mfm_alpha_target90_seed1_mu010_lam03_steps64_prod_atom_pb`
  - guide MAE `12.04 -> 10.00`
  - oracle MAE `12.07 -> 10.01`
  - property mean `78.90 -> 80.97`
- Baseline jump-8, `mu=0.1`:
  - unclipped:
    `outputs/glass_guided_sampling_experiment/mfm_jump8_alpha_target90_seed1_mu010_lam03_steps64_atom`
    - guide MAE `12.04 -> 11.47`
    - oracle MAE `12.07 -> 11.59`
  - clipped to per-sample coord/atom guidance RMS `80/30`:
    `outputs/glass_guided_sampling_experiment/mfm_jump8_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
    - guide MAE `12.04 -> 11.32`
    - oracle MAE `12.07 -> 11.55`
- Jump-8 rollout step 1000:
  - unclipped `mu=0.1`:
    `outputs/glass_guided_sampling_experiment/mfm_jump8roll_step1000_alpha_target90_seed1_mu010_lam03_steps64_atom`
    - guide MAE `12.04 -> 11.49`
    - oracle MAE `12.07 -> 11.59`
  - clipped `mu=0.1`:
    `outputs/glass_guided_sampling_experiment/mfm_jump8roll_step1000_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
    - guide MAE `12.04 -> 11.28`
    - oracle MAE `12.07 -> 11.47`
    - property mean `78.90 -> 79.66`
  - clipped `mu=0.2`:
    `outputs/glass_guided_sampling_experiment/mfm_jump8roll_step1000_alpha_target90_seed1_mu020_lam03_steps64_atom_clip80_30`
    - guide MAE `12.04 -> 11.32`
    - oracle MAE `12.07 -> 11.37`
  - clipped `mu=0.3`:
    `outputs/glass_guided_sampling_experiment/mfm_jump8roll_step1000_alpha_target90_seed1_mu030_lam03_steps64_atom_clip80_30`
    - guide MAE `12.04 -> 11.63`
    - oracle MAE `12.07 -> 11.70`
- Jump-8 rollout step 2000, clipped `mu=0.2`:
  `outputs/glass_guided_sampling_experiment/mfm_jump8roll_step2000_alpha_target90_seed1_mu020_lam03_steps64_atom_clip80_30`
  - guide MAE `12.04 -> 11.92`
  - oracle MAE `12.07 -> 12.05`

Guidance clipping diagnostic:
- Added optional `--guidance-max-coord-rms` and `--guidance-max-atom-rms`
  to `src/glass_guided_sampling_experiment.py` and the Slurm wrapper.
- Baseline jump-8 clipped run reduced severe per-step spikes:
  - max coordinate guidance RMS before clipping: `259.4`, after clipping:
    `26.3`
  - max atomics guidance RMS before clipping: `283.7`, after clipping: `8.23`
- Clipping helped slightly but did not recover strong guidance, so the failure is
  not just magnitude instability.

Interpretation after off-diagonal rollout:
- Direct 8-step off-diagonal rollout improves real-endpoint posterior fidelity
  and slightly improves alpha gradient alignment.
- Actual alpha guidance remains much weaker than jump-4 and GLASS.
- Increasing `mu` under clipping does not scale toward GLASS; `mu=0.3` is
  actively worse.
- Therefore the main remaining suspect is conditioning-distribution mismatch:
  training/evaluation conditions are noised real endpoints, but guidance
  differentiates through posterior samples conditioned on generated production
  trajectory states.

Code changes for next branch:
- Generalized `glass_rollout_distill_loss(..., cond_state=...)` so Stage 4 can
  train on a prebuilt conditioning state instead of always noising a real
  endpoint.
- Added `--rollout-condition-source {endpoint,generated}` and
  `--generated-condition-steps` to `src/mfm_student_train.py`.
- Added `ROLLOUT_CONDITION_SOURCE` and `GENERATED_CONDITION_STEPS` to
  `slurm/mfm_student_stage4_rollout.sbatch`.
- Generated condition source samples an unguided teacher production trajectory
  to a scalar batch `t_cond`, then trains the same property-agnostic GLASS
  rollout distillation objective from that condition.

Generated-condition job:
- Bad launch `20910851` was canceled because Slurm comma parsing truncated
  `T_COND_GRID` to `0.1`.
- Corrected launch `20911327`:
  - init:
    `outputs/mfm_jump8_rollout_20260610-100051/stage4_20260610-100640/student_step_1000.pt`
  - `rollout_condition_source=generated`
  - `generated_condition_steps=64`
  - `rollout_steps=8`, off-diagonal jump mode
  - LR `3e-7`, steps `1000`, batch `4`
  - `t_cond_grid=0.1,0.3,0.5,0.7,0.9`, jitter `0.04`
  - status at note time: pending on resources

Current best checkpoint for actual 8-step guidance:
- `outputs/mfm_jump8_rollout_20260610-100051/stage4_20260610-100640/student_step_1000.pt`
- Best current actual 8-step alpha run:
  `outputs/glass_guided_sampling_experiment/mfm_jump8roll_step1000_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
  - guide MAE `12.04 -> 11.28`
  - oracle MAE `12.07 -> 11.47`
- This is still far from the target and below the old jump-4 MFM run, so the
  goal is not achieved.

### Generated-condition branch results

Job `20911327` completed successfully.
- Output:
  `outputs/mfm_jump8_gencond_from_step1000_20260610/stage4_20260610-103610`
- Init:
  `outputs/mfm_jump8_rollout_20260610-100051/stage4_20260610-100640/student_step_1000.pt`
- The branch trains the same GLASS rollout distillation objective but uses
  unguided generated production-trajectory states as `x_cond`.

Added generated-trajectory posterior-fidelity diagnostic:
- `src/mfm_eval_diagonal.py` now supports
  `--condition-source generated_trajectory`.
- `slurm/mfm_eval_diagonal.sbatch` exposes `CONDITION_SOURCE` and
  `GENERATED_CONDITION_STEPS`.

Generated-condition posterior fidelity:
- Baseline checkpoint on generated-trajectory conditions:
  `outputs/mfm_student_eval/jump8roll_step1000_jump8_gencond_n16_k64`
  - t=0.3: spread `1.00`, meanRMSD `0.041`, pairRMSD `0.198`,
    atomTV `0.019`, atomMatch `0.972`
  - t=0.6: spread `0.97`, meanRMSD `0.036`, pairRMSD `0.089`,
    atomTV `0.026`, atomMatch `0.970`
  - t=0.9: spread `1.07`, meanRMSD `0.027`, pairRMSD `0.033`,
    atomTV `0.000`, atomMatch `1.000`
- Generated-condition step 500:
  `outputs/mfm_student_eval/jump8_gencond_step500_jump8_gencond_n16_k64`
  - t=0.3: spread `1.00`, meanRMSD `0.041`, pairRMSD `0.196`,
    atomTV `0.018`, atomMatch `0.974`
  - t=0.6: spread `0.98`, meanRMSD `0.035`, pairRMSD `0.088`,
    atomTV `0.026`, atomMatch `0.970`
  - t=0.9: spread `1.08`, meanRMSD `0.025`, pairRMSD `0.031`,
    atomTV `0.000`, atomMatch `1.000`
- Generated-condition step 1000:
  `outputs/mfm_student_eval/jump8_gencond_step1000_jump8_gencond_n16_k64`
  - t=0.3: spread `1.00`, meanRMSD `0.040`, pairRMSD `0.193`,
    atomTV `0.018`, atomMatch `0.975`
  - t=0.6: spread `0.96`, meanRMSD `0.028`, pairRMSD `0.077`,
    atomTV `0.022`, atomMatch `0.975`
  - t=0.9: spread `1.07`, meanRMSD `0.024`, pairRMSD `0.030`,
    atomTV `0.000`, atomMatch `1.000`

Generated-condition alpha gradient:
- Baseline jump-8-rollout step 1000:
  - coord `0.2142`
  - atom `0.1455`
- Generated-condition step 500:
  `outputs/mfm_guidance_gradient_diagnostic/jump8_gencond_step500_alpha_jump8_seed1_b16_vs4`
  - coord `0.2181`
  - atom `0.1469`
- Generated-condition step 1000:
  `outputs/mfm_guidance_gradient_diagnostic/jump8_gencond_step1000_alpha_jump8_seed1_b16_vs4`
  - coord `0.2151`
  - atom `0.1534`

Generated-condition actual alpha guidance:
- Step 500, clipped `mu=0.1`:
  `outputs/glass_guided_sampling_experiment/mfm_jump8_gencond_step500_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
  - guide MAE `12.04 -> 11.52`
  - oracle MAE `12.07 -> 11.72`
  - worse than the pre-generated-condition checkpoint despite a small cosine gain.
- Step 1000, clipped `mu=0.1`:
  `outputs/glass_guided_sampling_experiment/mfm_jump8_gencond_step1000_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
  - guide MAE `12.04 -> 11.05`
  - oracle MAE `12.07 -> 11.11`
  - best 8-step MFM result so far, but still worse than old jump-4
    (`oracle -> 10.01`) and far from GLASS (`oracle -> 8.69`).
- Step 1000, clipped `mu=0.15`:
  `outputs/glass_guided_sampling_experiment/mfm_jump8_gencond_step1000_alpha_target90_seed1_mu015_lam03_steps64_atom_clip80_30`
  - guide MAE `12.04 -> 11.24`
  - oracle MAE `12.07 -> 11.32`
- Step 1000, clipped `mu=0.2`:
  `outputs/glass_guided_sampling_experiment/mfm_jump8_gencond_step1000_alpha_target90_seed1_mu020_lam03_steps64_atom_clip80_30`
  - guide MAE `12.04 -> 11.88`
  - oracle MAE `12.07 -> 11.99`
- Decision: best scalar strength remains clipped `mu=0.1`; stronger guidance
  does not scale cleanly.

Added posterior-property diagnostic:
- `src/mfm_posterior_property_diagnostic.py`
- `slurm/mfm_posterior_property_diagnostic.sbatch`
- This compares GLASS vs MFM terminal property ensembles under common posterior
  noise along the unguided production trajectory.
- Alpha target 90, `posterior_samples=8`, `posterior_steps=8`:
  - baseline jump8roll1000:
    - alpha pair MAE `2.220`
    - alpha pair corr `0.753`
    - reward pair MAE `49.3`
    - softmax-weighted alpha MAE `1.794`
    - softmax weight L1 `0.875`
    - top posterior sample match `0.510`
  - generated step 500:
    - alpha pair MAE `2.206`
    - alpha pair corr `0.755`
    - reward pair MAE `48.9`
    - softmax-weighted alpha MAE `1.804`
    - softmax weight L1 `0.883`
    - top posterior sample match `0.514`
  - generated step 1000:
    - alpha pair MAE `2.192`
    - alpha pair corr `0.759`
    - reward pair MAE `48.7`
    - softmax-weighted alpha MAE `1.788`
    - softmax weight L1 `0.883`
    - top posterior sample match `0.505`

Interpretation:
- Generated-condition training is directionally useful and produced the best
  actual 8-step MFM alpha guidance so far.
- Plain posterior RMSD is still not enough. The alpha reward-tail mismatch is
  large: top posterior-sample agreement is close to random for `K=8`, and
  reward-weight L1 is about `0.88`.
- Next objective should continue improving generated-condition coordinates but
  should explicitly monitor reward-tail/property-ensemble metrics. A fully
  property-specific value function is still not the desired path.

New job:
- `20924360`: generated-condition continuation from generated step 1000.
  - init:
    `outputs/mfm_jump8_gencond_from_step1000_20260610/stage4_20260610-103610/student_step_1000.pt`
  - LR `2e-7`, steps `1000`, batch `4`
  - off-diagonal `rollout_steps=8`
  - generated condition source, `generated_condition_steps=64`
  - weaker real-endpoint diagonal anchor: `diag_weight=0.1`
  - stronger coordinate/final weighting: `coord_loss_weight=6.0`,
    `rollout_final_weight=6.0`
  - intent: improve generated-condition coordinate fidelity and alpha-relevant
    endpoint sensitivity without switching to a separate property value model.

### Coordinate-heavy generated continuation

Job `20924360` completed successfully.
- Output:
  `outputs/mfm_jump8_gencond_coordheavy_from_gen1000_20260610/stage4_20260610-110128`
- Step 500 diagnostics:
  - gradient:
    `outputs/mfm_guidance_gradient_diagnostic/jump8_gencond_coordheavy_step500_alpha_jump8_seed1_b16_vs4`
    - coord `0.2138`
    - atom `0.1491`
    - value bias `3.572`
    - reward bias `3.014`
  - generated-condition posterior:
    `outputs/mfm_student_eval/jump8_gencond_coordheavy_step500_jump8_gencond_n16_k64`
    - t=0.3: spread `1.00`, meanRMSD `0.038`, pairRMSD `0.193`,
      atomTV `0.018`, atomMatch `0.974`
    - t=0.6: spread `0.98`, meanRMSD `0.035`, pairRMSD `0.087`,
      atomTV `0.025`, atomMatch `0.969`
    - t=0.9: spread `1.07`, meanRMSD `0.023`, pairRMSD `0.029`,
      atomTV `0.000`, atomMatch `1.000`
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/jump8_gencond_coordheavy_step500_alpha_vs8_seed1_b16`
    - alpha pair MAE `2.163`
    - alpha pair corr `0.762`
    - reward pair MAE `47.8`
    - weighted alpha MAE `1.784`
    - weight L1 `0.880`
    - top sample match `0.514`
  - actual guidance:
    `outputs/glass_guided_sampling_experiment/mfm_jump8_gencond_coordheavy_step500_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
    - guide MAE `12.04 -> 11.34`
    - oracle MAE `12.07 -> 11.61`
    - worse than generated step 1000.
- Step 1000 diagnostics:
  - gradient:
    `outputs/mfm_guidance_gradient_diagnostic/jump8_gencond_coordheavy_step1000_alpha_jump8_seed1_b16_vs4`
    - coord `0.2099`
    - atom `0.1430`
    - value bias `3.614`
    - reward bias `3.440`
  - generated-condition posterior:
    `outputs/mfm_student_eval/jump8_gencond_coordheavy_step1000_jump8_gencond_n16_k64`
    - t=0.3: spread `1.00`, meanRMSD `0.040`, pairRMSD `0.194`,
      atomTV `0.017`, atomMatch `0.975`
    - t=0.6: spread `0.97`, meanRMSD `0.034`, pairRMSD `0.086`,
      atomTV `0.026`, atomMatch `0.970`
    - t=0.9: spread `1.06`, meanRMSD `0.021`, pairRMSD `0.026`,
      atomTV `0.000`, atomMatch `1.000`
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/jump8_gencond_coordheavy_step1000_alpha_vs8_seed1_b16`
    - alpha pair MAE `2.163`
    - alpha pair corr `0.760`
    - reward pair MAE `47.8`
    - weighted alpha MAE `1.758`
    - weight L1 `0.901`
    - top sample match `0.505`
- Decision:
  - coordinate-heavy continuation is not the next best checkpoint. It improves
    some terminal-property pair metrics but worsens gradient alignment and actual
    guidance.
  - Best 8-step checkpoint remains:
    `outputs/mfm_jump8_gencond_from_step1000_20260610/stage4_20260610-103610/student_step_1000.pt`

### New property-ensemble finetune branch

Added:
- `src/mfm_property_ensemble_finetune.py`
- `slurm/mfm_property_ensemble_finetune.sbatch`

Objective:
- Keep the MFM as the learned posterior integrator.
- Match GLASS terminal property ensembles under common posterior noise, on
  generated production-trajectory conditions.
- Use multiple TFG guide properties (`alpha,gap,homo,lumo,mu,cv`) rather than a
  separate per-property value model.
- Optionally up-weight the GLASS high-reward tail for alpha target 90, because
  the current bottleneck is reward-weight/top-sample mismatch.

Smoke job:
- `20932794`: 20-step property-ensemble smoke from current best checkpoint.
  Completed successfully in
  `outputs/mfm_property_ensemble_smoke_20260610/propens_20260610-112046`.
  - init:
    `outputs/mfm_jump8_gencond_from_step1000_20260610/stage4_20260610-103610/student_step_1000.pt`
  - tiny batch/posterior count: `batch=1`, `posterior_samples=2`
  - smoke validated GPU execution and checkpoint writing.
  - Slurm comma parsing meant this smoke trained only `alpha`; for real runs,
    pass comma-valued `PROPERTIES` through the inherited environment, not inside
    the `--export` list.

Active job:
- `20934547`: full multi-property property-ensemble finetune from the current
  best checkpoint. Completed successfully.
  - properties: `alpha,gap,homo,lumo,mu,cv`
  - steps `500`, LR `1e-7`, batch `2`, posterior samples `4`
  - properties per step `2`, endpoint loss weight `0.25`
  - output:
    `outputs/mfm_property_ensemble_from_gen1000_20260610/propens_20260610-112428`
  - hypothesis:
    8-step endpoint RMSD is now less diagnostic than reward-tail agreement. This
    branch directly trains the 8-step MFM posterior samples to match GLASS
    terminal property ensembles, while retaining a coordinate/atom endpoint
    anchor so the model stays an integrator rather than becoming a separate
    value-function surrogate.

Diagnostics:
- step 250:
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/propens_step250_alpha_vs8_seed1_b16`
    - alpha pair MAE `2.189`
    - alpha pair corr `0.758`
    - reward pair MAE `48.37`
    - weighted alpha MAE `1.780`
    - weight L1 `0.888`
    - top sample match `0.500`
  - alpha gradient:
    `outputs/mfm_guidance_gradient_diagnostic/propens_step250_alpha_jump8_seed1_b16_vs4`
    - coord `0.2310`
    - atom `0.1552`
- step 500:
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/propens_step500_alpha_vs8_seed1_b16`
    - alpha pair MAE `2.176`
    - alpha pair corr `0.759`
    - reward pair MAE `48.08`
    - weighted alpha MAE `1.775`
    - weight L1 `0.882`
    - top sample match `0.505`
  - alpha gradient:
    `outputs/mfm_guidance_gradient_diagnostic/propens_step500_alpha_jump8_seed1_b16_vs4`
    - coord `0.2317`
    - atom `0.1588`

Interpretation:
- Property-ensemble finetuning improved gradient alignment more than the
  reward-tail/top-sample metric.
- It is still not enough to explain GLASS-strength guidance: top sample match
  remains about `0.50` and weight L1 remains about `0.88`.
- Because step 500 is the best 8-step gradient checkpoint so far, it was gated
  with an actual alpha guidance run:
  `outputs/glass_guided_sampling_experiment/mfm_propens_step500_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
  (Slurm job `20938702`).
  - guide MAE `12.04 -> 11.98`
  - oracle MAE `12.07 -> 12.13`
  - property mean `78.90 -> 78.62`
  - reward `-188.3 -> -187.0`

Decision:
- Do not promote the property-ensemble step-500 checkpoint despite its better
  gradient cosine. Actual guidance is worse than the generated-condition step
  1000 checkpoint.
- Added reward-tail-specific losses to `src/mfm_property_ensemble_finetune.py`:
  - KL matching for the GLASS posterior reward softmax weights.
  - log-mean-exp reward value matching.
- Next branch should train the same MFM posterior integrator with these losses,
  focusing on alpha every step with larger posterior ensembles.

### Reward-tail branches

Active/recent jobs:
- `20939921`: reward-weight branch from the failed property-ensemble step-500
  checkpoint.
  - init:
    `outputs/mfm_property_ensemble_from_gen1000_20260610/propens_20260610-112428/student_step_500.pt`
  - output:
    `outputs/mfm_reward_weight_from_propens500_20260610/propens_20260610-113545`
  - steps `800`, LR `5e-8`, batch `4`, posterior samples `8`
  - alpha only, reward softmax KL weight `1.0`, reward value weight `0.05`
  - use as comparison only; the init checkpoint was not good in actual guidance.
- `20940631`: reward-tail branch from the current best generated-condition
  checkpoint. Completed successfully.
  - init:
    `outputs/mfm_jump8_gencond_from_step1000_20260610/stage4_20260610-103610/student_step_1000.pt`
  - output:
    `outputs/mfm_rewardtail_alpha_from_gen1000_20260610/propens_20260610-113734`
  - steps `500`, LR `7e-8`, batch `2`, posterior samples `8`
  - alpha only, reward softmax KL weight `2.0`, reward value weight `0.5`
  - this is the preferred branch because it starts from the best actual-guidance
    checkpoint.

Evaluation plan:
- Gate the `250/400/500/800` checkpoints with:
  - posterior-property alpha reward-tail diagnostic;
  - matched-noise alpha gradient diagnostic;
  - actual alpha guidance only if either reward-weight L1/top-match or gradient
    alignment improves without obvious endpoint collapse.

Results so far:
- `20940631` step 250:
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/rewardtail_gen1000_step250_alpha_vs8_seed1_b16`
    - alpha pair MAE `2.191`
    - weight L1 `0.890`
  - gradient:
    `outputs/mfm_guidance_gradient_diagnostic/rewardtail_gen1000_step250_alpha_jump8_seed1_b16_vs4`
    - coord `0.2299`
    - atom `0.1515`
- `20940631` step 500:
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/rewardtail_gen1000_step500_alpha_vs8_seed1_b16`
    - alpha pair MAE `2.190`
    - weight L1 `0.887`
  - gradient:
    `outputs/mfm_guidance_gradient_diagnostic/rewardtail_gen1000_step500_alpha_jump8_seed1_b16_vs4`
    - coord `0.2310`
    - atom `0.1522`
  - decision: not promoted; direct reward-tail KL from the best checkpoint is
    too noisy and does not improve the held-out reward-tail metric.
- `20939921` step 400:
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/rewardweight_propens500_step400_alpha_vs8_seed1_b16`
    - alpha pair MAE `2.179`
    - weight L1 `0.883`
  - gradient:
    `outputs/mfm_guidance_gradient_diagnostic/rewardweight_propens500_step400_alpha_jump8_seed1_b16_vs4`
    - coord `0.2340`
    - atom `0.1627`
  - actual alpha gate submitted:
    `outputs/glass_guided_sampling_experiment/mfm_rewardweight_propens500_step400_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
    (Slurm job `20943947`).
    - guide MAE `12.04 -> 11.78`
    - oracle MAE `12.07 -> 11.86`
- `20939921` step 800:
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/rewardweight_propens500_step800_alpha_vs8_seed1_b16`
    - alpha pair MAE `2.174`
    - weight L1 `0.883`
  - gradient:
    `outputs/mfm_guidance_gradient_diagnostic/rewardweight_propens500_step800_alpha_jump8_seed1_b16_vs4`
    - coord `0.2367`
    - atom `0.1617`
  - actual alpha:
    `outputs/glass_guided_sampling_experiment/mfm_rewardweight_propens500_step800_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
    - guide MAE `12.04 -> 12.11`
    - oracle MAE `12.07 -> 12.31`
  - decision: overtrained for actual guidance despite higher gradient cosine.

Noise-index diagnostic:
- Added paired-vs-sorted/top-sample metrics to
  `src/mfm_posterior_property_diagnostic.py`.
- Current best generated-condition checkpoint:
  - paired alpha MAE `2.196`
  - sorted alpha MAE `1.723`
  - top sample match `0.505`
  - GLASS top-sample alpha MAE against MFM's own top sample: `1.862`
  - GLASS top-sample alpha MAE against MFM at the same noise index: `2.330`
- Reward-weight step 800:
  - paired alpha MAE `2.174`
  - sorted alpha MAE `1.701`
  - top sample match `0.505`
  - GLASS top-sample alpha MAE against MFM's own top sample: `1.856`
  - GLASS top-sample alpha MAE against MFM at the same noise index: `2.278`

Interpretation:
- MFM is closer as an unordered posterior property set than as a common-noise
  GLASS map.
- The high-reward sample often exists approximately in the MFM ensemble, but not
  at the GLASS noise index. Pairwise reward-weight KL is therefore partly
  optimizing a noise-labeling problem rather than the set-valued posterior value
  used by guidance.
- This explains why reward-weight training can improve diagnostic cosine or a
  single actual gate while failing to close the GLASS gap.

### Direct guidance-gradient finetune

Added:
- `src/mfm_guidance_gradient_finetune.py`
- `slurm/mfm_guidance_gradient_finetune.sbatch`

Purpose:
- Keep the trainable object as the MFM posterior integrator.
- Backpropagate through the MFM 8-step value-gradient estimate and match the
  GLASS value-gradient direction/rms on generated conditions.
- This directly targets the vector field used by guidance.

Implementation notes:
- Requires second-order autograd through attention.
- PyTorch efficient attention kernels do not support that derivative, so the
  script forces math SDP attention locally.

Smoke:
- Local two-step smoke passed after forcing math attention.

Run:
- `20947418`: 100-step small gradient finetune from the best generated-condition
  checkpoint.
  - output:
    `outputs/mfm_guidance_gradient_ft_from_gen1000_20260610/gradft_20260610-115529`
  - batch `1`, posterior samples `4`, teacher/MFM steps `8`
  - atom guidance included
- Diagnostics:
  - step 50:
    `outputs/mfm_guidance_gradient_diagnostic/gradft_gen1000_step50_alpha_jump8_seed1_b16_vs4`
    - coord `0.2312`
    - atom `0.1542`
  - step 100:
    `outputs/mfm_guidance_gradient_diagnostic/gradft_gen1000_step100_alpha_jump8_seed1_b16_vs4`
    - coord `0.2311`
    - atom `0.1546`

Decision:
- The script is technically viable, but the first small run is not promoted.
- Batch `1`/`K=4` is too noisy or the objective is still underpowered. Do not
  spend a large run here until the active random-JVP / coord-direct architecture
  branches are evaluated.

### Active architecture/JVP branches

Active jobs:
- `20945190`: generated-condition off-diagonal random condition-JVP training
  from the current best checkpoint.
  - output:
    `outputs/mfm_gencond_offdiag_jvp_from_gen1000_20260610/randjvp_20260610-114918`
  - steps `1000`, LR `2e-7`, batch `4`, MFM steps `8`
  - goal: improve local sensitivity of the posterior map without property
    specificity.
  - step 500 diagnostics:
    - posterior-property:
      `outputs/mfm_posterior_property_diagnostic/randjvp_gen1000_step500_alpha_vs8_seed1_b16`
      - alpha pair MAE `2.170`
      - weight L1 `0.883`
    - gradient:
      `outputs/mfm_guidance_gradient_diagnostic/randjvp_gen1000_step500_alpha_jump8_seed1_b16_vs4`
      - coord `0.2105`
      - atom `0.1563`
    - decision: not an actual-guidance candidate yet because coordinate
      gradient alignment regressed. Continue to step 1000.
- `20947023`: generated-condition rollout training from teacher init with
  `velocity_parametrization=coord_direct_atom_endpoint`.
  - output:
    `outputs/mfm_coord_direct_gencond_rollout_teacherinit_20260610/stage4_20260610-115431`
  - steps `2000`, LR `2e-5`, batch `16`, rollout steps `8`
  - goal: test whether direct coordinate velocity parametrization is the missing
    architecture piece relative to MFM v2-style behavior.

Actual guidance result:
- `20938702` completed successfully.
- Output:
  `outputs/glass_guided_sampling_experiment/mfm_propens_step500_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
- Result:
  - guide MAE `12.04 -> 11.98`
  - oracle MAE `12.07 -> 12.13`
  - property mean `78.90 -> 78.62`
  - reward `-188.31 -> -186.99`
- Decision:
  - do not promote the property-ensemble step-500 checkpoint.
  - Better matched-noise gradient cosine alone did not transfer. The remaining
    bottleneck is still reward-tail/top-sample selection.

### Reward-weight finetune branch

Code change:
- `src/mfm_property_ensemble_finetune.py` now supports:
  - `--reward-weight-loss-weight`: KL from GLASS reward softmax weights to MFM
    reward softmax weights.
  - `--reward-value-loss-weight`: match GLASS log-mean-exp reward value.
  - `--reward-value-scale`: normalize value loss.
- `slurm/mfm_property_ensemble_finetune.sbatch` exposes these controls.

New job:
- `20939921`: reward-weight alpha branch.
  - init:
    `outputs/mfm_property_ensemble_from_gen1000_20260610/propens_20260610-112428/student_step_500.pt`
  - properties: `alpha` every step
  - steps `800`, LR `5e-8`, batch `4`, posterior samples `8`
  - endpoint loss weight `0.2`
  - reward-tail property weighting `2.0`
  - reward-weight KL weight `1.0`
  - reward-value loss weight `0.05`
  - output root:
    `outputs/mfm_reward_weight_from_propens500_20260610`
  - gate:
    reward-weight L1 and top sample match must improve materially before another
    actual guidance run.

Result:
- `20939921` completed in
  `outputs/mfm_reward_weight_from_propens500_20260610/propens_20260610-113545`.
- step 400:
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/reward_weight_step400_alpha_vs8_seed1_b16`
    - alpha pair MAE `2.179`
    - weight L1 `0.883`
  - gradient:
    `outputs/mfm_guidance_gradient_diagnostic/reward_weight_step400_alpha_jump8_seed1_b16_vs4`
    - coord `0.2340`
    - atom `0.1627`
- step 800:
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/reward_weight_step800_alpha_vs8_seed1_b16`
    - alpha pair MAE `2.174`
    - weight L1 `0.883`
  - gradient:
    `outputs/mfm_guidance_gradient_diagnostic/reward_weight_step800_alpha_jump8_seed1_b16_vs4`
    - coord `0.2367`
    - atom `0.1616`
- Decision:
  - no actual guidance run from this branch. It improves gradient cosine, but
    reward-weight/top-sample selection still does not move materially.

### Generated-condition off-diagonal random JVP branch

Reasoning:
- Terminal endpoint/property losses are not enough. Guidance depends on the
  Jacobian of the posterior endpoint map with respect to `x_cond` and atom
  types.
- The existing random-JVP script was effectively diagonal-only and trained on
  endpoint-noised real data. That does not match the 8-step off-diagonal MFM
  guidance path on generated production-trajectory conditions.

Code change:
- `src/mfm_random_condition_jvp_train.py`
  - added `--condition-source endpoint|generated`
  - added `--generated-condition-steps`
  - added `--offdiagonal` to actually train jump maps
- `slurm/mfm_random_condition_jvp.sbatch`
  - exposes `CONDITION_SOURCE`, `GENERATED_CONDITION_STEPS`, and
    `MFM_MODE_FLAG`.

New job:
- `20945190`: generated-condition, off-diagonal, random-condition JVP training.
  - init:
    `outputs/mfm_jump8_gencond_from_step1000_20260610/stage4_20260610-103610/student_step_1000.pt`
  - steps `1000`, LR `2e-7`, batch `4`
  - GLASS steps `64`, MFM steps `8`
  - generated condition source, generated condition steps `64`
  - random coordinate and atom perturbations, eps `0.03/0.03`
  - endpoint weight `0.5`, JVP weight `1.0`, cosine weight `0.2`
  - output root:
    `outputs/mfm_gencond_offdiag_jvp_from_gen1000_20260610`
  - gate:
    this should improve held-out gradient alignment without relying on
    property-specific losses; if it moves gradient but not reward-tail metrics,
    the next likely bottleneck is raw posterior sample accuracy/property
    sensitivity rather than value-gradient training.

Interim step-500 diagnostics:
- checkpoint:
  `outputs/mfm_gencond_offdiag_jvp_from_gen1000_20260610/randjvp_20260610-114918/student_step_500.pt`
- posterior-property:
  `outputs/mfm_posterior_property_diagnostic/gencond_offdiag_jvp_step500_alpha_vs8_seed1_b16`
  - alpha pair MAE `2.170`
  - weight L1 `0.883`
- gradient:
  `outputs/mfm_guidance_gradient_diagnostic/gencond_offdiag_jvp_step500_alpha_jump8_seed1_b16_vs4`
  - coord `0.2105`
  - atom `0.1563`
- Interim decision:
  - not a candidate for actual guidance. Random JVP training improves its own
    coordinate-JVP training cosine but hurts held-out alpha coordinate-gradient
    alignment.

Final step-1000 diagnostics:
- checkpoint:
  `outputs/mfm_gencond_offdiag_jvp_from_gen1000_20260610/randjvp_20260610-114918/student_step_1000.pt`
- posterior-property:
  `outputs/mfm_posterior_property_diagnostic/gencond_offdiag_jvp_step1000_alpha_vs8_seed1_b16`
  - alpha pair MAE `2.182`
  - weight L1 `0.892`
- gradient:
  `outputs/mfm_guidance_gradient_diagnostic/gencond_offdiag_jvp_step1000_alpha_jump8_seed1_b16_vs4`
  - coord `0.2070`
  - atom `0.1551`
- Decision:
  - do not run actual guidance.
  - generated/off-diagonal random-JVP training did not transfer to the
    reward-gradient direction. The random perturbation objective is too
    weak/misaligned for the current bottleneck.

### Coordinate-direct parametrization branch

ADiT MFM v2 comparison:
- MFM v2 predicts velocity directly in the molecular latent space.
- It uses a true-GLASS diagonal target, adaptive masked losses, and a
  long diagonal-to-offdiagonal curriculum.
- TABASCO's current best MFM mostly uses endpoint parametrization for
  coordinates: predict `x1_hat`, then divide by `1-s`.
- Since coordinates are the weak part, endpoint parametrization may be making
  the coordinate velocity/Jacobian too sensitive near late inner times.

Code change:
- `slurm/mfm_student_stage4_rollout.sbatch` now supports
  `INIT_FROM_TEACHER=1`, so parametrization changes can start from teacher
  initialization instead of incorrectly loading an endpoint-param checkpoint.

New job:
- `20947023`: coordinate-direct / atom-endpoint generated rollout branch.
  - init from teacher, not from an endpoint-param MFM checkpoint
  - velocity parametrization: `coord_direct_atom_endpoint`
  - generated condition source, off-diagonal jump-8 rollout
  - steps `2000`, LR `2e-5`, batch `16`
  - adaptive rollout/diagonal losses
  - coord loss weight `6.0`, atom loss weight `0.5`
  - rollout final weight `6.0`, diagonal weight `1.0`
  - output root:
    `outputs/mfm_coord_direct_gencond_rollout_teacherinit_20260610`
  - gate:
    compare generated-condition posterior RMSD, alpha gradient alignment, and
    actual alpha guidance against the endpoint-param best checkpoint.

Interim result and decision:
- step-1000 checkpoint:
  `outputs/mfm_coord_direct_gencond_rollout_teacherinit_20260610/stage4_20260610-115431/student_step_1000.pt`
- training loss stayed nearly flat around `53-55`.
- built-in posterior validity eval at step 1000:
  - t=0.3: MFM PB valid `0.00` vs GLASS `0.84`
  - t=0.6: MFM PB valid `0.00` vs GLASS `0.78`
  - t=0.9: MFM PB valid `0.00` vs GLASS `0.97`
- `20947023` was canceled after step 1000.
- Decision:
  - direct-coordinate from teacher init with this stage-4 setup is not learning
    a usable posterior.
  - If revisited, it needs a proper diagonal/direct-velocity pretraining stage
    or a different initialization schedule, not direct stage-4 rollout.

### Contrast with older stronger jump-4 MFM

Older actual guidance run:
- `outputs/glass_guided_sampling_experiment/mfm_alpha_target90_seed1_mu010_lam03_steps64_prod_atom_pb`
- checkpoint:
  `outputs/mfm_student_rollout_stage4_residual_gated_big_grid_mid2_spr075_s4_b16_lr1e6/stage4_20260609-090016/student_step_500.pt`
- settings:
  - value steps `4`
  - value samples `4`
  - atom guidance on
  - no clipping
- result:
  - guide MAE `12.04 -> 10.00`
  - oracle MAE `12.07 -> 10.01`
  - PB valid `1.00 -> 0.984`

Same older checkpoint with more/other value settings:
- 8 value steps, no clipping:
  `outputs/glass_guided_sampling_experiment/mfm_jump8_alpha_target90_seed1_mu010_lam03_steps64_atom`
  - guide MAE `12.04 -> 11.47`
  - oracle MAE `12.07 -> 11.59`
- 8 value steps, clipped `80/30`:
  `outputs/glass_guided_sampling_experiment/mfm_jump8_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
  - guide MAE `12.04 -> 11.32`
  - oracle MAE `12.07 -> 11.55`
- 4 value steps, 16 value samples:
  `outputs/glass_guided_sampling_experiment/mfm_alpha_target90_seed1_mu010_lam03_steps64_vs16_atom`
  - guide MAE `12.04 -> 11.09`
  - oracle MAE `12.07 -> 11.28`

Interpretation:
- The current best and older branch share the same residual/gated
  `coord_endpoint_residual_atom_endpoint` architecture.
- Moving that family to 8 value steps did not automatically improve guidance.
- Increasing posterior value samples can help, so current-best 8-step MFM is
  being tested with 16 value samples:
  `outputs/glass_guided_sampling_experiment/mfm_jump8_gencond_step1000_alpha_target90_seed1_mu010_lam03_steps64_atom_vs16_clip80_30`
  (Slurm job `20954917`).

Current-best 8-step with 16 value samples:
- `20954917` completed.
- Output:
  `outputs/glass_guided_sampling_experiment/mfm_jump8_gencond_step1000_alpha_target90_seed1_mu010_lam03_steps64_atom_vs16_clip80_30`
- Result:
  - guide MAE `12.04 -> 11.82`
  - oracle MAE `12.07 -> 11.84`
- Decision:
  - increasing MFM posterior value samples to 16 does not recover GLASS-strength
    guidance for the current 8-step model.

Reward-weight actual gates:
- step 400:
  `outputs/glass_guided_sampling_experiment/mfm_rewardweight_propens500_step400_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
  - guide MAE `12.04 -> 11.78`
  - oracle MAE `12.07 -> 11.86`
- step 800:
  `outputs/glass_guided_sampling_experiment/mfm_rewardweight_propens500_step800_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
  - guide MAE `12.04 -> 12.11`
  - oracle MAE `12.07 -> 12.31`
- Decision:
  - reward-weight finetuning is not a viable direction in its current form.

Old jump-4 stronger-guidance sweep:
- Rationale:
  - The old 4-step MFM is still the strongest MFM-guided result.
  - Before abandoning it as a practical few-step integrator, test whether it has
    headroom under stronger clipped guidance.
- Submitted:
  - `20957313`: old jump-4, atom guidance, clipped `80/30`, `mu=0.15`
    - output:
      `outputs/glass_guided_sampling_experiment/mfm_oldjump4_alpha_target90_seed1_mu015_lam03_steps64_atom_clip80_30`
  - `20957314`: old jump-4, atom guidance, clipped `80/30`, `mu=0.2`
    - output:
      `outputs/glass_guided_sampling_experiment/mfm_oldjump4_alpha_target90_seed1_mu020_lam03_steps64_atom_clip80_30`

Results:
- `mu=0.15`:
  - guide MAE `12.04 -> 10.74`
  - oracle MAE `12.07 -> 10.65`
  - PB valid `1.00 -> 0.984`
- `mu=0.2`:
  - guide MAE `12.04 -> 10.80`
  - oracle MAE `12.07 -> 10.94`
  - PB valid `1.00 -> 0.969`
- Decision:
  - stronger clipped guidance does not beat the original old-jump4 `mu=0.1`
    result (`guide 10.00`, oracle `10.01`).
  - old jump-4 remains the strongest MFM result, but it still does not reach
    GLASS (`guide 8.57`, oracle `8.69`).

### Old-recipe jump-8 generated-condition branch

Rationale:
- The old jump-4 checkpoint is the strongest MFM result.
- Our earlier jump-8 training changed two important old-recipe settings:
  - added diagonal anchor (`diag_weight=0.25`)
  - reduced spread matching (`0.75 -> 0.25`)
- New test: keep the old no-diagonal/high-spread recipe, but train an 8-step
  generated-condition rollout from the old jump-4 checkpoint.

New job:
- `20958810`
- init:
  `outputs/mfm_student_rollout_stage4_residual_gated_big_grid_mid2_spr075_s4_b16_lr1e6/stage4_20260609-090016/student_step_500.pt`
- settings:
  - generated condition source
  - rollout steps `8`
  - steps `1500`, LR `1e-6`, batch `8`
  - diagonal weight `0.0`
  - rollout spread weight `0.75`, spread samples `4`
  - rollout final weight `4.0`
  - coord/atom weights `4.0 / 0.5`
  - output root:
    `outputs/mfm_jump8_gencond_oldrecipe_from_oldjump4_20260610`
- gate:
  - built-in PB/posterior eval at steps 500/1000/1500
  - then alpha gradient and actual guidance if posterior quality is not degraded.

Status after step 1000:
- step-500 built-in posterior/PB check:
  - `t_cond=0.30`: MFM PB valid `0.53`, GLASS `0.69`
  - `t_cond=0.60`: MFM PB valid `0.69`, GLASS `0.81`
  - `t_cond=0.90`: MFM PB valid `1.00`, GLASS `1.00`
- step-1000 built-in posterior/PB check:
  - `t_cond=0.30`: MFM PB valid `0.69`, GLASS `0.75`
  - `t_cond=0.60`: MFM PB valid `0.72`, GLASS `0.69`
  - `t_cond=0.90`: MFM PB valid `1.00`, GLASS `1.00`
- step-500 posterior-property diagnostic:
  `outputs/mfm_posterior_property_diagnostic/oldrecipe_jump8_gencond_step500_alpha_vs8_seed1_b16`
  - mean alpha pair MAE `2.1798`
  - mean posterior weight L1 `0.8819`
- step-500 guidance-gradient diagnostic:
  `outputs/mfm_guidance_gradient_diagnostic/oldrecipe_jump8_gencond_step500_alpha_jump8_seed1_b16_vs4`
  - coordinate cosine `0.2267`
  - atom cosine `0.1385`

Interpretation:
- The old-recipe jump-8 branch is structurally stable and does not show the
  coordinate-direct collapse.
- But the step-500 guidance-gradient alignment is still weak, especially for
  atom logits, and is not yet clearly better than the current-best generated
  jump-8 branch.
- Step-1000 posterior-property and guidance-gradient diagnostics are running as
  Slurm jobs `20964840` and `20964854`. If those improve, run an actual alpha
  guidance sample from the step-1000 checkpoint; otherwise the branch mainly
  tells us that PB-valid rollouts alone are insufficient.

Step-1000 external diagnostics completed:
- posterior-property:
  `outputs/mfm_posterior_property_diagnostic/oldrecipe_jump8_gencond_step1000_alpha_vs8_seed1_b16`
  - mean alpha pair MAE `2.1415`
  - mean posterior weight L1 `0.8923`
- guidance-gradient:
  `outputs/mfm_guidance_gradient_diagnostic/oldrecipe_jump8_gencond_step1000_alpha_jump8_seed1_b16_vs4`
  - coordinate cosine `0.2336`
  - atom cosine `0.1375`

Step-1500 status:
- Training completed.
- built-in posterior/PB check:
  - `t_cond=0.30`: MFM PB valid `0.59`, GLASS `0.62`
  - `t_cond=0.60`: MFM PB valid `0.62`, GLASS `0.62`
  - `t_cond=0.90`: MFM PB valid `0.94`, GLASS `0.97`
- Final external diagnostics submitted:
  - `21004526`: posterior-property for step 1500
  - `21004527`: guidance-gradient for step 1500
- Final external diagnostics completed:
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/oldrecipe_jump8_gencond_step1500_alpha_vs8_seed1_b16`
    - mean alpha pair MAE `2.1284`
    - mean posterior weight L1 `0.8874`
  - guidance-gradient:
    `outputs/mfm_guidance_gradient_diagnostic/oldrecipe_jump8_gencond_step1500_alpha_jump8_seed1_b16_vs4`
    - coordinate cosine `0.2149`
    - atom cosine `0.1182`
- Decision:
  - do not run actual alpha guidance from this no-velocity branch;
  - it preserved plausible posterior samples but degraded the guidance-relevant
    gradients.

Corrected loss diagnosis:
- The old jump-4 checkpoint args show `rollout_velocity_weight=0.5`.
- The no-diagonal/high-spread jump-8 run above accidentally tested
  `rollout_velocity_weight=0.0`.
- That explains the failure mode: endpoints/spread can match while the
  conditional velocity field/Jacobian remains wrong.
- New corrected branch submitted as `21004534`:
  `outputs/mfm_jump8_gencond_oldrecipe_vel_from_oldjump4_20260610`
  - init old jump-4 best checkpoint
  - generated conditions, rollout steps `8`
  - diagonal weight `0.0`
  - final weight `4.0`
  - spread weight `0.75`, spread samples `4`
  - restored coarse GLASS velocity loss `0.5`
  - coord/atom weights `4.0 / 0.5`
- Step-500 status:
  - built-in PB valid at `t_cond=0.3/0.6/0.9`: `0.69/0.70/0.98`
  - GLASS references: `0.80/0.80/0.98`
  - external diagnostics submitted:
    - `21006741`: posterior-property
    - `21006742`: guidance-gradient
  - external diagnostics completed:
    - alpha pair MAE `2.1778`
    - posterior weight L1 `0.9054`
    - coordinate guidance-gradient cosine `0.2086`
    - atom guidance-gradient cosine `0.1462`
  - actual alpha guidance submitted as `21007738`:
    `outputs/glass_guided_sampling_experiment/mfm_oldrecipe_vel_step500_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30_pb`
    - run despite weak cosine because historical actual guidance is not
      perfectly ranked by this proxy.
  - actual alpha guidance completed:
    - guide MAE `12.04 -> 11.36`
    - oracle MAE `12.07 -> 11.44`
    - PB valid `1.00 -> 0.969`
  - Decision:
    - do not promote step 500; restoring coarse velocity supervision improves
      validity but not guidance strength.

Architecture/loss research direction:
- What MFM must learn for guidance is not only posterior endpoints
  `F(noise, x_cond, t_cond)`, but the conditional sensitivity
  `dF/dx_cond`; property guidance uses the transpose-Jacobian
  `J^T grad_reward`.
- Current endpoint/rollout losses mainly supervise samples and are weak gates
  on `J^T grad_reward`.
- Added a property-agnostic random-adjoint branch, submitted as `21004537`:
  `outputs/mfm_random_adjoint_from_gen1000_20260610`
  - trains random terminal vector-Jacobian products against GLASS
  - generated condition states
  - off-diagonal 8-step MFM
  - endpoint loss retained with weight `0.25`
- Random-adjoint step-300 held-out diagnostics:
  - posterior-property:
    `outputs/mfm_posterior_property_diagnostic/adjoint_gen1000_step300_alpha_vs8_seed1_b16`
    - mean alpha pair MAE `2.1642`
    - mean posterior weight L1 `0.8837`
  - guidance-gradient:
    `outputs/mfm_guidance_gradient_diagnostic/adjoint_gen1000_step300_alpha_jump8_seed1_b16_vs4`
    - coordinate cosine `0.2203`
    - atom cosine `0.1584`
  - Decision:
    - training random adjoints is feasible, but step 300 has not transferred to
      held-out alpha guidance gradients; do not promote yet.
- Random-adjoint final step-500 diagnostics submitted:
  - `21007860`: posterior-property
  - `21007884`: guidance-gradient
- Random-adjoint final diagnostics completed:
  - mean alpha pair MAE `2.1921`
  - posterior weight L1 `0.8830`
  - coordinate gradient cosine `0.2343`
  - atom gradient cosine `0.1550`
  - Decision: do not promote; still in the same weak held-out guidance regime.
- Added a split `s/u` conditioning architecture and teacher-initialized
  conditional embeddings:
  - `block_conditioning=split_gated`
  - encoder layers receive mostly `(s, t_cond)`
  - decoder layers receive mostly `(u, u-s, t_cond)`
  - `cond_coord_embed`/`cond_atom_embed` are initialized from the teacher input
    embeddings, instead of starting the conditional pathway fully random.
- Split diagonal pretrain submitted as `21004538`:
  `outputs/mfm_split_gated_stage1_diag_20260610`
  - stage 1 diagonal true-GLASS velocity target
  - adaptive diagonal loss, `p=0.5`, `c=0.01`
  - `t_cond_power=2.0`, `t_cond_zero_rate=0.1`
  - `velocity_parametrization=coord_endpoint_residual_atom_endpoint`
- Split pretrain early signal at step 2000:
  - diagonal coord loss `0.1185`, atom loss `0.0077`
  - PB validity at `t_cond=0.3/0.6/0.9`: `0.12/0.17/0.41`
  - GLASS references: `0.67/0.73/0.97`
- Split pretrain final signal:
  - step 4000: diagonal coord loss `0.0323`, atom loss `0.0037`;
    PB valid `0.12/0.31/0.84`
  - step 6000: diagonal coord loss `0.0251`, atom loss `0.0030`;
    PB valid `0.22/0.33/0.88`
  - Decision: do not promote; even after the diagonal loss improves, the
    low/mid-`t_cond` posterior geometry remains too weak.
- Gated conditional-copy control submitted as `21006113`:
  `outputs/mfm_gated_condcopy_stage1_diag_20260610`
  - same adaptive diagonal schedule as split pretrain
  - `block_conditioning=gated`
  - tests teacher-copied `x_cond` embeddings without changing the existing
    gated architecture.
- Gated conditional-copy final signal:
  - step 2000 PB valid `0.36/0.34/0.56`
  - step 4000 PB valid `0.27/0.36/0.94`
  - step 6000 PB valid `0.28/0.55/0.94`
  - diagonal coord loss improved to `0.0251`, atom loss `0.0035`
  - Decision: copying conditional embeddings helps relative to split at
    low/mid `t_cond`, but diagonal-only 6000-step training is still not enough;
    do not promote without the later rollout/consistency stages.

## 2026-06-10 architecture/loss reset

What the MFM must learn:
- The trainable object is still the 8-step posterior integrator
  `F_theta(z, x_cond, t_cond)`, not a separate value function.
- Strong guidance needs three things simultaneously:
  1. endpoint/trajectory fidelity:
     `F_theta(z, c) ~= F_GLASS(z, c)` under common posterior noise;
  2. conditioning sensitivity:
     `(dF_theta/dc)^T q ~= (dF_GLASS/dc)^T q`, because value guidance is exactly
     a terminal cotangent pushed back to `x_cond`;
  3. reward-tail ranking:
     the MFM posterior samples must put the same samples in the high-reward tail
     as GLASS, not just have the same mean RMSD/spread.

Current diagnosis:
- Plain rollout/final/spread losses can produce plausible posterior samples but
  leave the alpha reward-tail weight L1 near `0.88`.
- Random adjoints are too generic; they train but do not strongly transfer to
  alpha guidance.
- Multi-property guidance-gradient finetuning is technically stable but only
  weakly moved the held-out alpha operator:
  - step 100:
    - alpha property pair MAE `2.1728`
    - reward weight L1 `0.8861`
    - alpha guidance-gradient cosine coord/atom `0.2208/0.1547`
  - step 300:
    - alpha property pair MAE `2.1721`
    - reward weight L1 `0.8858`
    - alpha guidance-gradient cosine coord/atom `0.2181/0.1568`
  - base generated-condition step 1000 was about property MAE `2.192`,
    weight L1 `0.883`, cosine `0.215/0.153`, so the gain is marginal.

Losses to keep using, and why:
- Rollout node/final loss:
  masked coordinate/atom MSE between MFM coarse states and GLASS coarse states.
  This keeps the actual 8-step sampler close to GLASS and catches compounding
  integration error.
- Coarse velocity loss:
  masked MSE on `(x_{i+1}^GLASS - x_i^GLASS) / ds` at each coarse node. This is
  essential because endpoint/spread matching alone can leave the local vector
  field and Jacobian wrong.
- Spread loss:
  relative coordinate spread matching over repeated posterior samples. This
  protects multimodality and avoids a mean-estimator collapse.
- Diagonal anchor:
  true GLASS diagonal velocity loss, kept small during off-diagonal rollout
  continuation so the model does not forget the base posterior field.
- Property-tail ensemble loss:
  compare terminal property predictions for MFM vs GLASS under common posterior
  noise, with extra weight on the GLASS high-reward alpha tail. This teaches
  reward-tail ordering without introducing a separate property value model.
- Guidance-gradient loss:
  direct cosine + log-RMS matching of `grad_{x_cond} log E exp(lambda r)` for
  MFM vs GLASS. This is the closest loss to inference-time guidance, but it is
  expensive/noisy and must remain anchored by endpoint loss.

Code changes from this reset:
- Added `block_conditioning=gated_delta` in
  `src/tabasco/models/components/flow_map_transformer.py`.
  - It is the proven `gated` architecture plus zero-initialized embeddings of
    `(coords_cond - coords_x)` and `(atomics_cond - atomics_x)`.
  - Old `gated` checkpoints load non-strictly; the new delta path starts at
    norm zero and initially preserves checkpoint behavior.
  - Rationale: GLASS posterior velocity is a deterministic function of the
    current posterior state and condition. Exposing the relative coordinate/atom
    displacement directly should improve coordinate sensitivity/Jacobian
    learning without a wholesale architecture rebuild.
- Added explicit `--property-targets` and `--property-scales` to
  `src/mfm_property_ensemble_finetune.py` and the Slurm wrapper.
  - Rationale: the old ensemble loss used a large scale floor, so gap/homo/lumo
    losses were effectively weak. The normalized map keeps all properties
    comparable.

Jobs launched:
- `21014981`: `gated_delta` generated-condition Stage-4 rollout.
  - init:
    `outputs/mfm_jump8_gencond_from_step1000_20260610/stage4_20260610-103610/student_step_1000.pt`
  - output:
    `outputs/mfm_gated_delta_gencond_from_gen1000_20260610`
  - losses: rollout final `4.0`, coarse velocity `1.0`, spread `0.25`,
    diagonal anchor `0.1`, coordinate/atom weights `6.0/0.5`.
  - intent: test whether explicit relative conditioning improves the coordinate
    sensitivity needed by guidance while preserving the current best 8-step
    sampler.
- `21014982`: normalized multi-property posterior-tail finetune.
  - init: same generated-condition step 1000 checkpoint.
  - output:
    `outputs/mfm_propens_normtail_from_gen1000_20260610`
  - properties: `alpha,gap,homo,lumo,mu,cv`.
  - target/scale map:
    `alpha=90`, `gap=0.317`, `homo=-0.214`, `lumo=0.077`,
    `mu=4.627`, `cv=36.846`; scales
    `alpha=1`, `gap=0.126`, `homo=0.051`, `lumo=0.126`,
    `mu=3.582`, `cv=10.197`.
  - intent: improve terminal property/tail ranking under common GLASS posterior
    noise while retaining endpoint anchoring.
- `21014983`: actual alpha guidance check for multi-gradient step 300.
  - checkpoint:
    `outputs/mfm_multigrad_from_gen1000_20260610/gradft_20260610-135504/student_step_300.pt`
  - output:
    `outputs/glass_guided_sampling_experiment/mfm_multigrad_step300_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
  - intent: verify whether the tiny held-out metric improvement transfers to
    the real production guidance loop.

Results from this reset:
- Multi-property guidance-gradient finetune step 300:
  - actual alpha guidance:
    `outputs/glass_guided_sampling_experiment/mfm_multigrad_step300_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
    - guide MAE `12.04 -> 11.72`
    - oracle MAE `12.07 -> 11.75`
  - Decision: not promoted; small held-out proxy gains did not transfer.
- Normalized property-tail finetune from the generated-condition base:
  - output:
    `outputs/mfm_propens_normtail_from_gen1000_20260610/propens_20260610-140836`
  - step 200 held diagnostics:
    - alpha pair MAE `2.1618`
    - reward weight L1 `0.8834`
    - guidance-gradient cosine coord/atom `0.2223/0.1577`
  - step 200 actual alpha:
    `outputs/glass_guided_sampling_experiment/mfm_propens_normtail_step200_alpha_target90_seed1_mu010_lam03_steps64_atom_clip80_30`
    - guide MAE `12.04 -> 11.59`
    - oracle MAE `12.07 -> 11.71`
  - Decision: not promoted. The proxy improved slightly but actual guidance is
    weaker than the current best 8-step checkpoint.
- `gated_delta` generated-condition rollout:
  - output:
    `outputs/mfm_gated_delta_gencond_from_gen1000_20260610/stage4_20260610-140758`
  - step 600:
    - alpha pair MAE `2.0750`
    - reward weight L1 `0.8950`
    - guidance-gradient cosine coord/atom `0.2177/0.1623`
    - actual alpha guide MAE `12.04 -> 11.55`, oracle `12.07 -> 11.71`
  - step 1200:
    - alpha pair MAE `2.0585`
    - reward weight L1 `0.9073`
    - guidance-gradient cosine coord/atom `0.2402/0.1649`
    - actual alpha guide MAE `12.04 -> 11.96`, oracle `12.07 -> 12.12`
  - Decision: not promoted. Explicit delta conditioning improves pairwise
    property agreement and coordinate-gradient cosine, but worsens reward-tail
    weights and fails actual guidance.
- Added reward-tail-weighted endpoint loss to
  `src/mfm_property_ensemble_finetune.py`:
  - per-sample coordinate/atom endpoint MSE is weighted by the GLASS reward
    softmax for the alpha target.
  - intent: directly supervise the GLASS posterior samples that dominate
    guidance, rather than only scalar properties.
- Tail-endpoint finetune from generated-condition base:
  - output:
    `outputs/mfm_tail_endpoint_from_gen1000_20260610/propens_20260610-142402`
  - step 300 held diagnostics:
    - alpha pair MAE `2.1930`
    - reward weight L1 `0.8852`
    - guidance-gradient cosine coord/atom `0.2190/0.1556`
  - Decision: not promoted; no meaningful held-out gain.
- Tail-endpoint finetune from `gated_delta` step 600:
  - output:
    `outputs/mfm_tail_endpoint_from_gdelta600_20260610/propens_20260610-142533`
  - step 300 held diagnostics:
    - alpha pair MAE `2.0872`
    - reward weight L1 `0.8968`
    - guidance-gradient cosine coord/atom `0.2248/0.1582`
  - Decision: not promoted; preserves the `gated_delta` pairwise-property gain
    but still worsens reward-tail weights.

Updated diagnosis:
- We can now improve pairwise terminal property agreement without improving
  guidance. That means pairwise endpoint/property matching is not the missing
  ingredient.
- Reward-weight L1/top-tail identity is still not responding to the current
  scalar-tail or tail-endpoint losses.
- The likely next failure mode is condition-distribution mismatch: training and
  most diagnostics use unguided production states, while actual strong guidance
  differentiates posterior values along guided trajectory states.
- Next high-value diagnostic:
  run posterior-property and guidance-gradient alignment along GLASS-guided
  production trajectories, not only unguided trajectories. If the gap is much
  worse there, train the MFM on guided-condition states while keeping the model
  as the posterior integrator.

## 2026-06-10 diagonal velocity field diagnosis

Question:
- Is the local GLASS diagonal velocity field itself extremely hard to learn?
- Do we need to force stronger directional similarity on the diagonal?

Code changes:
- Added `src/mfm_glass_diagonal_field_diagnostic.py`.
  - Measures masked coordinate/atom cosine, relative error, norm ratio, target
    RMS, simple baselines, and condition-resample sensitivity.
  - Supports exact Stage-1 path states and generated-condition GLASS-posterior
    states.
- Added `slurm/mfm_glass_diagonal_field_diagnostic.sbatch`.
- Added `--diag-direction-weight` to `src/mfm_student_train.py` and
  `diagonal_distill_loss`.
  - Loss is `base diagonal loss + lambda * (1 - masked cosine)`.
  - Defaults to zero, so previous runs are unchanged.
- Exposed diagonal direction/coord/atom/endpoint-space knobs in
  `slurm/mfm_student_stage1.sbatch`.

Diagnostics run locally because Slurm controller was unavailable:
- Full Slurm submission failed with:
  `sbatch: error: Batch job submission failed: Unable to contact slurm controller`.
- Exact Stage-1 path distribution:
  `outputs/mfm_diag_field_diagnostic/cpu_diag_field_20260610_current/diagnostic.json`
  - current generated-condition 8-step checkpoint:
    - coord cosine `0.9945`
    - atom cosine `0.9988`
    - coord relative error `0.1057`
    - atom relative error `0.0292`
    - coord norm ratio `1.0142`
    - atom norm ratio `1.0016`
  - `gated_delta` step 600 and old jump-4 baseline are effectively the same.
  - Simple baselines:
    - endpoint-vector coord cosine `0.877`
    - unconditional teacher coord cosine `0.852`
    - condition-delta coord cosine `0.408`
- Generated-condition GLASS-posterior state distribution:
  `outputs/mfm_diag_field_diagnostic/cpu_diag_field_20260610_generated_posterior/diagnostic.json`
  - current generated-condition 8-step checkpoint:
    - coord cosine `0.9970`
    - atom cosine `0.9982`
    - coord relative error `0.0741`
    - atom relative error `0.0366`
    - coord norm ratio `1.0037`
    - atom norm ratio `1.0004`
  - Again, `gated_delta` step 600 and old jump-4 are essentially identical.

Interpretation:
- The tested diagonal GLASS velocity is not a hard local vector-field learning
  problem. Directional similarity is already very strong for both coordinates
  and atoms.
- The weak guidance is therefore unlikely to come from local diagonal angle
  error on ordinary path/posterior states.
- More diagonal angle forcing is still worth a controlled run, but it is now a
  falsification test, not the primary explanation.
- The leading failure remains an operator/Jacobian/reward-tail issue: the
  integrated posterior map must place the same reward-tail samples and expose
  the same condition sensitivity as GLASS.

Next controlled run once Slurm recovers:
- Direction-weighted Stage-1 continuation from the current 8-step checkpoint:
  - init:
    `outputs/mfm_jump8_gencond_from_step1000_20260610/stage4_20260610-103610/student_step_1000.pt`
  - `DIAG_DIRECTION_WEIGHT=2.0`
  - `COORD_LOSS_WEIGHT=6.0`
  - `ATOM_LOSS_WEIGHT=0.5`
  - `DIAG_LOSS_TYPE=adaptive`
  - short first run, then re-evaluate diagonal cosine and actual alpha guidance.

## 2026-06-10 many-step diagonal guidance check

Question:
- If the local diagonal velocity is good, does MFM guidance become GLASS-like
  when the value posterior sampler uses many diagonal Euler steps?

Existing larger evidence:
- Diagonal-16 actual alpha guidance from the diagonal-rollout checkpoint was
  still weak:
  - diag-8: guide MAE `12.04 -> 11.33`, oracle `12.07 -> 11.52`
  - diag-16: guide MAE `12.04 -> 11.61`, oracle `12.07 -> 11.73`
- This was already far from strong GLASS guidance:
  guide MAE `12.04 -> 8.57`, oracle `12.07 -> 8.69`.

New reduced CPU probe, current generated-condition checkpoint:
- Checkpoint:
  `outputs/mfm_jump8_gencond_from_step1000_20260610/stage4_20260610-103610/student_step_1000.pt`
- Matched-noise value-gradient diagnostic, batch `2`, value samples `2`,
  production steps `16`, diagonal MFM:
  - `value_steps=8`:
    `outputs/mfm_guidance_gradient_diagnostic/cpu_diag8_gen1000_b2_vs2/gradient_alignment.json`
    - coord cosine `0.1581`
    - atom cosine `0.0664`
    - MFM coord RMS `3.98` vs GLASS `2.86`
  - `value_steps=64`:
    `outputs/mfm_guidance_gradient_diagnostic/cpu_diag64_gen1000_b2_vs2/gradient_alignment.json`
    - coord cosine `0.1536`
    - atom cosine `0.0150`
    - MFM coord RMS `5.62` vs GLASS `3.42`
- Reduced actual guidance smoke, same seed/batch/settings, `mu=0.1`,
  atom-guided, clips `80/30`:
  - MFM diagonal-64:
    `outputs/glass_guided_sampling_experiment/cpu_mfm_diag64_gen1000_seed1_b2_s16_vs2_mu010`
    - guide MAE `1.79 -> 1.29`
    - oracle MAE `1.71 -> 1.05`
  - GLASS-64:
    `outputs/glass_guided_sampling_experiment/cpu_glass64_seed1_b2_s16_vs2_mu010`
    - guide MAE `1.79 -> 0.55`
    - oracle MAE is noisy on batch `2`, `1.71 -> 2.11`

Interpretation:
- More diagonal integration steps do not automatically recover GLASS value
  gradients. On this probe, 64 diagonal MFM steps are no better than 8 and have
  worse atom-gradient alignment.
- This resolves the apparent contradiction: high local velocity cosine does not
  imply high guidance-gradient cosine. Guidance differentiates through the whole
  posterior flow map and reward-tail softmax; small state/endpoint errors can
  change terminal reward weights and the condition Jacobian.
- The hypothesis "diagonal is good, therefore many-step MFM should be as good
  as GLASS" is false for the current checkpoint.

Pending GPU jobs:
- Tried to submit full batch `32`, production `64`, value steps `64` MFM/GLASS
  guidance runs and a batch `16` gradient diagnostic.
- All `sbatch` commands timed out without job ids because the Slurm controller
  was still unreachable from this shell.

## 2026-06-10 deeper diagonal failure investigation

Problem:
- Local diagonal velocity cosine is high, but many-step diagonal MFM guidance is
  not GLASS-like. The missing object must be either:
  1. condition sensitivity of the local field, `d v_diag / d x_cond`;
  2. terminal reward-tail/cotangent mismatch;
  3. accumulated posterior-map Jacobian error.

New diagnostic code:
- Extended `src/mfm_glass_diagonal_field_diagnostic.py` to finite-difference
  `x_cond` and compare:
  `Delta v_MFM / eps` vs `Delta v_GLASS / eps`.
- Added diagonal condition-sensitivity training knobs:
  - `--diag-condition-jvp-weight`
  - `--diag-condition-jvp-direction-weight`
  - `--diag-condition-jvp-eps`
  in `src/mfm_student_train.py` and `slurm/mfm_student_stage1.sbatch`.
- This trains the diagonal object we were actually missing: not just
  `v(s,s,x,c,t)`, but its local response to the condition.

Condition-sensitivity diagnostics:
- Exact Stage-1 path distribution:
  `outputs/mfm_diag_field_diagnostic/cpu_diag_sensitivity_path_20260610/diagnostic.json`
  - current generated-condition checkpoint:
    - velocity coord/atom cosine: `0.9940 / 0.9988`
    - velocity coord/atom relative error: `0.1094 / 0.0292`
    - condition-JVP coord/atom cosine: `0.9172 / 0.6805`
    - condition-JVP coord/atom relative error: `0.4776 / 2.0005`
    - condition-JVP coord/atom norm ratio: `1.1787 / 2.2098`
- Generated-condition GLASS-posterior states:
  `outputs/mfm_diag_field_diagnostic/cpu_diag_sensitivity_generated_posterior_20260610/diagnostic.json`
  - velocity coord/atom cosine: `0.9971 / 0.9973`
  - velocity coord/atom relative error: `0.0736 / 0.0429`
  - condition-JVP coord/atom cosine: `0.9149 / 0.6691`
  - condition-JVP coord/atom relative error: `0.4341 / 14.7542`
  - atom target condition-JVP RMS is tiny, so the huge atom relative error is a
    scale warning; the cosine is still only about `0.67`.

Factor diagnostic, reduced CPU setting:
- `outputs/mfm_guidance_factor_diagnostic/cpu_gen1000_diag8_b2_vs2/factor.json`
  - actual coord/atom cosine: `0.1581 / 0.0664`
  - same terminal cotangent through MFM Jacobian: `0.9054 / 0.8470`
  - same GLASS Jacobian with MFM terminal cotangent: `-0.7616 / -0.8237`
  - terminal cotangent cosine: `0.0637 / 0.2026`
- `outputs/mfm_guidance_factor_diagnostic/cpu_gen1000_diag64_b2_vs2/factor.json`
  - actual coord/atom cosine: `0.1536 / 0.0150`
  - same terminal cotangent through MFM Jacobian: `0.0968 / 0.1245`
  - same GLASS Jacobian with MFM terminal cotangent: `-0.7307 / -0.6965`
  - terminal cotangent cosine: `-0.2596 / 0.0697`

Interpretation:
- The local velocity value is learned, but `d v / d x_cond` is not accurate
  enough: coordinate sensitivity has about `0.43-0.48` relative error, and atom
  sensitivity is much worse.
- The terminal reward cotangent is also wrong. For diag-8, giving MFM the GLASS
  terminal cotangent makes its Jacobian look good in this small probe, but using
  MFM's own terminal cotangent through GLASS is anti-aligned. That means reward
  tail/cotangent mismatch dominates the actual gradient.
- For diag-64, the MFM trajectory/cotangent becomes worse rather than better;
  many small steps amplify the wrong sensitivity instead of fixing it.

Next training branch:
- Force the diagonal to learn condition sensitivity:
  - init current generated-condition checkpoint;
  - Stage 1 diagonal continuation;
  - `DIAG_CONDITION_JVP_WEIGHT=0.25`
  - `DIAG_CONDITION_JVP_DIRECTION_WEIGHT=0.5`
  - `DIAG_DIRECTION_WEIGHT=0.5`
  - `COORD_LOSS_WEIGHT=6.0`, `ATOM_LOSS_WEIGHT=0.25`
  - small LR first (`1e-6` or lower), then re-run:
    1. diagonal sensitivity diagnostic;
    2. factor diagnostic;
    3. many-step guidance gradient;
    4. actual alpha guidance only if sensitivity/cotangent metrics move.

Slurm status update:
- The earlier `sbatch` failures were caused by the restricted Codex shell not
  resolving/contacting `holy-slurm02`. With normal network access:
  - `scontrol ping` reports `Slurmctld(primary) at holy-slurm02 is UP`;
  - `squeue` works;
  - `getent hosts holy-slurm02` resolves.
- Launched the condition-sensitivity Stage-1 continuation:
  - job `21159331`
  - node `holygpu8a25406`
  - output dir:
    `outputs/mfm_diag_condition_jvp_from_gen1000_20260610/stage1_20260610-201931`
  - init:
    `outputs/mfm_jump8_gencond_from_step1000_20260610/stage4_20260610-103610/student_step_1000.pt`
  - key weights:
    `DIAG_CONDITION_JVP_WEIGHT=0.25`,
    `DIAG_CONDITION_JVP_DIRECTION_WEIGHT=0.5`,
    `DIAG_DIRECTION_WEIGHT=0.5`,
    `COORD_LOSS_WEIGHT=6.0`,
    `ATOM_LOSS_WEIGHT=0.25`.

## 2026-06-10 ESD comparison against MFM, Norway, and ADiT

Question: are we using ESD, and is it implemented the same way as the reference
repos?

Answer:
- Yes, TABASCO has Stage-2 ESD in `src/tabasco/flow/mfm_losses.py`.
- The mathematical ESD target matches the MFM and Norway implementations:
  `vss = GLASS posterior velocity`, then compute a student JVP of
  `v(s,u,I_s)` along `(ds=1, du=0, dI_s=vss)`, and train
  `v(s,u,I_s)` toward `vss + (u-s) * JVP`.
- We also run the JVP under math SDPA, matching the JVP workaround used in the
  adjacent repos.

Important differences:
- Before this note, TABASCO Stage-2 ESD used `coord_loss + atom_loss` directly,
  while Norway weights the coordinate and atom/type components separately. This
  is now patched: `esd_consistency_loss` accepts `coord_weight` and
  `atom_weight`, `mfm_student_train.py` passes the global modality weights, and
  `slurm/mfm_student_stage2.sbatch` exposes `COORD_LOSS_WEIGHT` /
  `ATOM_LOSS_WEIGHT`.
- The TABASCO flow-map output is still endpoint-biased by default:
  off-diagonal velocities are often derived with a `(1-s)` denominator and atom
  endpoints are softmaxed. The reference MFM-style ESD objective assumes a clean
  direct velocity map for `s -> u`. This may make our ESD formula correct but a
  poor match to the parametrization.
- Prior Stage-2 ESD runs showed very large JVP/target norms. The issue is not
  an algebra bug; the ESD teacher target is stiff in this model.

Consistency-loss direction:
- Do not blindly scale up ESD as the primary fix. Keep it as a small-gap,
  clipped, weighted diagnostic/training term.
- The more promising consistency loss from ADiT MFM v2 is semigroup/PSD:
  train the big jump `X_{s,u}` to match a two-step composition
  `X_{w,u}(X_{s,w})`, preferably using an EMA student as the composed target.
  This avoids JVP stiffness and directly trains the few-step integrator
  property we care about.
- Direct GLASS transition distillation and GLASS rollout distillation remain
  the strongest teacher-supervised consistency losses in this repo. The next
  serious branch should combine diagonal GLASS anchors with weighted transition
  or semigroup consistency, then evaluate condition sensitivity and reward-tail
  cotangent alignment before expensive alpha guidance.

## 2026-06-11 diagonal-64 GLASS guidance gate

User reset: before more architecture work, prove whether the diagonal is strong
enough to recover GLASS guidance when we use many posterior integration steps.

Gate definition:
- Local diagonal field on generated-condition GLASS-posterior states:
  - coordinate and atom velocity cosine should stay near `>=0.99`;
  - coordinate relative error should be low across `s` and `t_cond`;
  - finite-difference condition sensitivity should not be badly wrong.
- Diagonal-64 posterior fidelity vs GLASS-64:
  - coordinate spread ratio should be near `1`;
  - coordinate pair RMSD should not grow with 64 steps;
  - atom TV/match should stay close to GLASS.
- Differentiated guidance-gradient alignment:
  - diagonal MFM-64 value gradient should align with GLASS-64, not merely
    produce similar posterior samples;
  - this is the real gate for guided sampling.
- Actual alpha guidance:
  - compare GLASS-64 value guidance vs diagonal MFM-64 value guidance on the
    same seed, alpha target `90`, atom guidance enabled, and gradient clips
    `coord=80`, `atom=30`.

Jobs launched:
- Failed first attempt `21399096`: wrong student spec format for the field
  diagnostic (`label:path:key` instead of expected `label=path`).
- `21399354`: field diagnostic for `gen1000`, generated condition +
  GLASS-posterior states.
  output: `outputs/mfm_diagonal_gate_20260611/diag_field_compare`
- `21399804`: field diagnostic for `sens2000` with the same settings.
  output: `outputs/mfm_diagonal_gate_20260611/diag_field_sens2000`
- `21399352`: diagonal-64 posterior fidelity for `gen1000`;
  output: `outputs/mfm_diagonal_gate_20260611/eval_gen1000_diag64_gencond`
- `21399351`: diagonal-64 posterior fidelity for `sens2000`;
  output: `outputs/mfm_diagonal_gate_20260611/eval_sens2000_diag64_gencond`
- `21399496`: GLASS-64 vs MFM-64 guidance-gradient alignment for `gen1000`,
  batch `8`, value samples `4`.
  output: `outputs/mfm_diagonal_gate_20260611/grad_gen1000_diag64_b8_vs4`
- `21399494`: same guidance-gradient alignment for `sens2000`.
  output: `outputs/mfm_diagonal_gate_20260611/grad_sens2000_diag64_b8_vs4`
- `21399492`: actual GLASS-64 alpha guidance, batch `16`.
  output: `outputs/mfm_diagonal_gate_20260611/guided_glass64_alpha_b16_mu010_atom_clip80_30`
- `21399493`: actual diagonal MFM-64 alpha guidance for `gen1000`, batch `16`.
  output: `outputs/mfm_diagonal_gate_20260611/guided_mfm_gen1000_diag64_alpha_b16_mu010_atom_clip80_30`

Current prior evidence before these GPU jobs:
- Local velocity value is already strong: generated-posterior diagnostic had
  coordinate/atom velocity cosines around `0.997 / 0.998`.
- This is not sufficient. Reduced CPU guidance-gradient probes with diagonal
  MFM did not match GLASS:
  - `gen1000`, diag-8: coord/atom cosine `0.158 / 0.066`.
  - `gen1000`, diag-64: coord/atom cosine `0.154 / 0.015`.
  - `sens2000`, diag-8: coord/atom cosine `-0.550 / -0.348`.
- Therefore the gate is expected to fail unless the small CPU probes were too
  noisy. If it fails at the gradient/actual-guidance stage, the diagonal value
  fit is not the bottleneck; condition sensitivity and reward-tail cotangents
  are.

Results:
- Local diagonal velocity value passes the easy part but not the sensitivity
  part.
  - `gen1000` field average:
    - coord velocity cosine `0.993`, coord relative error `0.102`
    - condition-JVP coord cosine `0.865`, relative error `0.963`
    - condition-JVP atom cosine `0.427`; atom relative error is huge because
      the atom sensitivity target is often tiny.
  - `sens2000` field average:
    - coord velocity cosine `0.995`, coord relative error `0.089`
    - condition-JVP coord cosine `0.877`, relative error `0.943`
    - condition-JVP atom cosine `0.430`
- Diagonal-64 posterior fidelity is reasonable but not exact:
  - `gen1000` coordinate pair RMSD at `t_cond=0.3/0.6/0.9`:
    `0.292 / 0.121 / 0.067`; spread `1.026 / 0.992 / 1.202`.
  - `sens2000` pair RMSD:
    `0.221 / 0.126 / 0.057`; spread `1.022 / 1.042 / 1.136`.
  - `sens2000` is the better diagonal posterior checkpoint.
- Diagonal-64 guidance-gradient alignment fails:
  - `gen1000`: coord/atom cosine `0.115 / 0.186`.
  - `sens2000`: coord/atom cosine `0.281 / 0.229`.
  - This is much better than the tiny CPU negative result for `sens2000`, but
    still nowhere near a strong diagonal mimic of GLASS.
- Weak actual-guidance setting, atom guidance, clips `80/30`, `mu=0.1`,
  guide-every `8` was not a useful gate: GLASS-64 itself worsened
  (`12.05 -> 13.28` guide MAE), so ignore it as a strength test.
- Strong actual-guidance setting, coordinate-only, `mu=3.0`, guide-every `4`:
  - GLASS-64:
    `outputs/mfm_diagonal_gate_20260611/guided_glass64_alpha_b64_mu300_coord`
    - guide MAE `12.04 -> 9.09`
    - oracle MAE `12.07 -> 9.12`
  - diagonal MFM-64 `gen1000`:
    `outputs/mfm_diagonal_gate_20260611/guided_mfm_gen1000_diag64_alpha_b64_mu300_coord`
    - guide MAE `12.04 -> 9.80`
    - oracle MAE `12.07 -> 9.83`
  - diagonal MFM-64 `sens2000`:
    `outputs/mfm_diagonal_gate_20260611/guided_mfm_sens2000_diag64_alpha_b64_mu300_coord`
    - guide MAE `12.04 -> 9.69`
    - oracle MAE `12.07 -> 9.69`

PB-valid rerun of the same strong setting:
- The non-PB run above is not sufficient on its own; we need target-property
  MAE and PB validity together.
- Reran coordinate-only `mu=3.0`, guide-every `4`, batch `64`,
  `value_steps=64`, `value_samples=4`, with `--compute-pb`.
- GLASS-64:
  `outputs/mfm_diagonal_gate_20260611/guided_glass64_alpha_b64_mu300_coord_pb`
  - guide MAE `12.04 -> 7.78`
  - oracle MAE `12.07 -> 7.38`
  - PB valid `1.000 -> 0.781`
  - PB intersection `1.000 -> 0.781`
- diagonal MFM-64 `sens2000`:
  `outputs/mfm_diagonal_gate_20260611/guided_mfm_sens2000_diag64_alpha_b64_mu300_coord_pb`
  - guide MAE `12.04 -> 9.45`
  - oracle MAE `12.07 -> 9.85`
  - PB valid `1.000 -> 0.797`
  - PB intersection `1.000 -> 0.797`
- Interpretation:
  - GLASS-64 is much stronger on target alpha MAE.
  - MFM diagonal-64 preserves roughly the same PB validity, slightly higher in
    this run, but is materially weaker on the target property.
  - The proper scorecard is therefore `(target MAE, oracle MAE, PB valid)`, not
    guidance strength alone.

Benchmark-disambiguation note:
- The old strong GLASS guidance benchmark was not only the tiny batch-2 CPU
  smoke. The large PB-valid campaign lives under
  `outputs/qm9_length_conditioned_guidance/*_n1000*/summary.json` and used
  `num_samples=1000`, PB computation, length-conditioned targets, soft atom
  guidance, `value_glass_steps=4`, and `value_samples=4`.
- Best alpha `n=1000` run found:
  `outputs/qm9_length_conditioned_guidance/n1000_mu045_s128/summary.json`
  - guide MAE `9.30 -> 1.33`
  - oracle MAE `9.10 -> 1.81`
  - PB valid `0.987 -> 0.936`
- Other `n=1000` property-conditioned GLASS runs are indeed well below MAE `1`
  with PB validity recorded:
  - `cv`: guide/oracle `0.521 / 0.670`, PB valid `0.944`
  - `mu`: guide/oracle `0.648 / 0.827`, PB valid `0.883`
  - `gap`: guide/oracle `0.022 / 0.027`, PB valid `0.873`
  - `homo`: guide/oracle `0.0077 / 0.0105`, PB valid `0.866`
  - `lumo`: guide/oracle `0.014 / 0.018`, PB valid `0.910`
- The fixed-target alpha-90 batch-64 gate above is a separate stress test. It
  is still useful because current MFM work has been centered on reproducing
  that fixed-target guidance path, but it should not replace the historical
  `n=1000` PB-valid GLASS scorecard. The right MFM evaluation target is to run
  the same length-conditioned `n=1000` scorecard, or a smaller exact-schema
  version during iteration.

Current MFM on the length-conditioned scorecard:
- Patched `src/qm9_length_conditioned_guidance_experiment.py` and
  `slurm/qm9_length_conditioned_guidance.sbatch` to expose the existing
  `value_sampler=mfm` path, MFM checkpoint/key, diagonal mode, and guidance
  clipping options.
- Ran direct GPU `n=100` alpha, seed/target seed `8/2026`, sample steps `128`,
  guide-every `1`, soft atom guidance, PB enabled. This is the exact
  length-conditioned alpha schema, not the fixed alpha-90 stress test.

| value sampler | value steps | mu | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|---:|---:|
| GLASS | 4 | 0.45 | `8.59 -> 1.38` | `8.30 -> 1.80` | `0.96 -> 0.93` |
| MFM diagonal `sens2000` | 8 | 0.45 | `8.59 -> 2.15` | `8.30 -> 3.02` | `0.96 -> 0.85` |
| MFM diagonal `sens2000` | 8 | 0.35 | `8.59 -> 2.56` | `8.30 -> 3.39` | `0.96 -> 0.88` |
| MFM diagonal `sens2000` | 16 | 0.45 | `8.59 -> 2.87` | `8.30 -> 4.09` | `0.96 -> 0.82` |

Outputs:
- `outputs/qm9_length_conditioned_guidance/glass_len_alpha_n100_s128_mu045_ge1_pb_direct`
- `outputs/qm9_length_conditioned_guidance/mfm_len_alpha_n100_s128_mu045_diag8_sens2000_pb_direct`
- `outputs/qm9_length_conditioned_guidance/mfm_len_alpha_n100_s128_mu035_diag8_sens2000_pb_direct`
- `outputs/qm9_length_conditioned_guidance/mfm_len_alpha_n100_s128_mu045_diag16_sens2000_pb_direct`

Interpretation:
- Current MFM does move the length-conditioned alpha distribution strongly, so
  it is not dead on this scorecard.
- It does **not** match GLASS: at the same `mu=0.45`, diagonal-8 leaves about
  `+0.77` guide MAE and `+1.21` oracle MAE relative to GLASS and has lower PB
  validity (`0.85` vs `0.93`).
- Lowering MFM strength improves PB but worsens MAE; increasing diagonal steps
  to 16 is worse. The problem is still guidance-map fidelity, not just the
  scalar guidance strength or too few diagonal steps.

Large `n=1000` alpha scorecard for current MFM:
- `outputs/qm9_length_conditioned_guidance/mfm_len_alpha_n1000_s128_mu045_diag8_sens2000_pb`
- Same length-conditioned alpha setup as the historical GLASS run:
  sample steps `128`, `mu=0.45`, value samples `4`, diagonal MFM value steps
  `8`, PB enabled.
- Result:
  - guide MAE `9.29 -> 2.49`
  - oracle MAE `9.10 -> 3.34`
  - PB valid `0.987 -> 0.824`
- Reference GLASS `n=1000`:
  `outputs/qm9_length_conditioned_guidance/n1000_mu045_s128/summary.json`
  - guide MAE `9.30 -> 1.33`
  - oracle MAE `9.10 -> 1.81`
  - PB valid `0.987 -> 0.936`
- Conclusion: current MFM does not match GLASS on the production
  length-conditioned scorecard.

Forced diagonal-rollout training:
- Objective: train the realized 8-step diagonal Euler rollout against GLASS-64
  coarse nodes on generated conditioning states, while keeping a strong local
  diagonal anchor and a small condition-JVP term.
- Short direct GPU run:
  `outputs/mfm_diag_rollout8_forced_20260611/stage4_20260611-165222`
  from `sens2000`, batch `2`, steps `600`.
- Built-in training loss was noisy, but the diagonal anchor stayed tight:
  `diag_coord` stayed around `0.012-0.021`.
- Actual length-conditioned alpha `n=100` evaluation:

| checkpoint | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|
| before forced rollout, `sens2000` | `8.59 -> 2.15` | `8.30 -> 3.02` | `0.96 -> 0.85` |
| forced rollout step 200 | `8.59 -> 1.99` | `8.30 -> 2.71` | `0.96 -> 0.88` |
| forced rollout step 600 | `8.59 -> 1.99` | `8.30 -> 2.77` | `0.96 -> 0.87` |
| GLASS reference | `8.59 -> 1.38` | `8.30 -> 1.80` | `0.96 -> 0.93` |

- Interpretation: this is the first diagonal training branch in this round that
  improved the length-conditioned guidance scorecard. It is still not enough.
  The next move is scaling this exact objective with a larger batch and more
  steps, not returning to local-only JVP training.
- Submitted long batch-4 continuation:
  - job `21601396`
  - wrapper `slurm/mfm_diag_forced_rollout.sbatch`
  - output root `outputs/mfm_diag_rollout8_forced_20260611_long`
  - status at launch: pending for resources.

MC-sample check:
- All earlier meaningful MFM guidance evaluations used only `value_samples=4`.
  That is small for a log-sum-exp reward tail over a multimodal posterior.
- Sweep on forced-rollout step 200, same `n=100` alpha/PB setup:

| value samples | guide MAE | oracle MAE | PB valid |
|---:|---:|---:|---:|
| 4 | `8.59 -> 1.99` | `8.30 -> 2.71` | `0.96 -> 0.88` |
| 8 | `8.59 -> 2.02` | `8.30 -> 2.95` | `0.96 -> 0.80` |
| 16 | `8.59 -> 1.98` | `8.30 -> 2.62` | `0.96 -> 0.92` |

- Interpretation: `value_samples=4` is not enough for a reliable MFM gate.
  `value_samples=16` gives a materially better oracle/PB tradeoff, though it
  still does not close the GLASS gap (`1.38 / 1.80 / 0.93`).
- Evaluation policy going forward: use `value_samples=16` for MFM guidance
  scorecards when feasible; keep `4` only for quick smoke tests.
- Training policy going forward: increase posterior MC in rollout/spread
  training. Patched `slurm/mfm_diag_forced_rollout.sbatch` to expose
  `ROLLOUT_SPREAD_SAMPLES`; submitted a spread-4 variant:
  - job `21603965`
  - output root `outputs/mfm_diag_rollout8_forced_20260611_long_spr4`.

Conclusion:
- The diagonal is good as a posterior sampler, but not "super strong" as a
  differentiable GLASS guidance map.
- Many diagonal steps do recover a large fraction of the strong GLASS guidance
  improvement, but not all of it:
  - GLASS-64 improves guide MAE by `2.95`.
  - best diagonal MFM-64 improves by `2.35`.
  - remaining gap is about `0.61` guide MAE / `0.57` oracle MAE.
- In the PB-valid rerun, the gap is larger:
  - GLASS-64 improves guide MAE by `4.26` at PB valid `0.781`;
  - best diagonal MFM-64 improves guide MAE by `2.59` at PB valid `0.797`.
- The remaining gap is aligned with the bad value-gradient and condition-JVP
  diagnostics. Training only the velocity value is not enough.

Next branch:
- Launched stronger diagonal condition-sensitivity training from `sens2000`:
  - job `21402973`
  - output:
    `outputs/mfm_diag_condition_jvp_strong_from_sens2000_20260611`
  - init:
    `outputs/mfm_diag_condition_jvp_from_gen1000_20260610/stage1_20260610-201931/student_step_2000.pt`
  - weights:
    `DIAG_CONDITION_JVP_WEIGHT=1.0`,
    `DIAG_CONDITION_JVP_DIRECTION_WEIGHT=2.0`,
    `DIAG_DIRECTION_WEIGHT=1.0`,
    `COORD_LOSS_WEIGHT=8.0`,
    `ATOM_LOSS_WEIGHT=0.25`,
    `LR=5e-7`,
    `STEPS=6000`.
- Evaluation criterion for this branch:
  - do not judge it by diagonal loss alone;
  - re-run field condition-JVP, diagonal-64 guidance-gradient cosine, and
    strong actual guidance at each useful checkpoint.

Branch status:
- Stopped `21402973` early after step `4000` was written. Reason:
  - step-2000 training log already showed diagonal drift
    (`diag_coord 0.0182`, coord cosine `0.993`);
  - internal condition-JVP cosine was only `0.596`;
  - PB eval degraded by step 3000.
- Evaluated `student_step_2000.pt`:
  - field average:
    - coord velocity cosine `0.993`, coord rel `0.097`
    - condition-JVP coord cosine `0.896`, condition-JVP coord rel `1.024`
    - condition-JVP atom cosine `0.437`
  - diagonal-64 posterior fidelity:
    - pair RMSD `0.225 / 0.159 / 0.058` at `t_cond=0.3/0.6/0.9`
    - spread `1.013 / 1.081 / 1.181`
  - diagonal-64 guidance-gradient alignment:
    - coord/atom cosine `0.209 / 0.155`
- Verdict: stronger local condition-JVP training moved one local field metric
  but hurt the guidance-gradient metric relative to `sens2000`
  (`0.281 / 0.229`). This is not the right standalone fix.

Updated conclusion:
- Best diagonal checkpoint for this gate remains `sens2000`, not the stronger
  continuation.
- The next objective should explicitly train the integrated value-gradient /
  adjoint of the diagonal sampler, or use a consistency loss on the integrated
  posterior map. Local velocity and local finite-difference sensitivity are not
  sufficient proxies.

## 2026-06-11 continued: forced-rollout long-run gate

Actual MFM guidance scorecard remains the deciding metric: length-conditioned
alpha, `n=100`, PB enabled, `sample_steps=128`, `guide_every=1`, `mu=0.45`,
diagonal MFM value sampler with `value_glass_steps=8`.

Current measured rows:

| sampler/checkpoint | value samples | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|---:|
| GLASS reference | 4 | `8.59 -> 1.38` | `8.30 -> 1.80` | `0.96 -> 0.93` |
| `sens2000` | 4 | `8.59 -> 2.15` | `8.30 -> 3.02` | `0.96 -> 0.85` |
| short forced rollout step 200 | 4 | `8.59 -> 1.99` | `8.30 -> 2.71` | `0.96 -> 0.88` |
| short forced rollout step 600 | 4 | `8.59 -> 1.99` | `8.30 -> 2.77` | `0.96 -> 0.87` |
| short forced rollout step 200 | 16 | `8.59 -> 1.98` | `8.30 -> 2.62` | `0.96 -> 0.92` |
| long spread-2 step 1000 | 16 | `8.59 -> 2.11` | `8.30 -> 2.64` | `0.96 -> 0.87` |

Interpretation:
- Increasing posterior MC to `value_samples=16` is necessary for a serious
  gate. It improves the oracle/PB tradeoff for the short forced-rollout
  checkpoint.
- Longer forced-rollout training did not monotonically improve the actual
  guidance score. Spread-2 step 1000 is worse than the short step-200 gate even
  though the training losses look acceptable.
- Therefore checkpoint selection must be by actual guidance MAE plus PB valid,
  not by rollout loss or internal PB probes alone.

Long jobs still running / producing checkpoints:
- `21601396`: spread-2 long forced rollout,
  `outputs/mfm_diag_rollout8_forced_20260611_long/stage4_20260611-170747`;
  checkpoints observed through step `7000`.
- `21603965`: spread-4 long forced rollout,
  `outputs/mfm_diag_rollout8_forced_20260611_long_spr4/stage4_20260611-171509`;
  checkpoints observed through step `6000`.

Pending evaluation once Slurm submission works again:
- spread-2 steps `4000`, `6000`, and `7000`;
- spread-4 steps `3000`, `5000`, and `6000`;
- all with `value_samples=16`, PB enabled, same alpha length-conditioned gate.

Infrastructure note:
- The current shell reports a stale/invalid Slurm allocation
  (`SLURM_JOB_ID=21623110`) on `holygpu7c26202`, but has no `/dev/nvidia*`
  device nodes and `nvidia-smi` cannot reach the driver. `module load cuda` does
  not fix this. `srun` reports the allocation as expired/invalid.
- New `sbatch` submissions from this compute shell failed with
  `Unable to contact slurm controller`. Submit the pending scorecards from a
  login node or a fresh allocation.

## 2026-06-11 continued: forced-rollout checkpoint selection

Slurm recovered and the actual alpha/PB scorecards were run. Same gate as
above: `n=100`, `sample_steps=128`, `guide_every=1`, `mu=0.45`, PB enabled,
diagonal MFM value sampler.

| sampler/checkpoint | value samples | clip coord/atom | guide MAE | oracle MAE | PB valid |
|---|---:|---|---:|---:|---:|
| GLASS reference | 4 | none | `1.38` | `1.80` | `0.93` |
| short forced rollout step 200 | 16 | none | `1.98` | `2.62` | `0.92` |
| spread-2 step 2000 | 16 | none | `1.58` | `2.16` | `0.88` |
| spread-2 step 2000 | 32 | none | `1.91` | `2.64` | `0.93` |
| spread-2 step 2000 | 16 | `15 / 8` | `2.32` | `2.99` | `0.91` |
| spread-2 step 3000 | 16 | none | `1.90` | `2.68` | `0.89` |
| spread-2 step 4000 | 16 | none | `1.87` | `2.41` | `0.88` |
| spread-2 step 6000 | 16 | none | `2.34` | `3.05` | `0.95` |
| spread-2 step 7000 | 16 | none | `1.90` | `2.64` | `0.90` |
| spread-4 step 1000 | 16 | none | `2.04` | `2.75` | `0.91` |
| spread-4 step 2000 | 16 | none | `1.99` | `2.74` | `0.90` |
| spread-4 step 3000 | 16 | none | `1.82` | `2.44` | `0.91` |
| spread-4 step 3000 | 32 | none | `1.80` | `2.51` | `0.93` |
| spread-4 step 5000 | 16 | none | `2.14` | `3.02` | `0.91` |
| spread-4 step 6000 | 16 | none | `2.20` | `2.98` | `0.91` |

Current reading:
- `spread-2 step 2000` proves the MFM can get close to GLASS MAE, but it loses
  geometry. PB failures are mostly bond length/angle related.
- Increasing MC to `value_samples=32` improves PB, but weakens the actual
  target improvement; it is not a free fix.
- Guidance-gradient RMS has heavy tails. For spread-2 step 2000,
  coordinate RMS p50/p90/p95/max is about `5.0/15.9/23.1/68.8`;
  atom RMS p50/p90/p95/max is about `1.85/7.34/14.1/72.7`.
- Simple gradient clipping is too blunt. Clip `15/8` improved PB from `0.88`
  to `0.91` but worsened oracle MAE to `2.99`; clip `10/5` was worse.
- Best balanced checkpoint right now is `spread-4 step 3000` with
  `value_samples=32`: guide/oracle/PB `1.80 / 2.51 / 0.93`.

Next training branch:
- Launched `21632711`, output
  `outputs/mfm_diag_rollout8_forced_20260611_spr8_anchor_lr2e7`.
- Intent: preserve the early forced-rollout gain while reducing geometry drift:
  `ROLLOUT_SPREAD_SAMPLES=8`, `LR=2e-7`, `DIAG_WEIGHT=2.0`,
  `DIAG_DIRECTION_WEIGHT=1.5`, lower `ROLLOUT_FINAL_WEIGHT=6.0`, and
  checkpoint/eval every `500` steps.
- First checkpoint observed at step `500`; internal PB probe is not yet clearly
  better than the prior branches, so select by actual scorecard once step
  `1000` or `1500` is available.

## 2026-06-12: 4-step diagonal target

Important correction: the strong GLASS length-conditioned scorecard uses
`value_glass_steps=4`. In the GLASS value sampler this is exactly four
forward-Euler steps over posterior time `s`.

Implication:
- The immediate MFM target should not only be "8-step MFM matches GLASS-64".
- We should test and train "4-step diagonal MFM matches the actual 4-step GLASS
  posterior/value map used by guidance".
- This is a better-aligned target for the strong guidance scorecard.

Additional results before switching:
- spr8-anchor step 1000, `value_samples=32`: guide/oracle/PB
  `2.15 / 2.95 / 0.90`.
- spr8-anchor step 2500, `value_samples=32`: `1.93 / 2.44 / 0.91`.
- spr8-anchor step 4000, `value_samples=32`: `1.81 / 2.45 / 0.94`.
- spr8-anchor is a good balanced result, but still not GLASS MAE.

Jobs launched:
- 4-step evaluation of existing checkpoints with `VALUE_GLASS_STEPS=4`
  (this means 4 MFM diagonal steps when `VALUE_SAMPLER=mfm`):
  - `21881383`: spread-4 step3000, `value_samples=32`.
  - `21881381`: spread-2 step2000, `value_samples=32`.
  - `21881382`: spr8-anchor step4000, `value_samples=32`.
- true 4-step diagonal rollout training:
  - `21881474`, output
    `outputs/mfm_diag_rollout4_forced_from_spr4_step3000_20260612`;
  - init: spread-4 step3000;
  - `ROLLOUT_STEPS=4`, `TRANSITION_TEACHER_STEPS=4`,
    `GENERATED_CONDITION_STEPS=4`, `MFM_STEPS=4`, `GLASS_STEPS=4`;
  - spread samples `8`, LR `2e-7`, strong diagonal anchor.

Completed 4-step scorecard:

| sampler/checkpoint | value samples | value steps | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|---:|---:|
| GLASS reference | 4 | 4 | `1.381` | `1.803` | `0.93` |
| spread-4 step3000 | 32 | 4 | `1.224` | `1.738` | `0.99` |
| spread-2 step2000 | 32 | 4 | `1.418` | `1.787` | `0.97` |
| spr8-anchor step4000 | 32 | 4 | `1.560` | `2.047` | `0.96` |

This is the first MFM result that is actually GLASS-level on the
length-conditioned alpha gate. It also confirms the working hypothesis: with an
adequate diagonal, four MFM value steps can reproduce the strong GLASS guidance
behavior. The prior 8-step MFM evaluations were testing the wrong numerical map
for this scorecard and over-integrated the learned diagonal.

Current priority:
- validate `spread-4 step3000`, diagonal-4, `value_samples=32` at `n=1000`;
- score the true 4-step-trained continuation checkpoints on the same actual
  guidance/PB gate before trusting internal rollout losses;
- then continue diagonal training only if it improves over the already strong
  spread-4 step3000 diagonal-4 result.

4-step continuation scorecard:

| checkpoint | value samples | value steps | guide MAE | oracle MAE | PB valid |
|---|---:|---:|---:|---:|---:|
| init: spread-4 step3000 | 32 | 4 | `1.224` | `1.738` | `0.99` |
| train4 step1000 | 32 | 4 | `1.628` | `1.904` | `0.96` |
| train4 step2000 | 32 | 4 | `1.507` | `1.888` | `0.93` |
| train4 step4000 | 32 | 4 | `1.380` | `1.872` | `0.94` |
| train4 step5000 | 32 | 4 | `1.347` | `1.589` | `0.97` |

The continuation initially damages the useful guidance map, then recovers. Step
5000 is not clearly better than the init on the small guide-regressor MAE, but
it is better on oracle MAE. Both `spread-4 step3000` and `train4 step5000` need
the `n=1000` gate before choosing a checkpoint.

Jobs launched:
- `21890703`: `n=1000`, spread-4 step3000, diagonal-4,
  `value_samples=32`.
- `21891912`: `n=1000`, train4 step5000, diagonal-4,
  `value_samples=32`.

Large validation result:

| checkpoint | n | guide MAE | oracle MAE | PB valid | bond lengths | bond angles |
|---|---:|---:|---:|---:|---:|---:|
| GLASS reference | 1000 | `1.33` | `1.81` | `0.936` | not recorded here | not recorded here |
| spread-4 step3000 diagonal-4 | 1000 | `1.450` | `2.042` | `0.946` | `0.961` | `0.983` |
| train4 step5000 diagonal-4 | 1000 | `1.514` | `2.021` | `0.952` | `0.967` | `0.981` |
| coord-res step2000 diagonal-4 | 1000 | `1.510` | `2.060` | `0.954` | `0.968` | `0.981` |

The MFM diagonal-4 result is close to GLASS at `n=1000`, but the n=100 win did
not fully hold. The remaining gap is now small enough that checkpoint/loss
selection matters: validate the train4 step5000 checkpoint before deciding
whether continued 4-step training actually improves the diagonal.

Scale verdict:
- spread-4 step3000 remains the alpha guide-MAE leader at `n=1000`;
- train4 step5000 has slightly better oracle/PB but worse guide MAE;
- coord-res step2000 looked decent at `n=100` but does not improve the large
  alpha scorecard;
- neither all-parameter 4-step branch has yet beaten GLASS on the large alpha
  scorecard.

New conservative training branch:
- `21893050`, output
  `outputs/mfm_diag4_coord_residual_from_spr4_step3000_20260612`;
- init: spread-4 step3000, but model parametrization is
  `coord_endpoint_residual_atom_endpoint`;
- `--train-residual-only`, so only the coordinate residual head trains
  (`33,408` parameters); backbone and atom head stay fixed;
- generated conditioning uses `128` production steps, while the posterior target
  remains the exact 4-step GLASS map;
- loss emphasis: coordinate-only (`ATOM_LOSS_WEIGHT=0`), strong diagonal
  direction anchor, lower spread weight (`0.05`).

Purpose: improve the weak coordinate diagonal without overwriting the already
good atom model or the backbone learned by spread-4 step3000.

Coordinate residual-only branch results, `n=100` alpha/PB:

| checkpoint | guide MAE | oracle MAE | PB valid | bond lengths | bond angles |
|---|---:|---:|---:|---:|---:|
| init: spread-4 step3000 | `1.224` | `1.738` | `0.99` | not in summary row | not in summary row |
| coord-res step500 | `1.304` | `1.805` | `0.94` | not checked here | not checked here |
| coord-res step1000 | `1.316` | `1.746` | `0.95` | not checked here | not checked here |
| coord-res step1500 | `1.429` | `1.911` | `0.94` | `0.960` | `0.970` |
| coord-res step2000 | `1.299` | `1.811` | `0.99` | `0.990` | `0.990` |
| coord-res step2500 | `1.340` | `1.876` | `0.96` | `0.970` | `0.980` |
| coord-res step3000 | `1.311` | `1.836` | `0.95` | `0.960` | `0.970` |

Interim verdict:
- coordinate residual training can preserve PB if checkpointed carefully
  (`step2000`), but it has not beaten the original spread-4 step3000 diagonal;
- the coordinate-only branch sharpens or shifts the value map without giving a
  robust improvement, so the next residual branch adds atom/geometry pressure
  and lower LR rather than pushing the same branch longer.
- submitted `21902711` for `n=1000` validation of coord-res step2000 because it
  is the only residual checkpoint with both decent MAE and PB `0.99` at `n=100`.

Geometry-preserving residual branch:
- `21898175`, output
  `outputs/mfm_diag4_coord_residual_geom_from_spr4_step3000_20260612`;
- same residual-only architecture, but `LR=1e-5`, `ATOM_LOSS_WEIGHT=0.5`,
  `ROLLOUT_SPREAD_WEIGHT=0.25`, `DIAG_WEIGHT=8`, `DIAG_DIRECTION_WEIGHT=4`.
- early step500 actual score is poor: guide/oracle/PB
  `1.801 / 2.285 / 0.97`; this branch is not promising unless later checkpoints
  recover.
- step2000 actual score also underperforms:
  `1.337 / 1.900 / 0.93`; close this branch unless final step2500
  unexpectedly differs. It does not beat the plain residual branch or the
  spread-4 step3000 base checkpoint.

Next full-model branch:
- `21906561`, output
  `outputs/mfm_diag4_full_gen128_from_spr4_step3000_20260612`;
- init: spread-4 step3000;
- endpoint parametrization, all parameters trainable;
- generated conditioning uses `128` production steps to match the guidance
  trajectory distribution better than the previous generated-4 run;
- posterior target is still the exact 4-step GLASS map;
- conservative LR `8e-8`, `DIAG_WEIGHT=4`, `DIAG_DIRECTION_WEIGHT=2`,
  `ROLLOUT_SPREAD_WEIGHT=0.10`, spread samples `4`.
- first checkpoint step500 is saved and queued for actual alpha/PB scorecard as
  job `21908065`.
- step500 actual score: `1.441 / 1.918 / 0.96`, worse than the base checkpoint.
  Step1000 is queued as `21909494`; if it also underperforms, this branch is
  likely not the right direction.
- step1000 actual score: `1.229 / 1.772 / 0.96`. This is close on guide MAE but
  still below the base checkpoint on oracle and PB. Step1500 queued as
  `21910592`; continue watching only because step1000 is near the base guide MAE.
- step1500 actual score degraded to `1.461 / 1.928 / 0.97`.
- step2000 queued as `21911508`, but this branch is not ahead of the base
  checkpoint unless a later checkpoint unexpectedly recovers.
- step2000 actual score: `1.546 / 2.024 / 1.00`. Geometry is excellent, but the
  property guidance is materially worse than the base checkpoint.
- step2500 queued as `21912181`; continue to collect, but current checkpoint
  choice remains spread-4 step3000 diagonal-4.
- step2500 actual score: `1.580 / 2.059 / 0.93`; close this branch as a
  non-improvement unless the final checkpoint unexpectedly reverses the trend.
- later checkpoints:
  - step3000: `1.447 / 1.686 / 0.96`;
  - step3500: `1.248 / 1.684 / 0.96`;
  - step4000: `1.502 / 2.003 / 0.96`.
- step3500 is not better than the base on small-gate guide MAE, but it has the
  best small-gate oracle MAE among recent branches. Submitted `21920177` for
  `n=1000` validation.
- n=1000 validation for step3500 failed to improve:
  `1.497 / 2.032 / 0.938`. Do not promote this branch.

## 2026-06-12 continued: new diagonal forcing branches

The base checkpoint is still:

```text
outputs/mfm_diag_rollout8_forced_20260611_long_spr4/stage4_20260611-171509/student_step_3000.pt
```

Current base score:
- `n=100`: guide/oracle/PB `1.224 / 1.738 / 0.99`.
- `n=1000`: guide/oracle/PB `1.450 / 2.042 / 0.946`.

New architecture branches:
- `21917844`, `outputs/mfm_diag4_gateddelta_gen128_from_spr4_step3000_20260612`
  - `BLOCK_CONDITIONING=gated_delta`;
  - initialized from the additive base checkpoint with missing gated/delta
    modules left at init;
  - early log is stable: diagonal coord around `0.015-0.018`.
  - step500 actual score queued as `21920181`.
  - step500 actual score: `1.299 / 1.759 / 0.93`; not better than base, but
    plausible enough to score step1000 (`21921117`).
- `21917845`, `outputs/mfm_diag4_splitgated_gen128_from_spr4_step3000_20260612`
  - `BLOCK_CONDITIONING=split_gated`, `ENCODER_DEPTH=2`;
  - initialized from the additive base checkpoint with missing gates left at
    init;
  - early diagonal anchor is worse (`~0.027-0.038` coord), so this is lower
    priority unless actual guidance surprises.
  - step500 actual score queued as `21920183`.
  - step500 actual score: `2.595 / 3.037 / 0.94`; cancelled job `21917845`.

New loss branch:
- Added a Stage-4 teacher-forced velocity loss:
  - the student is queried at the actual GLASS coarse nodes;
  - target is the GLASS coarse-step velocity;
  - this directly forces the diagonal field on the 4-step GLASS manifold, not
    only on the student rollout state.
- `21919172`, `outputs/mfm_diag4_tfvel_gen128_from_spr4_step3000_20260612`
  - additive architecture, generated-128 conditions;
  - `ROLLOUT_TEACHER_FORCED_VELOCITY_WEIGHT=3.0`;
  - `ROLLOUT_FINAL_WEIGHT=2.0`, `ROLLOUT_VELOCITY_WEIGHT=0.5`,
    `DIAG_WEIGHT=5`, `DIAG_DIRECTION_WEIGHT=2.5`.
  - first checkpoint step500 queued for actual score as `21921118`.

Direct-coordinate parametrization branches:
- Added `--skip-init-coord-head` so a direct coordinate velocity head is not
  initialized from an endpoint-prediction head.
- Added `--train-coord-output-only` for a low-risk head-only direct velocity
  experiment.
- `21922623`, output
  `outputs/mfm_diag4_coord_direct_headonly_from_spr4_step3000_20260612`
  - `VELOCITY_PARAMETRIZATION=coord_direct_atom_endpoint`;
  - skip old coordinate head;
  - train only `out_coord_linear`;
  - atom endpoint path stays initialized from the base checkpoint.
- `21922624`, output
  `outputs/mfm_diag4_coord_direct_full_from_spr4_step3000_20260612`
  - same parametrization and initialization;
  - full model trains at low LR.

Purpose: test the user's direct-coordinate-velocity hypothesis without throwing
away the strong atom endpoint behavior learned by the base diagonal model.

Selection rule remains unchanged: actual length-conditioned alpha scorecard
with PB validity decides checkpoint quality. Internal rollout/PB probes are only
screening signals.

Hypothesis: the failed all-parameter true4 branch may have used the wrong
conditioning-state distribution (`generated_condition_steps=4`). This branch
keeps the exact 4-step diagonal target but trains on a production-state
distribution closer to the actual 128-step guided sampler.

## 2026-06-12 generated-condition diagonal diagnosis

Teacher-forced Stage-4 result:
- `21919172`, `outputs/mfm_diag4_tfvel_gen128_from_spr4_step3000_20260612`
  - step500: `1.761 / 2.200 / 0.92`;
  - step1000: `1.245 / 1.815 / 0.95`;
  - step1500: `1.502 / 1.850 / 0.97`.
- Verdict: not better than the base checkpoint (`1.224 / 1.738 / 0.99` at
  `n=100`), so this branch was stopped.

Destructive direct-coordinate test:
- `21922623`, direct coordinate head-only with skipped coord head;
- `21922624`, direct coordinate full-model with skipped coord head.
- Both started two orders of magnitude off the usable diagonal range:
  - head-only: `diag_coord 1.68 -> 0.81` by step350;
  - full-model: `diag_coord ~1.6-1.8` through step300.
- Both were cancelled. This was a useful negative result: zeroing/skipping the
  inherited coordinate head destroys the already useful coordinate prior. Direct
  coordinate velocity should only be tested with the inherited head retained or
  as a residual correction.

New code change:
- Added `diagonal_state_distill_loss(...)` in
  `src/tabasco/flow/mfm_losses.py` so diagonal supervision can be applied to an
  explicit `(bar_x, x_cond, s, t_cond)` tuple.
- Added `glass_posterior_diagonal_distill_loss(...)`: sample `x_cond` from a
  generated production trajectory, sample `bar_x` by integrating the GLASS
  posterior path only up to the selected inner time `s`, and supervise the
  instantaneous GLASS diagonal velocity at that point. There is no final rollout
  matching term in this objective.
- Wired Stage 1 training to use this generated-condition diagonal objective
  when `--rollout-condition-source generated`.
- Added `STAGE` to `slurm/mfm_diag_forced_rollout.sbatch` so the same GPU
  wrapper can run clean diagonal Stage 1 forcing.

Smoke result:
- one-step CPU smoke on generated-condition Stage 1 passed;
- the initial smoke used only `generated_condition_steps=2` and
  `posterior_steps=2`, which produced an artificial large loss
  (`diag_coord 11.28`). The production-like generated-128 / posterior-16
  training runs do not reproduce that failure.

Full generated-posterior field diagnostic:
- `21926842`, output
  `outputs/mfm_diag_field_diagnostic/base_spr4_generated_posterior_20260612/diagnostic.json`.
- Base checkpoint has strong local coordinate velocity on generated-posterior
  states:
  - coordinate cosine mostly `0.98-1.00`;
  - coordinate relative error mostly `0.02-0.23`.
- The weak spot is condition sensitivity, not first-order velocity:
  - at `t_cond=0.10, s=0.95`, condition-JVP cosine `-0.060`, rel error `3.88`;
  - at `t_cond=0.30, s=0.95`, condition-JVP cosine `0.207`, rel error `5.40`;
  - at `t_cond=0.50, s=0.95`, condition-JVP cosine `0.685`, rel error `3.87`.

Interpretation:
- The base checkpoint is good at the local diagonal velocity on the generated
  condition manifold, so continuing to reduce ordinary diagonal MSE alone is
  unlikely to move the guidance score much.
- Strong property guidance needs the derivative of the posterior map with
  respect to `x_cond`. The generated-condition condition-JVP is the most direct
  property-agnostic target for that.
- The current priority is therefore generated-condition diagonal field learning
  with heavier condition-JVP pressure, followed by the actual alpha/PB
  scorecard.

Active jobs:
- `21926337`: endpoint-param generated-condition Stage 1 diagonal forcing
  - output:
    `outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612`
  - init: spread-4 step3000;
  - `STAGE=1`, generated condition steps `128`, posterior path steps `16`;
  - `LR=1e-6`, `COORD_LOSS_WEIGHT=8`, `DIAG_DIRECTION_WEIGHT=5`,
    condition-JVP weights kept on.
- `21926336`: inherited-head direct-coordinate generated-condition Stage 1
  - output:
    `outputs/mfm_gencond_diag_stage1_coorddirect_from_spr4_step3000_20260612`
  - `VELOCITY_PARAMETRIZATION=coord_direct_atom_endpoint`;
  - no skipped coordinate head; low LR `7e-7`.
- `21926336` was cancelled at step300 because `diag_coord` stayed around
  `0.77-0.82`, far from the endpoint-param diagonal range.
- `21928200`: JVP-heavy generated-condition Stage 1 diagonal forcing
  - output:
    `outputs/mfm_gencond_diag_jvpheavy_from_spr4_step3000_20260612`;
  - `DIAG_CONDITION_JVP_WEIGHT=1.0`,
    `DIAG_CONDITION_JVP_DIRECTION_WEIGHT=1.0`;
  - lower ordinary direction weight so the branch focuses on condition
    sensitivity instead of already-good local velocity.
- `21929536`: actual alpha/PB scorecard for `21926337` step500.
- `21929537`: generated-posterior field diagnostic comparing base vs
  `21926337` step500.

Endpoint generated-condition Stage 1 step500:
- alpha/PB n=100, diagonal-4, `value_samples=32`:
  `1.283 / 1.589 / 0.99`.
- Compared with base (`1.224 / 1.738 / 0.99`), this is worse on guide MAE but
  better on oracle MAE with PB preserved.
- Field diagnostic vs base is almost unchanged: local velocity and
  condition-JVP metrics are essentially identical across the grid. The small
  oracle gain may be from subtle trajectory changes not visible in the coarse
  field diagnostic, or from n=100 variance.
- Do not promote yet. Continue to step1000 and evaluate JVP-heavy step500.

Jobs launched:
- `21931358`: actual alpha/PB scorecard for JVP-heavy step500.
- `21931376`: generated-posterior field diagnostic for JVP-heavy step500 vs
  base.

JVP-heavy step500:
- alpha/PB n=100, diagonal-4, `value_samples=32`:
  `1.304 / 1.910 / 0.96`.
- This is worse than both base and endpoint generated-condition step500.
- Field diagnostic is also essentially unchanged versus base; heavier random
  condition-JVP pressure did not move the measured generated-posterior JVP by
  step500.
- Step1000 is still worth scoring before cancelling because the training loss is
  noisy and the branch checkpoint already exists.

Jobs launched:
- `21932979`: actual alpha/PB scorecard for endpoint generated-condition
  step1000.
- `21932978`: actual alpha/PB scorecard for JVP-heavy step1000.

Endpoint generated-condition Stage 1 step1000:
- alpha/PB n=100, diagonal-4, `value_samples=32`:
  `1.107 / 1.513 / 0.98`.
- This is the best n=100 result so far in this branch family and beats the
  base checkpoint on guide and oracle MAE:
  - base: `1.224 / 1.738 / 0.99`;
  - dgen step1000: `1.107 / 1.513 / 0.98`.
- PB drops by one point relative to base, so this needs n=1000 validation
  before promotion.

JVP-heavy step1000:
- alpha/PB n=100, diagonal-4, `value_samples=32`:
  `1.498 / 1.918 / 0.93`.
- Verdict: close this branch. Random condition-JVP pressure at this weighting
  damaged actual guidance and PB and did not visibly move the diagnostic field.
- Cancelled `21928200`.

Large validation:
- `21934996`: n=1000 alpha/PB validation for endpoint generated-condition
  step1000.
- Result: `1.451 / 1.983 / 0.943`.
- This is essentially tied with the base guide MAE (`1.450`) and slightly
  improves oracle MAE (`2.042 -> 1.983`), but it does not solve the large-gate
  target. Do not promote as a decisive replacement.

Endpoint generated-condition continuation:
- step1500 checkpoint exists:
  `outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612/stage1_20260612-112012/student_step_1500.pt`.
- Internal Stage-1 PB probe at step1500:
  - t=0.30 MFM/GLASS PB `0.50 / 0.25`;
  - t=0.60 MFM/GLASS PB `0.75 / 0.66`;
  - t=0.90 MFM/GLASS PB `1.00 / 0.94`.
- `21935981`: actual n=100 alpha/PB scorecard for step1500.
- Result: `1.340 / 1.681 / 0.97`.
- Step1500 regressed relative to step1000 on the n=100 scorecard, so step1000
  remains the best generated-condition diagonal checkpoint so far.

New tuning checks for step1000:
- `21938285`: `value_samples=64`, `mu=0.45`, n=100.
- `21938286`: `value_samples=32`, `mu=0.55`, n=100.
- Purpose: test whether the apparent n=100 improvement is MC-estimator limited
  or guidance-strength limited before changing the training objective again.
- Results:
  - `value_samples=64`, `mu=0.45`: `1.570 / 1.919 / 1.00`;
  - `value_samples=32`, `mu=0.55`: `1.111 / 1.577 / 0.97`.
- More MC samples did not help; stronger guidance did not beat the original
  step1000 score (`1.107 / 1.513 / 0.98`). The remaining gap is in the learned
  posterior map, not simple MC noise or scalar guidance strength.

Terminal condition-JVP branch:
- The local generated-condition diagonal objective improved the n=100 metric
  but did not scale to n=1000, and local field diagnostics barely moved. The
  next sharper property-agnostic objective is terminal posterior-map
  condition-JVP matching:
  `F_MFM(noise, x_cond + eps d) - F_MFM(noise, x_cond)` vs the same GLASS
  endpoint finite difference.
- `21939514`, output
  `outputs/mfm_random_condition_jvp_dgen4_from_dgen_step1000_20260612`;
  - init: endpoint generated-condition step1000;
  - generated conditions, `generated_condition_steps=128`;
  - diagonal MFM-4 vs GLASS-4;
  - endpoint weight `1.0`, JVP weight `1.0`, cosine weight `0.25`;
  - full t grid `0.1,0.3,0.5,0.7,0.9`.
- An earlier submission (`21939324`) had a truncated t grid due Slurm comma
  export and was cancelled.
- Step250 real alpha/PB scorecard:
  `1.578 / 2.090 / 0.93`.
- This is worse than both the base diagonal checkpoint and generated Stage-1
  step1000. The proxy terminal-JVP loss also did not show a clean improving
  trend, so job `21939514` was cancelled. Checkpoints exist through step750,
  but this branch is deprioritized unless a later diagnostic gives a concrete
  reason to revisit it.

Exact-node diagonal/JVP branch:
- Added `--s-grid`, `--s-grid-jitter`, and `--s-power` to
  `src/mfm_student_train.py`, and wired them through
  `slurm/mfm_diag_forced_rollout.sbatch`.
- Reason: 4-step diagonal posterior sampling queries exact inner nodes
  `s = 0, 0.25, 0.5, 0.75`. Uniform Stage-1 training under-samples the late
  nodes that dominate accumulated condition sensitivity.
- Generated-posterior field diagnostic at the actual 4-step node `s=0.75`
  shows the current base has excellent local velocity but weak condition
  derivative:
  - `t_cond=0.10`: local coord cosine `0.990`, condition-JVP cosine `0.795`,
    rel error `0.84`;
  - `t_cond=0.30`: local coord cosine `0.992`, condition-JVP cosine `0.879`,
    rel error `0.54`;
  - `t_cond=0.50`: local coord cosine `0.994`, condition-JVP cosine `0.943`,
    rel error `0.34`.
- New job `21943769`, output
  `outputs/mfm_gencond_diag_lates_from_dgen_step1000_20260612`;
  - init: generated-condition Stage-1 step1000;
  - `STAGE=1`, generated conditions, `generated_condition_steps=128`;
  - GLASS posterior-state target with `transition_teacher_steps=16`;
  - focused training grid: `t_cond=0.1,0.3,0.5`, `s=0.5,0.75`;
  - weights: coord `8`, atom `0.5`, direction `2`, condition-JVP `0.3`,
    condition-JVP direction `1.0`;
  - saves/evaluates every 250 steps.
- Early log is numerically stable:
  - step50: diagonal coord `0.0205`, coord cosine `0.994`,
    condition-JVP coord loss `1.5245`, condition-JVP cosine `0.112`;
  - step100: diagonal coord `0.0248`, coord cosine `0.995`,
    condition-JVP coord loss `1.3517`, condition-JVP cosine `0.125`.
- Interpretation: ordinary diagonal velocity is still easy, but the exact-node
  condition derivative is the hard missing target. This branch is worth scoring
  at step250/500 because it directly attacks the 4-step sampler nodes.
- Step250 actual alpha/PB scorecard:
  `1.444 / 1.966 / 0.96`.
- Step500 internal PB did not recover:
  - t=0.30 MFM/GLASS PB `0.50 / 0.58`;
  - t=0.60 MFM/GLASS PB `0.67 / 0.75`;
  - t=0.90 MFM/GLASS PB `1.00 / 1.00`.
- Verdict: mixed coordinate+atom condition perturbations with `eps=0.03` are
  too noisy or too blunt. They damage the real scorecard despite preserving
  high ordinary diagonal velocity. Job `21943769` was cancelled after step500.

Coordinate-only exact-node JVP branch:
- Added `--diag-condition-jvp-mode {both,coords,atomics}` and wrapper env
  `DIAG_CONDITION_JVP_MODE`.
- Added wrapper env `DIAG_CONDITION_JVP_EPS` so epsilon is explicit in the job
  log.
- New job `21945717`, output
  `outputs/mfm_gencond_diag_lates_coordjvp_from_dgen_step1000_20260612`;
  - init: generated-condition Stage-1 step1000;
  - same focused grid: `t_cond=0.1,0.3,0.5`, `s=0.5,0.75`;
  - coordinate-only condition perturbation: `DIAG_CONDITION_JVP_MODE=coords`;
  - smaller finite-difference scale: `DIAG_CONDITION_JVP_EPS=0.01`;
  - weights: coord `8`, atom `0.5`, ordinary direction `1`,
    condition-JVP `1`, condition-JVP direction `2`;
  - LR `3e-7`, saves/evaluates every 250 steps.
- Rationale: alpha guidance is coordinate-weak, and the mixed condition
  perturbation couples coordinate and atom-logit sensitivities. This branch asks
  a cleaner question: can the student learn the coordinate conditioning
  derivative at the actual 4-step diagonal nodes without atom-perturbation
  noise?
- Result for the small-epsilon variant (`eps=0.01`, job `21945717`):
  - step50/100/150 losses stayed around `140-148`;
  - condition-JVP coordinate loss was `9.7-20.3`;
  - condition-JVP cosine stayed weak (`-0.146`, `0.102`, `0.146`);
  - cancelled at step150. The small epsilon amplifies finite-difference noise
    too much for this objective.
- Softer coordinate-only variant (`21946737`):
  - output:
    `outputs/mfm_gencond_diag_lates_coordjvp_soft_from_dgen_step1000_20260612`;
  - `eps=0.03`, condition-JVP weight `0.2`, condition-JVP direction `0.5`;
  - early step50 looked better (`cjvpC 1.2777`, `cjvpCos 0.564`), but the
    signal did not persist;
  - step250: `diag_coord 0.0227`, coord cosine `0.993`,
    `cjvpC 1.2171`, `cjvpCos 0.068`;
  - internal PB at step250 was poor at low t:
    t=0.30 MFM/GLASS PB `0.42 / 0.83`.
  - actual alpha/PB scorecard: `1.440 / 1.993 / 0.97`;
  - cancelled after saving step250.

More-integration diagnostic:
- Tested whether current diagonal students recover by using more value-posterior
  integration steps in the real length-conditioned alpha guidance loop.
- Generated-condition Stage-1 step1000 with diagonal-8:
  `1.663 / 2.442 / 0.94`.
- Base spread-4 step3000 with diagonal-8:
  `1.936 / 2.586 / 0.90`.
- Both are worse than their diagonal-4 scores:
  - generated-condition step1000 diagonal-4: `1.107 / 1.513 / 0.98`;
  - base spread-4 step3000 diagonal-4: `1.224 / 1.738 / 0.99`.
- Conclusion: for the current learned fields, more inner integration steps do
  not make the MFM more GLASS-like in the guidance loop. They compound learned
  field/Jacobian errors. The useful target is the exact 4-step diagonal map, not
  simply increasing MFM value steps.

Exact GLASS-4 rollout refinement:
- Job `21947924`, output
  `outputs/mfm_diag4_exactroll_from_dgen_step1000_20260612`;
  - init: generated-condition Stage-1 step1000;
  - `STAGE=4`, diagonal rollout, generated conditions;
  - `rollout_steps=4`, `transition_teacher_steps=4`, so the teacher matches
    the value sampler's GLASS-4 integrator exactly;
  - pure rollout training (`diag_weight=0`), final weight `10`, velocity weight
    `1`, spread weight `0.25`, spread samples `4`.
- Step250 scorecard: `1.286 / 1.798 / 0.96`.
- Step500 scorecard: `1.431 / 1.798 / 0.97`.
- Both are worse than generated Stage-1 step1000 (`1.107 / 1.513 / 0.98`).
- Training diagnosis:
  - endpoint losses sometimes improve, but spread ratio oscillates badly
    (`sprR` reached `2.87` at step300 and `2.92` at step550);
  - pure endpoint-heavy rollout refinement damages the posterior distribution
    even when PB stays superficially acceptable in the actual guided sample.
- Job was cancelled after step500.

Next rollout refinement:
- Ran one gentler exact-GLASS4 branch from the same init:
  - much lower LR;
  - lower final/velocity weights;
  - higher spread weight;
  - small diagonal anchor to prevent local-field drift.
- Job `21951564`, output
  `outputs/mfm_diag4_exactroll_gentle_from_dgen_step1000_20260612`;
  - LR `1e-7`, final weight `2`, velocity weight `0.5`, spread weight `2`,
    `diag_weight=0.5`.
- Step250 actual alpha/PB scorecard: `1.584 / 2.049 / 0.96`.
- Verdict: worse than the aggressive exact-rollout step250 and much worse than
  generated Stage-1 step1000. Cancelled after step250.
- Conclusion: exact rollout refinement from the best checkpoint, at both
  aggressive and gentle settings, does not improve the real guidance score.
  More weight tuning in this family is low priority.

Adapter-only architectural branch:
- Added `--train-conditioning-adapter-only` to `src/mfm_student_train.py` and
  `TRAIN_CONDITIONING_ADAPTER_ONLY_FLAG` to the Slurm wrapper.
- This freezes the existing checkpoint and trains only zero-initialized
  `block_cond_gates` plus gated-delta condition embeddings.
- New job `21954042`, output
  `outputs/mfm_gateddelta_adapter_from_dgen_step1000_20260612`;
  - init: generated Stage-1 step1000;
  - `BLOCK_CONDITIONING=gated_delta`;
  - `STAGE=1`, generated conditions, posterior target steps `16`;
  - adapter-only training, LR `3e-5`, batch `8`;
  - objective weights match the successful dgen Stage-1 setup:
    coord `8`, atom `0.5`, direction `5`, condition-JVP `0.1`,
    condition-JVP direction `0.25`;
  - save/evaluate every 250 steps.
- Rationale: the additive architecture appears to be at its diagonal-4 limit.
  This branch adds conditional capacity while preserving the trained map at
  initialization, instead of perturbing all weights or training noisy
  finite-difference objectives.

Next gate:
- Score adapter-only gated-delta checkpoints at step250/500.
- Only promote a checkpoint if it beats or matches base on all three:
  guide MAE, oracle MAE, and PB valid.

## 2026-06-12 diagonal-4 follow-up

Adapter-only gated-delta results:
- Job `21954042`, output
  `outputs/mfm_gateddelta_adapter_from_dgen_step1000_20260612/stage1_20260612-122420`.
- Actual length-conditioned alpha/PB scorecards:
  - step250: `1.293 / 1.733 / 0.97`;
  - step500: `1.348 / 1.739 / 0.95`;
  - step1000: `1.582 / 1.893 / 0.98`.
- Verdict: adapter-only preserves useful guidance but does not beat the current
  best generated-condition step1000 (`1.107 / 1.513 / 0.98`). Cancelled.

Exact 4-node diagonal branch:
- Job `21956177`, output
  `outputs/mfm_diag4_nodes_from_dgen_step1000_20260612/stage1_20260612-123122`.
- Config: generated conditions, `s={0,0.25,0.5,0.75}`,
  `transition_teacher_steps=4`, no finite-difference condition-JVP.
- Actual scorecards:
  - step250: `1.290 / 1.746 / 0.96`;
  - step750: `1.151 / 1.746 / 0.94`.
- Verdict: exact-node training can improve guide MAE relative to several
  variants, but PB validity and oracle MAE are worse than the current best.
  Step1000 had poor internal mid-t PB, so the branch was cancelled.

Full gated-delta finetune:
- Job `21956162`, output
  `outputs/mfm_gateddelta_full_from_dgen_step1000_20260612/stage1_20260612-123114`.
- Config: `BLOCK_CONDITIONING=gated_delta`, full low-LR finetune from dgen
  step1000, generated diagonal target with small condition-JVP weight.
- Actual scorecards:
  - step250: `1.457 / 1.890 / 0.95`;
  - step500: `1.211 / 1.665 / 0.95`;
  - step750: `1.125 / 1.621 / 0.95`.
- Verdict: this is the best architectural branch from this batch but still does
  not beat dgen step1000 because PB validity and oracle MAE remain worse. It was
  cancelled after step1000 when the internal low/mid-t PB probe degraded.

Length-conditioned factor diagnostic:
- Patched `src/mfm_guidance_factor_diagnostic.py` and
  `slurm/mfm_guidance_factor_diagnostic.sbatch` so the diagnostic can sample the
  same per-length property targets as the scorecard.
- Job `21957998`, output
  `outputs/mfm_guidance_factor_diagnostic/dgen_step1000_lencond_diag4_20260612`.
- Current best dgen step1000, diagonal-4, length-conditioned targets:
  - actual coordinate guidance cosine: `0.176`;
  - same GLASS terminal cotangent through MFM Jacobian: `0.660`;
  - MFM terminal cotangent through GLASS Jacobian: `0.185`;
  - terminal coordinate cotangent cosine: `0.312`;
  - actual atom guidance cosine: `0.063`.
- Interpretation: the failure is not explained by ordinary local diagonal
  velocity alone. The largest error is that MFM posterior endpoints put the
  reward/regressor in a different terminal-cotangent regime; the MFM posterior
  Jacobian is also imperfect, but replacing only the terminal cotangent improves
  coordinate guidance much more than replacing only the Jacobian.

Parameterization tests:
- Direct coordinate head-only branch:
  `outputs/mfm_coorddirect_head_diag4nodes_from_dgen_step1000_20260612`;
  `coord_direct_atom_endpoint`, skipped coord head init, trained only
  `out_coord_linear`.
  - Failed quickly: by step200 `diag_coord` was still about `1.71`, far above
    the endpoint-parametrized diagonal loss. Cancelled.
- Active branch `21959100`:
  `outputs/mfm_coordresid_diag4nodes_from_dgen_step1000_20260612`;
  `coord_endpoint_residual_atom_endpoint`, residual-only training, generated
  exact 4-node diagonal target.
  - Rationale: starts exactly from the current best endpoint model and learns a
    zero-init direct coordinate-velocity correction, so it can test whether a
    small coordinate residual improves the diagonal-4 adjoint without damaging
    atoms/backbone.
  - Results:
    - step250: `1.359 / 1.855 / 0.98`;
    - step500: `1.623 / 2.091 / 0.97`.
  - Verdict: residual-only coordinate correction keeps PB acceptable but makes
    guidance/oracle MAE worse. The terminal-cotangent mismatch is not fixed by a
    small direct coordinate residual trained only with diagonal MSE/direction.
    Cancelled.

Guidance-gradient diagnostic branch:
- Patched `src/mfm_guidance_gradient_finetune.py` and
  `slurm/mfm_guidance_gradient_finetune.sbatch` to support:
  - diagonal MFM posterior integration via `--mfm-diagonal`;
  - length-conditioned property targets matching the real scorecard.
- New job `21961050`, output
  `outputs/mfm_guidance_gradient_lencond_diag4_from_dgen_step1000_20260612`;
  - init: current best generated-condition diagonal checkpoint;
  - teacher/MFM value steps: `4/4`;
  - posterior samples: `8`;
  - property: alpha;
  - length-conditioned targets with `TARGET_SEED=2026`;
  - atom guidance gradients enabled.
- Rationale: the factor diagnostic says the main error is in the actual
  scorecard terminal-cotangent regime. This branch is property-specific as a
  diagnostic, not the final general solution; if it moves the real score, the
  next general version should train on multi-property terminal-cotangent
  families rather than a separate value function per property.

## 2026-06-12 terminal-regime and condition-distribution tests

Guidance-gradient diagnostic result:
- Job `21961050`, output
  `outputs/mfm_guidance_gradient_lencond_diag4_from_dgen_step1000_20260612`.
- Actual length-conditioned alpha/PB scorecards:
  - step25: `1.418 / 1.975 / 0.96`;
  - step75: `1.577 / 2.001 / 0.98`;
  - step100: `1.271 / 1.657 / 0.97`.
- Verdict: second-order property-gradient finetuning is noisy and did not beat
  the current best (`1.107 / 1.513 / 0.98`). It is useful evidence that directly
  matching scorecard gradients is not the first route to pursue.

Property-ensemble alpha-tail branch:
- Patched `src/mfm_property_ensemble_finetune.py` and wrapper support for
  diagonal MFM plus length-conditioned reward targets.
- First run `21963108`, output
  `outputs/mfm_propens_lencond_alpha_diag4_from_dgen_step1000_20260612/propens_20260612-130116`.
- Actual scorecards:
  - step50: `1.633 / 1.998 / 0.93`;
  - step100: `1.567 / 1.976 / 0.91`;
  - step150: `1.507 / 1.994 / 0.94`;
  - step200: `1.151 / 1.659 / 0.95`.
- Continuation from step200, lower LR and stronger reward-tail weight:
  `outputs/mfm_propens_lencond_alpha_diag4_cont_from_prop200_20260612/propens_20260612-131013`.
- Actual scorecards:
  - step100: `1.469 / 1.868 / 0.98`;
  - step200: `1.195 / 1.624 / 0.98`;
  - step300: `1.322 / 1.813 / 0.96`.
- Step200 n=1000 scorecard:
  `1.513 / 2.000 / 0.948`.
- Verdict: terminal reward-distribution alignment can move the real scorecard
  while preserving PB, but it still does not beat the current best. The best
  result in this family is continuation step200, close on PB but worse on guide
  and oracle MAE at n=100 and n=1000. This is a diagnostic branch, not the
  desired general MFM solution.

Guided-condition diagonal branch:
- Patched `src/mfm_student_train.py` and
  `slurm/mfm_diag_forced_rollout.sbatch` with
  `ROLLOUT_CONDITION_SOURCE=guided`.
- The new condition source samples a production trajectory from prior noise,
  applies GLASS value guidance up to the selected `t_cond`, then trains the same
  deterministic GLASS diagonal velocity on that guided intermediate state.
- Job `21964847`, output
  `outputs/mfm_guidedcond_diag4_from_dgen_step1000_20260612/stage1_20260612-130925`.
- Training was easy in ordinary diagonal metrics even on guided states:
  coord cosine stayed about `0.995-1.000`, atom cosine about `0.996-1.000`.
- Actual scorecards:
  - step100: `1.543 / 2.110 / 0.95`;
  - step200: `1.420 / 1.746 / 0.94`;
  - step300: `1.471 / 1.988 / 0.96`.
- Verdict: guided-condition support is not the missing ingredient by itself.
  The model can learn local GLASS diagonal velocities on those states, but the
  real posterior guidance remains worse than the unguided-condition diagonal
  baseline. This strengthens the conclusion that ordinary local diagonal
  velocity cosine is too weak a success criterion for the value-guidance task.

Endpoint-space diagonal branch:
- Current best generated-condition checkpoint did not use
  `--endpoint-space-loss`:
  `endpoint_space_loss=False`, `diag_loss_type=mse`,
  `diag_direction_weight=5`, generated conditions, teacher posterior steps
  `16`.
- Added `ENDPOINT_SPACE_LOSS_FLAG` to
  `slurm/mfm_diag_forced_rollout.sbatch`.
- Active job `21967804`, output
  `outputs/mfm_diag4_endpointspace_from_dgen_step1000_20260612`.
- Config:
  - init: current best dgen step1000;
  - generated conditions, exact 4 nodes `s={0,0.25,0.5,0.75}`;
  - `--endpoint-space-loss`, coord weight `8`, atom weight `0.5`,
    direction weight `8`;
  - checkpoint/eval every 250 steps.
- Rationale: the endpoint-parametrized diagonal velocity is an endpoint
  estimate divided by `1-s`; endpoint-space loss asks for the endpoint map used
  by the 4-step sampler to be correct directly, rather than only the scaled
  velocity.
- Result:
  - step250 actual scorecard: `1.443 / 1.910 / 0.96`;
  - internal step500 low-`t_cond` PB degraded to `0.28`.
- Verdict: endpoint-space weighting is not the missing ingredient. It keeps
  ordinary local diagonal metrics strong, but worsens the real guidance gate.
  Job `21967804` was cancelled.

Multi-property terminal ensemble branch:
- New job `21969413`, output
  `outputs/mfm_propens_multiprop_diag4_from_dgen_step1000_20260612`.
- Config:
  - init: current best dgen step1000;
  - diagonal MFM and GLASS, 4 value steps each;
  - generated conditions, posterior samples `8`;
  - properties: `alpha,gap,homo,lumo,mu,cv`, two sampled per step;
  - no reward-tail weighting; endpoint loss weight `0.5`, property loss weight
    `0.5`.
- Rationale: property-tail alpha helps somewhat but is too property-specific
  and does not beat the baseline. This branch asks whether a broader terminal
  regressor-regime match can repair the MFM posterior without overfitting to one
  target guidance task.
- Actual scorecards:
  - step100: `1.690 / 2.042 / 0.95`;
  - step200: `1.353 / 1.962 / 0.93`;
  - step300: `1.265 / 1.807 / 0.99`.
- Verdict: broad terminal-property ensemble matching can preserve or improve PB
  by step300, but it still does not close the alpha guidance gap. It is worse
  than the current best on both guide and oracle MAE.

Split-gated architecture branch:
- New job `21969984`, output
  `outputs/mfm_splitgated_diag_from_dgen_step1000_20260612/stage1_20260612-132824`.
- Config:
  - init: current best dgen step1000, non-strict load;
  - `BLOCK_CONDITIONING=split_gated`, `ENCODER_DEPTH=4`;
  - original successful generated-condition diagonal recipe, low LR `1e-7`.
- Result:
  - non-strict load added 8 missing parameters; student size increased from
    about `363k` to `397k`;
  - local diagonal losses were worse than the baseline by step200
    (`diag_coord 0.036`, `diag_atom 0.018`);
  - internal step250 PB at `t_cond=0.30/0.60/0.90` was `0.34/0.56/0.97`.
- Verdict: split-gated does not recover the diagonal from the current checkpoint
  and is not worth a full scorecard. Job cancelled.

## 2026-06-12 diagonal integrator stability and adjoint tests

Random posterior-adjoint branch:
- Patched `slurm/mfm_random_adjoint.sbatch` so the default mode is now
  `--diagonal`; the old wrapper defaulted to `--offdiagonal`, which was wrong
  for the current diagonal-4 objective.
- Job `21971850`, output
  `outputs/mfm_random_adjoint_diag4_from_dgen_step1000_20260612/randadj_20260612-133709`.
- Config:
  - init: current best generated-condition diagonal checkpoint;
  - generated conditions, `glass_steps=4`, `mfm_steps=4`;
  - random terminal cotangents, full posterior VJP matching
    `(dF/dx_cond)^T v`;
  - `LR=3e-8`, batch `1`, endpoint weight `0.25`,
    adjoint weight `1.0`, coord/atom weights `8.0/0.5`.
- Training behavior:
  - early windows were numerically sane, with coordinate adjoint cosine around
    `0.94-0.95` and atom adjoint cosine around `0.7`;
  - later windows were noisy and sometimes had endpoint spikes or bad atom
    adjoint windows.
- Actual length-conditioned alpha/PB scorecards:
  - step50: `1.285 / 1.830 / 0.98`;
  - step100: `1.364 / 1.915 / 0.95`.
- Verdict: matching random full-posterior adjoints is directionally relevant
  but does not beat the current best (`1.107 / 1.513 / 0.98`). It also drifts
  quickly, so blindly scaling this exact loss is not justified.

Value-step stability diagnostic on the current best:
- Current best checkpoint:
  `outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612/stage1_20260612-112012/student_step_1000.pt`.
- Actual length-conditioned alpha/PB scorecards with more diagonal MFM value
  steps:
  - diagonal 4-step: `1.107 / 1.513 / 0.98`;
  - diagonal 8-step: `1.617 / 2.642 / 0.94`;
  - diagonal 16-step: `2.979 / 4.097 / 0.83`.
- Interpretation: the learned "diagonal" is not a stable continuous GLASS
  velocity field. Extra integration steps expose accumulated field error rather
  than reducing discretization error. This also explains why ordinary local
  diagonal cosine is misleading: the endpoint-parametrized model can be useful
  as a coarse 4-step fitted map while failing as an integrator.

Architecture implication:
- `TabascoFlowMap` default `velocity_parametrization=endpoint` predicts an
  endpoint and returns `(endpoint - x) / (1 - s)`.
- This parametrization amplifies small endpoint errors near late inner times.
  The 8/16-step collapse is consistent with late-time endpoint error being
  converted into a bad velocity.
- The next architecture gate is therefore not more endpoint-space weighting,
  which already failed, but a direct-coordinate velocity model that can behave
  as a continuous field.

Coord-direct checkpoint gate:
- Existing dedicated Stage-1 checkpoint:
  `outputs/mfm_student_coord_direct_atom_endpoint_continue/stage1_20260604-155227/student_step_10000.pt`.
- It uses `velocity_parametrization=coord_direct_atom_endpoint`; older internal
  16-step posterior diagnostics showed good coordinate spread and pair RMSD,
  but weak PB in small probes.
- Queued actual length-conditioned alpha/PB scorecards for this checkpoint:
  - job `21985212`: diagonal 4-step;
  - job `21985215`: diagonal 8-step;
  - job `21985217`: diagonal 16-step.
- Decision rule: if this direct-coordinate model degrades less severely with
  8/16 value steps than the endpoint-parametrized model, continue from it with
  generated-condition diagonal training plus the terminal-cotangent diagnostics.
  If it also fails the real gate, the issue is not just endpoint
  parametrization and the next branch should train the full generated-condition
  posterior map with property-cotangent or targeted adjoint losses.

Generated-condition coord-direct continuation:
- Queued job `21985546`, output
  `outputs/mfm_coorddirect_gencond_stage1_from_cd10k_20260612`.
- Config:
  - init:
    `outputs/mfm_student_coord_direct_atom_endpoint_continue/stage1_20260604-155227/student_step_10000.pt`;
  - `velocity_parametrization=coord_direct_atom_endpoint`;
  - Stage-1 generated-condition diagonal GLASS target;
  - generated production states from 128-step teacher sampler;
  - posterior path target steps `16`;
  - `LR=1e-5`, batch `8`, coord/atom weights `8.0/0.5`,
    direction weight `5`, condition-JVP weights `0.1/0.25`;
  - checkpoint/internal eval every 250 steps.
- Rationale: this is the first direct-coordinate continuation that starts from
  a properly trained direct-coordinate diagonal checkpoint and adapts it to the
  generated-condition distribution. Earlier coord-direct branches either loaded
  an endpoint-param checkpoint into a direct head or trained only the direct
  coordinate head, both of which produced high diagonal loss and poor PB.

## 2026-06-12 all-property length-conditioned MFM scorecard

Setup:
- Checkpoint:
  `outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612/stage1_20260612-112012/student_step_1000.pt`.
- Original flawed run: `n=1000`, length-conditioned targets,
  `sample_steps=128`, `mu=0.45`, reward scale `0.3`, guide every step, atom
  guidance enabled.
- MFM value sampler: diagonal, `value_steps=4`, `value_samples=32`.
- PB enabled for every run.

Unit correction:
- The TFG QM9 loader stores `homo`, `lumo`, and `gap` directly from the
  dataset, and the classifier/scorecard losses are raw L1 losses in those
  units. The cached QM9 values are Hartree-scale (`homo` mean about `-0.240`,
  `gap` mean about `0.251`), not eV.
- Correct conversion for orbital/gap MAE is `1 Ha = 27211.386 meV`. Earlier
  discussion that treated `raw * 1000` as meV was wrong.
- Evaluation correction: this all-property MFM table is not a fair comparison
  to the historical GLASS property scorecard. The GLASS runs used
  property-normalized reward scales (`homo=41500`, `lumo=9088`, `gap=8925`,
  `mu=8.638`, `cv=1.209`, `alpha=0.3`), while this MFM table used
  `reward_scale=0.3` for every property. The energy MFM gradients were therefore
  suppressed by four to five orders of magnitude.
- Cache correction: the default histogram cache path was hard-coded to
  `alpha_train_flow_bins100.json`. Non-alpha runs could overwrite that file;
  at diagnosis time the alpha cache contained `gap` metadata. Patched
  `src/qm9_length_conditioned_guidance_experiment.py` to redirect the default
  cache to `{property}_{partition}_bins{bins}.json` and rebuilt the alpha cache.

| property | unit | unguided guide/oracle MAE | MFM guided guide/oracle MAE | PB valid guided | guided - unguided guide MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| alpha | raw | `9.291 / 9.098` | `1.451 / 1.983` | `0.943` | `-7.840` |
| homo | Ha / meV | `0.02832 / 0.02718` (`770.7 / 739.7`) | `0.02836 / 0.02717` (`771.7 / 739.5`) | `0.983` | `+0.00003 Ha` (`+0.9 meV`) |
| lumo | Ha / meV | `0.05571 / 0.05566` (`1515.9 / 1514.6`) | `0.05537 / 0.05553` (`1506.6 / 1510.9`) | `0.982` | `-0.00034 Ha` (`-9.3 meV`) |
| mu | raw | `1.664 / 1.699` | `1.206 / 1.308` | `0.982` | `-0.458` |
| cv | raw | `3.477 / 3.485` | `1.390 / 1.561` | `0.979` | `-2.087` |
| gap | Ha / meV | `0.07260 / 0.07033` (`1975.5 / 1913.9`) | `0.07258 / 0.07027` (`1974.9 / 1912.0`) | `0.983` | `-0.00002 Ha` (`-0.6 meV`) |

Comparison to historical GLASS n=1000 length-conditioned guided scorecard:
- GLASS guided values recorded previously:
  - alpha: `1.33 / 1.81`, PB `0.936`;
  - cv: `0.521 / 0.670`, PB `0.944`;
  - mu: `0.648 / 0.827`, PB `0.883`;
  - gap: `0.0223 / 0.0272 Ha` (`607 / 740 meV`), PB `0.873`;
  - homo: `0.0077 / 0.0105 Ha` (`210 / 286 meV`), PB `0.866`;
  - lumo: `0.0144 / 0.0183 Ha` (`392 / 498 meV`), PB `0.910`.
- Interpretation:
  - The raw-unit energy errors are hundreds to about two thousand meV, not tens
    of meV.
  - The previous conclusion that MFM does not move `gap/homo/lumo` is not yet
    valid, because those MFM runs used the wrong reward scale.
  - The correct next gate is MFM with the historical GLASS per-property
    `(mu, reward_scale)` settings.

Corrected jobs launched on 2026-06-19:
- MFM n=100 scorecards, same checkpoint, diagonal-4, `value_samples=32`,
  `sample_steps=128`, PB enabled:
  - `gap`: job `23543402`,
    `mu=0.15`, `reward_scale=8925`,
    output `outputs/qm9_length_conditioned_guidance/mfm_len_gap_n100_s128_mu015_rs8925_vs32_diag4_dgen_step1000_pb_fix`.
  - `homo`: job `23543434`,
    `mu=0.35`, `reward_scale=41500`,
    output `outputs/qm9_length_conditioned_guidance/mfm_len_homo_n100_s128_mu035_rs41500_vs32_diag4_dgen_step1000_pb_fix`.
  - `lumo`: job `23543433`,
    `mu=0.25`, `reward_scale=9088`,
    output `outputs/qm9_length_conditioned_guidance/mfm_len_lumo_n100_s128_mu025_rs9088_vs32_diag4_dgen_step1000_pb_fix`.
  - `mu`: job `23543435`,
    `mu=0.35`, `reward_scale=8.638`,
    output `outputs/qm9_length_conditioned_guidance/mfm_len_mu_n100_s128_mu035_rs8638_vs32_diag4_dgen_step1000_pb_fix`.
  - `cv`: job `23543401`,
    `mu=0.70`, `reward_scale=1.209`,
    output `outputs/qm9_length_conditioned_guidance/mfm_len_cv_n100_s128_mu070_rs1209_vs32_diag4_dgen_step1000_pb_fix`.
- Corrected factor diagnostics, same checkpoint, diagonal-4,
  `value_samples=8`, length-conditioned targets:
  - `gap`: job `23543484`,
    output `outputs/mfm_guidance_factor_diagnostic/mfm_factor_gap_len_rs8925_diag4_dgen_step1000_20260619`.
  - `homo`: job `23543486`,
    output `outputs/mfm_guidance_factor_diagnostic/mfm_factor_homo_len_rs41500_diag4_dgen_step1000_20260619`.
  - `lumo`: job `23543485`,
    output `outputs/mfm_guidance_factor_diagnostic/mfm_factor_lumo_len_rs9088_diag4_dgen_step1000_20260619`.

Corrected n=100 scorecard results:

| property | mu | reward scale | unguided guide/oracle MAE | MFM guided guide/oracle MAE | PB valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| gap | `0.15` | `8925` | `0.07151 / 0.07021 Ha` (`1946 / 1910 meV`) | `0.01407 / 0.02403 Ha` (`383 / 654 meV`) | `0.96 -> 0.91` |
| homo | `0.35` | `41500` | `0.02683 / 0.02658 Ha` (`730 / 723 meV`) | `0.00652 / 0.00893 Ha` (`177 / 243 meV`) | `0.96 -> 0.95` |
| lumo | `0.25` | `9088` | `0.05410 / 0.05336 Ha` (`1472 / 1452 meV`) | `0.01197 / 0.01478 Ha` (`326 / 402 meV`) | `0.96 -> 0.97` |
| mu | `0.35` | `8.638` | `1.401 / 1.421` | `0.767 / 0.871` | `0.96 -> 0.95` |
| cv | `0.70` | `1.209` | `3.300 / 3.305` | `0.621 / 0.760` | `0.96 -> 0.97` |

Immediate conclusion:
- With the historical GLASS reward scales, the current diagonal MFM does improve
  all tested non-alpha properties at `n=100`.
- Energy guidance is no longer "dead"; the previous table was primarily a
  reward-scale/configuration bug.
- Remaining question is whether the same result holds at `n=1000` and how it
  compares to GLASS at the same large-sample gate.

Corrected factor diagnostic results:

| property | actual coord cos | same-terminal coord cos | same-jacobian coord cos | terminal-cot coord cos | actual atom cos | same-terminal atom cos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gap | `0.183` | `0.679` | `0.217` | `0.228` | `0.154` | `0.663` |
| homo | `0.301` | `0.590` | `0.357` | `0.283` | `0.155` | `0.512` |
| lumo | `0.194` | `0.638` | `0.195` | `0.209` | `0.161` | `0.582` |

Interpretation of the factor diagnostic:
- The corrected scorecard can work even though local GLASS-vs-MFM guidance
  cosine remains modest.
- The same-terminal-cotangent numbers are much better than the actual numbers,
  so the remaining architectural weakness is still endpoint/reward-cotangent
  fidelity more than raw local diagonal velocity alone.
- This supports continuing diagonal training, but with explicit endpoint
  property/cotangent consistency in the generated-condition regime.

Corrected n=1000 jobs launched:
- `homo`: job `23544130`,
  output `outputs/qm9_length_conditioned_guidance/mfm_len_homo_n1000_s128_mu035_rs41500_vs32_diag4_dgen_step1000_pb_fix`.
- `lumo`: job `23544129`,
  output `outputs/qm9_length_conditioned_guidance/mfm_len_lumo_n1000_s128_mu025_rs9088_vs32_diag4_dgen_step1000_pb_fix`.
- `gap`: job `23544132`,
  output `outputs/qm9_length_conditioned_guidance/mfm_len_gap_n1000_s128_mu015_rs8925_vs32_diag4_dgen_step1000_pb_fix`.
- `mu`: job `23544131`,
  output `outputs/qm9_length_conditioned_guidance/mfm_len_mu_n1000_s128_mu035_rs8638_vs32_diag4_dgen_step1000_pb_fix`.
- `cv`: job `23544133`,
  output `outputs/qm9_length_conditioned_guidance/mfm_len_cv_n1000_s128_mu070_rs1209_vs32_diag4_dgen_step1000_pb_fix`.

Corrected n=1000 comparison to historical GLASS:

| property | MFM guided guide/oracle MAE | MFM PB | GLASS guided guide/oracle MAE | GLASS PB |
| --- | ---: | ---: | ---: | ---: |
| alpha | `1.451 / 1.983` | `0.943` | `1.331 / 1.810` | `0.936` |
| gap | `0.01764 / 0.02491 Ha` (`480 / 678 meV`) | `0.921` | `0.02234 / 0.02718 Ha` (`608 / 740 meV`) | `0.873` |
| homo | `0.00769 / 0.01044 Ha` (`209 / 284 meV`) | `0.926` | `0.00766 / 0.01054 Ha` (`208 / 287 meV`) | `0.866` |
| lumo | `0.01124 / 0.01656 Ha` (`306 / 450 meV`) | `0.936` | `0.01438 / 0.01835 Ha` (`391 / 499 meV`) | `0.910` |
| mu | `0.798 / 0.959` | `0.934` | `0.648 / 0.827` | `0.883` |
| cv | `0.536 / 0.739` | `0.962` | `0.521 / 0.670` | `0.944` |

Conclusion after corrected scaling:
- MFM is already competitive with GLASS on the length-conditioned n=1000
  property scorecard.
- MFM is clearly better than the historical GLASS run for `gap` and `lumo` on
  both guide/oracle MAE and PB validity, essentially tied for `homo`, and close
  for `cv`.
- `alpha` and `mu` remain the main residual gaps. They improve strongly, but
  GLASS still has lower MAE.

## 2026-06-19 tuned MFM inference sweeps

Question:
- Can the current diagonal MFM be brought to GLASS-level property guidance by
  inference tuning alone, once the reward-scale and unit issues are fixed?

Procedure:
- Ran `n=100` sweeps over property-specific `mu` and `reward_scale` for the
  current diagonal-4 MFM value sampler.
- Reran the best or most plausible settings at `n=1000` with PB validity.
- Kept energy MAE in Hartree internally and reported `gap/homo/lumo` in meV
  with `1 Ha = 27211.386245988 meV`.
- Canonical reproduction note:
  `docs/qm9_mfm_corrected_guidance_scorecard.md`.

Tuned n=1000 result:

| property | tuned MFM setting | tuned MFM guide/oracle | MFM PB | GLASS guide/oracle | GLASS PB |
| --- | --- | ---: | ---: | ---: | ---: |
| alpha | `mu=0.60, rs=0.45` | `1.165 / 1.674` | `0.939` | `1.331 / 1.810` | `0.936` |
| gap | `mu=0.15, rs=12000` | `428 / 583 meV` | `0.888` | `608 / 740 meV` | `0.873` |
| homo | `mu=0.35, rs=41500` | `209 / 284 meV` | `0.926` | `208 / 287 meV` | `0.866` |
| lumo | `mu=0.25, rs=12000` | `289 / 440 meV` | `0.924` | `391 / 499 meV` | `0.910` |
| mu | `mu=0.55, rs=8.638` | `0.643 / 0.822` | `0.920` | `0.648 / 0.827` | `0.883` |
| cv | `mu=0.80, rs=1.5` | `0.449 / 0.652` | `0.948` | `0.521 / 0.670` | `0.944` |

Interpretation:
- The corrected/tuned MFM now beats or matches the historical GLASS scorecard
  across the all-property length-conditioned benchmark.
- The original diagnosis that "MFM does not move the energy properties" was a
  configuration bug: wrong reward scales plus wrong energy-unit reporting.
- The remaining real research question is no longer whether this checkpoint can
  produce useful posterior guidance. It can. The next question is whether we can
  make the MFM integrator intrinsically better so it needs less inference tuning
  and gives a more stable PB/MAE tradeoff.

Follow-up launched:
- Script: `slurm/submit_mfm_guidance_followup2_20260619.sh`.
- Jobs: `23557166` through `23557193`.
- These are `n=100` probes, not the final scorecard:
  - alpha interpolation around the large-gate optimum;
  - homo local sweeps around the only non-clearly-better GLASS row;
  - `value_samples=64` checks for alpha/gap/homo/lumo/mu/cv;
  - small local `mu`/`reward_scale` probes for `mu` and `cv`.

Follow-up `n=100` results:

| property | setting | value samples | guide/oracle | PB |
| --- | --- | ---: | ---: | ---: |
| alpha | `mu=0.60, rs=0.45` | `32` | `0.878 / 1.373` | `0.960` |
| alpha | `mu=0.55, rs=0.50` | `32` | `1.033 / 1.612` | `0.920` |
| alpha | `mu=0.50, rs=0.50` | `32` | `1.152 / 1.554` | `0.940` |
| alpha | `mu=0.55, rs=0.45` | `64` | `1.164 / 1.572` | `0.920` |
| alpha | `mu=0.525, rs=0.45` | `32` | `1.268 / 1.740` | `0.920` |
| gap | `mu=0.15, rs=14000` | `32` | `372 / 545 meV` | `0.870` |
| gap | `mu=0.15, rs=12000` | `64` | `447 / 675 meV` | `0.900` |
| homo | `mu=0.40, rs=55000` | `32` | `203 / 242 meV` | `0.900` |
| homo | `mu=0.40, rs=50000` | `32` | `205 / 282 meV` | `0.930` |
| homo | `mu=0.35, rs=41500` | `64` | `224 / 296 meV` | `0.960` |
| lumo | `mu=0.25, rs=14000` | `32` | `293 / 440 meV` | `0.960` |
| lumo | `mu=0.25, rs=12000` | `64` | `303 / 460 meV` | `0.960` |
| mu | `mu=0.60, rs=8.638` | `32` | `0.597 / 0.747` | `0.910` |
| mu | `mu=0.55, rs=8.638` | `64` | `0.605 / 0.909` | `0.910` |
| mu | `mu=0.55, rs=10.0` | `32` | `0.630 / 0.735` | `0.920` |
| cv | `mu=0.80, rs=1.5` | `64` | `0.446 / 0.620` | `0.960` |
| cv | `mu=0.85, rs=1.5` | `32` | `0.489 / 0.610` | `0.960` |
| cv | `mu=0.75, rs=1.5` | `32` | `0.509 / 0.639` | `0.970` |

Large validation launched:
- Script: `slurm/submit_mfm_guidance_followup2_winners_20260619.sh`.
- Jobs: `23557351` through `23557357`.
- Scaled candidates:
  - `alpha`: `mu=0.60, rs=0.45`, `value_samples=32`;
  - `alpha`: `mu=0.50, rs=0.50`, `value_samples=32`;
  - `homo`: `mu=0.40, rs=55000`, `value_samples=32`;
  - `homo`: `mu=0.40, rs=50000`, `value_samples=32`;
  - `mu`: `mu=0.60, rs=8.638`, `value_samples=32`;
  - `mu`: `mu=0.55, rs=10.0`, `value_samples=32`;
  - `cv`: `mu=0.80, rs=1.5`, `value_samples=64`.

Large validation results:

| property | setting | value samples | guide/oracle | PB | decision |
| --- | --- | ---: | ---: | ---: | --- |
| alpha | `mu=0.60, rs=0.45` | `32` | `1.165 / 1.674` | `0.939` | new best |
| alpha | `mu=0.50, rs=0.50` | `32` | `1.205 / 1.761` | `0.928` | worse PB/MAE tradeoff |
| homo | `mu=0.40, rs=55000` | `32` | `217 / 278 meV` | `0.904` | worse than current |
| homo | `mu=0.40, rs=50000` | `32` | `216 / 276 meV` | `0.909` | worse than current |
| mu | `mu=0.60, rs=8.638` | `32` | `0.692 / 0.867` | `0.924` | worse than current |
| mu | `mu=0.55, rs=10.0` | `32` | `0.730 / 0.970` | `0.912` | worse than current |

CV `value_samples=64`, `48`, and `40` all OOM at `n=1000` on the current
all-at-once value-gradient path. The practical CV row remains
`mu=0.80, rs=1.5, value_samples=32` unless we add chunked value-gradient
evaluation.

Chunked value-gradient implementation:
- Added `--value-batch-size` to `glass_guided_sampling_experiment.py` and
  `qm9_length_conditioned_guidance_experiment.py`.
- Added `VALUE_BATCH_SIZE` to `slurm/qm9_length_conditioned_guidance.sbatch`.
- The chunking splits the original conditioning batch and keeps the requested
  posterior sample count per molecule. With fixed prior noise, a synthetic MFM
  test matched unchunked gradients exactly (`coord_maxdiff=0.0`,
  `atom_maxdiff=0.0`).
- CV retry:
  - job `23564314`, completed in `00:07:06`;
  - `value_samples=64`, `VALUE_BATCH_SIZE=500`;
  - output
    `outputs/qm9_length_conditioned_guidance/mfm_follow2win_cv_n1000_s128_m080_rs1500_vs64_vbs500_diag4_dgen_step1000_pb`;
  - guide/oracle `0.485 / 0.666`, PB `0.960`.
- Decision: chunking fixes the OOM, but `value_samples=64` is worse than the
  current `value_samples=32` CV row (`0.449 / 0.652`, PB `0.948`). Keep
  `value_samples=32` in the main scorecard.

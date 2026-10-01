# QM9 MFM Corrected Guidance Scorecard

Date: 2026-06-19

This note records the corrected and tuned length-conditioned QM9
property-guidance scorecard for the diagonal MFM posterior sampler, plus the
exact reproduction commands. It is intended as the clean reference for a future
GitHub push.

## Main Result

Setup:
- Dataset/targets: QM9 train-flow length-conditioned property histograms.
- DFM generator:
  `outputs/qm9_dfm_long/qm9_dfm_trainflow_long100k_20260531/checkpoints/model_step_100000.pt`
- MFM checkpoint:
  `outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612/stage1_20260612-112012/student_step_1000.pt`
- Sampling: `n=1000`, `sample_steps=128`, guide every step, atom guidance on,
  PB validity enabled.
- MFM value sampler: diagonal, `value_steps=4`, `value_samples=32`.
- GLASS reference: `value_steps=4`, `value_samples=4`.
- Tuned MFM settings were selected by `n=100` inference sweeps and then rerun at
  `n=1000`.

Energy rows in the raw JSON/CSV summaries are Hartree. The table below reports
energy MAE in meV using `1 Ha = 27211.386245988 meV`.

| property | tuned MFM setting | tuned MFM guide/oracle | MFM PB | GLASS guide/oracle | GLASS PB |
| --- | --- | ---: | ---: | ---: | ---: |
| alpha | `mu=0.60, rs=0.45` | `1.165 / 1.674` | `0.939` | `1.331 / 1.810` | `0.936` |
| gap | `mu=0.15, rs=12000` | `428 / 583 meV` | `0.888` | `608 / 740 meV` | `0.873` |
| homo | `mu=0.35, rs=41500` | `209 / 284 meV` | `0.926` | `208 / 287 meV` | `0.866` |
| lumo | `mu=0.25, rs=12000` | `289 / 440 meV` | `0.924` | `391 / 499 meV` | `0.910` |
| mu | `mu=0.55, rs=8.638` | `0.643 / 0.822` | `0.920` | `0.648 / 0.827` | `0.883` |
| cv | `mu=0.80, rs=1.5` | `0.449 / 0.652` | `0.948` | `0.521 / 0.670` | `0.944` |

Raw Hartree energy rows:

| property | unguided guide/oracle | tuned MFM guide/oracle | GLASS guided guide/oracle |
| --- | ---: | ---: | ---: |
| gap | `0.072584 / 0.070349` | `0.015726 / 0.021440` | `0.022335 / 0.027179` |
| homo | `0.028324 / 0.027189` | `0.007691 / 0.010440` | `0.007657 / 0.010540` |
| lumo | `0.055720 / 0.055667` | `0.010636 / 0.016157` | `0.014380 / 0.018349` |

Conclusion:
- With corrected per-property reward scales and light inference tuning, the
  current diagonal MFM beats or matches the historical GLASS reference on every
  property in the n=1000 scorecard.
- MFM beats GLASS on `alpha`, `gap`, `lumo`, `mu`, and `cv` for both guide and
  oracle MAE while retaining higher PB validity.
- `homo` is essentially tied on guide MAE, slightly better on oracle MAE, and
  much better on PB validity.

## Critical Corrections

Two issues caused the earlier all-property MFM table to look wrong:

1. Energy units:
   - QM9 `homo`, `lumo`, and `gap` are raw Hartree values in this pipeline.
   - Correct conversion is `1 Ha = 27211.386245988 meV`.
   - A raw `0.028 Ha` HOMO error is about `762 meV`, not `28 meV`.

2. Reward scale:
   - The strong GLASS property scorecard used per-property reward scales.
   - The bad all-property MFM table used `reward_scale=0.3` for every property,
     suppressing energy guidance by four to five orders of magnitude.

Correct reward settings:

| property | mu | reward_scale |
| --- | ---: | ---: |
| alpha | `0.45` | `0.3` |
| gap | `0.15` | `8925` |
| homo | `0.35` | `41500` |
| lumo | `0.25` | `9088` |
| mu | `0.35` | `8.638` |
| cv | `0.70` | `1.209` |

The histogram cache routing was also fixed in
`src/qm9_length_conditioned_guidance_experiment.py`: if the caller leaves the
default hist-cache path, the script now routes non-alpha properties to
`cache/qm9_property_histograms/{property}_{partition}_bins{bins}.json`.
This prevents `gap/homo/lumo` runs from overwriting
`alpha_train_flow_bins100.json`.

## Existing Output Directories

Tuned MFM runs:

| property | output directory |
| --- | --- |
| alpha | `outputs/qm9_length_conditioned_guidance/mfm_follow2win_alpha_n1000_s128_m060_rs045_vs32_diag4_dgen_step1000_pb` |
| gap | `outputs/qm9_length_conditioned_guidance/mfm_winner_gap_n1000_s128_m015_rs12000_vs32_diag4_dgen_step1000_pb` |
| homo | `outputs/qm9_length_conditioned_guidance/mfm_len_homo_n1000_s128_mu035_rs41500_vs32_diag4_dgen_step1000_pb_fix` |
| lumo | `outputs/qm9_length_conditioned_guidance/mfm_winner_lumo_n1000_s128_m025_rs12000_vs32_diag4_dgen_step1000_pb` |
| mu | `outputs/qm9_length_conditioned_guidance/mfm_winner_mu_n1000_s128_m055_rs8638_vs32_diag4_dgen_step1000_pb` |
| cv | `outputs/qm9_length_conditioned_guidance/mfm_winner_cv_n1000_s128_m080_rs1500_vs32_diag4_dgen_step1000_pb` |

Earlier corrected MFM baselines:

| property | output directory |
| --- | --- |
| alpha | `outputs/qm9_length_conditioned_guidance/mfm_len_alpha_n1000_s128_mu045_vs32_diag4_dgen_step1000_pb_slurm` |
| gap | `outputs/qm9_length_conditioned_guidance/mfm_len_gap_n1000_s128_mu015_rs8925_vs32_diag4_dgen_step1000_pb_fix` |
| homo | `outputs/qm9_length_conditioned_guidance/mfm_len_homo_n1000_s128_mu035_rs41500_vs32_diag4_dgen_step1000_pb_fix` |
| lumo | `outputs/qm9_length_conditioned_guidance/mfm_len_lumo_n1000_s128_mu025_rs9088_vs32_diag4_dgen_step1000_pb_fix` |
| mu | `outputs/qm9_length_conditioned_guidance/mfm_len_mu_n1000_s128_mu035_rs8638_vs32_diag4_dgen_step1000_pb_fix` |
| cv | `outputs/qm9_length_conditioned_guidance/mfm_len_cv_n1000_s128_mu070_rs1209_vs32_diag4_dgen_step1000_pb_fix` |

GLASS reference runs:

| property | output directory |
| --- | --- |
| alpha | `outputs/qm9_length_conditioned_guidance/n1000_mu045_s128` |
| gap | `outputs/qm9_length_conditioned_guidance/propc_gap_mu015_s128_n1000` |
| homo | `outputs/qm9_length_conditioned_guidance/propc_homo_mu035_s128_n1000` |
| lumo | `outputs/qm9_length_conditioned_guidance/propc_lumo_mu025_s128_n1000` |
| mu | `outputs/qm9_length_conditioned_guidance/propc_mu_mu035_s128_n1000` |
| cv | `outputs/qm9_length_conditioned_guidance/propc_cv_mu070_s128_n1000` |

## Reproduction

Run from the repository root:

```bash
cd /n/netscratch/kozinsky_lab/Lab/uunneberg/repos/tabasco
```

The Slurm script used for all runs is:

```bash
slurm/qm9_length_conditioned_guidance.sbatch
```

The sweep submission scripts are:

```bash
slurm/submit_mfm_guidance_sweeps_20260619.sh
slurm/submit_mfm_guidance_winners_20260619.sh
slurm/submit_mfm_alpha_followup_20260619.sh
slurm/submit_mfm_guidance_followup2_20260619.sh
slurm/submit_mfm_guidance_followup2_winners_20260619.sh
```

To reproduce the exact tuned MFM table directly:

```bash
DFM_CKPT=outputs/qm9_dfm_long/qm9_dfm_trainflow_long100k_20260531/checkpoints/model_step_100000.pt
MFM_CKPT=outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612/stage1_20260612-112012/student_step_1000.pt

COMMON_MFM="CHECKPOINT=${DFM_CKPT},NUM_SAMPLES=1000,SEED=8,TARGET_SEED=2026,VALUE_SAMPLES=32,VALUE_GLASS_STEPS=4,VALUE_SAMPLER=mfm,MFM_CHECKPOINT=${MFM_CKPT},MFM_DIAGONAL_FLAG=--mfm-diagonal,SAMPLE_STEPS=128,GUIDE_EVERY=1,GUIDE_MIN_T=0.05,GUIDE_MAX_T=0.95,SOFT_ATOM_TEMPERATURE=0.25,COMPUTE_PB_FLAG=--compute-pb,GUIDE_ATOMICS_FLAG=--guide-atomics"

sbatch --job-name=mfm_alpha_best --export=ALL,${COMMON_MFM},RUN_NAME=mfm_follow2win_alpha_n1000_s128_m060_rs045_vs32_diag4_dgen_step1000_pb,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_follow2win_alpha_n1000_s128_m060_rs045_vs32_diag4_dgen_step1000_pb,PROPERTY_NAME=alpha,MU=0.60,REWARD_SCALE=0.45 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=mfm_gap_best --export=ALL,${COMMON_MFM},RUN_NAME=mfm_winner_gap_n1000_s128_m015_rs12000_vs32_diag4_dgen_step1000_pb,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_winner_gap_n1000_s128_m015_rs12000_vs32_diag4_dgen_step1000_pb,PROPERTY_NAME=gap,MU=0.15,REWARD_SCALE=12000 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=mfm_homo_best --export=ALL,${COMMON_MFM},RUN_NAME=mfm_len_homo_n1000_s128_mu035_rs41500_vs32_diag4_dgen_step1000_pb_fix,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_len_homo_n1000_s128_mu035_rs41500_vs32_diag4_dgen_step1000_pb_fix,PROPERTY_NAME=homo,MU=0.35,REWARD_SCALE=41500 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=mfm_lumo_best --export=ALL,${COMMON_MFM},RUN_NAME=mfm_winner_lumo_n1000_s128_m025_rs12000_vs32_diag4_dgen_step1000_pb,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_winner_lumo_n1000_s128_m025_rs12000_vs32_diag4_dgen_step1000_pb,PROPERTY_NAME=lumo,MU=0.25,REWARD_SCALE=12000 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=mfm_mu_best --export=ALL,${COMMON_MFM},RUN_NAME=mfm_winner_mu_n1000_s128_m055_rs8638_vs32_diag4_dgen_step1000_pb,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_winner_mu_n1000_s128_m055_rs8638_vs32_diag4_dgen_step1000_pb,PROPERTY_NAME=mu,MU=0.55,REWARD_SCALE=8.638 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=mfm_cv_best --export=ALL,${COMMON_MFM},RUN_NAME=mfm_winner_cv_n1000_s128_m080_rs1500_vs32_diag4_dgen_step1000_pb,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_winner_cv_n1000_s128_m080_rs1500_vs32_diag4_dgen_step1000_pb,PROPERTY_NAME=cv,MU=0.80,REWARD_SCALE=1.5 slurm/qm9_length_conditioned_guidance.sbatch
```

The commands below reproduce the earlier corrected MFM baseline. They intentionally
set `OUTPUT_DIR` to the archived paths above. Change `RUN_NAME`/`OUTPUT_DIR`
if you do not want to overwrite the existing artifacts.

```bash
DFM_CKPT=outputs/qm9_dfm_long/qm9_dfm_trainflow_long100k_20260531/checkpoints/model_step_100000.pt
MFM_CKPT=outputs/mfm_gencond_diag_stage1_from_spr4_step3000_20260612/stage1_20260612-112012/student_step_1000.pt

COMMON_MFM="CHECKPOINT=${DFM_CKPT},NUM_SAMPLES=1000,SEED=8,TARGET_SEED=2026,VALUE_SAMPLES=32,VALUE_GLASS_STEPS=4,VALUE_SAMPLER=mfm,MFM_CHECKPOINT=${MFM_CKPT},MFM_DIAGONAL_FLAG=--mfm-diagonal,SAMPLE_STEPS=128,GUIDE_EVERY=1,GUIDE_MIN_T=0.05,GUIDE_MAX_T=0.95,SOFT_ATOM_TEMPERATURE=0.25,COMPUTE_PB_FLAG=--compute-pb,GUIDE_ATOMICS_FLAG=--guide-atomics"

sbatch --job-name=mfm_alpha1k --export=ALL,${COMMON_MFM},RUN_NAME=mfm_len_alpha_n1000_s128_mu045_vs32_diag4_dgen_step1000_pb_slurm,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_len_alpha_n1000_s128_mu045_vs32_diag4_dgen_step1000_pb_slurm,PROPERTY_NAME=alpha,MU=0.45,REWARD_SCALE=0.3 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=mfm_gap1k --export=ALL,${COMMON_MFM},RUN_NAME=mfm_len_gap_n1000_s128_mu015_rs8925_vs32_diag4_dgen_step1000_pb_fix,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_len_gap_n1000_s128_mu015_rs8925_vs32_diag4_dgen_step1000_pb_fix,PROPERTY_NAME=gap,MU=0.15,REWARD_SCALE=8925 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=mfm_homo1k --export=ALL,${COMMON_MFM},RUN_NAME=mfm_len_homo_n1000_s128_mu035_rs41500_vs32_diag4_dgen_step1000_pb_fix,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_len_homo_n1000_s128_mu035_rs41500_vs32_diag4_dgen_step1000_pb_fix,PROPERTY_NAME=homo,MU=0.35,REWARD_SCALE=41500 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=mfm_lumo1k --export=ALL,${COMMON_MFM},RUN_NAME=mfm_len_lumo_n1000_s128_mu025_rs9088_vs32_diag4_dgen_step1000_pb_fix,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_len_lumo_n1000_s128_mu025_rs9088_vs32_diag4_dgen_step1000_pb_fix,PROPERTY_NAME=lumo,MU=0.25,REWARD_SCALE=9088 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=mfm_mu1k --export=ALL,${COMMON_MFM},RUN_NAME=mfm_len_mu_n1000_s128_mu035_rs8638_vs32_diag4_dgen_step1000_pb_fix,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_len_mu_n1000_s128_mu035_rs8638_vs32_diag4_dgen_step1000_pb_fix,PROPERTY_NAME=mu,MU=0.35,REWARD_SCALE=8.638 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=mfm_cv1k --export=ALL,${COMMON_MFM},RUN_NAME=mfm_len_cv_n1000_s128_mu070_rs1209_vs32_diag4_dgen_step1000_pb_fix,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/mfm_len_cv_n1000_s128_mu070_rs1209_vs32_diag4_dgen_step1000_pb_fix,PROPERTY_NAME=cv,MU=0.70,REWARD_SCALE=1.209 slurm/qm9_length_conditioned_guidance.sbatch
```

The historical GLASS references can be reproduced with:

```bash
DFM_CKPT=outputs/qm9_dfm_long/qm9_dfm_trainflow_long100k_20260531/checkpoints/model_step_100000.pt

COMMON_GLASS="CHECKPOINT=${DFM_CKPT},NUM_SAMPLES=1000,SEED=8,TARGET_SEED=2026,VALUE_SAMPLES=4,VALUE_GLASS_STEPS=4,VALUE_SAMPLER=glass,SAMPLE_STEPS=128,GUIDE_EVERY=1,GUIDE_MIN_T=0.05,GUIDE_MAX_T=0.95,SOFT_ATOM_TEMPERATURE=0.25,COMPUTE_PB_FLAG=--compute-pb,GUIDE_ATOMICS_FLAG=--guide-atomics"

sbatch --job-name=glass_alpha1k --export=ALL,${COMMON_GLASS},RUN_NAME=n1000_mu045_s128,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/n1000_mu045_s128,PROPERTY_NAME=alpha,MU=0.45,REWARD_SCALE=0.3 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=glass_gap1k --export=ALL,${COMMON_GLASS},RUN_NAME=propc_gap_mu015_s128_n1000,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/propc_gap_mu015_s128_n1000,PROPERTY_NAME=gap,MU=0.15,REWARD_SCALE=8925 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=glass_homo1k --export=ALL,${COMMON_GLASS},RUN_NAME=propc_homo_mu035_s128_n1000,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/propc_homo_mu035_s128_n1000,PROPERTY_NAME=homo,MU=0.35,REWARD_SCALE=41500 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=glass_lumo1k --export=ALL,${COMMON_GLASS},RUN_NAME=propc_lumo_mu025_s128_n1000,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/propc_lumo_mu025_s128_n1000,PROPERTY_NAME=lumo,MU=0.25,REWARD_SCALE=9088 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=glass_mu1k --export=ALL,${COMMON_GLASS},RUN_NAME=propc_mu_mu035_s128_n1000,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/propc_mu_mu035_s128_n1000,PROPERTY_NAME=mu,MU=0.35,REWARD_SCALE=8.638 slurm/qm9_length_conditioned_guidance.sbatch

sbatch --job-name=glass_cv1k --export=ALL,${COMMON_GLASS},RUN_NAME=propc_cv_mu070_s128_n1000,OUTPUT_DIR=outputs/qm9_length_conditioned_guidance/propc_cv_mu070_s128_n1000,PROPERTY_NAME=cv,MU=0.70,REWARD_SCALE=1.209 slurm/qm9_length_conditioned_guidance.sbatch
```

## Validation Command

After jobs finish, this command regenerates the comparison table from the JSON
artifacts:

```bash
.envs/tabasco-gpu/bin/python - <<'PY'
import json
from pathlib import Path

HARTREE_TO_MEV = 27211.386245988

mfm = {
    "alpha": "outputs/qm9_length_conditioned_guidance/mfm_follow2win_alpha_n1000_s128_m060_rs045_vs32_diag4_dgen_step1000_pb",
    "gap": "outputs/qm9_length_conditioned_guidance/mfm_winner_gap_n1000_s128_m015_rs12000_vs32_diag4_dgen_step1000_pb",
    "homo": "outputs/qm9_length_conditioned_guidance/mfm_len_homo_n1000_s128_mu035_rs41500_vs32_diag4_dgen_step1000_pb_fix",
    "lumo": "outputs/qm9_length_conditioned_guidance/mfm_winner_lumo_n1000_s128_m025_rs12000_vs32_diag4_dgen_step1000_pb",
    "mu": "outputs/qm9_length_conditioned_guidance/mfm_winner_mu_n1000_s128_m055_rs8638_vs32_diag4_dgen_step1000_pb",
    "cv": "outputs/qm9_length_conditioned_guidance/mfm_winner_cv_n1000_s128_m080_rs1500_vs32_diag4_dgen_step1000_pb",
}

glass = {
    "alpha": "outputs/qm9_length_conditioned_guidance/n1000_mu045_s128",
    "gap": "outputs/qm9_length_conditioned_guidance/propc_gap_mu015_s128_n1000",
    "homo": "outputs/qm9_length_conditioned_guidance/propc_homo_mu035_s128_n1000",
    "lumo": "outputs/qm9_length_conditioned_guidance/propc_lumo_mu025_s128_n1000",
    "mu": "outputs/qm9_length_conditioned_guidance/propc_mu_mu035_s128_n1000",
    "cv": "outputs/qm9_length_conditioned_guidance/propc_cv_mu070_s128_n1000",
}

def guided(path):
    data = json.loads((Path(path) / "summary.json").read_text())
    rows = {row["run"]: row for row in data["summary"] if "run" in row}
    return rows["unguided"], rows["guided"]

def fmt(prop, guide, oracle):
    if prop in {"gap", "homo", "lumo"}:
        return f"`{guide * HARTREE_TO_MEV:.0f} / {oracle * HARTREE_TO_MEV:.0f} meV`"
    return f"`{guide:.3f} / {oracle:.3f}`"

print("| property | unguided guide/oracle | MFM guided guide/oracle | MFM PB | GLASS guided guide/oracle | GLASS PB |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
for prop in ["alpha", "gap", "homo", "lumo", "mu", "cv"]:
    mfm_u, mfm_g = guided(mfm[prop])
    _, glass_g = guided(glass[prop])
    print(
        f"| {prop} | "
        f"{fmt(prop, mfm_u['guide_mae'], mfm_u['oracle_mae'])} | "
        f"{fmt(prop, mfm_g['guide_mae'], mfm_g['oracle_mae'])} | "
        f"`{mfm_g['pb_pb_valid']:.3f}` | "
        f"{fmt(prop, glass_g['guide_mae'], glass_g['oracle_mae'])} | "
        f"`{glass_g['pb_pb_valid']:.3f}` |"
    )
PY
```

## Notes For Future Sweeps

The current tuned MFM has higher PB validity than GLASS for every property in
the corrected table and lower or equal MAE on essentially all rows. The next
sweeps should be framed as PB/MAE tradeoff tuning, not as a dead-guidance
diagnostic.

Priority sweeps:
- `gap`: compare the best-MAE tuned row (`mu=0.15`, `reward_scale=12000`,
  PB `0.888`) with the more PB-balanced row (`mu=0.25`,
  `reward_scale=8925`, guide/oracle `445 / 606 meV`, PB `0.911`).
- `homo`: run a small local sweep around `mu=0.35`, `reward_scale=41500`
  because this is the only row that is not clearly better than GLASS on guide
  MAE, although oracle MAE and PB are already better.
- `value_samples=64`: test only after fixing the best property-specific
  settings, because the `n=100` to `n=1000` scaling already showed that small
  gates can overstate the best setting.

Follow-up batch launched after this scorecard:
- Script: `slurm/submit_mfm_guidance_followup2_20260619.sh`.
- Jobs: `23557166` through `23557193`.
- Scope: `n=100` probes around the tuned settings, including alpha local
  interpolation, homo local sweeps, and `value_samples=64` checks for the tuned
  rows.
- Status: all completed successfully.
- Best/promising small-gate rows:
  - `alpha`: `mu=0.60`, `reward_scale=0.45`, `value_samples=32`,
    guide/oracle `0.878 / 1.373`, PB `0.960`.
  - `homo`: `mu=0.40`, `reward_scale=55000`, `value_samples=32`,
    guide/oracle `203 / 242 meV`, PB `0.900`.
  - `mu`: `mu=0.60`, `reward_scale=8.638`, `value_samples=32`,
    guide/oracle `0.597 / 0.747`, PB `0.910`.
  - `cv`: `mu=0.80`, `reward_scale=1.5`, `value_samples=64`,
    guide/oracle `0.446 / 0.620`, PB `0.960`.
- Scaled validation script:
  `slurm/submit_mfm_guidance_followup2_winners_20260619.sh`.
- Scaled validation jobs: `23557351` through `23557357`.
- Scaled validation results:
  - `alpha mu=0.60, rs=0.45`: guide/oracle `1.165 / 1.674`, PB `0.939`;
    this replaces the previous alpha row.
  - `alpha mu=0.50, rs=0.50`: guide/oracle `1.205 / 1.761`, PB `0.928`.
  - `homo mu=0.40, rs=55000`: guide/oracle `217 / 278 meV`, PB `0.904`.
  - `homo mu=0.40, rs=50000`: guide/oracle `216 / 276 meV`, PB `0.909`.
  - `mu mu=0.60, rs=8.638`: guide/oracle `0.692 / 0.867`, PB `0.924`.
  - `mu mu=0.55, rs=10.0`: guide/oracle `0.730 / 0.970`, PB `0.912`.
- CV `value_samples=64`, `48`, and `40` all OOMed at `n=1000` on an 80GB
  A100 with the current all-at-once value-gradient implementation. Keep the
  `value_samples=32` CV row unless value-gradient chunking is added.
- Value-gradient chunking was then added via `--value-batch-size` /
  `VALUE_BATCH_SIZE`. A CV `value_samples=64`, `VALUE_BATCH_SIZE=500` retry
  completed as job `23564314`, output
  `outputs/qm9_length_conditioned_guidance/mfm_follow2win_cv_n1000_s128_m080_rs1500_vs64_vbs500_diag4_dgen_step1000_pb`.
  Result: guide/oracle `0.485 / 0.666`, PB `0.960`, so the main table keeps
  the better `value_samples=32` CV row (`0.449 / 0.652`, PB `0.948`).

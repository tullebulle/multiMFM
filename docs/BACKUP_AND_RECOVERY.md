# Backup and recovery

The working copy lives on **netscratch, which is purged**. That has already destroyed this
project's source repositories once (see `REPRODUCIBILITY_PLAN.md` §1), so treat the netscratch copy
as disposable and these backups as the record.

## Where the copies are, as of 2026-09-30

| Location | Holds | Notes |
|---|---|---|
| `/n/netscratch/albergo_lab/Lab/uunneberg/MultiMFM` | everything, 16 GB | working copy; **purge-prone** |
| `/n/holylabs/kozinsky_lab/Users/uunneberg/paper_backup_dmfm_20260929/` | the paper artifacts as of the morning of 2026-09-29 (7.2 GB, `SHA256SUMS.txt` inside) | not purged; **but under `kozinsky_lab`** |
| `…/paper_backup_dmfm_20260929/MultiMFM_20260930_full/` | the whole repo minus `cache/` (3.2 GB), including `.git` with full history | not purged; same `kozinsky_lab` caveat |
| `~/mfm_backup_20260930/` | compact: `.git`, `outputs/`, `logs/`, `checkpoints_toy/` (~1.3 GB) | home directory; independent of lab quotas |

Two constraints worth knowing before adding to these:

- **`kozinsky_lab`'s holylabs quota is exhausted.** The 2026-09-30 copy stopped 185 files short for
  that reason. All 98 of the omitted tracked files are recoverable from the copy's own `.git`; the
  other 87 are `__pycache__`, `.pytest_cache` and `.lock` files. Nothing irreplaceable is missing —
  but a further copy there will fail.
- **The lab has changed.** The repo moved to `albergo_lab` on 2026-09-29, while both backups sit
  under `kozinsky_lab`. `/n/holylabs/albergo_lab/` grants no write access to this user
  (`Users/`, `Lab/` and `Everyone/` are all unwritable), so there is currently no `albergo_lab`
  equivalent. If `kozinsky_lab` access lapses, `~/mfm_backup_20260930/` is the surviving copy.

## What is deliberately not backed up

`cache/` — 13 GB of the repo's 16 GB, almost entirely the public Evo 2 7B weights
(`arcinstitute/evo2_7b_base`, revision `074097e9…`) plus Hugging Face, matplotlib and torch caches.
All re-downloadable. `scripts/dna/regen/evo2_environment.yml` rebuilds that environment.

## Recovering

From the full copy, the whole repo including history:

```bash
rsync -a /n/holylabs/.../MultiMFM_20260930_full/ <destination>/
cd <destination> && git status          # the 185 omitted files show as deleted
git checkout -- .                      # restores every tracked one from .git
```

From the compact home copy, code and history plus the run outputs that are not in git:

```bash
git clone ~/mfm_backup_20260930/.git <destination>
cp -a ~/mfm_backup_20260930/outputs ~/mfm_backup_20260930/logs <destination>/
cp -a ~/mfm_backup_20260930/checkpoints_toy <destination>/checkpoints/toy
```

Then restore the large artifacts and verify them:

```bash
python scripts/manifest.py verify --fetch-missing   # 59 artifacts, 1.57 GB, sha256-checked
```

`checkpoints/MANIFEST.json` is tracked in git, so the manifest survives in any copy that has `.git`;
its `source` fields point at the morning backup. Note that the DNA and GEOM checkpoints under
`checkpoints/` and `data/` originate from that backup, so they are not uniquely held by the
2026-09-30 copies.

## Branch state

Everything from 2026-09-29/30 is on branch **`reproducibility-package`** (9 commits off `main`,
**not merged**), authored as Ulrik Unneberg with Claude as co-author. The author was set
explicitly because the repo's configured git identity is a different one. The commits carry
decisions that have not been through co-author review; merge only after that review.

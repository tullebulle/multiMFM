# Rerun status and handoff

Updated 2026-09-29 17:30. Everything below is queued Slurm work plus the single command that
finishes each piece. Every run is sharded and skips shards whose output already exists, so a
preempted or interrupted run resumes by resubmitting, the collection commands are safe to run
repeatedly and while shards are still arriving, and nothing depends on a Claude session staying
alive.

## Scheduling: read this before resubmitting anything

Nothing was scheduling as of 17:30, on any partition, and the reason is priority rather than
capacity. Facts worth keeping:

| Partition | Depth when measured | Notes |
|---|---|---|
| `kempner_h100` | Table 1 array was 86th | H100 only |
| `kempner_requeue` | ~1,400 jobs ahead of ours | H100 only; preemptible, use `--requeue` |
| `serial_requeue` | 6,728 pending / 3,832 running | CPU; rejects a job that still carries `--gres=gpu:1` or an architecture constraint (use `--gres=none --constraint=`) |
| `gpu_requeue` | 2,447 pending / **61 running** | Mixed pool (A100/H100/H200/RTX Pro); shallowest running set |
| `gpu_test` | 2 jobs per user | Fine for smoke tests, 12 h limit |
| `seas_gpu` | used as a third route for one GEOM job | |

Accounts, which are not interchangeable:
- `kempner_albergo_lab` — best fairshare (0.27) but **invalid on `gpu_requeue` and `serial_requeue`**.
- `albergo_lab` — valid on the CPU and general GPU pools, but its fairshare factor is **0.000000**
  against a RawUsage of 3.29 billion, so it is not a fast lane.
- `kozinsky_lab` — RawUsage only 337,666, essentially unused, so it would likely schedule sooner.
  **Not authorised**: the authors moved this project off that account and have not said to use it.

Device policy the agents settled on, worth preserving:
- One device class per plotted panel or table column. CUDA (TF32 matmuls, CUDA RNG) and CPU (true
  fp32, CPU RNG) do not draw the same samples, so a panel is all-GPU or all-CPU. Shards from
  different CUDA card models share a tree; the card name is recorded per shard.
- Every shard records device, TF32 setting, torch version, account, partition and host.
- Where a run moved to CPU, the provenance says so and justifies it. The strongest example is
  Fig 6: its noise is drawn by a seeded CPU generator before being moved to the device, so CPU and
  GPU share identical randomness, and the decode margin (min top-1 minus top-2 of 0.999999 against
  an fp32 noise floor near 1e-7) shows no token's argmax can flip.

## Live jobs: one route per item

Consolidated 2026-09-29 18:00 after the authors objected to holding GPU slots we were not using.
17 job groups were cancelled: duplicate routes, GPU copies of work already moved to CPU, and one
smoke test. **The rule is one route per item.** Every run skips shards whose output exists, so any
of these can be resubmitted and will pick up where it stopped.

| Item | Job | Where | Collect with |
|---|---|---|---|
| **Table 1 with dMFM** | `49277977` | kempner_h100 (#111) | `PARTIAL=1 bash scripts/dna/collect_table1_dmfm.sh 49277977` |
| Table 1, CPU route | `49293510` | serial_requeue, albergo_lab | `RUN_ROOT=outputs/dna/table1_dmfm_n1000_cpu PARTIAL=1 bash scripts/dna/collect_table1_dmfm.sh 49293510` |
| **GEOM grad-sign** | `49293660` | serial_requeue (CPU) | writes `results/geom/grad_sign.json` |
| GEOM constant-mu / paper coefficient | `49293829`, `49293831` | serial_requeue (CPU) | `python scripts/geom/collect_asdescribed.py tune --dir results/geom/runs/asdescribed_tuning results/geom/runs/asdescribed_tuning_cpu` |
| GEOM Table 3 as described, 3 x 4,096 | `49274410` | kempner_requeue | `bash scripts/geom/collect_table3_asdescribed.sh` |
| Fig 4 panel B (and all stages) | `49298117` | gpu_requeue, albergo_lab | `bash scripts/dna/fig4_collect.sh` |
| Fig 4 pilot + panel A | `49296674` | serial_requeue (CPU) | same |
| Fig 5 / Tables 14-15, dMFM | `49298891` (GPU), `49297229` (CPU) | gpu_requeue / serial_requeue | `bash scripts/dna/scalefree/collect.sh` (`OUTPUT_ROOT=outputs/dna/scale_free_mc_dmfm_cpu` for CPU) |
| Fig 6 + Table 16 | `49294651` | serial_requeue (CPU) | `bash scripts/dna/regen/collect.sh` |
| Fig 7 sweep | `49298146` | gpu_requeue | same |
| Fig 8 trace | `49298115` | gpu_requeue | same |
| Table 6 (Evo 2 7B) | `49298971` | gpu_requeue, constrained to a100/h100/h200 | same |
| Table 18 parity diagnostic | `49278666` | kempner_h100 | `docs/provenance/dna/verification.md` |
| Toy Fig 1 recalibration | `49285374` (running), `49300106` (CPU) | gpu_test / serial_requeue | `bash scripts/toy/collect_fig1.sh` |

### Deferred by the consolidation, not missing

Record these as deliberate if anyone writes them up, and resubmit if the authors want them:
- The GEOM (lambda, mu) sweep is now the CPU points, not the full 15-point grid.
- Fig 5's Tables 14-15 will come from the diagonal students only (App F.2's posterior sampler, and
  the students Tables 13 and 17 use), so the tables must not be described as covering both students.
  The apples-to-apples 4-step comparison (the cancelled `49281005`) is deferred; it is one command
  whenever GPUs free up, and it reuses the diagonal run's reference gradients so it adds no
  reference compute:
  `STUDENT=dmfm4 REFERENCE_FROM=outputs/dna/scale_free_mc_dmfm/diag1step sbatch scripts/dna/scalefree/fig5_dmfm_gradient_mc.sbatch`
  **Do not quote the existing 4-step rescore beside Table 14/15 cells.** It used a different
  reference (128 samples, 50-step RK4 to 0.999), held-out rather than base-flow conditioning states,
  4 flow-map steps, nested pools and a different seed, so placing them side by side would make dMFM
  look far worse than the matched comparison shows.
- Fig 6 has no GPU fallback any more; the CPU array is its only route.

## QM9 — complete

All 36 3-seed shards and 120 baseline runs finished; `results/qm9` and notebook 07 are final.
Re-print the paper cells any time with `python scripts/qm9/report_cells.py`.

## After collecting: copy the figures the paper points at

The paper refers to `figures/<name>`, the convention its other figures use, while the notebooks
write regenerated figures under `results/`. After any collection command has run:

```bash
bash scripts/sync_paper_figures.sh
```

It copies only what exists and lists what is still pending, so it is safe to run at any point.
Until a figure has landed, pdftex draws a draft box for it and reports "file not found" — that is
expected, not a build failure, and the tex is deliberately wired ahead of the runs.

## After collecting: the paper

`docs/paper_changes.md` §1.12 is a placeholder inventory: 18 rows giving each marked line in
`iclr2027_conferencemod.tex`, what fills it and who owns it. Once a collection command has run,
substituting is mechanical. The paper currently builds at 38 pages with 0 errors; stripping the
markers gives 34.

Open items for the authors are in `docs/REPRODUCIBILITY_PLAN.md` §7.9, plus two that surfaced
later:
- The main text grew by one page from substantive additions (the GEOM negative-result passage,
  the Fig 4 caption, two §3.3 sentences). If ICLR caps the main text, the natural trims are moving
  the GEOM detail to App F.9 and shortening the Fig 4 caption.
- `iclr2027_conference.sty` exists nowhere on this system, so page counts were measured with a
  2026 substitute and are indicative only. Two Fig 5 PDFs are missing and 14 cited keys are absent
  from `bibliography.bib`.

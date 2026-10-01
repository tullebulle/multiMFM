# results/qm9/ss_diagnostics/ — provenance

Diagnostics behind `docs/ss_guide_oracle_gap.md` (the multiMFM-SS guide-vs-oracle gap).
Produced 2026-09-29. All four scripts live in this directory and take repo-relative defaults;
the verdict-carrying ones are CPU-only.

| file | produced by | inputs | date / job |
|---|---|---|---|
| `regressor_anchor.json` | `regressor_anchor.py --limit 2000 --partition test --cross` | `yairschiff/qm9` (HF cache in `data/qm9/`), `checkpoints/regressors/{guide,oracle}_clf_ckpt.zip` | 2026-09-29, login node, CPU |
| `regressor_anchor_validation.json` | same, `--limit 1800 --partition validation` | " | 2026-09-29, CPU |
| `regressor_anchor_train_flow.json` | same, `--limit 2000 --partition train_flow` | " | 2026-09-29, CPU |
| `regressor_anchor_train_classifier.json` | same, `--limit 2000 --partition train_classifier` | " | 2026-09-29, CPU |
| `guide_oracle_paired.csv` | `selection_bias.py` | `results/qm9/rerun_3seed/runs/{ss,multimfm}/<prop>_s<seed>/per_sample.csv.gz` (Slurm 49276185 / 49279470 / 49279656, the §7 items 4 + 5 3-seed rerun) | 2026-09-29, CPU, no model loaded |
| `guide_oracle_dose_response.{csv,json}` | `selection_bias.py` | same | " |
| `ss_advantage_decomposition.json` | `selection_bias.py` | same | " |
| `ss_input_identity.json` | `ss_input_identity.py` | `checkpoints/{base_flow,mfm_student,regressors}`; writes a throwaway run to `outputs/ss_input_identity/` | 2026-09-29, CPU, n = 4, 8 ODE steps |
| `lookahead_pool_{gap,alpha,mu}.json` | `lookahead_pool.py --property <p> --num-samples 64 --pool-subsample 24 --device cuda --pb-workers 12` | arguments copied from `results/qm9/rerun_3seed/runs/ss/<p>_s1/summary.json`; same checkpoints | Slurm **49348724**, `kempner_requeue`, A100-SXM4-40GB, 2026-09-29 21:45 |
| `lookahead_pool_gap_cpu_reduced.json` | `lookahead_pool.py --property gap --num-samples 16 --pool-subsample 16 --device cpu --pb-workers 8 --sample-steps 32 --value-samples 8 --tag _cpu_reduced` | same | 2026-09-29, login node, CPU. **Reduced scale** (`reduced_scale: true` in the file): 232 candidates instead of 3680, weaker guidance; absolute levels do not transfer. |

Notes.

* Nothing here modified `src/` — no defect was found, so `steer_search.py` and `qm9_data.py`
  are untouched and every stored QM9 result remains valid. See
  `docs/ss_guide_oracle_gap.md` §5.
* `selection_bias.py` needs no model and no GPU: it re-reads the per-molecule guide/oracle
  predictions the rerun already stored, so both regressors are compared on identical molecules
  by construction. Its seed-averaged selected oracle MAEs reproduce
  `results/qm9/rerun_3seed/rerun_3seed_summary.csv` `oracle_mae_mean` to 10 significant
  figures, which is the check that this re-derivation is faithful.
* `lookahead_pool.py` reports an `oracle_sel_subsample` row (argmin of the *oracle* error over
  a random 24-candidate subsample). It is a diagnostic upper bound only, is itself optimistically
  biased, and must never be used for a reported number — selecting on the held-out evaluator
  would leak it into the method.
* `guide_oracle_paired.csv` is 108 rows (6 properties × 3 seeds × up to 6 run types);
  `guide_oracle_dose_response.csv` is its seed average, 36 rows.

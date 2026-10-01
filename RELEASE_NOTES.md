# What is in this repository

Code, results and notebooks for MultiMFM / dMFM inference-time steering.

## What is deliberately not here

* **The manuscript.** The ICLR submission, its figures' LaTeX sources and the
  bibliography are kept in the working repository, not here.
* **Large artifacts.** Model checkpoints and the QM9/GEOM datasets are tracked by
  `scripts/manifest.py` rather than by git -- three of the dataset files exceed GitHub's
  100 MB hard limit on their own. `checkpoints/MANIFEST.json` and `MANIFEST.lock` carry a
  sha256 for every one of them, so a fetched copy can be verified byte for byte:

  ```bash
  python scripts/manifest.py verify     # checks everything listed in the manifest
  ```

  Without those artifacts the analysis notebooks will not run. Fetch them from the lab
  storage location and verify before use.
* **Superseded results.** Earlier runs that later work replaced have been left out: the
  June 2026 `split65k` DNA split, the May 2026 recovered C0 runs, the digitised copies of
  previously published figures, the refuted GEOM composition rescue, and the figure
  variants under `notebooks/figures/diagnostics/` marked `superseded_`, `published_` or
  `UNSAFE_`. They remain in the working repository where the provenance documents
  reference them.

## What is here, and what state it is in

`results/` holds the measurements behind the current figures, and `docs/` records how each
was arrived at, including the claims that were withdrawn. Several documents are candid
about results that did not survive scrutiny -- that is deliberate, and those files are the
reason the surviving numbers can be trusted.

Headline results, all on held-out seeds at full sample size:

| | |
|---|---|
| QM9 value estimation | multiMFM 27.3 -> 0.19 over N = 1 -> 512, about 110x below DPS, which is flat |
| QM9 steering | multiMFM beats DPS like for like by 1.07-2.21x; multiMFM-SS beats best-of-N at every budget |
| GEOM fingerprint | the earlier configuration scores -3.24% +- 1.14 against its own unguided arm; the corrected one +16.58% +- 0.95 at PoseBusters validity 0.889 |
| GEOM vs TFG-Flow | on a matched protocol (their 30-atom cap, their averaging convention) our guided arm reaches 0.2516 +- 0.0073 against their 0.208, clearing the metric's random-pair floor by 0.042 where theirs sits at it -- but we do not win on relative improvement, and `results/geom/MATCHED30.md` says so |
| DNA steering | guidance beats search baselines by 1.6-6x at rare targets; DPS is a tie we do not win |
| GEOM radius of gyration | a negative: best-of-N wins at every budget |

Numbers measured at n = 64 on a single seed shrank by roughly half when replicated, so only
the confirmed figures above are quoted.

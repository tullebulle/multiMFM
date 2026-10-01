# Third-party components

MultiMFM builds on and vendors code from the following projects. All are used
under their original licenses; copyright remains with their respective authors.

## TABASCO (`src/tabasco/`)
The base flow-matching backbone, GLASS posterior sampler, and meta-flow-map
transformer are from TABASCO — a transformer-based flow-matching model for 3D
molecule generation.
- License: MIT (Copyright (c) 2025 Carlos Vonessen)
- Paper: `tabasco_paper.pdf` in the original repository.

## TFG-Flow (`third_party/tfg_flow/`)
The property **guide** and **oracle** regressors (`Predictor` = EGNN head) are
from TFG-Flow: *Training-Free Guidance in Multi-Modal Generative Flow*
(Lin, Li, Ye, Yang, Ermon, Liang, Ma — ICLR 2025).
- License: MIT
- Only the subset needed to instantiate and run the regressors is vendored
  (`diffusion/predictor.py`, `networks/egnn.py`, `dataloader/constants.py`).
- Trained regressor weights: `checkpoints/regressors/{guide,oracle}_clf_ckpt.zip`.

## MFM reference (`mfm/src/`)
The GLASS posterior-velocity distillation target used to train the meta flow map
is imported from the reference Meta Flow Map implementation (Potaptchik, Albergo
et al.). Vendored so that `tabasco.sample.glass.extract_glass_velocity` resolves
`mfm.losses.extract_posterior_velocity` at training time.

## QM9 (`data/qm9/`)
QM9 molecular dataset, loaded via the Hugging Face dataset `yairschiff/qm9`.

## DNA-MFM / Dirichlet Flow Matching (`src/dmfm/`)
The DNA discrete-flow-matching library, dMFM distillation, cyclizability
regressors and experiment scripts derive from `tullebulle/DNA-MFM` at commit
`187fe7b`, itself built on **Dirichlet Flow Matching** (Stark, Jing, Wang,
Corso, Berger, Barzilay, Jaakkola).
- License: MIT (Copyright (c) 2024 Hannes Stärk)
- A complete clone of `187fe7b` is kept at
  `paper_backup_dmfm_20260929/DNA-MFM_187fe7b` on non-purged storage; verbatim
  copies of the original producer scripts, with sha256s, are in
  `results/dna/original_scripts/` (`_sources.json`).
- Ported into an importable package: flat script-to-script imports removed,
  repo-relative defaults, and one fix (a redundant second network evaluation per
  base-flow step; the original behaviour remains available via
  `legacy_double_eval=True` and the outputs are bit-identical).

## Loop-seq yeast cyclizability data (`data/dna/`)
Intrinsic cyclizability measurements of yeast genomic DNA are from Basu et al.,
*Measuring DNA mechanics on the genome scale* (Nature, 2021). The
parent-disjoint windows used here are derived from that assay; the raw record
file is not redistributed, and `data/dna/.../parent_split_seed0.json` carries
its sha256 so a re-download can be verified.

## GEOM-Drugs (`data/geom/`)
GEOM-Drugs conformers (Axelrod & Gómez-Bombarelli, *GEOM, energy-annotated
molecular conformations*, Sci. Data 2022), consumed via the preprocessed
release `carlosinator/tabasco-geom-drugs` on the Hugging Face Hub (CC0-1.0 for
the underlying GEOM data).

## TFG-Flow additions (`third_party/tfg_flow/`, beyond the vendored regressors)
The GEOM structural-fingerprint dataset builder, the structure-head trainer and
a `num_atom_types` fix to `diffusion/predictor.py` were written by the authors of
this work, following TFG-Flow's architecture (a six-layer EGNN with mean pooling
and a linear head, trained with binary cross-entropy against 1024-bit OpenBabel
FP2 fingerprints). TFG-Flow itself releases **no** GEOM fingerprint predictor:
`checkpoints/geom/…/structure_head` is trained here, not an upstream release.
- TFG-Flow license: MIT.

## Evo 2 (`scripts/dna/regen/`, optional environment)
Sequence-model plausibility scores use **Evo 2** (Brixi et al., *Genome
modeling and design across all domains of life*, 2026) via the `evo2` package
(0.5.5) and `vtx` (1.0.8), with the `arcinstitute/evo2_7b_base` weights from the
Hugging Face Hub. Run in a separate environment
(`scripts/dna/regen/evo2_environment.yml`); weights are not redistributed here.
- Evo 2 code and weights are used under their upstream (Apache-2.0 code /
  Arc Institute model) terms.

## Open Babel (via `posebusters` / fingerprinting)
Open Babel 3.1 is used for FP2 fingerprints and is licensed **GPL-2.0**. It is
an environment dependency (conda-forge `openbabel`), not vendored, and no
GPL code is included in this repository.

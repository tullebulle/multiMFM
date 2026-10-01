"""GEOM-Drugs experiments (paper Table 3, GEOM rows of Tables 19-21, App. F.8).

Modules
-------
``paths``            repo-relative default locations of the GEOM artifacts.
``teacher``          load the Lightning GEOM-Drugs base flow model (EMA weights).
``reward``           differentiable FP2-fingerprint reward for the GLASS engine.
``fingerprint``      structure head, target sampling, FP2 scoring (Tanimoto).
``glass_steering``   GLASS (teacher look-ahead) fingerprint steering driver.
``mfm_steering``     multiMFM (flow-map look-ahead) fingerprint steering driver -> Table 3.
``train_student``    multi-GPU MFM student distillation (stages 1-4).
``train_base``       Hydra entry point for base-model training (configs/geom).
``check_pb_validity``PoseBusters validity of a base checkpoint -> Table 21 "unsteered".
``diagnostics``      directional check that the value gradient ascends the guide reward.
``train_structure_head`` wrapper around the vendored TFG-Flow head trainer.

Every module is runnable as ``python -m multimfm.geom.<module> --help`` and takes
repo-relative defaults, so nothing has to be run from a particular directory.
"""

from multimfm.geom.paths import (  # noqa: F401
    DEFAULT_GEOM_FP_DIR,
    DEFAULT_PB_CONFIG,
    DEFAULT_STRUCTURE_HEAD,
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    REPO_ROOT,
)

"""Train the GEOM-Drugs base flow model (paper Table 19, GEOM-Drugs column).

Hydra + Lightning entry point over ``configs/geom``. It reproduces the run that
produced ``checkpoints/geom/teacher/epoch=38-step=89310.ckpt`` (1 GPU, batch 512,
7 extra rotation copies, lr 2e-3, EMA 0.999, ~89k steps / 39 epochs)::

    python -m multimfm.geom.train_base experiment=geom_dfm trainer=gpu
    python -m multimfm.geom.train_base experiment=geom_dfm trainer=ddp trainer.devices=2

Checkpoints land in ``outputs/geom/base/<SLURM_JOB_ID|manual>/`` and the resolved
config in the Hydra run directory. Requires the processed GEOM-Drugs splits in
``data/geom/`` -- see ``docs/provenance/geom/data.md``.

Ported from ``geom-drug-tabasco/src/train.py`` (the generic lightning-hydra
template runner); only the config path and the root-dir handling differ.
"""

from __future__ import annotations

import os
from typing import Any

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from multimfm.geom.paths import CONFIGS, REPO_ROOT
from tabasco.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

torch.set_float32_matmul_precision("high")
log = RankedLogger(__name__, rank_zero_only=True)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
# configs/geom/paths/default.yaml reads MULTIMFM_ROOT, so every relative path in
# the config tree resolves against the repository regardless of the cwd.
os.environ.setdefault("MULTIMFM_ROOT", str(REPO_ROOT))


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train (and optionally test) the model described by ``cfg``."""
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.lightning_module._target_}>")
    lightning_module: LightningModule = hydra.utils.instantiate(cfg.lightning_module)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": lightning_module.model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=lightning_module, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
    test_metrics = trainer.callback_metrics

    return {**train_metrics, **test_metrics}, object_dict


@hydra.main(version_base="1.3", config_path=str(CONFIGS), config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)
    metric_dict, _ = train(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()

"""Train the TFG structure head (fingerprint predictor) on GEOM-Drugs.

The head is TFG's ``diffusion.predictor.Predictor`` (EGNN + masked mean-pool +
linear -> 1024). It maps ``(coords, atom_types, mask)`` to a 1024-bit structure
fingerprint and is trained with binary cross-entropy against the OpenBabel FP2
label. Accuracy is reported as Tanimoto similarity (binarising the prediction at
0.5), matching how TFG scores its ``structure`` target.

Supports single-GPU and multi-GPU DistributedDataParallel. For DDP, launch with
torchrun and this script reads ``RANK``/``LOCAL_RANK``/``WORLD_SIZE`` from the
environment, e.g.:

    torchrun --standalone --nproc_per_node=3 train_structure_head.py \
        --data_dir storage/datasets/geom_fp --batch_size 256 --lr 3e-4 --wandb

``--batch_size`` is per-GPU, so global batch = batch_size * num_gpus.

Prepare the dataset first with ``dataloader.datasets.build_geom_fingerprints``.
"""

import argparse
import glob
import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataloader.datasets.geom_fingerprint import (
    GEOMFingerprintDataset,
    NUM_GEOM_ATOM_TYPES,
    tanimoto_similarity,
)
from diffusion.predictor import Predictor


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #
def is_dist():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank():
    return int(os.environ.get("RANK", "0"))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main():
    return get_rank() == 0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="storage/datasets/geom_fp")
    ap.add_argument("--log_dir", default="storage/ckpts/geom_structure_head")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val")

    # optimisation
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64, help="per-GPU batch size")
    ap.add_argument("--lr", type=float, default=1e-4, help="peak learning rate")
    ap.add_argument("--lr_schedule", default="none", choices=["none", "cosine"],
                    help="cosine = linear warmup then cosine decay to 0 over the run")
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None,
                    help="override padded length (default: global split max)")

    # data / model geometry
    ap.add_argument("--coord_normalizer", type=float, default=2.0,
                    help="divide coords by this (tabasco dataset_normalizer)")
    ap.add_argument("--h_embed_size", type=int, default=128)
    ap.add_argument("--e_embed_size", type=int, default=128)
    ap.add_argument("--cls_embed_size", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=6)

    ap.add_argument("--init_ckpt", default=None,
                    help="load model weights from this .pth before training (warm start)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="geom-drugs")
    ap.add_argument("--wandb_entity", default="ulrikunneberg-harvard-university")
    ap.add_argument("--wandb_name", default="structure-head")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, ddp):
    """Global (all-rank) mean val BCE and Tanimoto over the val set."""
    model.eval()
    tot_bce = torch.zeros((), device=device)
    tot_tan = torch.zeros((), device=device)
    n = torch.zeros((), device=device)
    for batch in loader:
        coors = batch["coors"].float().to(device)
        atom_types = batch["atom_types"].to(device)
        mask = batch["mask"].to(device)
        target = batch["structure"].float().to(device)

        logits = model(coors, atom_types, mask)
        bce_ps = nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none").mean(dim=1)          # (bs,)
        pred_bits = (torch.sigmoid(logits) > 0.5).float()
        tan_ps = tanimoto_similarity(pred_bits, target)            # (bs,)

        tot_bce += bce_ps.sum()
        tot_tan += tan_ps.sum()
        n += target.shape[0]

    if ddp:
        for t in (tot_bce, tot_tan, n):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (tot_bce / n).item(), (tot_tan / n).item()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    ddp = is_dist()
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    if is_main():
        os.makedirs(args.log_dir, exist_ok=True)

    # The head handles any molecule size up to max_len (masked mean-pool). Pad
    # width must cover the largest molecule across every split, so use the global
    # max over the available split metas unless overridden.
    max_len = args.max_len
    if max_len is None:
        maxima = []
        for meta_path in glob.glob(os.path.join(args.data_dir, "*_meta.json")):
            with open(meta_path) as f:
                maxima.append(int(json.load(f)["max_num_atoms"]))
        max_len = max(maxima)
        if is_main():
            print(f"auto max_len (global over splits) = {max_len}")

    train_set = GEOMFingerprintDataset(
        args.data_dir, split=args.train_split, max_len=max_len,
        coord_normalizer=args.coord_normalizer, limit=args.limit_train,
    )
    val_set = GEOMFingerprintDataset(
        args.data_dir, split=args.val_split, max_len=max_len,
        coord_normalizer=args.coord_normalizer, limit=args.limit_val,
    )

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if is_main():
        print(f"train={len(train_set)}  val={len(val_set)}  max_len={max_len}  "
              f"atom_types={NUM_GEOM_ATOM_TYPES}  world_size={world}  "
              f"per_gpu_bs={args.batch_size}  global_bs={args.batch_size * world}  "
              f"lr={args.lr}")

    train_sampler = DistributedSampler(train_set, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=args.num_workers,
        drop_last=True, persistent_workers=args.num_workers > 0, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=True)

    model = Predictor(
        max_len=max_len, device=device,
        h_embed_size=args.h_embed_size, num_layers=args.num_layers,
        cls_embed_size=args.cls_embed_size, e_embed_size=args.e_embed_size,
        class_num=1024, num_atom_types=NUM_GEOM_ATOM_TYPES,
    ).to(device)
    if args.init_ckpt is not None:
        state = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(state)
        if is_main():
            print(f"warm start: loaded weights from {args.init_ckpt}")
    if ddp:
        # Predictor discards the EGNN's coordinate-update output (uses only the
        # pooled node features), so some coord-head params get no gradient ->
        # DDP needs find_unused_parameters to skip them in the reduction.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Optional linear-warmup + cosine-decay schedule, stepped per optimizer step.
    # Deterministic across ranks (same total_steps), so DDP stays in sync.
    scheduler = None
    if args.lr_schedule == "cosine":
        steps_per_epoch = len(train_loader)
        total_steps = args.epochs * steps_per_epoch
        warmup = min(max(args.warmup_steps, 1), max(1, total_steps - 1))

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            prog = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if is_main():
            print(f"cosine LR: peak={args.lr} warmup={warmup} total_steps={total_steps}")

    run = None
    if args.wandb and is_main():
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, config=vars(args))

    best_tan = -1.0
    for epoch in range(args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_main())
        for batch in pbar:
            coors = batch["coors"].float().to(device)
            atom_types = batch["atom_types"].to(device)
            mask = batch["mask"].to(device)
            target = batch["structure"].float().to(device)

            logits = model(coors, atom_types, mask)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            running += loss.item()
            if is_main():
                cur_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"bce": loss.item(), "lr": cur_lr})
                if run is not None:
                    run.log({"train/bce": loss.item(), "train/lr": cur_lr})

        train_bce = running / max(len(train_loader), 1)
        if is_main():
            print(f"epoch {epoch} | train_bce {train_bce:.4f}")

        if epoch % args.eval_every == 0:
            val_bce, val_tan = evaluate(model, val_loader, device, ddp)
            if is_main():
                print(f"epoch {epoch} | val_bce {val_bce:.4f} | val_tanimoto {val_tan:.4f}")
                if run is not None:
                    run.log({"epoch": epoch, "val/bce": val_bce,
                             "val/tanimoto": val_tan, "train/bce_epoch": train_bce})
                torch.save(raw_model.state_dict(), os.path.join(args.log_dir, "last.pth"))
                if val_tan > best_tan:
                    best_tan = val_tan
                    torch.save(raw_model.state_dict(), os.path.join(args.log_dir, "best.pth"))
                    with open(os.path.join(args.log_dir, "best.txt"), "w") as f:
                        f.write(f"epoch={epoch}\nval_tanimoto={val_tan}\nval_bce={val_bce}\n")
                    print(f"  saved best (tanimoto={val_tan:.4f})")

    if is_main():
        print(f"done. best val tanimoto = {best_tan:.4f}")
        if run is not None:
            run.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

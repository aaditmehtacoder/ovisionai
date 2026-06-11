"""
train.py — train the v0 baseline and report honest metrics.

Usage:
    python src/train.py
    python src/train.py --task classification --epochs 20
    python src/train.py --task regression --backbone efficientnet_b0
    python src/train.py --wandb            # optional W&B logging

Picks task automatically from the data when --task is omitted (config.TASK
defaults to "auto"). Saves the best checkpoint to checkpoints/ and writes
val/test metrics to results/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from data import prepare_data  # noqa: E402
from model import OVisionModel, build_loss, save_checkpoint  # noqa: E402
from utils import (  # noqa: E402
    classification_metrics,
    get_device,
    pretty_print_metrics,
    regression_metrics,
    save_json,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="OVision v0 trainer")
    p.add_argument("--task", choices=["auto", "classification", "regression"],
                   default=config.TASK)
    p.add_argument("--backbone", default=config.BACKBONE)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging (optional; off by default).")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass. Train if optimizer is given, else evaluate. Returns
    (mean_loss, y_true, y_score) where y_score is prob (cls) or Hb (reg)."""
    is_train = optimizer is not None
    model.train(is_train)

    losses, y_true, y_score = [], [], []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)            # logits (cls) or Hb (reg)
            loss = criterion(outputs, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        y_true.append(targets.detach().cpu().numpy())
        if model.task == "classification":
            y_score.append(torch.sigmoid(outputs).detach().cpu().numpy())
        else:
            y_score.append(outputs.detach().cpu().numpy())

    return (
        float(np.mean(losses)) if losses else 0.0,
        np.concatenate(y_true) if y_true else np.array([]),
        np.concatenate(y_score) if y_score else np.array([]),
    )


def compute_metrics(task, y_true, y_score):
    if task == "classification":
        return classification_metrics(y_true, y_score)
    return regression_metrics(y_true, y_score, config.ANEMIA_HB_THRESHOLD)


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device()
    print(f"[train] device = {device}")

    # Allow CLI overrides of a few config values.
    config.BATCH_SIZE = args.batch_size
    config.BACKBONE = args.backbone

    df, task, splits, loaders = prepare_data(requested_task=args.task, verbose=True)
    print(f"[train] resolved task = {task}")
    print(f"[train] split sizes -> "
          f"train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")

    model = OVisionModel(task=task, backbone=args.backbone).to(device)
    criterion = build_loss(task)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=config.WEIGHT_DECAY)

    run = _init_wandb(args, task) if args.wandb else None

    # Lower-is-better selection metric for both tasks: val loss.
    best_val = float("inf")
    ckpt_path = config.CHECKPOINT_DIR / f"ovision_v0_{task}.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, *_ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, vy, vs = run_epoch(model, loaders["val"], criterion, device)
        val_metrics = compute_metrics(task, vy, vs)

        key = "auc" if task == "classification" else "hb_mae"
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_{key}={val_metrics[key]:.4f}")

        if run is not None:
            run.log({"epoch": epoch, "train_loss": train_loss,
                     "val_loss": val_loss, **{f"val_{k}": v
                     for k, v in val_metrics.items() if isinstance(v, (int, float))}})

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, ckpt_path,
                            extra={"epoch": epoch, "val_loss": val_loss,
                                   "backbone": args.backbone})

    print(f"\n[train] best val_loss={best_val:.4f} -> saved {ckpt_path}")

    # Final validation metrics from the best checkpoint, saved to results/.
    pretty_print_metrics(f"Final validation metrics ({task})", val_metrics)
    save_json(
        {"task": task, "backbone": args.backbone, "epochs": args.epochs,
         "best_val_loss": best_val, "val_metrics": val_metrics},
        config.RESULTS_DIR / f"train_{task}.json",
    )
    print(f"\n[train] Done. Evaluate the held-out test set with:\n"
          f"        python src/evaluate.py --task {task}")

    if run is not None:
        run.finish()


def _init_wandb(args, task):
    try:
        import wandb
    except ImportError:
        print("[train] --wandb passed but wandb is not installed; skipping.")
        return None
    return wandb.init(project=config.WANDB_PROJECT,
                      config={"task": task, "backbone": args.backbone,
                              "epochs": args.epochs, "lr": args.lr,
                              "batch_size": args.batch_size, "seed": config.SEED})


if __name__ == "__main__":
    main()

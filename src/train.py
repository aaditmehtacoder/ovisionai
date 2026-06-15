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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import data  # noqa: E402  (ConjunctivaDataset/SOURCE_ORDER for the balanced loader)
from data import genders_for_split, prepare_data  # noqa: E402
from model import OVisionModel, build_loss, save_checkpoint  # noqa: E402
from utils import (  # noqa: E402
    classification_metrics,
    get_device,
    pretty_print_metrics,
    regression_metrics,
    save_json,
    seed_worker,
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
    # Balanced-sampling A/B switches (default to config). Use e.g.
    #   --no-balance-classes --no-balance-sources   for the unbalanced baseline.
    p.add_argument("--balance-classes", action=argparse.BooleanOptionalAction,
                   default=config.BALANCE_CLASSES,
                   help="Balance anemic vs non-anemic per batch (default on).")
    p.add_argument("--balance-sources", action=argparse.BooleanOptionalAction,
                   default=config.BALANCE_SOURCES,
                   help="Balance india/italy/ghana per batch (default on).")
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging (optional; off by default).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Balanced sampling — reshape ONLY what the train loader draws (val/test stay
# as-is). Lives here (not data.py) so the shared data loader/crop are untouched;
# crossval.py imports make_train_loader so both paths balance identically.
# ---------------------------------------------------------------------------
def _anemia_labels(df, indices):
    """Per-sample anemic flag, derived from hgb + the gender-aware cutoff.
    Computed directly (not read off df['anemic']) so unknown-gender rows still
    get a class via anemia_cutoff's scalar fallback."""
    return np.array(
        [float(df.loc[i, "hgb"]) < config.anemia_cutoff(df.loc[i, "gender"])
         for i in indices],
        dtype=bool,
    )


def compute_sample_weights(df, indices):
    """Sampling weight per train sample: inverse-frequency over a balancing KEY.

    The key is the derived anemia class (BALANCE_CLASSES only), the source
    (BALANCE_SOURCES only), or — when BOTH are on — the JOINT (source, class)
    cell. Giving every cell equal total mass is what "class-balance within a
    source-even draw" actually means: equal cells => 50/50 anemic AND even
    sources at the same time. (Multiplying the two marginal inverse-frequencies
    does NOT achieve this — a source skewed toward one class throws the global
    class rate off.) Returns (weights, labels, sources)."""
    n = len(indices)
    weights = np.ones(n, dtype=np.float64)
    labels = _anemia_labels(df, indices)
    sources = np.array([df.loc[i, "source"] for i in indices], dtype=object)

    if config.BALANCE_CLASSES and config.BALANCE_SOURCES:
        key = np.array([f"{s}|{int(a)}" for s, a in zip(sources, labels)], dtype=object)
    elif config.BALANCE_CLASSES:
        key = labels.astype(int).astype(object)
    elif config.BALANCE_SOURCES:
        key = sources
    else:
        return weights, labels, sources

    for grp in np.unique(key):
        m = key == grp
        c = int(m.sum())
        if c:
            weights[m] = 1.0 / c
    return weights, labels, sources


def _report_balance(weights, labels, sources, tag=""):
    """Draw a stream from the SAME weights the sampler uses and print the
    resulting anemic % and per-source % — proof it's actually balancing, with the
    raw (unsampled) pool shown for contrast."""
    n_stream = max(2000, 4 * len(weights))
    gen = torch.Generator()
    gen.manual_seed(config.SEED + 1)
    drawn = np.array(list(WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=n_stream, replacement=True, generator=gen)))
    d_lab, d_src = labels[drawn], sources[drawn]

    def src_line(src_arr):
        return "  ".join(
            f"{s}={100.0 * float((src_arr == s).mean()):4.1f}%"
            for s in data.SOURCE_ORDER if (sources == s).any())

    head = f"[balance]{(' ' + tag) if tag else ''} "
    print(f"{head}flags: classes={config.BALANCE_CLASSES} "
          f"sources={config.BALANCE_SOURCES}  (stream of {n_stream} draws)")
    print(f"{head}  raw pool : anemic={100.0 * labels.mean():4.1f}%   "
          f"{src_line(sources)}")
    print(f"{head}  sampled  : anemic={100.0 * d_lab.mean():4.1f}%   "
          f"{src_line(d_src)}")


def make_train_loader(df, indices, task, tag=""):
    """Train DataLoader. With balancing off (both flags False) this is exactly
    data.make_loader(train=True). With either flag on, swap the shuffle for a
    WeightedRandomSampler over the same dataset (crop/loader logic unchanged)."""
    if not (config.BALANCE_CLASSES or config.BALANCE_SOURCES):
        return data.make_loader(df, indices, task, train=True)

    weights, labels, sources = compute_sample_weights(df, indices)
    _report_balance(weights, labels, sources, tag=tag)

    generator = torch.Generator()
    generator.manual_seed(config.SEED)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(indices), replacement=True, generator=generator)
    ds = data.ConjunctivaDataset(df, indices, task, train=True)
    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,           # mutually exclusive with shuffle=True
        num_workers=config.NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


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


def compute_metrics(task, y_true, y_score, genders=None):
    if task == "classification":
        return classification_metrics(y_true, y_score)
    # Gender-aware anemia cutoff (<12 F, <13 M) when genders are available.
    cutoff = config.ANEMIA_HB_THRESHOLD
    if genders is not None:
        cutoff = np.array([config.anemia_cutoff(g) for g in genders], dtype=float)
    return regression_metrics(y_true, y_score, cutoff)


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device()
    print(f"[train] device = {device}")

    # Allow CLI overrides of a few config values.
    config.BATCH_SIZE = args.batch_size
    config.BACKBONE = args.backbone
    config.BALANCE_CLASSES = args.balance_classes
    config.BALANCE_SOURCES = args.balance_sources

    df, task, splits, loaders = prepare_data(requested_task=args.task, verbose=True)
    print(f"[train] resolved task = {task}")
    print(f"[train] split sizes -> "
          f"train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")

    # Swap the train loader for the balanced one (val/test stay exactly as built).
    loaders["train"] = make_train_loader(df, splits["train"], task, tag="train")

    model = OVisionModel(task=task, backbone=args.backbone).to(device)
    criterion = build_loss(task)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=config.WEIGHT_DECAY)

    # Genders aligned with the (shuffle=False) val loader order, for the
    # gender-aware anemia cutoff in regression metrics.
    val_genders = genders_for_split(df, splits["val"]) if task == "regression" else None

    run = _init_wandb(args, task) if args.wandb else None

    # Lower-is-better selection metric for both tasks: val loss.
    best_val = float("inf")
    ckpt_path = config.CHECKPOINT_DIR / f"ovision_v0_{task}.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, *_ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, vy, vs = run_epoch(model, loaders["val"], criterion, device)
        val_metrics = compute_metrics(task, vy, vs, val_genders)

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

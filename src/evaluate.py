"""
evaluate.py — score the held-out TEST split with the saved checkpoint.

Kept separate from train.py so the test set is only ever touched on purpose
(not peeked at during model selection). Prints screening metrics and writes
them to results/.

Usage:
    python src/evaluate.py
    python src/evaluate.py --task classification
    python src/evaluate.py --checkpoint checkpoints/ovision_v0_regression.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from data import genders_for_split, prepare_data  # noqa: E402
from model import load_checkpoint  # noqa: E402
from utils import (  # noqa: E402
    append_results_csv,
    classification_metrics,
    get_device,
    pretty_print_metrics,
    regression_metrics,
    save_json,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="OVision v0 evaluator")
    p.add_argument("--task", choices=["auto", "classification", "regression"],
                   default=config.TASK)
    p.add_argument("--checkpoint", default=None,
                   help="Path to a .pt checkpoint. Defaults to the one matching the task.")
    return p.parse_args()


@torch.no_grad()
def predict(model, loader, device):
    """Returns (y_true, y_score) where y_score is prob (cls) or Hb (reg)."""
    model.eval().to(device)
    y_true, y_score = [], []
    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        if model.task == "classification":
            scores = torch.sigmoid(outputs)
        else:
            scores = outputs
        y_true.append(targets.numpy())
        y_score.append(scores.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_score)


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device()

    df, task, splits, loaders = prepare_data(requested_task=args.task, verbose=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
        config.CHECKPOINT_DIR / f"ovision_v0_{task}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run train.py first."
        )

    model, payload = load_checkpoint(ckpt_path, map_location=device)
    print(f"[eval] loaded {ckpt_path} (task={model.task}, "
          f"epoch={payload.get('epoch')})")

    y_true, y_score = predict(model, loaders["test"], device)

    if model.task == "classification":
        metrics = classification_metrics(y_true, y_score)
    else:
        # Gender-aware anemia cutoff (<12 F, <13 M), aligned with the
        # (shuffle=False) test loader order.
        test_genders = genders_for_split(df, splits["test"])
        cutoff = np.array([config.anemia_cutoff(g) for g in test_genders], dtype=float)
        metrics = regression_metrics(y_true, y_score, cutoff)

    pretty_print_metrics(f"TEST metrics ({model.task}, n={metrics['n']})", metrics)

    # Save both a per-run JSON and an appended CSV log.
    result_row = {"task": model.task, "checkpoint": str(ckpt_path), **metrics}
    save_json(result_row, config.RESULTS_DIR / f"test_{model.task}.json")
    append_results_csv(result_row, config.RESULTS_DIR / "all_test_runs.csv")
    print(f"\n[eval] metrics saved to {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()

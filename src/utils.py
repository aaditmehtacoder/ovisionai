"""
utils.py — reproducibility, device selection, metrics, and small helpers.

Kept deliberately dependency-light and readable. The metric functions return
plain dicts so train.py / evaluate.py can print them and dump them to JSON.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility (HARD REQUIREMENT: fix seed everywhere)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Trade a little speed for determinism. Fine for a small v0 dataset.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding so augmentation is reproducible too."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Device (HARD REQUIREMENT: single GPU, auto CPU fallback)
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon GPU, if present — harmless fallback chain.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """
    Screening metrics for binary anemia classification.

    y_true : array of 0/1 ground-truth labels.
    y_prob : array of predicted probabilities for the positive (anemic) class.

    Returns accuracy, sensitivity (recall for anemic), specificity, AUC,
    and the confusion matrix. "Positive" = anemic, because for a screening
    tool the costly miss is a false negative (missed anemia).
    """
    from sklearn.metrics import confusion_matrix, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Force a 2x2 matrix even if a class is missing in a tiny test split.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)   # true positive rate (catch anemics)
    specificity = tn / max(tn + fp, 1)   # true negative rate

    # AUC needs both classes present in y_true.
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "threshold": float(threshold),
    }


def regression_metrics(y_true_hb, y_pred_hb, anemia_threshold: float) -> dict:
    """
    Metrics for hemoglobin regression.

    Reports MAE/RMSE on Hb, then thresholds both true and predicted Hb at the
    anemia cutoff to derive the same screening sensitivity/specificity a
    clinician cares about.
    """
    y_true_hb = np.asarray(y_true_hb, dtype=float)
    y_pred_hb = np.asarray(y_pred_hb, dtype=float)

    mae = float(np.mean(np.abs(y_pred_hb - y_true_hb)))
    rmse = float(np.sqrt(np.mean((y_pred_hb - y_true_hb) ** 2)))

    # Below threshold => anemic (positive class).
    true_anemic = (y_true_hb < anemia_threshold).astype(int)
    pred_anemic = (y_pred_hb < anemia_threshold).astype(int)
    # Use predicted Hb's "distance below threshold" as a pseudo-probability so
    # we can still get an AUC-style number from the regression head.
    pseudo_prob = anemia_threshold - y_pred_hb
    pseudo_prob = (pseudo_prob - pseudo_prob.min()) / (np.ptp(pseudo_prob) + 1e-8)

    cls = classification_metrics(true_anemic, pseudo_prob, threshold=_prob_at_cutoff(
        y_pred_hb, anemia_threshold, pseudo_prob))

    return {
        "hb_mae": mae,
        "hb_rmse": rmse,
        "anemia_threshold": float(anemia_threshold),
        # Screening metrics derived from thresholded Hb:
        "sensitivity": cls["sensitivity"],
        "specificity": cls["specificity"],
        "accuracy": cls["accuracy"],
        "auc": cls["auc"],
        "confusion_matrix": cls["confusion_matrix"],
        "n": int(len(y_true_hb)),
    }


def _prob_at_cutoff(y_pred_hb, anemia_threshold, pseudo_prob) -> float:
    """The pseudo-probability value that corresponds exactly to Hb == cutoff."""
    raw = anemia_threshold - np.asarray(y_pred_hb, dtype=float)
    lo, span = raw.min(), (np.ptp(raw) + 1e-8)
    return float((0.0 - lo) / span)


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------
def save_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_results_csv(row: dict, path: Path) -> None:
    """Append one flat row of metrics to a CSV (created on first write)."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = _flatten(row)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix=f"{key}_"))
        else:
            flat[key] = v
    return flat


def pretty_print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in metrics.items():
        if isinstance(v, dict):
            inner = "  ".join(f"{ik}={iv}" for ik, iv in v.items())
            print(f"  {k:18s}: {inner}")
        elif isinstance(v, float):
            print(f"  {k:18s}: {v:.4f}")
        else:
            print(f"  {k:18s}: {v}")

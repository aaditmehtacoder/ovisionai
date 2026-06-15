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


def is_cuda_oom(err: Exception) -> bool:
    """True if `err` is a CUDA out-of-memory error (message-sniffed, since torch
    raises a plain RuntimeError for it)."""
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


def run_with_oom_retry(fn, set_batch_size, get_batch_size, min_batch: int = 1):
    """Call `fn()`; on CUDA OOM, halve the batch size and retry until it fits.

    Heavier backbones (e.g. EfficientNet-B3) may not fit at the default batch
    size. `get_batch_size`/`set_batch_size` read and write the live batch-size
    knob (config.BATCH_SIZE) so the retried `fn` rebuilds its loaders smaller.
    Returns whatever `fn` returns. Re-raises any non-OOM error, or OOM once the
    batch size can't shrink further."""
    while True:
        try:
            return fn()
        except RuntimeError as err:
            bs = get_batch_size()
            if is_cuda_oom(err) and bs > min_batch:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                new_bs = max(min_batch, bs // 2)
                set_batch_size(new_bs)
                print(f"[oom] CUDA OOM at batch_size={bs} -> retrying at {new_bs}")
                continue
            raise


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


def regression_metrics(y_true_hb, y_pred_hb, anemia_threshold, genders=None) -> dict:
    """
    Metrics for hemoglobin regression.

    Reports MAE/RMSE on Hb, then thresholds both true and predicted Hb at the
    anemia cutoff to derive the same screening sensitivity/specificity a
    clinician cares about.

    `anemia_threshold` is either a scalar cutoff (g/dL) OR, when `genders` is
    given, a per-sample cutoff array — pass `config.anemia_cutoff(g)` per gender
    so the anemic call is gender-aware (e.g. <12 for F, <13 for M).
    """
    y_true_hb = np.asarray(y_true_hb, dtype=float)
    y_pred_hb = np.asarray(y_pred_hb, dtype=float)

    mae = float(np.mean(np.abs(y_pred_hb - y_true_hb)))
    rmse = float(np.sqrt(np.mean((y_pred_hb - y_true_hb) ** 2)))

    # Per-sample cutoff vector (gender-aware when `genders` is supplied).
    cutoff = np.asarray(anemia_threshold, dtype=float)
    if cutoff.ndim == 0:
        cutoff = np.full_like(y_true_hb, float(anemia_threshold))

    # Below the cutoff => anemic (positive class).
    true_anemic = (y_true_hb < cutoff).astype(int)
    # "Distance below cutoff" doubles as a pseudo-probability so we can still get
    # an AUC-style number from the regression head; 0 distance == exactly at the
    # cutoff, which is the decision boundary.
    raw = cutoff - y_pred_hb
    lo, span = raw.min(), (np.ptp(raw) + 1e-8)
    pseudo_prob = (raw - lo) / span
    boundary = float((0.0 - lo) / span)

    cls = classification_metrics(true_anemic, pseudo_prob, threshold=boundary)

    reported_threshold = (
        float(anemia_threshold) if cutoff.size and np.all(cutoff == cutoff[0])
        else "gender-aware"
    )
    return {
        "hb_mae": mae,
        "hb_rmse": rmse,
        "anemia_threshold": reported_threshold,
        # Screening metrics derived from thresholded Hb:
        "sensitivity": cls["sensitivity"],
        "specificity": cls["specificity"],
        "accuracy": cls["accuracy"],
        "auc": cls["auc"],
        "confusion_matrix": cls["confusion_matrix"],
        "n": int(len(y_true_hb)),
    }


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

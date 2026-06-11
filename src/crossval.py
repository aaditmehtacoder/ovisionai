"""
crossval.py — patient-level, source-stratified K-fold cross-validation.

Why: with a small pooled dataset, a single train/test split is noisy. K-fold gives
a steadier estimate, and breaking the metrics down by SOURCE (india / italy /
ghana) shows whether pooling Ghana (CP-AnemiC) helps or hurts performance on the
India + Italy (Eyes-defy) data.

Design:
  * Folds are built over PATIENTS (a patient never spans folds) and stratified by
    source via round-robin within each source, so every fold carries India + Italy
    + Ghana.
  * For each fold: train on the other folds, predict the held-out fold. Predictions
    are pooled out-of-fold, then scored OVERALL and PER SOURCE with the same
    gender-aware regression metrics evaluate.py uses.

Reuses the existing model/transform/training code (model.OVisionModel,
train.run_epoch, data.make_loader). Does NOT touch the frozen splits.json.

Usage:
    python src/crossval.py
    python src/crossval.py --folds 5 --epochs 20 --backbone efficientnet_b0
    python src/crossval.py --datasets eyes_defy        # India+Italy only
    python src/crossval.py --datasets eyes_defy,cp_anemic
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import data  # noqa: E402
from model import OVisionModel, build_loss  # noqa: E402
from train import run_epoch  # noqa: E402  (reuse the exact train/eval loop)
from utils import get_device, regression_metrics, set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="OVision source-stratified cross-validation")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--backbone", default=config.BACKBONE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--datasets", default=None,
                   help="comma list overriding config.DATASETS, e.g. 'eyes_defy'")
    return p.parse_args()


def make_source_folds(df, k: int, seed: int) -> dict:
    """patient_id -> fold index. Round-robin within each source so every fold is
    balanced across india / italy / ghana."""
    rng = np.random.default_rng(seed)
    source_of = (df.drop_duplicates("patient_id")
                   .set_index("patient_id")["source"].to_dict())

    by_source = defaultdict(list)
    for pid, src in source_of.items():
        by_source[src].append(pid)

    fold_of = {}
    for src in sorted(by_source):
        order = np.array(sorted(by_source[src]), dtype=object)
        rng.shuffle(order)
        for i, pid in enumerate(order):
            fold_of[pid] = i % k
    return fold_of


def _metrics_block(y_true, y_pred, genders) -> dict:
    cutoffs = np.array([config.anemia_cutoff(g) for g in genders], dtype=float)
    return regression_metrics(np.asarray(y_true), np.asarray(y_pred), cutoffs)


def main():
    args = parse_args()
    if args.datasets:
        config.DATASETS = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    config.BATCH_SIZE = args.batch_size
    config.BACKBONE = args.backbone

    set_seed(config.SEED)
    device = get_device()
    print(f"[crossval] device={device}  datasets={config.DATASETS}  "
          f"folds={args.folds}  epochs={args.epochs}  backbone={args.backbone}")

    df = data.build_dataframe(verbose=True)
    task = data.resolve_task(df, "regression")
    fold_of = make_source_folds(df, args.folds, config.SEED)

    # Out-of-fold predictions, pooled across folds.
    oof = defaultdict(lambda: {"y_true": [], "y_pred": [], "gender": [], "source": []})

    for f in range(args.folds):
        test_patients = {p for p, fold in fold_of.items() if fold == f}
        train_patients = {p for p, fold in fold_of.items() if fold != f}
        test_idx = data._rows_for(df, test_patients)
        train_idx = data._rows_for(df, train_patients)
        if not test_idx or not train_idx:
            print(f"[crossval] fold {f}: empty train/test — skipping.")
            continue

        train_loader = data.make_loader(df, train_idx, task, train=True)
        test_loader = data.make_loader(df, test_idx, task, train=False, shuffle=False)

        model = OVisionModel(task=task, backbone=args.backbone).to(device)
        criterion = build_loss(task)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=config.WEIGHT_DECAY)
        for _epoch in range(args.epochs):
            run_epoch(model, train_loader, criterion, device, optimizer)

        _loss, y_true, y_pred = run_epoch(model, test_loader, criterion, device)
        genders = data.genders_for_split(df, test_idx)
        sources = data.sources_for_split(df, test_idx)

        for s in np.unique(sources):
            m = sources == s
            oof[s]["y_true"].extend(np.asarray(y_true)[m].tolist())
            oof[s]["y_pred"].extend(np.asarray(y_pred)[m].tolist())
            oof[s]["gender"].extend(np.asarray(genders)[m].tolist())

        fold_mae = float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))
        print(f"[crossval] fold {f}: train={len(train_idx)} test={len(test_idx)} "
              f"img  MAE={fold_mae:.2f} g/dL")

    # ---- per-source + overall report over pooled out-of-fold predictions ----
    print("\n=========== CROSS-VAL RESULTS (out-of-fold) ===========")
    header = f"{'source':8} {'n':>5} {'MAE':>6} {'RMSE':>6} {'acc':>6} {'sens':>6} {'spec':>6} {'AUC':>6}"
    print(header)
    print("-" * len(header))

    all_true, all_pred, all_gender = [], [], []
    for s in data.SOURCE_ORDER:
        if s not in oof:
            continue
        b = oof[s]
        all_true += b["y_true"]
        all_pred += b["y_pred"]
        all_gender += b["gender"]
        m = _metrics_block(b["y_true"], b["y_pred"], b["gender"])
        print(f"{s:8} {m['n']:>5} {m['hb_mae']:>6.2f} {m['hb_rmse']:>6.2f} "
              f"{m['accuracy']:>6.2f} {m['sensitivity']:>6.2f} {m['specificity']:>6.2f} "
              f"{m['auc']:>6.2f}")

    if all_true:
        mo = _metrics_block(all_true, all_pred, all_gender)
        print("-" * len(header))
        print(f"{'OVERALL':8} {mo['n']:>5} {mo['hb_mae']:>6.2f} {mo['hb_rmse']:>6.2f} "
              f"{mo['accuracy']:>6.2f} {mo['sensitivity']:>6.2f} {mo['specificity']:>6.2f} "
              f"{mo['auc']:>6.2f}")

    print("\nNOTE: per-source rows show how each dataset fares under pooled training "
          "— compare india/italy here vs a --datasets eyes_defy run to see if "
          "adding Ghana helps or hurts.")


if __name__ == "__main__":
    main()

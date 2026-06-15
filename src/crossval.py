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
from model import OVisionModel, build_loss, count_parameters  # noqa: E402
from train import make_train_loader, run_epoch  # noqa: E402  (reuse train loop + balanced loader)
from utils import (  # noqa: E402
    get_device,
    regression_metrics,
    run_with_oom_retry,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="OVision source-stratified cross-validation")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--backbone", default=config.BACKBONE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--datasets", default=None,
                   help="comma list overriding config.DATASETS, e.g. 'eyes_defy'")
    # Same balanced-sampling A/B switches as train.py (default to config). Run
    # once with both on and once with --no-balance-classes --no-balance-sources
    # to compare the per-source table directly.
    p.add_argument("--balance-classes", action=argparse.BooleanOptionalAction,
                   default=config.BALANCE_CLASSES,
                   help="Balance anemic vs non-anemic per batch (default on).")
    p.add_argument("--balance-sources", action=argparse.BooleanOptionalAction,
                   default=config.BALANCE_SOURCES,
                   help="Balance india/italy/ghana per batch (default on).")
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


# ---------------------------------------------------------------------------
# Reusable cross-validation core (shared by main() and backbone_sweep.py).
# ---------------------------------------------------------------------------
def prepare_cv(folds: int, datasets=None):
    """Build the dataframe, resolve the task, and assign patient-level,
    source-stratified folds ONCE. The sweep reuses these so every backbone sees
    the identical folds."""
    if datasets:
        config.DATASETS = tuple(s.strip() for s in datasets.split(",") if s.strip())
    df = data.build_dataframe(verbose=True)
    task = data.resolve_task(df, "regression")
    fold_of = make_source_folds(df, folds, config.SEED)
    return df, task, fold_of


def _train_and_eval_fold(df, task, train_idx, test_idx, *, backbone, epochs, lr,
                         device, report_params, fold):
    """Train one fold and return (y_true, y_pred) on its held-out images.

    Wrapped in run_with_oom_retry: on CUDA OOM the batch size is halved and the
    whole fold (loaders + model) is rebuilt smaller, so heavy backbones still
    finish. Loaders are built INSIDE the retried closure for that reason."""
    def _attempt():
        train_loader = make_train_loader(df, train_idx, task, tag=f"fold{fold}")
        test_loader = data.make_loader(df, test_idx, task, train=False, shuffle=False)
        print(f"[crossval] fold {fold}: effective batch_size={config.BATCH_SIZE}")

        model = OVisionModel(task=task, backbone=backbone).to(device)
        if report_params:
            total, trainable = count_parameters(model)
            print(f"[crossval] backbone={backbone}  params={total/1e6:.2f}M "
                  f"(trainable {trainable/1e6:.2f}M)")
        criterion = build_loss(task)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=config.WEIGHT_DECAY)
        for _epoch in range(epochs):
            run_epoch(model, train_loader, criterion, device, optimizer)
        _loss, y_true, y_pred = run_epoch(model, test_loader, criterion, device)
        return y_true, y_pred

    return run_with_oom_retry(
        _attempt,
        set_batch_size=lambda b: setattr(config, "BATCH_SIZE", b),
        get_batch_size=lambda: config.BATCH_SIZE,
    )


def run_crossval(df, task, fold_of, *, folds, epochs, backbone, lr, device):
    """Train each fold for `backbone`, pool out-of-fold predictions, return the
    oof dict {source: {y_true, y_pred, gender}}. Re-seeds first so each backbone
    runs under identical RNG conditions (backbone is the only variable)."""
    set_seed(config.SEED)
    oof = defaultdict(lambda: {"y_true": [], "y_pred": [], "gender": [], "source": []})

    for f in range(folds):
        test_patients = {p for p, fold in fold_of.items() if fold == f}
        train_patients = {p for p, fold in fold_of.items() if fold != f}
        test_idx = data._rows_for(df, test_patients)
        train_idx = data._rows_for(df, train_patients)
        if not test_idx or not train_idx:
            print(f"[crossval] fold {f}: empty train/test — skipping.")
            continue

        y_true, y_pred = _train_and_eval_fold(
            df, task, train_idx, test_idx, backbone=backbone, epochs=epochs,
            lr=lr, device=device, report_params=(f == 0), fold=f)
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

    return oof


def per_source_results(oof) -> list:
    """Flatten pooled OOF predictions into per-source + OVERALL metric rows:
    [{source, n, MAE, RMSE, acc, sens, spec, AUC}, ...]."""
    results = []
    all_true, all_pred, all_gender = [], [], []
    for s in data.SOURCE_ORDER:
        if s not in oof:
            continue
        b = oof[s]
        all_true += b["y_true"]
        all_pred += b["y_pred"]
        all_gender += b["gender"]
        results.append(_result_row(s, _metrics_block(b["y_true"], b["y_pred"], b["gender"])))
    if all_true:
        results.append(_result_row("OVERALL", _metrics_block(all_true, all_pred, all_gender)))
    return results


def _result_row(source: str, m: dict) -> dict:
    return {"source": source, "n": int(m["n"]), "MAE": m["hb_mae"], "RMSE": m["hb_rmse"],
            "acc": m["accuracy"], "sens": m["sensitivity"], "spec": m["specificity"],
            "AUC": m["auc"]}


def print_results_table(results: list, title: str = "CROSS-VAL RESULTS (out-of-fold)"):
    """Print the per-source + OVERALL table (identical format across runs)."""
    print(f"\n=========== {title} ===========")
    header = f"{'source':8} {'n':>5} {'MAE':>6} {'RMSE':>6} {'acc':>6} {'sens':>6} {'spec':>6} {'AUC':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["source"] == "OVERALL":
            print("-" * len(header))
        print(f"{r['source']:8} {r['n']:>5} {r['MAE']:>6.2f} {r['RMSE']:>6.2f} "
              f"{r['acc']:>6.2f} {r['sens']:>6.2f} {r['spec']:>6.2f} {r['AUC']:>6.2f}")


def main():
    args = parse_args()
    if args.datasets:
        config.DATASETS = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    config.BATCH_SIZE = args.batch_size
    config.BACKBONE = args.backbone
    config.BALANCE_CLASSES = args.balance_classes
    config.BALANCE_SOURCES = args.balance_sources

    set_seed(config.SEED)
    device = get_device()
    print(f"[crossval] device={device}  datasets={config.DATASETS}  "
          f"folds={args.folds}  epochs={args.epochs}  backbone={args.backbone}  "
          f"balance(classes={config.BALANCE_CLASSES}, sources={config.BALANCE_SOURCES})")

    df, task, fold_of = prepare_cv(args.folds)
    oof = run_crossval(df, task, fold_of, folds=args.folds, epochs=args.epochs,
                       backbone=args.backbone, lr=args.lr, device=device)
    results = per_source_results(oof)
    print_results_table(results)

    print("\nNOTE: per-source rows show how each dataset fares under pooled training "
          "— compare india/italy here vs a --datasets eyes_defy run to see if "
          "adding Ghana helps or hurts.")


if __name__ == "__main__":
    main()

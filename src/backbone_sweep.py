"""
backbone_sweep.py — run the SAME cross-validation across several backbones so you
can compare them with the backbone as the ONLY variable.

Everything except the backbone is held fixed: the same patient-level,
source-stratified folds (built ONCE and reused), the same rule crop, the same
balanced sampling (classes + sources ON), the same epochs, and the same
gender-aware per-source metrics. For each backbone we run K-fold crossval, then —
IMMEDIATELY, before moving on — persist its results so a timeout can never lose a
finished run:
  * results/sweep_<backbone>.json   (full per-source table + run metadata)
  * results/sweep_summary.csv       (one appended row per source)

Resumable: a backbone whose results/sweep_<backbone>.json already exists is
skipped, so restarting after a timeout never redoes finished backbones.

At the end it prints a combined comparison table (per-source AUC + specificity
side by side, including a hardcoded resnet18 baseline reference) so you can see
which backbone wins on the HARD populations (ghana + india).

Usage:
    python src/backbone_sweep.py
    python src/backbone_sweep.py --backbones resnet50,efficientnet_b3 --folds 5
    python src/backbone_sweep.py --epochs 20 --batch-size 16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import crossval  # noqa: E402  (reuse the exact folds + train/eval loop)
from utils import append_results_csv, get_device, save_json, set_seed  # noqa: E402

# resnet18 baseline (the current default backbone) — hardcoded reference row so
# every sweep is read against the same yardstick without re-running it.
REFERENCE_RESNET18 = {
    "india": {"AUC": 0.60, "spec": 0.41},
    "ghana": {"AUC": 0.74, "spec": 0.28},
    "italy": {"AUC": 0.94, "spec": 0.90},
}

SUMMARY_CSV = config.RESULTS_DIR / "sweep_summary.csv"


def parse_args():
    p = argparse.ArgumentParser(description="OVision backbone sweep (crossval per backbone)")
    p.add_argument("--backbones", default=",".join(config.SWEEP_BACKBONES),
                   help="comma list of backbones to sweep, in order.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                   help="starting batch size; auto-halved on CUDA OOM per backbone.")
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--datasets", default=None,
                   help="comma list overriding config.DATASETS, e.g. 'eyes_defy'")
    return p.parse_args()


def _json_path(backbone: str) -> Path:
    return config.RESULTS_DIR / f"sweep_{backbone}.json"


def _save_backbone_results(backbone, results, *, folds, epochs):
    """Persist one backbone's results the moment it finishes (json + csv append)."""
    save_json(
        {"backbone": backbone, "folds": folds, "epochs": epochs,
         "balance_classes": config.BALANCE_CLASSES,
         "balance_sources": config.BALANCE_SOURCES,
         "results": results},
        _json_path(backbone),
    )
    for r in results:
        append_results_csv(
            {"backbone": backbone, "source": r["source"], "MAE": r["MAE"],
             "RMSE": r["RMSE"], "acc": r["acc"], "sens": r["sens"],
             "spec": r["spec"], "AUC": r["AUC"]},
            SUMMARY_CSV,
        )
    print(f"[sweep] saved {_json_path(backbone).name} and appended rows to "
          f"{SUMMARY_CSV.name}")


def _by_source(results: list) -> dict:
    return {r["source"]: r for r in results}


def _print_comparison(all_results: dict):
    """Combined table: per-source AUC + spec for each backbone, side by side, with
    the resnet18 baseline as a reference row. Sources ordered hard -> easy
    (ghana, india, italy) so the populations that matter are read first."""
    order = ("ghana", "india", "italy")
    print("\n\n=================== BACKBONE COMPARISON "
          "(per-source AUC / specificity) ===================")
    head = f"{'backbone':18}"
    for s in order:
        head += f" | {s+' AUC':>9} {s+' spec':>9}"
    print(head)
    print("-" * len(head))

    def fmt_row(name, by_src):
        line = f"{name:18}"
        for s in order:
            r = by_src.get(s)
            if r is None:
                line += f" | {'-':>9} {'-':>9}"
            else:
                line += f" | {r['AUC']:>9.2f} {r['spec']:>9.2f}"
        return line

    # Reference row first.
    ref = {s: {"AUC": REFERENCE_RESNET18[s]["AUC"], "spec": REFERENCE_RESNET18[s]["spec"]}
           for s in order}
    print(fmt_row("resnet18 (ref)", ref))
    print("-" * len(head))
    for backbone, results in all_results.items():
        print(fmt_row(backbone, _by_source(results)))

    print("\nHARD populations are ghana + india — the backbone that lifts their AUC "
          "and specificity above the resnet18 reference is the real winner.")


def main():
    args = parse_args()
    if args.datasets:
        config.DATASETS = tuple(s.strip() for s in args.datasets.split(",") if s.strip())
    # Hold balancing ON so the sweep isolates the backbone effect.
    config.BALANCE_CLASSES = True
    config.BALANCE_SOURCES = True

    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    set_seed(config.SEED)
    device = get_device()
    print(f"[sweep] device={device}  backbones={backbones}  folds={args.folds}  "
          f"epochs={args.epochs}  batch_size={args.batch_size}  "
          f"balance(classes=True, sources=True)")

    # Folds are built ONCE and reused for every backbone (same folds guarantee).
    df, task, fold_of = crossval.prepare_cv(args.folds, datasets=args.datasets)

    all_results = {}
    for backbone in backbones:
        jpath = _json_path(backbone)
        if jpath.exists():
            import json
            saved = json.loads(jpath.read_text())
            all_results[backbone] = saved["results"]
            print(f"\n[sweep] {backbone}: found {jpath.name} — skipping (resume).")
            crossval.print_results_table(all_results[backbone],
                                         title=f"SWEEP {backbone} (cached)")
            continue

        # Fresh batch-size headroom per backbone (auto-reduced on OOM inside).
        config.BATCH_SIZE = args.batch_size
        print(f"\n[sweep] ===== training backbone: {backbone} =====")
        oof = crossval.run_crossval(df, task, fold_of, folds=args.folds,
                                    epochs=args.epochs, backbone=backbone,
                                    lr=args.lr, device=device)
        results = crossval.per_source_results(oof)
        crossval.print_results_table(results, title=f"SWEEP {backbone}")
        _save_backbone_results(backbone, results, folds=args.folds, epochs=args.epochs)
        all_results[backbone] = results

    _print_comparison(all_results)


if __name__ == "__main__":
    main()

"""
compare.py — read-only PREDICTED vs ACTUAL hemoglobin on the held-out TEST set.

A quick eyeball check that complements evaluate.py: it prints a per-patient table
(true vs predicted Hgb, abs error, true/predicted anemia label, and whether the
classification matched) plus the same summary metrics evaluate.py reports, and
saves the table to results/compare_test.csv.

It reuses the exact evaluation code paths so it can't drift from training:
  * model + checkpoint loaded via model.load_checkpoint (same as evaluate.py),
    from config.CHECKPOINT_DIR / "ovision_v0_regression.pt".
  * preprocessing is data._build_transforms(train=False).
  * TEST patients come from the frozen split in splits.json.
  * metrics come from utils.regression_metrics with the gender-aware cutoff.

Run:
    python src/compare.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import data  # noqa: E402
from data import _build_transforms  # noqa: E402  (exact test transform)
from model import load_checkpoint  # noqa: E402  (same loader evaluate.py uses)
from utils import get_device, regression_metrics  # noqa: E402

from PIL import Image  # noqa: E402

TASK = "regression"
CKPT_PATH = config.CHECKPOINT_DIR / f"ovision_v0_{TASK}.pt"


@torch.no_grad()
def _predict_hb(model, transform, image_path, device) -> float:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return float(model(tensor).item())


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CKPT_PATH}. Train the v0 model first:\n"
            f"    python src/train.py --task regression"
        )

    device = get_device()
    model, payload = load_checkpoint(CKPT_PATH, map_location=device)
    model.eval().to(device)
    print(f"[compare] loaded {CKPT_PATH} (task={model.task}, "
          f"epoch={payload.get('epoch')}) on {device}")

    df = data.build_dataframe(verbose=False)
    task = data.resolve_task(df, TASK)
    splits = data.make_or_load_splits(df, task)
    test_idx = splits["test"]
    transform = _build_transforms(train=False)

    # ---- per-image predictions on the TEST split ----
    records = []
    for i in test_idx:
        row = df.loc[i]
        true_hb = float(row["hgb"])
        gender = row["gender"]
        cutoff = config.anemia_cutoff(gender)  # gender-aware: <12 F, <13 M
        pred_hb = _predict_hb(model, transform, row["image_path"], device)

        true_label = "anemic" if true_hb < cutoff else "healthy"
        pred_label = "anemic" if pred_hb < cutoff else "healthy"
        records.append({
            "patient_id": row["patient_id"],
            "gender": gender or "?",
            "true_Hgb": round(true_hb, 1),
            "pred_Hgb": round(pred_hb, 1),
            "abs_error": round(abs(pred_hb - true_hb), 1),
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": "yes" if true_label == pred_label else "no",
        })

    table = pd.DataFrame(records).sort_values("true_Hgb").reset_index(drop=True)

    # ---- aligned table ----
    cols = ["patient_id", "gender", "true_Hgb", "pred_Hgb", "abs_error",
            "true_label", "pred_label", "correct"]
    widths = {"patient_id": 14, "gender": 7, "true_Hgb": 9, "pred_Hgb": 9,
              "abs_error": 10, "true_label": 11, "pred_label": 11, "correct": 8}
    header = "".join(f"{c:>{widths[c]}}" for c in cols)
    print(f"\nPREDICTED vs ACTUAL — TEST set (n={len(table)}), sorted by true Hgb")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for _, r in table.iterrows():
        mark = "✓" if r["correct"] == "yes" else "✗"
        cells = [
            f'{r["patient_id"]:>{widths["patient_id"]}}',
            f'{r["gender"]:>{widths["gender"]}}',
            f'{r["true_Hgb"]:>{widths["true_Hgb"]}.1f}',
            f'{r["pred_Hgb"]:>{widths["pred_Hgb"]}.1f}',
            f'{r["abs_error"]:>{widths["abs_error"]}.1f}',
            f'{r["true_label"]:>{widths["true_label"]}}',
            f'{r["pred_label"]:>{widths["pred_label"]}}',
            f'{mark:>{widths["correct"]}}',
        ]
        print("".join(cells))
    print("-" * len(header))

    # ---- summary metrics (same defs as evaluate.py) ----
    # y_true/y_pred come from the SORTED table, so align cutoffs to it too.
    y_true = table["true_Hgb"].to_numpy(dtype=float)
    y_pred = table["pred_Hgb"].to_numpy(dtype=float)
    cutoffs = np.array([config.anemia_cutoff(g) for g in table["gender"]], dtype=float)
    metrics = regression_metrics(y_true, y_pred, cutoffs)
    cm = metrics["confusion_matrix"]
    print(f"\nTEST summary (n={metrics['n']}):")
    print(f"  Hgb MAE      : {metrics['hb_mae']:.2f} g/dL")
    print(f"  Hgb RMSE     : {metrics['hb_rmse']:.2f} g/dL")
    print(f"  accuracy     : {metrics['accuracy']:.3f}")
    print(f"  sensitivity  : {metrics['sensitivity']:.3f}  (catches anemics)")
    print(f"  specificity  : {metrics['specificity']:.3f}")
    print(f"  AUC          : {metrics['auc']:.3f}")
    print(f"  confusion    : tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}  "
          f"(positive = anemic)")

    # ---- save table ----
    out = config.RESULTS_DIR / "compare_test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n[compare] table saved to {out}")
    print(f"NOTE: the test set is small (n={metrics['n']}, ~30) — these numbers "
          f"are indicative, not definitive.")


if __name__ == "__main__":
    main()

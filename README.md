# OVision v0 — anemia screening baseline

Quick fix on new branch thanks....

A minimal, reproducible PyTorch baseline for estimating **anemia risk** from a
smartphone photo of the **palpebral conjunctiva** (inner lower eyelid).
Conjunctival pallor correlates with low hemoglobin, so: *photo in → anemia
risk / hemoglobin estimate out.*

This is **Phase 1, v0** — the simplest thing that trains honestly and reports
screening metrics. It is a **whole-image transfer-learning baseline**, not the
final system. The eventual two-stage pipeline (conjunctiva segmentation → Hb
regression) is intentionally left as a `TODO` in [`src/model.py`](src/model.py)
and not implemented yet.

Dataset: [Eyes-defy-anemia](https://www.kaggle.com/datasets/harshwardhanfartale/eyes-defy-anemia)
(~218 smartphone conjunctiva images with anemia and/or hemoglobin labels).

---

## What it does

- **Auto-detects the dataset structure** — you do not hardcode folder/column
  names. [`src/explore_data.py`](src/explore_data.py) discovers the real layout;
  [`src/data.py`](src/data.py) adapts via candidate names in
  [`config.py`](config.py).
- **Both task modes**, chosen automatically from what the data supports:
  - `classification` — anemic vs non-anemic
  - `regression` — predict Hb (g/dL), then threshold at the anemia cutoff
- **Screening-grade metrics**: accuracy, sensitivity, specificity, AUC,
  confusion matrix (classification); Hb MAE/RMSE plus sensitivity/specificity at
  the cutoff (regression).
- **Reproducible**: fixed seeds (python/numpy/torch/cuda) and a frozen
  train/val/test split saved to `splits.json`.
- **Runs on one GPU, falls back to CPU/MPS automatically.**
- **Optional** Weights & Biases logging behind `--wandb` (off by default).

---

## Repo layout

```
ovision/
  README.md            this file
  requirements.txt
  .gitignore           ignores data/ checkpoints/ results/ wandb/ splits.json
  config.py            ALL knobs: data path, column/folder names, task, training
  src/
    explore_data.py    RUN FIRST — discovers dataset structure + label stats
    data.py            Dataset + DataLoaders + frozen split
    model.py           ResNet18 / EfficientNet-B0 with switchable head
    train.py           training loop + checkpointing
    evaluate.py        held-out test scoring
    utils.py           seeds, device, metrics, result saving
  notebooks/
    kaggle_v0.ipynb    thin wrapper: explore -> train -> evaluate on Kaggle
```

---

## Run it locally

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Get the data
#    Download the Eyes-defy-anemia dataset and unzip it into ./data
#    (or point OVISION_DATA_ROOT at wherever it lives).
#    By default config.py looks for /kaggle/input/eyes-defy-anemia, then ./data.
export OVISION_DATA_ROOT=/path/to/eyes-defy-anemia   # optional override

# 3. ALWAYS run this first — learn the real folder/label structure
python src/explore_data.py

# 4. If explore_data.py shows different column/folder names than the defaults,
#    edit config.py (IMAGE_COL / ANEMIA_COL / HB_COL / *_FOLDER_HINTS).

# 5. Train (task auto-detected; override with --task)
python src/train.py
python src/train.py --task classification --epochs 20
python src/train.py --task regression --backbone efficientnet_b0 --wandb

# 6. Score the held-out test split
python src/evaluate.py
```

Outputs:
- `checkpoints/ovision_v0_<task>.pt` — best model weights
- `results/train_<task>.json`, `results/test_<task>.json`,
  `results/all_test_runs.csv` — metrics
- `splits.json` — the frozen split (delete it to regenerate)

---

## Run it on Kaggle

1. Create a new Kaggle Notebook and **Add Data → Eyes-defy-anemia**. It mounts at
   `/kaggle/input/eyes-defy-anemia`, which `config.py` auto-detects — no path
   editing needed.
2. Make the `src/` code available, either by:
   - uploading this repo as a Kaggle *Dataset* / *Utility Script*, or
   - pasting the cells from [`notebooks/kaggle_v0.ipynb`](notebooks/kaggle_v0.ipynb),
     which clones/imports `src/` and runs **explore → train → evaluate**.
3. Enable GPU (Settings → Accelerator → GPU) — or leave it on CPU; the code falls
   back automatically.

`notebooks/kaggle_v0.ipynb` is intentionally thin: it sets the repo path on
`sys.path` and calls the same functions used locally, so behavior matches.

---

## Configuring for the real data

After `explore_data.py`, the only file you should normally touch is
[`config.py`](config.py):

| If explore shows…                          | Set in config.py                         |
| ------------------------------------------ | ---------------------------------------- |
| a metadata CSV with an image column named `X` | `IMAGE_COL = "X"`                      |
| an anemia label column named `Y`           | `ANEMIA_COL = "Y"`                       |
| a hemoglobin column named `Z`              | `HB_COL = "Z"`                           |
| no CSV; labels are in folder names         | `ANEMIC_FOLDER_HINTS` / `NONANEMIC_FOLDER_HINTS` |
| a documented anemia cutoff ≠ 12 g/dL       | `ANEMIA_HB_THRESHOLD`                    |

The discovery logic matches column names case-insensitively and tries a list of
common aliases, so often it just works without edits.

---

## Honest-metrics notes

- **Sensitivity is the metric that matters most** for screening: a false
  negative is a missed anemia case. The confusion matrix is printed so you can
  see the failure mode, not just a single headline number.
- The dataset is **small (~218 images)** — expect high variance. The frozen
  split keeps runs comparable; don't over-read a single test number.
- v0 uses the **whole image**. Improving past this is what Phase 2's
  segmentation stage is for.

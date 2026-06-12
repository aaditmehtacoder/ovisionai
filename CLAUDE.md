# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

OVision v0 — a reproducible PyTorch baseline that estimates **anemia risk / hemoglobin** from a smartphone photo of the **palpebral conjunctiva** (inner lower eyelid). Conjunctival pallor correlates with low Hb, so: *photo in → Hb estimate / anemia flag out.* This is deliberately the simplest honest baseline: a **whole-image transfer-learning regressor**, not the eventual two-stage (segment → regress) system (left as a `TODO` in [src/model.py](src/model.py)).

## Commands

```bash
# Setup (Python 3.10+; on Kaggle most deps are preinstalled)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ALWAYS run first on a new dataset mount — discovers the real folder/label layout (read-only)
python src/explore_data.py

# Train (task auto-detected; checkpoint -> checkpoints/, metrics -> results/)
python src/train.py
python src/train.py --task regression --backbone efficientnet_b0 --epochs 20 --wandb

# Score the held-out test split (only ever touch test on purpose)
python src/evaluate.py

# Patient-level, source-stratified K-fold (steadier estimate on this tiny dataset)
python src/crossval.py --folds 5
python src/crossval.py --datasets eyes_defy          # India+Italy only, to A/B against pooling Ghana

# Eyeball predicted vs actual Hb on the test set
python src/compare.py

# Gradio sanity-check UI
python src/demo.py        # __main__ runs self_check(); build_ui().launch() to serve

# SAM cropper (OPTIONAL, opt-in): offline synthetic self-check, then on-Kaggle preview
python src/sam_crop.py            # synthetic self-check (safe offline, no weights)
python src/sam_crop.py --smoke    # 4 imgs/source -> results/sam_crop_preview.png + timing
```

There is no test runner, linter, or build step — verification is done by running the scripts above (the self-checks in `demo.py` / `sam_crop.py` are the smoke tests).

## Dataset location (no hardcoded paths)

Data lives **outside** the repo and is gitignored. `config.get_data_root()` resolves, in order: `OVISION_DATA_ROOT` env var → the Kaggle mount → local `./data`. CP-AnemiC uses `OVISION_CPANEMIC_ROOT` the same way. **Never hardcode a dataset path elsewhere** — import `config.DATA_ROOT` / `config.CPANEMIC_ROOT`. Set `export OVISION_DATA_ROOT=/path/...` to run locally.

## Architecture

The whole pipeline funnels through **`config.py` → `data.py`** and reuses one model/loop everywhere, so behavior can't drift between train / crossval / evaluate / compare / demo.

- **`config.py`** is the single source of truth for *all* knobs (paths, dataset structure, column/folder names, task, crop params, training hyperparams). Most "configuration" is adjusting names here after `explore_data.py`, not editing code. CLI flags and `crossval.py` mutate a few `config.*` values at runtime (`config.BATCH_SIZE`, `config.DATASETS`, etc.) — config is treated as live, mutable global state.

- **`data.py`** merges **two datasets into one unified schema** (`build_dataframe()` → columns `image_path, patient_id, hgb, gender, source, age_group, anemic`):
  - *Eyes-defy-anemia* (adults; India + Italy) — per-country `.xlsx`, pre-cropped `*_palpebral.png` eyelid cutouts. No anemia label in data; `anemic` is **derived** from a gender-aware Hb cutoff (`config.ANEMIA_CUTOFFS`: F<12, M<13).
  - *CP-AnemiC* (children; Ghana) — one PNG per row under `Anemic/` or `Non-anemic/`; the **folder name is the ground-truth label**.
  - `patient_id` is prefixed by source (`"india/12"`, `"ghana/Image_001"`) so ids never collide across datasets.

- **Splitting is PATIENT-level and SOURCE-stratified.** A patient never spans train/val/test (multiple images per patient), and each split carries India+Italy+Ghana in proportion. The split is computed once, frozen to `splits.json` (as patient-id lists), and reused so runs stay comparable. **Delete `splits.json` to regenerate** — it auto-recomputes if the patient set changes. `crossval.py` builds its own folds and does **not** touch `splits.json`.

- **Task is auto-resolved** (`resolve_task`): regression when Hb values are present (they are), else classification. The model is **one backbone + single-output head shared by both tasks** ([src/model.py](src/model.py)): the output is P(anemic) via sigmoid for classification, or the raw Hb value for regression. Loss switches in `build_loss` (BCE vs L1).

- **Gender-aware metric.** Regression predicts Hb, then thresholds at the per-gender anemia cutoff to compute screening metrics. `genders_for_split` / `sources_for_split` return per-sample arrays **aligned to a `shuffle=False` loader's order** so predictions can be mapped back to the right cutoff and source — keep that ordering invariant intact.

- **Shared training loop.** `train.run_epoch` is the single train/eval pass; `crossval.py` and `evaluate.py` import it rather than reimplementing. `data.make_loader` is the shared DataLoader factory.

## Crop preprocessing (stage one)

Different sources frame the eyelid differently (lots of black border vs eyelid-fills-frame), which leaks source identity to the model instead of pallor. `PREPROCESS_TIGHT_CROP` (on by default) equalizes framing to a fixed square. **`config.CROP_METHOD` selects the cropper** (`data._make_cropper`):

- **`"rule"` (default):** `tight_crop_image` in [src/data.py](src/data.py). Type-aware — detects *cutout* (eyelid on black → bbox the non-black region) vs *raw photo* (no black border → bbox the reddish/saturated conjunctiva via the HSV heuristic), with a center-crop fallback when confidence is low. Returns `(image, info)` where `info` carries `type` and `fallback`.

- **`"sam"` (opt-in, [src/sam_crop.py](src/sam_crop.py)):** Meta's Segment-Anything, intended for **raw whole-eye photos** the rule cropper can't frame. Prompts SAM near center-bottom, scores each candidate mask by overlap with the **same HSV conjunctiva mask** (`data._conjunctiva_mask`), crops the best mask's bbox, and **falls back to `tight_crop_image`** (flagged) when SAM finds nothing confident. The `segment_anything` import is **guarded** — the module imports fine with no SAM / no internet, just always falling back. vit_b weights (~375MB) download free (no API key) on first use to `checkpoints/`. Runs on CUDA, warns + uses CPU otherwise.

Both croppers are wrapped in picklable transform classes (`TightCrop` / `SamCrop`) because DataLoader workers can't pickle lambdas; `SamCrop` keeps only scalars and lazily builds a **process-local** SAM predictor so it still pickles to workers. When changing crop behavior, keep the `(cropped, info)`-style contract and the picklability intact.

## Conventions that matter here

- **`src/` scripts add the repo root to `sys.path`** (`sys.path.insert(0, parent.parent)`) and `import config` as a top-level module — run them as `python src/<x>.py` from the repo root, not as a package.
- **Bad images must never kill a run.** Loaders catch decode errors and fall back to a known-good sample; crop/preview helpers skip unreadable files. Preserve this resilience.
- **Outputs are gitignored** (`data/ checkpoints/ results/ wandb/ splits.json`) — keep the dataset read-only (required on Kaggle, where `/kaggle/input` is read-only); write only under `checkpoints/` and `results/`.
- This is a **screening** tool: **sensitivity is the headline metric** (a false negative is a missed case). The dataset is small (~218 Eyes-defy images + CP-AnemiC) — expect high variance; don't over-read a single test number. Prefer `crossval.py` for honest estimates.

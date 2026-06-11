"""
data.py — load the OVision datasets into ONE unified PyTorch pipeline.

Two datasets feed the same schema:
  * Eyes-defy-anemia  (config.DATA_ROOT)      — adults, India + Italy, per-country
    spreadsheets, pre-cropped *_palpebral.png eyelid images.
  * CP-AnemiC         (config.CPANEMIC_ROOT)  — children, Ghana, one PNG per row
    under Anemic/ or Non-anemic/, labelled by Anemia_Data_Collection_Sheet.xlsx.

build_dataframe() merges whichever datasets config.DATASETS selects into a single
frame with EXACTLY these columns (plus a derived `anemic` flag used downstream):
    image_path : absolute path to the image
    patient_id : unique across datasets, prefixed by source ("india/12",
                 "ghana/Image_001") so ids never collide
    hgb        : float hemoglobin (g/dL) — the regression target
    gender     : "M" / "F"
    source     : "india" | "italy" | "ghana"
    age_group  : "adult" (Eyes-defy) | "child" (CP-AnemiC)

Splitting stays PATIENT-level (a patient never spans train/test) and is now also
STRATIFIED by source, so every fold carries India + Italy + Ghana.

The flow:
  1. build_dataframe()      -> unified, per-source kept/skipped report + summary.
  2. resolve_task()         -> "regression" (Hgb values present).
  3. make_or_load_splits()  -> frozen, patient-level, source-stratified split.
  4. get_dataloaders()      -> ready-to-train DataLoaders.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# Tolerate slightly truncated images rather than erroring on the last few bytes.
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from utils import seed_worker  # noqa: E402

# Stable display / stratification order for sources.
SOURCE_ORDER = ("india", "italy", "ghana")

# Unified schema (the `anemic` flag is derived and kept for downstream use).
UNIFIED_COLUMNS = [
    "image_path", "patient_id", "hgb", "gender", "source", "age_group", "anemic",
]


# ---------------------------------------------------------------------------
# 1. Unified loader  ->  one tidy dataframe (one row per usable image)
# ---------------------------------------------------------------------------
def build_dataframe(verbose: bool = False) -> pd.DataFrame:
    rows, stats = [], {}

    if "eyes_defy" in config.DATASETS:
        ed_rows, ed_stats = _load_eyes_defy(verbose)
        rows += ed_rows
        stats.update(ed_stats)
    if "cp_anemic" in config.DATASETS:
        cp_rows, cp_stats = _load_cp_anemic(verbose)
        rows += cp_rows
        stats.update(cp_stats)

    # Per-source kept/skipped report (honest usable N per dataset).
    for src in SOURCE_ORDER:
        if src in stats:
            s = stats[src]
            print(f"[data] {src:5s}: seen {s['seen']:4d}  kept {s['kept']:4d}  "
                  f"skipped (missing img/row {s['missing']}, bad Hgb {s['bad_hb']})"
                  f"  unreadable dropped {s['unreadable']}")

    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    if df.empty:
        raise ValueError(
            "Loaded 0 usable rows. Check OVISION_DATA_ROOT / OVISION_CPANEMIC_ROOT "
            "and config.DATASETS."
        )

    _print_summary(df)
    return df


def _blank_stats() -> dict:
    return {"seen": 0, "kept": 0, "missing": 0, "bad_hb": 0, "unreadable": 0}


def _row(image_path, pid, hb, gender, source, age_group, anemic) -> dict:
    return {
        "image_path": image_path,
        "patient_id": pid,
        "hgb": float(hb),
        "gender": gender,
        "source": source,
        "age_group": age_group,
        "anemic": anemic,
    }


# ---- Eyes-defy-anemia (adults; India + Italy) -----------------------------
def _load_eyes_defy(verbose: bool):
    rows = []
    stats = {"india": _blank_stats(), "italy": _blank_stats()}
    root = config.DATA_ROOT
    if not root.exists():
        if verbose:
            print(f"[data] eyes_defy: root not found ({root}) — skipping. "
                  f"Set OVISION_DATA_ROOT.")
        return rows, stats

    for country in config.COUNTRIES:
        src = country.lower()
        st = stats[src]
        sheet = config.spreadsheet_path(root, country)
        if not sheet.exists():
            if verbose:
                print(f"[data] {country}: spreadsheet not found at {sheet} — skipping.")
            continue

        table = _read_spreadsheet(sheet)
        if verbose:
            print(f"[data] {country}: {len(table)} spreadsheet rows from {sheet.name}")

        for _, r in table.iterrows():
            st["seen"] += 1
            number = _clean_number(r["number"])
            if number is None:
                st["missing"] += 1
                continue
            hb = _to_float(r["hb"])
            if hb is None or not np.isfinite(hb):
                st["bad_hb"] += 1
                continue
            folder = root / country / number
            if not folder.is_dir():
                st["missing"] += 1
                continue
            images = _palpebral_images(folder)
            if not images:
                st["missing"] += 1
                continue

            readable = []
            for img in images:
                if _is_readable(img):
                    readable.append(img)
                else:
                    st["unreadable"] += 1
                    if verbose:
                        print(f"[data] unreadable image skipped: {img}")
            if not readable:
                st["missing"] += 1
                continue

            gender = _clean_gender(r["gender"])
            # Eyes-defy has no folder label; derive anemic from the gender cutoff.
            anemic = float(hb < config.anemia_cutoff(gender)) if gender else np.nan
            pid = f"{src}/{number}"
            st["kept"] += 1
            for img in readable:
                rows.append(_row(str(img), pid, hb, gender, src, "adult", anemic))

    return rows, stats


# ---- CP-AnemiC (children; Ghana) ------------------------------------------
def _load_cp_anemic(verbose: bool):
    rows = []
    src = config.CPANEMIC_SOURCE
    stats = {src: _blank_stats()}
    st = stats[src]

    root = config.CPANEMIC_ROOT
    if not root.exists():
        if verbose:
            print(f"[data] cp_anemic: root not found ({root}) — skipping. "
                  f"Set OVISION_CPANEMIC_ROOT.")
        return rows, stats
    sheet = root / config.CPANEMIC_SHEET
    if not sheet.exists():
        if verbose:
            print(f"[data] cp_anemic: sheet not found at {sheet} — skipping.")
        return rows, stats

    table = _read_cpanemic_sheet(sheet)
    if verbose:
        print(f"[data] {src}: {len(table)} spreadsheet rows from {sheet.name}")

    # folder name -> (path, anemic label). The folder the file lives in is the
    # ground-truth label for CP-AnemiC.
    folders = [(name, root / name, 1.0 if name.lower() == "anemic" else 0.0)
               for name in config.CPANEMIC_IMAGE_FOLDERS]

    for _, r in table.iterrows():
        st["seen"] += 1
        image_id = _clean_image_id(r["image_id"])
        if image_id is None:
            st["missing"] += 1
            continue
        hb = _to_float(r["hb"])
        if hb is None or not np.isfinite(hb):
            st["bad_hb"] += 1
            continue

        found, anemic = None, np.nan
        for _name, folder, lab in folders:
            cand = folder / f"{image_id}.png"
            if cand.is_file():
                found, anemic = cand, lab
                break
        if found is None:
            st["missing"] += 1
            continue
        if not _is_readable(found):
            st["unreadable"] += 1
            if verbose:
                print(f"[data] unreadable image skipped: {found}")
            continue

        gender = _clean_gender(r["gender"])  # "Male"/"Female" -> M/F
        pid = f"{src}/{image_id}"
        st["kept"] += 1
        rows.append(_row(str(found), pid, hb, gender, src, "child", anemic))

    return rows, stats


def _print_summary(df: pd.DataFrame) -> None:
    uniq = df.drop_duplicates("patient_id")
    print("\n[data] ===== Unified dataset summary =====")
    for src in SOURCE_ORDER:
        sub = df[df["source"] == src]
        if sub.empty:
            continue
        hb = sub["hgb"]
        an = sub["anemic"].dropna()
        anemic_pct = 100.0 * an.mean() if len(an) else float("nan")
        age_group = sub["age_group"].iloc[0]
        print(f"  {src:5s} [{age_group:5s}]  patients={sub['patient_id'].nunique():4d}  "
              f"images={len(sub):4d}  "
              f"Hgb min/mean/max = {hb.min():4.1f}/{hb.mean():4.1f}/{hb.max():4.1f}  "
              f"anemic = {anemic_pct:5.1f}%")
    ages = uniq["age_group"].value_counts().to_dict()
    print(f"  age groups (patients): adult={ages.get('adult', 0)}  "
          f"child={ages.get('child', 0)}")
    print(f"  TOTAL  patients={uniq['patient_id'].nunique()}  images={len(df)}")


# ---- spreadsheet readers ---------------------------------------------------
def _read_spreadsheet(path: Path) -> pd.DataFrame:
    """Eyes-defy sheet: keep ONLY Number/Hgb/Gender/Age (ignore junk Unnamed)."""
    return _read_selected(path, {
        "number": config.NUMBER_COL,
        "hb": config.HB_COL,
        "gender": config.GENDER_COL,
        "age": config.AGE_COL,
    })


def _read_cpanemic_sheet(path: Path) -> pd.DataFrame:
    """CP-AnemiC sheet: keep IMAGE_ID / HB_LEVEL / GENDER (ignore the rest)."""
    return _read_selected(path, {
        "image_id": config.CPANEMIC_IMAGE_ID_COL,
        "hb": config.CPANEMIC_HB_COL,
        "gender": config.CPANEMIC_GENDER_COL,
    })


def _read_selected(path: Path, wanted: dict) -> pd.DataFrame:
    """Read an .xlsx and return columns renamed per `wanted` (canonical -> header),
    matched case-insensitively. Missing headers become all-NaN columns."""
    raw = pd.read_excel(path)
    lower = {str(c).strip().lower(): c for c in raw.columns}

    def col(header):
        c = lower.get(header.lower())
        return raw[c] if c is not None else pd.Series([np.nan] * len(raw))

    return pd.DataFrame({canon: col(header) for canon, header in wanted.items()})


# ---- image helpers ---------------------------------------------------------
def _palpebral_images(folder: Path) -> list:
    """All usable palpebral crops in an Eyes-defy patient folder.

    A file qualifies if its name ends in config.PALPEBRAL_SUFFIX and contains
    none of config.PALPEBRAL_EXCLUDE (so "_forniceal_palpebral.png" is dropped).
    """
    out = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not name.endswith(config.PALPEBRAL_SUFFIX):
            continue
        if any(bad in name for bad in config.PALPEBRAL_EXCLUDE):
            continue
        out.append(p)
    return out


def _is_readable(path: Path) -> bool:
    """True if PIL can open and verify the image (catches corrupt/truncated files)."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001 - any decode error means "skip this file"
        return False


# ---- value cleaners --------------------------------------------------------
def _clean_number(value):
    """Normalize an Eyes-defy Number into a folder-name string ('12'), or None."""
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    s = str(value).strip()
    if not s:
        return None
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s


def _clean_image_id(value):
    """Normalize a CP-AnemiC IMAGE_ID into the PNG stem (e.g. 'Image_001'), or None."""
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    s = str(value).strip()
    return s or None


def _clean_gender(value):
    """Normalize gender to 'M' / 'F' (handles 'M'/'F' and 'Male'/'Female'), or ''."""
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    return s[:1] if s[:1] in ("M", "F") else ""


def _to_float(value):
    """Coerce a possibly-string value (e.g. '15') to float, or None."""
    if pd.isna(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 2. Resolve task mode
# ---------------------------------------------------------------------------
def resolve_task(df: pd.DataFrame, requested: str = None) -> str:
    requested = requested or config.TASK
    if requested == "classification":
        if df["anemic"].notna().sum() < 2:
            raise ValueError("Classification requested but no anemia labels found.")
        return "classification"
    if requested == "regression":
        if df["hgb"].notna().sum() < 2:
            raise ValueError("Regression requested but no Hgb values found.")
        return "regression"
    # auto: prefer regression when real Hgb values exist.
    if df["hgb"].notna().sum() >= 2:
        return "regression"
    return "classification"


# ---------------------------------------------------------------------------
# 3. Frozen, patient-level, SOURCE-STRATIFIED split
# ---------------------------------------------------------------------------
def make_or_load_splits(df: pd.DataFrame, task: str = None) -> dict:
    """
    Returns {"train": [idx...], "val": [...], "test": [...]} indexing into df.

    Keyed by patient_id so a patient never spans splits, and stratified by source
    so each split carries India + Italy + Ghana in proportion. Computed once and
    cached at config.SPLIT_PATH (as patient_id lists), reused after.
    """
    patients = sorted(df["patient_id"].unique())
    source_of = (df.drop_duplicates("patient_id")
                   .set_index("patient_id")["source"].to_dict())

    if config.SPLIT_PATH.exists():
        saved = json.loads(config.SPLIT_PATH.read_text())
        saved_patients = {p for grp in saved.values() for p in grp}
        if saved_patients == set(patients):
            splits = {name: _rows_for(df, saved[name]) for name in ("train", "val", "test")}
            print(f"[data] Loaded frozen patient-level split from {config.SPLIT_PATH}")
            return splits
        print(f"[data] Saved split no longer matches the patient set "
              f"({len(saved_patients)} saved vs {len(patients)} now) — recomputing.")

    patient_splits = _split_patients(patients, source_of)
    config.SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.SPLIT_PATH.write_text(json.dumps(patient_splits, indent=2))
    print(f"[data] Saved frozen patient-level, source-stratified split to "
          f"{config.SPLIT_PATH}")

    return {name: _rows_for(df, patient_splits[name]) for name in ("train", "val", "test")}


def _split_patients(patients: list, source_of: dict) -> dict:
    """Seeded patient-level split, stratified by source (each split gets a share
    of every source's patients)."""
    rng = np.random.default_rng(config.SEED)
    ratios = config.SPLIT_RATIOS

    by_source = defaultdict(list)
    for p in patients:
        by_source[source_of[p]].append(p)

    train, val, test = [], [], []
    for src in sorted(by_source):  # deterministic source order
        order = np.array(sorted(by_source[src]), dtype=object)
        rng.shuffle(order)
        n = len(order)
        n_train = int(round(ratios["train"] * n))
        n_val = int(round(ratios["val"] * n))
        # Guard tiny per-source counts so val/test aren't starved.
        n_train = min(n_train, max(n - 2, 0))
        n_val = min(n_val, max(n - n_train - 1, 0))
        train += order[:n_train].tolist()
        val += order[n_train:n_train + n_val].tolist()
        test += order[n_train + n_val:].tolist()

    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def _rows_for(df: pd.DataFrame, patient_ids) -> list:
    """All df row indices belonging to the given patient_ids."""
    wanted = set(patient_ids)
    return df.index[df["patient_id"].isin(wanted)].tolist()


# ---------------------------------------------------------------------------
# 3b. Tight-crop preprocessing — equalize framing across sources
# ---------------------------------------------------------------------------
# Two image types exist:
#   * CLEAN CUTOUT — eyelid on a BLACK background. The non-black region IS the
#     eyelid, so we bbox the non-black pixels.
#   * RAW PHOTO    — a close-up with full skin/background and NO black border.
#     The non-black trick fails (everything is "foreground"), so we find the
#     conjunctiva by COLOR (reddish + saturated in HSV) and bbox that instead.
# We detect the type from the black-pixel fraction, then crop accordingly so the
# eyelid fills the frame the same way for every source.
def _gray(img) -> np.ndarray:
    return np.asarray(img.convert("L"))


def detect_image_type(img, thresh: int = None) -> str:
    """'raw' if there is almost no black border, else 'cutout'."""
    thresh = config.TIGHT_CROP_BLACK_THRESHOLD if thresh is None else thresh
    black_frac = float((_gray(img) <= thresh).mean())
    return "raw" if black_frac < config.RAW_PHOTO_BLACK_FRAC else "cutout"


def _conjunctiva_mask(img) -> np.ndarray:
    """Reddish, saturated pixels (the conjunctiva) — for raw photos with no black."""
    hsv = np.asarray(img.convert("HSV"))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    reddish = (h <= config.CONJ_HUE_RED_HI) | (h >= config.CONJ_HUE_RED_LO_WRAP)
    return reddish & (s >= config.CONJ_MIN_SAT) & (v >= config.CONJ_MIN_VAL)


def eyelid_mask(img, thresh: int = None):
    """Return (mask, image_type). Cutouts use the non-black mask; raw photos use
    the conjunctiva-color mask. `foreground_fraction` and the crop both use this,
    so the 'eyelid fraction' is measured the right way for each image type."""
    thresh = config.TIGHT_CROP_BLACK_THRESHOLD if thresh is None else thresh
    rgb = img.convert("RGB")
    image_type = detect_image_type(rgb, thresh)
    if image_type == "cutout":
        return (_gray(rgb) > thresh), "cutout"
    return _conjunctiva_mask(rgb), "raw"


def foreground_fraction(img) -> float:
    """Eyelid fraction, measured per image type (non-black for cutouts,
    conjunctiva-color for raw photos)."""
    mask, _ = eyelid_mask(img)
    return float(mask.mean())


def _center_square(rgb, size: int):
    """Largest centered square, resized — the low-confidence fallback crop."""
    W, H = rgb.size
    side = min(W, H)
    left, top = (W - side) // 2, (H - side) // 2
    return rgb.crop((left, top, left + side, top + side)).resize((size, size))


def _bbox_crop(rgb, mask, margin: float, size: int):
    H, W = mask.shape
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    mx = int(round((x1 - x0 + 1) * margin))
    my = int(round((y1 - y0 + 1) * margin))
    x0 = max(0, x0 - mx); y0 = max(0, y0 - my)
    x1 = min(W - 1, x1 + mx); y1 = min(H - 1, y1 + my)
    return rgb.crop((x0, y0, x1 + 1, y1 + 1)).resize((size, size))


def tight_crop_image(img, size: int = None, margin: float = None,
                     thresh: int = None, min_fg: int = None):
    """Type-aware tight crop to the eyelid, resized to a square.

    CUTOUT -> bbox of non-black pixels. RAW -> bbox of conjunctiva-color pixels.
    If the mask is too small to trust (all-black cutout, or no confident
    conjunctiva in a raw photo), fall back to a center crop.

    Returns (cropped_resized_RGB_image, info) where
        info = {"type": "cutout"|"raw", "fallback": bool}.
    """
    size = config.IMAGE_SIZE if size is None else size
    margin = config.TIGHT_CROP_MARGIN if margin is None else margin
    thresh = config.TIGHT_CROP_BLACK_THRESHOLD if thresh is None else thresh
    min_fg = config.TIGHT_CROP_MIN_FG_PIXELS if min_fg is None else min_fg

    rgb = img.convert("RGB")
    mask, image_type = eyelid_mask(rgb, thresh)
    H, W = mask.shape
    fg = int(mask.sum())

    if image_type == "raw":
        min_needed = max(min_fg, int(config.CONJ_MIN_FRAC * H * W))
    else:
        min_needed = min_fg

    if fg < min_needed:
        # Low confidence — center-crop and flag it.
        return _center_square(rgb, size), {"type": image_type, "fallback": True}
    return _bbox_crop(rgb, mask, margin, size), {"type": image_type, "fallback": False}


class TightCrop:
    """Picklable transform wrapper (lambdas aren't picklable for DataLoader
    workers). Applies tight_crop_image, returning just the cropped image."""

    def __init__(self, size=None, margin=None, thresh=None, min_fg=None):
        self.size = config.IMAGE_SIZE if size is None else size
        self.margin = config.TIGHT_CROP_MARGIN if margin is None else margin
        self.thresh = config.TIGHT_CROP_BLACK_THRESHOLD if thresh is None else thresh
        self.min_fg = config.TIGHT_CROP_MIN_FG_PIXELS if min_fg is None else min_fg

    def __call__(self, img):
        return tight_crop_image(img, self.size, self.margin, self.thresh, self.min_fg)[0]


def _sample_paths(sub: pd.DataFrame, cap):
    """Deterministic, evenly-spaced sample of image_paths (or all if under cap)."""
    paths = sub.drop_duplicates("image_path")["image_path"].tolist()
    if cap and len(paths) > cap:
        step = len(paths) / cap
        paths = [paths[int(i * step)] for i in range(cap)]
    return paths


def foreground_report(df: pd.DataFrame, sample_per_source: int = 250,
                      verbose: bool = True) -> dict:
    """Median eyelid (foreground) fraction per source, BEFORE vs AFTER the crop —
    after the type-aware crop all three sources should land at a SIMILAR value.
    Also counts how many images were cutout vs raw vs fallback per source (so you
    can see how many India images are the raw type). Samples up to
    `sample_per_source` images per source to stay fast on the real data."""
    if verbose:
        print("\n[crop] Eyelid (foreground) fraction — median per source, "
              "before vs after, with type counts:")
        print(f"  {'source':6} {'n':>5} {'before':>8} {'after':>8}   "
              f"{'cutout':>6} {'raw':>5} {'fallback':>8}")
    out = {}
    for src in SOURCE_ORDER:
        sub = df[df["source"] == src]
        if sub.empty:
            continue
        before, after = [], []
        counts = {"cutout": 0, "raw": 0, "fallback": 0}
        for p in _sample_paths(sub, sample_per_source):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            before.append(foreground_fraction(img))
            cropped, info = tight_crop_image(img)
            after.append(foreground_fraction(cropped))
            if info["fallback"]:
                counts["fallback"] += 1
            else:
                counts[info["type"]] += 1
        if not before:
            continue
        mb, ma = float(np.median(before)), float(np.median(after))
        out[src] = {"before_median": mb, "after_median": ma,
                    "n": len(before), **counts}
        if verbose:
            print(f"  {src:6} {len(before):>5} {mb:>8.3f} {ma:>8.3f}   "
                  f"{counts['cutout']:>6} {counts['raw']:>5} {counts['fallback']:>8}")
    return out


def save_crop_preview(df: pd.DataFrame, path=None, per_source: int = 4):
    """Save a (sources x per_source) grid of CROPPED images, each tile labelled
    with its detected type (cutout / raw / fallback). Default: results/crop_preview.png."""
    from PIL import ImageDraw

    path = Path(path) if path else (config.RESULTS_DIR / "crop_preview.png")
    sources = [s for s in SOURCE_ORDER if (df["source"] == s).any()]
    cell = config.IMAGE_SIZE
    grid = Image.new("RGB", (per_source * cell, len(sources) * cell), (15, 15, 18))
    draw = ImageDraw.Draw(grid)

    for r, src in enumerate(sources):
        paths = (df[df["source"] == src].drop_duplicates("patient_id")
                 ["image_path"].tolist()[:per_source])
        for c, p in enumerate(paths):
            try:
                cropped, info = tight_crop_image(Image.open(p).convert("RGB"))
            except Exception:  # noqa: BLE001
                continue
            grid.paste(cropped, (c * cell, r * cell))
            # Per-tile type label, bottom-left.
            tile_label = "fallback" if info["fallback"] else info["type"]
            ty = r * cell + cell - 18
            draw.rectangle([c * cell + 2, ty, c * cell + 78, ty + 16], fill=(0, 0, 0))
            draw.text((c * cell + 5, ty + 3), tile_label, fill=(255, 220, 120))
        # Source label, top-left of the row.
        draw.rectangle([2, r * cell + 2, 86, r * cell + 20], fill=(0, 0, 0))
        draw.text((6, r * cell + 6), src, fill=(255, 255, 255))

    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
    print(f"[crop] preview grid saved to {path}")
    return path


# ---------------------------------------------------------------------------
# 4. Dataset + DataLoaders
# ---------------------------------------------------------------------------
def _build_transforms(train: bool):
    base = []
    if config.PREPROCESS_TIGHT_CROP:
        # Equalize framing first; outputs a config.IMAGE_SIZE square.
        base.append(TightCrop())
    # No-op when already cropped to size; the real resize when cropping is off.
    base.append(transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
    if train:
        # Light, label-preserving augmentation. Conjunctiva color is the signal,
        # so we avoid aggressive color jitter that could destroy pallor cues.
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ]
    return transforms.Compose(base)


class ConjunctivaDataset(Dataset):
    """Yields (image_tensor, target). Target is the Hgb value for regression,
    or the anemic label for classification."""

    def __init__(self, df: pd.DataFrame, indices, task: str, train: bool):
        self.df = df
        self.indices = list(indices)
        self.task = task
        self.transform = _build_transforms(train)
        self._warned = set()  # image paths we've already logged as bad

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.df.loc[self.indices[i]]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:  # noqa: BLE001 - a single bad file must not kill a run
            path = row["image_path"]
            if path not in self._warned:
                print(f"[data] WARNING: failed to load {path}; "
                      f"falling back to a valid sample.")
                self._warned.add(path)
            # Fall back to a known-good sample so the batch stays intact.
            if i != 0:
                return self[0]
            raise  # index 0 itself is unreadable — nothing safe to fall back to
        image = self.transform(image)
        if self.task == "classification":
            target = torch.tensor(float(row["anemic"]), dtype=torch.float32)
        else:
            target = torch.tensor(float(row["hgb"]), dtype=torch.float32)
        return image, target


def make_loader(df: pd.DataFrame, indices, task: str, train: bool, shuffle=None):
    """One DataLoader over `indices`. Reused by get_dataloaders and crossval."""
    if shuffle is None:
        shuffle = train
    generator = torch.Generator()
    generator.manual_seed(config.SEED)
    ds = ConjunctivaDataset(df, indices, task, train=train)
    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def get_dataloaders(df: pd.DataFrame, task: str, splits: dict):
    """Build train/val/test DataLoaders from a df + frozen split indices."""
    return {
        name: make_loader(df, splits[name], task, train=(name == "train"))
        for name in ("train", "val", "test")
    }


def genders_for_split(df: pd.DataFrame, indices) -> np.ndarray:
    """Gender per sample in the SAME order a (shuffle=False) loader yields them.

    Lets train/evaluate apply the gender-aware anemia cutoff to predictions.
    """
    return np.array([df.loc[i, "gender"] for i in indices], dtype=object)


def sources_for_split(df: pd.DataFrame, indices) -> np.ndarray:
    """Source per sample, aligned to a (shuffle=False) loader's order."""
    return np.array([df.loc[i, "source"] for i in indices], dtype=object)


# Convenience: assemble everything in one call.
def prepare_data(requested_task: str = None, verbose: bool = False):
    df = build_dataframe(verbose=verbose)
    task = resolve_task(df, requested_task)
    splits = make_or_load_splits(df, task)
    loaders = get_dataloaders(df, task, splits)
    return df, task, splits, loaders

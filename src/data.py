"""
data.py — turn the (unknown-structure) dataset into PyTorch DataLoaders.

The flow:
  1. build_dataframe()  -> a tidy DataFrame with columns:
        image_path : absolute path to the image
        anemic     : 0/1 or NaN
        hb         : float g/dL or NaN
     It discovers metadata via config.METADATA_CANDIDATES + column candidates,
     and falls back to folder-name labels if no usable CSV exists.
  2. resolve_task()     -> picks "classification" or "regression" when TASK="auto".
  3. make_or_load_splits() -> frozen train/val/test split saved to disk.
  4. get_dataloaders()  -> ready-to-train DataLoaders.

WHERE TO ADJUST after running explore_data.py: almost everything is driven by
config.py (METADATA_CANDIDATES, IMAGE_COL, ANEMIA_COL, HB_COL, *_FOLDER_HINTS).
You should rarely need to touch the logic here.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from utils import seed_worker  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Discover labels -> tidy dataframe
# ---------------------------------------------------------------------------
def build_dataframe(verbose: bool = False) -> pd.DataFrame:
    root = config.DATA_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist: {root}. Download the dataset or set "
            f"OVISION_DATA_ROOT."
        )

    # Index every image by basename so we can join metadata rows to real files.
    image_paths = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {root}")
    by_name = {p.name: p for p in image_paths}
    by_stem = {p.stem: p for p in image_paths}  # filename without extension

    meta_path = _find_metadata(root)
    if meta_path is not None:
        if verbose:
            print(f"[data] Using metadata file: {meta_path}")
        df = _dataframe_from_metadata(meta_path, by_name, by_stem, verbose)
    else:
        if verbose:
            print("[data] No metadata file found — using folder-name labels.")
        df = _dataframe_from_folders(image_paths, verbose)

    # Keep only rows that have at least one usable label.
    has_label = df["anemic"].notna() | df["hb"].notna()
    df = df[has_label].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "Found images but could not attach any labels. Check the column / "
            "folder-name settings in config.py against explore_data.py output."
        )
    if verbose:
        print(f"[data] {len(df)} labeled images assembled.")
    return df


def _find_metadata(root: Path):
    """Return the first existing metadata file from config.METADATA_CANDIDATES."""
    # Exact-name candidates first (anywhere in the tree).
    for name in config.METADATA_CANDIDATES:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    # Otherwise: any CSV at all, so a differently-named file still gets noticed.
    csvs = sorted(root.rglob("*.csv"))
    return csvs[0] if csvs else None


def _pick_column(df: pd.DataFrame, preferred: str, candidates) -> str | None:
    """Find a column matching `preferred` or any candidate, case-insensitively."""
    lower = {c.lower(): c for c in df.columns}
    for name in (preferred, *candidates):
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _dataframe_from_metadata(meta_path, by_name, by_stem, verbose) -> pd.DataFrame:
    raw = pd.read_csv(meta_path)

    img_col = _pick_column(raw, config.IMAGE_COL, config.IMAGE_COL_CANDIDATES)
    anemia_col = _pick_column(raw, config.ANEMIA_COL, config.ANEMIA_COL_CANDIDATES)
    hb_col = _pick_column(raw, config.HB_COL, config.HB_COL_CANDIDATES)

    if verbose:
        print(f"[data] columns -> image:{img_col}  anemic:{anemia_col}  hb:{hb_col}")
    if img_col is None:
        raise ValueError(
            f"No image-filename column found in {meta_path}. Columns are "
            f"{list(raw.columns)}. Set IMAGE_COL in config.py."
        )

    rows = []
    unmatched = 0
    for _, r in raw.iterrows():
        path = _match_image(str(r[img_col]), by_name, by_stem)
        if path is None:
            unmatched += 1
            continue
        rows.append({
            "image_path": str(path),
            "anemic": _to_binary(r[anemia_col]) if anemia_col else np.nan,
            "hb": _to_float(r[hb_col]) if hb_col else np.nan,
        })
    if verbose and unmatched:
        print(f"[data] {unmatched} metadata rows had no matching image file.")

    df = pd.DataFrame(rows)
    # If we have Hb but no explicit anemic label, derive it from the threshold.
    if "hb" in df and df["anemic"].isna().all() and df["hb"].notna().any():
        df["anemic"] = (df["hb"] < config.ANEMIA_HB_THRESHOLD).astype(float)
        df.loc[df["hb"].isna(), "anemic"] = np.nan
    return df


def _dataframe_from_folders(image_paths, verbose) -> pd.DataFrame:
    """Label each image from hints in its folder path (no CSV available)."""
    rows = []
    for p in image_paths:
        path_str = str(p).lower()
        anemic = np.nan
        if any(h in path_str for h in config.NONANEMIC_FOLDER_HINTS):
            anemic = 0.0
        if any(h in path_str for h in config.ANEMIC_FOLDER_HINTS):
            anemic = 1.0  # anemic hint wins if both somehow match
        rows.append({"image_path": str(p), "anemic": anemic, "hb": np.nan})
    return pd.DataFrame(rows)


def _match_image(name: str, by_name: dict, by_stem: dict):
    name = name.strip()
    base = Path(name).name
    if base in by_name:
        return by_name[base]
    stem = Path(base).stem
    if stem in by_stem:
        return by_stem[stem]
    return None


def _to_binary(value):
    """Coerce assorted anemia encodings (1/0, yes/no, anemic/normal) to 0/1."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in ("1", "1.0", "yes", "y", "true", "anemic", "anaemic", "positive", "pos"):
        return 1.0
    if s in ("0", "0.0", "no", "n", "false", "non-anemic", "normal", "healthy",
             "negative", "neg", "control"):
        return 0.0
    try:
        return float(float(s) > 0)
    except ValueError:
        return np.nan


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# 2. Resolve task mode
# ---------------------------------------------------------------------------
def resolve_task(df: pd.DataFrame, requested: str = None) -> str:
    requested = requested or config.TASK
    if requested == "regression":
        if df["hb"].notna().sum() < 2:
            raise ValueError("Regression requested but no Hb values found.")
        return "regression"
    if requested == "classification":
        if df["anemic"].notna().sum() < 2:
            raise ValueError("Classification requested but no anemia labels found.")
        return "classification"
    # auto: prefer regression when real Hb values exist, else classification.
    if df["hb"].notna().sum() >= max(10, 0.5 * len(df)):
        return "regression"
    return "classification"


# ---------------------------------------------------------------------------
# 3. Frozen splits (HARD REQUIREMENT: saved to disk, stable across runs)
# ---------------------------------------------------------------------------
def make_or_load_splits(df: pd.DataFrame, task: str) -> dict:
    """
    Returns {"train": [idx...], "val": [...], "test": [...]} indexing into df.

    Split is keyed by image_path so it stays stable even if df row order changes.
    Computed once and cached at config.SPLIT_PATH; reused on every later run.
    """
    path_to_idx = {row.image_path: i for i, row in df.iterrows()}

    if config.SPLIT_PATH.exists():
        saved = json.loads(config.SPLIT_PATH.read_text())
        splits = {}
        for name in ("train", "val", "test"):
            splits[name] = [path_to_idx[p] for p in saved[name] if p in path_to_idx]
        missing = len(df) - sum(len(v) for v in splits.values())
        if missing == 0:
            print(f"[data] Loaded frozen split from {config.SPLIT_PATH}")
            return splits
        print(f"[data] Saved split is stale ({missing} new/changed images) — "
              f"recomputing.")

    splits = _stratified_split(df, task)

    # Persist by image_path (not index) so the split survives reordering.
    saveable = {
        name: [df.loc[i, "image_path"] for i in idxs]
        for name, idxs in splits.items()
    }
    config.SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.SPLIT_PATH.write_text(json.dumps(saveable, indent=2))
    print(f"[data] Saved frozen split to {config.SPLIT_PATH}")
    return splits


def _stratified_split(df: pd.DataFrame, task: str) -> dict:
    """Seeded split; stratified on the anemia label when available."""
    rng = np.random.default_rng(config.SEED)
    ratios = config.SPLIT_RATIOS

    # Stratify on the binary label so each split has both classes when possible.
    if df["anemic"].notna().all() and df["anemic"].nunique() > 1:
        groups = [df.index[df["anemic"] == c].to_numpy() for c in sorted(df["anemic"].unique())]
    else:
        groups = [df.index.to_numpy()]

    train, val, test = [], [], []
    for g in groups:
        g = g.copy()
        rng.shuffle(g)
        n = len(g)
        n_train = int(round(ratios["train"] * n))
        n_val = int(round(ratios["val"] * n))
        train.extend(g[:n_train])
        val.extend(g[n_train:n_train + n_val])
        test.extend(g[n_train + n_val:])

    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


# ---------------------------------------------------------------------------
# 4. Dataset + DataLoaders
# ---------------------------------------------------------------------------
def _build_transforms(train: bool):
    base = [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    ]
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
    """Yields (image_tensor, target). Target is the anemia label for
    classification, or the Hb value for regression."""

    def __init__(self, df: pd.DataFrame, indices, task: str, train: bool):
        self.df = df
        self.indices = list(indices)
        self.task = task
        self.transform = _build_transforms(train)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.df.loc[self.indices[i]]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        if self.task == "classification":
            target = torch.tensor(float(row["anemic"]), dtype=torch.float32)
        else:
            target = torch.tensor(float(row["hb"]), dtype=torch.float32)
        return image, target


def get_dataloaders(df: pd.DataFrame, task: str, splits: dict):
    """Build train/val/test DataLoaders from a df + frozen split indices."""
    generator = torch.Generator()
    generator.manual_seed(config.SEED)

    loaders = {}
    for name in ("train", "val", "test"):
        is_train = name == "train"
        ds = ConjunctivaDataset(df, splits[name], task, train=is_train)
        loaders[name] = DataLoader(
            ds,
            batch_size=config.BATCH_SIZE,
            shuffle=is_train,
            num_workers=config.NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        )
    return loaders


# Convenience: assemble everything in one call.
def prepare_data(requested_task: str = None, verbose: bool = False):
    df = build_dataframe(verbose=verbose)
    task = resolve_task(df, requested_task)
    splits = make_or_load_splits(df, task)
    loaders = get_dataloaders(df, task, splits)
    return df, task, splits, loaders

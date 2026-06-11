"""
data.py — turn the real Eyes-defy-anemia dataset into PyTorch DataLoaders.

Real layout (under config.DATA_ROOT):
    India/  India.xlsx   India/<Number>/<images>
    Italy/  Italy.xlsx   Italy/<Number>/<images>

Each spreadsheet row (columns Number, Hgb, Gender, Age) describes one patient
whose images live in the folder named after their Number. We use ONLY the
pre-cropped eyelid image ending in "_palpebral.png".

The flow:
  1. build_dataframe()      -> one row per usable image, with columns:
         image_path : absolute path to the *_palpebral.png crop
         patient_id : "<Country>/<Number>" — unique per patient
         country, number, gender, age
         hb         : float Hgb (regression target)
         anemic     : ground-truth anemic flag from the gender-aware cutoff
     Patients missing a folder, a _palpebral.png, or a usable Hgb are skipped
     gracefully and counted (printed at the end).
  2. resolve_task()         -> "regression" (Hgb values present).
  3. make_or_load_splits()  -> frozen PATIENT-LEVEL train/val/test split. All
     of a patient's images stay on the SAME side. Saved to config.SPLIT_PATH.
  4. get_dataloaders()      -> ready-to-train DataLoaders.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# Tolerate slightly truncated images rather than erroring on the last few bytes.
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from utils import seed_worker  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Walk the real structure -> tidy dataframe (one row per usable image)
# ---------------------------------------------------------------------------
def build_dataframe(verbose: bool = False) -> pd.DataFrame:
    root = config.DATA_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist: {root}. Download the dataset or set "
            f"OVISION_DATA_ROOT."
        )

    rows = []
    # Patient-level bookkeeping so we can report honest usable N.
    kept = 0
    skipped_no_folder = 0
    skipped_no_image = 0
    skipped_bad_hb = 0
    skipped_unreadable = 0  # per-image: corrupt/truncated/not-really-a-PNG files
    seen_patients = 0

    for country in config.COUNTRIES:
        sheet = config.spreadsheet_path(root, country)
        if not sheet.exists():
            if verbose:
                print(f"[data] {country}: spreadsheet not found at {sheet} — skipping country.")
            continue

        table = _read_spreadsheet(sheet)
        if verbose:
            print(f"[data] {country}: {len(table)} spreadsheet rows from {sheet.name}")

        for _, r in table.iterrows():
            seen_patients += 1
            number = _clean_number(r["number"])
            if number is None:
                skipped_no_folder += 1
                continue

            hb = _to_float(r["hb"])
            if hb is None or not np.isfinite(hb):
                skipped_bad_hb += 1
                continue

            folder = root / country / number
            if not folder.is_dir():
                skipped_no_folder += 1
                continue

            images = _palpebral_images(folder)
            if not images:
                skipped_no_image += 1
                continue

            # Verify each crop actually opens; drop corrupt/truncated files so a
            # bad image can't crash training later.
            readable = []
            for img in images:
                if _is_readable(img):
                    readable.append(img)
                else:
                    skipped_unreadable += 1
                    if verbose:
                        print(f"[data] unreadable image skipped: {img}")
            if not readable:
                # Every crop for this patient was unreadable -> no usable images.
                skipped_no_image += 1
                continue
            images = readable

            gender = _clean_gender(r["gender"])
            age = _to_float(r["age"])
            patient_id = f"{country}/{number}"
            anemic = float(hb < config.anemia_cutoff(gender)) if gender else np.nan

            kept += 1
            for img in images:
                rows.append({
                    "image_path": str(img),
                    "patient_id": patient_id,
                    "country": country,
                    "number": number,
                    "gender": gender,
                    "age": age,
                    "hb": hb,
                    "anemic": anemic,
                })

    skipped = skipped_no_folder + skipped_no_image + skipped_bad_hb
    print(
        f"[data] Patients seen: {seen_patients}  kept: {kept}  skipped: {skipped} "
        f"(no/invalid folder: {skipped_no_folder}, no _palpebral.png: "
        f"{skipped_no_image}, unusable Hgb: {skipped_bad_hb})  "
        f"unreadable images dropped: {skipped_unreadable}"
    )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "Walked the dataset but assembled 0 usable rows. Check DATA_ROOT / "
            "OVISION_DATA_ROOT and the structure constants in config.py."
        )

    if verbose:
        per_country = Counter(df["country"])
        print(f"[data] Usable images: {len(df)} from {df['patient_id'].nunique()} "
              f"patients  ({dict(per_country)} images per country)")
    return df


def _read_spreadsheet(path: Path) -> pd.DataFrame:
    """Read a country spreadsheet, keeping ONLY Number/Hgb/Gender/Age.

    Italy's sheet carries extra junk "Unnamed" columns; matching the wanted
    columns case-insensitively by name ignores them. Returns a frame with
    canonical lowercase column names: number, hb, gender, age.
    """
    raw = pd.read_excel(path)
    lower = {str(c).strip().lower(): c for c in raw.columns}

    def col(name):
        c = lower.get(name.lower())
        return raw[c] if c is not None else pd.Series([np.nan] * len(raw))

    return pd.DataFrame({
        "number": col(config.NUMBER_COL),
        "hb": col(config.HB_COL),
        "gender": col(config.GENDER_COL),
        "age": col(config.AGE_COL),
    })


def _palpebral_images(folder: Path) -> list:
    """All usable palpebral crops in a patient folder.

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
    """True if PIL can open and verify the image (catches corrupt/truncated files).

    verify() consumes the file object, so it must be called on a fresh handle and
    the image re-opened before any real use — which __getitem__ does separately.
    """
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001 - any decode error means "skip this file"
        return False


def _clean_number(value):
    """Normalize a spreadsheet Number into a folder-name string ('12'), or None."""
    if pd.isna(value):
        return None
    # Numbers often read back as floats (12.0); render whole numbers cleanly.
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


def _clean_gender(value):
    """Normalize gender to 'M' / 'F' (single uppercase letter), or '' if unknown."""
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
        if df["hb"].notna().sum() < 2:
            raise ValueError("Regression requested but no Hgb values found.")
        return "regression"
    # auto: prefer regression when real Hgb values exist.
    if df["hb"].notna().sum() >= 2:
        return "regression"
    return "classification"


# ---------------------------------------------------------------------------
# 3. Frozen PATIENT-LEVEL split (HARD REQUIREMENT)
# ---------------------------------------------------------------------------
def make_or_load_splits(df: pd.DataFrame, task: str = None) -> dict:
    """
    Returns {"train": [idx...], "val": [...], "test": [...]} indexing into df.

    The split is keyed by patient_id: every image of a patient lands in exactly
    one split, so a patient can NEVER appear in both train and test. Computed
    once and cached at config.SPLIT_PATH (as patient_id lists), reused after.
    """
    patients = sorted(df["patient_id"].unique())

    saved_patients = None
    if config.SPLIT_PATH.exists():
        saved = json.loads(config.SPLIT_PATH.read_text())
        saved_patients = {p for grp in saved.values() for p in grp}
        if saved_patients == set(patients):
            splits = {name: _rows_for(df, saved[name]) for name in ("train", "val", "test")}
            print(f"[data] Loaded frozen patient-level split from {config.SPLIT_PATH}")
            return splits
        print(f"[data] Saved split no longer matches the patient set "
              f"({len(saved_patients)} saved vs {len(patients)} now) — recomputing.")

    patient_splits = _split_patients(patients)
    config.SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.SPLIT_PATH.write_text(json.dumps(patient_splits, indent=2))
    print(f"[data] Saved frozen patient-level split to {config.SPLIT_PATH}")

    return {name: _rows_for(df, patient_splits[name]) for name in ("train", "val", "test")}


def _split_patients(patients: list) -> dict:
    """Seeded patient-level split into train/val/test by config.SPLIT_RATIOS."""
    rng = np.random.default_rng(config.SEED)
    order = np.array(patients, dtype=object)
    rng.shuffle(order)

    n = len(order)
    ratios = config.SPLIT_RATIOS
    n_train = int(round(ratios["train"] * n))
    n_val = int(round(ratios["val"] * n))
    # Guard tiny datasets so val/test aren't starved when n is small.
    n_train = min(n_train, max(n - 2, 0))
    n_val = min(n_val, max(n - n_train - 1, 0))

    return {
        "train": sorted(order[:n_train].tolist()),
        "val": sorted(order[n_train:n_train + n_val].tolist()),
        "test": sorted(order[n_train + n_val:].tolist()),
    }


def _rows_for(df: pd.DataFrame, patient_ids) -> list:
    """All df row indices belonging to the given patient_ids."""
    wanted = set(patient_ids)
    return df.index[df["patient_id"].isin(wanted)].tolist()


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


def genders_for_split(df: pd.DataFrame, indices) -> np.ndarray:
    """Gender per sample in the SAME order a (shuffle=False) loader yields them.

    Lets train/evaluate apply the gender-aware anemia cutoff to predictions.
    """
    return np.array([df.loc[i, "gender"] for i in indices], dtype=object)


# Convenience: assemble everything in one call.
def prepare_data(requested_task: str = None, verbose: bool = False):
    df = build_dataframe(verbose=verbose)
    task = resolve_task(df, requested_task)
    splits = make_or_load_splits(df, task)
    loaders = get_dataloaders(df, task, splits)
    return df, task, splits, loaders

"""
Central configuration for OVision v0.

Everything that might change between machines, datasets, or experiments lives
here so you never have to hunt through the code. The most important knobs:

  * DATA_ROOT        - dataset location. OVISION_DATA_ROOT is the source of
                       truth; we auto-detect the Kaggle path as a fallback.
  * TASK             - this dataset is hemoglobin REGRESSION (predict Hgb).
  * The DATASET STRUCTURE block - how we walk the real Eyes-defy-anemia layout
                       (country subfolders, per-country spreadsheets, numbered
                       patient folders, *_palpebral.png crops).
  * ANEMIA_CUTOFFS   - gender-aware Hgb cutoffs used to turn a predicted Hgb
                       into an anemic / not-anemic call.
"""

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# 1. Dataset location  (OVISION_DATA_ROOT is the source of truth)
# ---------------------------------------------------------------------------
# Priority:
#   1. OVISION_DATA_ROOT environment variable (explicit override — preferred).
#   2. The real Kaggle input path, if it exists (note the SPACE in the last
#      folder name: ".../dataset anemia").
#   3. A local ./data folder next to this file.
#
# Never hardcode an absolute dataset path anywhere else in the repo. Import
# DATA_ROOT from here instead.

_KAGGLE_DATA_ROOT = Path(
    "/kaggle/input/datasets/harshwardhanfartale/eyes-defy-anemia/dataset anemia"
)
_LOCAL_DATA_ROOT = Path(__file__).resolve().parent / "data"


def get_data_root() -> Path:
    env_override = os.environ.get("OVISION_DATA_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    if _KAGGLE_DATA_ROOT.exists():
        return _KAGGLE_DATA_ROOT
    return _LOCAL_DATA_ROOT


DATA_ROOT = get_data_root()

# Where we write things. Kept out of the dataset folder so the dataset stays
# read-only (important on Kaggle, where /kaggle/input is read-only).
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
SPLIT_PATH = PROJECT_ROOT / "splits.json"  # frozen, patient-level split

# ---------------------------------------------------------------------------
# 2. Dataset structure  (the REAL Eyes-defy-anemia layout)
# ---------------------------------------------------------------------------
# Layout (under DATA_ROOT):
#   India/   India.xlsx   India/<Number>/<images>
#   Italy/   Italy.xlsx   Italy/<Number>/<images>
#
# Each spreadsheet has columns: Number, Hgb, Gender, Age, Note  (+ junk
# "Unnamed" columns in Italy that we ignore). A patient's images live in the
# folder whose name equals their Number. We use ONLY the pre-cropped eyelid
# image whose filename ends in "_palpebral.png" — ignoring the raw .jpg, the
# "_forniceal.png", and the "_forniceal_palpebral.png".

COUNTRIES = ("India", "Italy")

# Per-country spreadsheet lives at <DATA_ROOT>/<Country>/<Country>.xlsx.
def spreadsheet_path(root: Path, country: str) -> Path:
    return root / country / f"{country}.xlsx"


# Only these spreadsheet columns are read; everything else (Note, Unnamed:*) is
# dropped. Matched case-insensitively against the real headers.
NUMBER_COL = "Number"   # patient id == folder name
HB_COL = "Hgb"          # hemoglobin, g/dL  (regression target)
GENDER_COL = "Gender"   # "M" / "F"
AGE_COL = "Age"

# Image selection. We keep files ending in PALPEBRAL_SUFFIX, but skip any whose
# name contains an EXCLUDE substring (so "_forniceal_palpebral.png" is dropped
# even though it also ends in "_palpebral.png").
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
PALPEBRAL_SUFFIX = "_palpebral.png"
PALPEBRAL_EXCLUDE = ("forniceal",)

# ---------------------------------------------------------------------------
# 2b. CP-AnemiC dataset (second source — children, Ghana)
# ---------------------------------------------------------------------------
# Layout (under CPANEMIC_ROOT):
#   Anemic/<IMAGE_ID>.png        (label = anemic)
#   Non-anemic/<IMAGE_ID>.png    (label = non-anemic)
#   Anemia_Data_Collection_Sheet.xlsx  with columns
#       IMAGE_ID, HB_LEVEL, Severity, Age(Months), GENDER, REMARK, COUNTRY, ...
# Each row links to <IMAGE_ID>.png in whichever of the two folders holds it.
# HB_LEVEL is the regression target; GENDER is "Male"/"Female" (normalized M/F);
# Age is in MONTHS (children) — so these rows are age_group "child".
#
# Configurable via OVISION_CPANEMIC_ROOT, defaulting to the Kaggle path below.
_KAGGLE_CPANEMIC_ROOT = Path(
    "/kaggle/input/datasets/karankumar4090/cp-anemic-dataset-same"
)


def get_cpanemic_root() -> Path:
    env_override = os.environ.get("OVISION_CPANEMIC_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return _KAGGLE_CPANEMIC_ROOT


CPANEMIC_ROOT = get_cpanemic_root()
CPANEMIC_SHEET = "Anemia_Data_Collection_Sheet.xlsx"
CPANEMIC_IMAGE_FOLDERS = ("Anemic", "Non-anemic")  # folder name decides the label
CPANEMIC_IMAGE_ID_COL = "IMAGE_ID"
CPANEMIC_HB_COL = "HB_LEVEL"
CPANEMIC_GENDER_COL = "GENDER"
CPANEMIC_SOURCE = "ghana"  # CP-AnemiC rows all carry this source label

# ---------------------------------------------------------------------------
# 2c. Which datasets to load / pool
# ---------------------------------------------------------------------------
# Controls build_dataframe(). Default loads BOTH and pools them. To train on one
# alone, set to a single-element tuple, e.g. ("eyes_defy",) or ("cp_anemic",).
DATASETS = ("eyes_defy", "cp_anemic")

# ---------------------------------------------------------------------------
# 3. Task mode  (this dataset is regression)
# ---------------------------------------------------------------------------
# "regression"     -> predict Hgb (g/dL), then threshold to a class for metrics.
# "classification" -> kept for completeness; not used by this dataset.
# "auto"           -> resolves to regression because Hgb values are present.
TASK = "regression"

# Gender-aware anemia cutoffs (g/dL): a (predicted) Hgb below the cutoff for
# that gender counts as anemic. Easy to change here. ANEMIA_HB_THRESHOLD is the
# scalar fallback used when a sample's gender is unknown.
ANEMIA_CUTOFFS = {"F": 12.0, "M": 13.0}
ANEMIA_HB_THRESHOLD = 12.0


def anemia_cutoff(gender) -> float:
    """Hgb cutoff for a gender. Falls back to ANEMIA_HB_THRESHOLD if unknown."""
    key = str(gender).strip().upper()[:1]
    return ANEMIA_CUTOFFS.get(key, ANEMIA_HB_THRESHOLD)


# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------
# "resnet18" or "efficientnet_b0" (both pretrained on ImageNet via torchvision).
BACKBONE = "resnet18"
PRETRAINED = True
IMAGE_SIZE = 224  # square resize fed to the backbone

# ImageNet normalization (matches the pretrained backbones).
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# 4b. Tight-crop preprocessing  (v1 stage one: equalize framing across sources)
# ---------------------------------------------------------------------------
# All images are conjunctiva cutouts on a BLACK background, but framing differs
# by source (India/Italy: small eyelid in lots of black; Ghana: eyelid fills the
# frame). That lets the model ID the source from aspect ratio / black-fraction
# instead of reading pallor. Tight-cropping to the eyelid bbox + resizing to a
# fixed square makes every source look the same.
#
# Toggle this to run crossval with cropping ON vs OFF and compare.
PREPROCESS_TIGHT_CROP = True
TIGHT_CROP_MARGIN = 0.05          # ~5% padding around the detected eyelid bbox
TIGHT_CROP_BLACK_THRESHOLD = 20   # grayscale brightness > this counts as eyelid
TIGHT_CROP_MIN_FG_PIXELS = 64     # fewer foreground px -> treat as all-black, fall back

# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

# Frozen split ratios. The split is PATIENT-level (all of a patient's images
# stay on the same side), computed once, saved to SPLIT_PATH, and reused on
# every later run so results are comparable.
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------------------
# 6. Logging
# ---------------------------------------------------------------------------
WANDB_PROJECT = "ovision-v0"  # only used when --wandb is passed

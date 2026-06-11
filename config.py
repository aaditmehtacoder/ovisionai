"""
Central configuration for OVision v0.

Everything that might change between machines, datasets, or experiments lives
here so you never have to hunt through the code. The most important knobs:

  * DATA_ROOT        - auto-detected dataset location (Kaggle or local)
  * TASK             - "auto" | "classification" | "regression"
  * The *_CANDIDATES / *_COL settings - how we discover folders and CSV columns.
                       The dataset's real structure is unknown until you run
                       `python src/explore_data.py`. Once you know the real
                       folder/column names, adjust the values below.
"""

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# 1. Dataset location  (HARD REQUIREMENT: auto-detecting + configurable)
# ---------------------------------------------------------------------------
# Priority:
#   1. OVISION_DATA_ROOT environment variable (explicit override)
#   2. The standard Kaggle input path, if it exists
#   3. A local ./data folder next to this file
#
# Never hardcode an absolute dataset path anywhere else in the repo. Import
# DATA_ROOT from here instead.

_KAGGLE_DATA_ROOT = Path("/kaggle/input/eyes-defy-anemia")
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
SPLIT_PATH = PROJECT_ROOT / "splits.json"  # frozen train/val/test split

# ---------------------------------------------------------------------------
# 2. Dataset structure discovery
# ---------------------------------------------------------------------------
# We DON'T assume exact folder/file names. Instead we list plausible candidates
# and let data.py pick whichever actually exists. After running explore_data.py
# you can trim these lists down to the real names (or just leave them).

# Image file extensions we treat as conjunctiva photos.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Candidate metadata files (CSV/JSON/Excel). First match wins.
METADATA_CANDIDATES = (
    "metadata.csv",
    "labels.csv",
    "data.csv",
    "anemia.csv",
    "eyes_defy_anemia.csv",
    "India.csv",
    "Italy.csv",
)

# ---- CSV column names (adjust after explore_data.py reveals the real ones) ----
# data.py matches these case-insensitively and also tries the *_CANDIDATES list,
# so you usually only need to set the canonical name once you know it.
IMAGE_COL = "image"          # column holding the image filename
IMAGE_COL_CANDIDATES = ("image", "filename", "file", "name", "img", "image_id", "id")

ANEMIA_COL = "anemic"        # binary anemia label column (0/1 or yes/no)
ANEMIA_COL_CANDIDATES = ("anemic", "anaemic", "anemia", "label", "class", "diagnosis", "target")

HB_COL = "hb"                # hemoglobin value column (g/dL)
HB_COL_CANDIDATES = ("hb", "hgb", "hemoglobin", "haemoglobin", "hb_value", "hb_level")

# If there is NO metadata file and labels live in folder names instead
# (e.g. .../Anemic/ and .../Non_Anemic/), list the "positive" folder name
# fragments here. data.py falls back to this when no CSV is found.
ANEMIC_FOLDER_HINTS = ("anemic", "anaemic", "anemia", "positive", "pos")
NONANEMIC_FOLDER_HINTS = ("non", "normal", "healthy", "negative", "neg", "control")

# ---------------------------------------------------------------------------
# 3. Task mode  (HARD REQUIREMENT: support both, default to what data supports)
# ---------------------------------------------------------------------------
# "auto"           -> data.py picks regression if Hb values exist, else
#                     classification. Resolved task is printed at runtime.
# "classification" -> anemic vs non-anemic.
# "regression"     -> predict Hb (g/dL), then threshold to a class for metrics.
TASK = "auto"

# WHO-style hemoglobin cutoff for "anemic". The real cutoff is age/sex specific
# (women <12, men <13, children <11 g/dL). For a v0 screening baseline we use a
# single configurable threshold. Adjust if the dataset documents its own cutoff.
ANEMIA_HB_THRESHOLD = 12.0  # g/dL; Hb below this => anemic

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
# 5. Training
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

# Frozen split ratios. The actual split is computed once, saved to SPLIT_PATH,
# and reused on every later run so results are comparable.
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------------------
# 6. Logging
# ---------------------------------------------------------------------------
WANDB_PROJECT = "ovision-v0"  # only used when --wandb is passed

"""
explore_data.py — RUN THIS FIRST.

The Eyes-defy-anemia dataset's exact folder layout and label format are unknown
until we look. This script discovers and prints the real structure so we can
configure data.py correctly. It is read-only: it never modifies the dataset.

It prints:
  1. A recursive directory tree (folders + file counts per extension).
  2. The schema + a preview of every CSV / JSON / TXT metadata file found.
  3. A handful of sample image filenames with their pixel dimensions + file size.
  4. If labels are discoverable, the anemia-label and/or hemoglobin distribution.

Usage:
    python src/explore_data.py
    OVISION_DATA_ROOT=/path/to/dataset python src/explore_data.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make `config` importable whether run from repo root or src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is in requirements.txt
    pd = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


SEP = "=" * 72


def banner(title: str) -> None:
    print("\n" + SEP)
    print(title)
    print(SEP)


# ---------------------------------------------------------------------------
# 1. Directory tree
# ---------------------------------------------------------------------------
def print_directory_tree(root: Path, max_depth: int = 6) -> None:
    banner(f"1. DIRECTORY TREE  (root = {root})")
    if not root.exists():
        print(f"!! Dataset root does not exist: {root}")
        print("   Download the dataset into ./data or set OVISION_DATA_ROOT.")
        return

    for dirpath, dirnames, filenames in _walk_sorted(root):
        rel = Path(dirpath).relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth > max_depth:
            continue
        indent = "    " * depth
        name = root.name if str(rel) == "." else rel.parts[-1]

        # Count files by extension so big image folders stay one readable line.
        ext_counts = Counter(Path(f).suffix.lower() or "<none>" for f in filenames)
        summary = ", ".join(f"{n}×{ext}" for ext, n in sorted(ext_counts.items()))
        suffix = f"  [{len(filenames)} files: {summary}]" if filenames else ""
        print(f"{indent}{name}/{suffix}")


def _walk_sorted(root: Path):
    """os.walk-style traversal with deterministic ordering."""
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        yield dirpath, dirnames, filenames


# ---------------------------------------------------------------------------
# 2. Metadata files (CSV / JSON / TXT)
# ---------------------------------------------------------------------------
def explore_metadata_files(root: Path) -> None:
    banner("2. METADATA FILES (CSV / JSON / TXT)")
    if not root.exists():
        return

    meta_files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in (".csv", ".json", ".txt", ".xlsx", ".xls")
    ]
    if not meta_files:
        print("No CSV/JSON/TXT/XLSX metadata files found.")
        print("Labels may be encoded in folder names — see sections 1 and 4.")
        return

    for path in sorted(meta_files):
        rel = path.relative_to(root)
        print(f"\n--- {rel}  ({path.stat().st_size} bytes) ---")
        suffix = path.suffix.lower()
        if suffix in (".csv", ".xlsx", ".xls"):
            _preview_table(path)
        elif suffix == ".json":
            _preview_json(path)
        else:  # .txt
            _preview_text(path)


def _preview_table(path: Path) -> None:
    if pd is None:
        print("  (pandas not installed — cannot preview)")
        return
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    except Exception as exc:  # noqa: BLE001 - want to keep going on bad files
        print(f"  (could not read: {exc})")
        return

    print(f"  shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  columns: {list(df.columns)}")
    print(f"  dtypes:\n{df.dtypes.to_string()}")
    print("  first rows:")
    print(df.head().to_string(max_cols=12))


def _preview_json(path: Path) -> None:
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not read: {exc})")
        return

    if isinstance(data, list):
        print(f"  JSON list of {len(data)} items. First item:")
        print("  " + json.dumps(data[0], indent=2)[:800] if data else "  (empty)")
    elif isinstance(data, dict):
        print(f"  JSON object with keys: {list(data.keys())[:20]}")
        print("  " + json.dumps(data, indent=2)[:800])
    else:
        print(f"  JSON scalar: {data!r}")


def _preview_text(path: Path, n_lines: int = 10) -> None:
    try:
        with open(path, errors="replace") as f:
            lines = [next(f, None) for _ in range(n_lines)]
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not read: {exc})")
        return
    for line in lines:
        if line is None:
            break
        print("  " + line.rstrip())


# ---------------------------------------------------------------------------
# 3. Sample images
# ---------------------------------------------------------------------------
def explore_images(root: Path, n_samples: int = 8) -> list:
    banner("3. SAMPLE IMAGES")
    if not root.exists():
        return []

    images = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    ]
    images.sort()
    print(f"Found {len(images)} image files total "
          f"(extensions: {sorted(set(p.suffix.lower() for p in images))})")

    if not images:
        return images

    # Show how images are distributed across folders.
    per_folder = Counter(str(p.parent.relative_to(root)) for p in images)
    print("\nImages per folder:")
    for folder, count in sorted(per_folder.items()):
        print(f"  {folder or '.'}/  -> {count}")

    # Evenly spaced samples so we see more than just the first folder.
    step = max(1, len(images) // n_samples)
    samples = images[::step][:n_samples]
    print(f"\n{len(samples)} sample images:")
    for p in samples:
        size_kb = p.stat().st_size / 1024
        dims = _image_dims(p)
        print(f"  {p.relative_to(root)}  ({dims}, {size_kb:.0f} KB)")

    return images


def _image_dims(path: Path) -> str:
    if Image is None:
        return "PIL not installed"
    try:
        with Image.open(path) as im:
            return f"{im.width}x{im.height} {im.mode}"
    except Exception as exc:  # noqa: BLE001
        return f"unreadable: {exc}"


# ---------------------------------------------------------------------------
# 4. Label / Hb distribution
# ---------------------------------------------------------------------------
def explore_label_distribution(root: Path) -> None:
    banner("4. LABEL / HEMOGLOBIN DISTRIBUTION")
    if not root.exists():
        return

    # Try the same discovery data.py will use, so this section previews exactly
    # what the training pipeline will see.
    try:
        from data import build_dataframe  # local import; src/ is on sys.path
    except Exception as exc:  # noqa: BLE001
        print(f"(data.build_dataframe unavailable: {exc})")
        return

    try:
        df = build_dataframe(verbose=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not assemble a labeled dataframe yet: {exc}")
        print("Use sections 1-3 above to set the right names in config.py.")
        return

    print(f"\nAssembled dataframe: {len(df)} rows. Columns: {list(df.columns)}")

    if "anemic" in df.columns and df["anemic"].notna().any():
        counts = df["anemic"].value_counts(dropna=False).to_dict()
        print(f"\nAnemia label counts (1=anemic, 0=non-anemic): {counts}")

    if "hb" in df.columns and df["hb"].notna().any():
        hb = df["hb"].dropna()
        print("\nHemoglobin (g/dL) summary:")
        print(f"  n={len(hb)}  min={hb.min():.1f}  max={hb.max():.1f}  "
              f"mean={hb.mean():.1f}  median={hb.median():.1f}")
        below = (hb < config.ANEMIA_HB_THRESHOLD).sum()
        print(f"  below threshold {config.ANEMIA_HB_THRESHOLD} g/dL "
              f"(=> anemic): {below} / {len(hb)}")


def main() -> None:
    root = config.DATA_ROOT
    print(f"OVision data exploration")
    print(f"DATA_ROOT = {root}")
    print(f"(override with the OVISION_DATA_ROOT environment variable)")

    print_directory_tree(root)
    explore_metadata_files(root)
    explore_images(root)
    explore_label_distribution(root)

    banner("NEXT STEPS")
    print("Use what you saw above to set the real names in config.py:")
    print("  * METADATA_CANDIDATES / IMAGE_COL / ANEMIA_COL / HB_COL")
    print("  * or the *_FOLDER_HINTS if labels live in folder names.")
    print("Then run:  python src/train.py")


if __name__ == "__main__":
    main()

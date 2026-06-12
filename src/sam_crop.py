"""
sam_crop.py — OPTIONAL SAM-based conjunctiva cropper (alternative to the
rule-based tight_crop_image in data.py).

Why this exists: the rule cropper in data.py frames *pre-cropped* eyelid cutouts
well, but struggles on RAW whole-eye photos (full face/skin, no black border).
This module runs Meta's Segment-Anything (SAM) to segment the eye, picks the
mask that best overlaps the reddish conjunctiva (reusing data.py's HSV eyelid
heuristic to score the candidate masks), and crops to that mask's bbox. When SAM
is unavailable or finds nothing confident, it transparently falls back to the
existing rule-based crop and flags it.

It is fully GATED and NON-BREAKING:
  * Nothing here runs unless config.CROP_METHOD == "sam" (default "rule").
  * The segment_anything import is guarded — importing this module never crashes
    even with no SAM installed and no internet (weights can't download). In that
    case every call simply falls back to tight_crop_image.

Public API:
  * sam_crop_image(img) -> (cropped_224, used_fallback)   # the core entry point
  * SamCrop                                               # picklable transform
  * smoke_test()                                          # eyeball + timing report
  * self_check()                                          # offline synthetic check

vit_b weights are ~375MB and download on first use to config.SAM_CHECKPOINT_PATH
(see requirements.txt / config.py). Runs on Kaggle GPU; falls back to CPU with a
warning. Internet must be ON for the first download.
"""

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# Reuse the rule-based cropper, the HSV conjunctiva heuristic, the bbox crop, and
# the per-source foreground/eyelid-fraction measure — so SAM masks are scored the
# SAME way the rest of the pipeline judges "is this the eyelid".
from data import (  # noqa: E402
    SOURCE_ORDER,
    _bbox_crop,
    _conjunctiva_mask,
    foreground_fraction,
    tight_crop_image,
)

# ---------------------------------------------------------------------------
# Guarded SAM import — module must import cleanly even with no SAM / no weights.
# ---------------------------------------------------------------------------
try:
    from segment_anything import SamPredictor, sam_model_registry
    _SAM_IMPORT_OK = True
    _SAM_IMPORT_ERR = ""
except Exception as exc:  # noqa: BLE001 - any import failure -> fall back to rule crop
    SamPredictor = None
    sam_model_registry = None
    _SAM_IMPORT_OK = False
    _SAM_IMPORT_ERR = repr(exc)

# Process-local predictor cache. Built lazily on first crop so that (a) importing
# this module is cheap, and (b) DataLoader workers each load their own copy
# (SamPredictor holds a CUDA model and isn't safely picklable across processes).
# Value sentinels: None = not built yet, False = build failed (stick to fallback).
_PREDICTOR = None
_PREDICTOR_TRIED = False


# ---------------------------------------------------------------------------
# Weights + predictor setup
# ---------------------------------------------------------------------------
def _sam_device() -> str:
    """SAM runs on CUDA when present, else CPU with a one-time warning. (We skip
    MPS — SAM's ops are flaky there; CPU is the safe Mac fallback.)"""
    if torch.cuda.is_available():
        return "cuda"
    print("[sam] WARNING: no CUDA GPU found — running SAM on CPU (slow, ~seconds "
          "per image). On Kaggle, enable the GPU accelerator.")
    return "cpu"


def _download_weights(dest: Path) -> bool:
    """Download the vit_b checkpoint to `dest` if missing. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = config.SAM_CHECKPOINT_URL
    print(f"[sam] downloading {config.SAM_MODEL_TYPE} weights (~375MB, first use)\n"
          f"      {url}\n      -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            read, chunk, last_pct = 0, 1 << 20, -10
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                read += len(buf)
                if total:
                    pct = int(100 * read / total)
                    if pct >= last_pct + 10:
                        print(f"[sam]   {pct:3d}%  ({read >> 20}MB/{total >> 20}MB)")
                        last_pct = pct
        tmp.replace(dest)
        print("[sam] download complete.")
        return True
    except Exception as exc:  # noqa: BLE001 - offline / URL change -> use fallback crop
        print(f"[sam] WARNING: weight download failed ({exc!r}). "
              f"Falling back to the rule-based crop.")
        tmp.unlink(missing_ok=True)
        return False


def _get_predictor():
    """Build (once per process) and return a SamPredictor, or None if SAM can't
    be used (no package, no weights, offline). Cached so we pay setup only once."""
    global _PREDICTOR, _PREDICTOR_TRIED
    if _PREDICTOR_TRIED:
        return _PREDICTOR or None
    _PREDICTOR_TRIED = True

    if not _SAM_IMPORT_OK:
        print(f"[sam] segment_anything not importable ({_SAM_IMPORT_ERR}); "
              f"using the rule-based crop. `pip install segment-anything`.")
        _PREDICTOR = False
        return None

    if not _download_weights(config.SAM_CHECKPOINT_PATH):
        _PREDICTOR = False
        return None

    try:
        device = _sam_device()
        sam = sam_model_registry[config.SAM_MODEL_TYPE](
            checkpoint=str(config.SAM_CHECKPOINT_PATH))
        sam.to(device)
        _PREDICTOR = SamPredictor(sam)
        print(f"[sam] predictor ready ({config.SAM_MODEL_TYPE} on {device}).")
    except Exception as exc:  # noqa: BLE001 - bad weights / OOM -> fall back
        print(f"[sam] WARNING: failed to build predictor ({exc!r}); "
              f"using the rule-based crop.")
        _PREDICTOR = False
        return None
    return _PREDICTOR


# ---------------------------------------------------------------------------
# Mask scoring — reuse data.py's reddish-conjunctiva heuristic
# ---------------------------------------------------------------------------
def _red_overlap_score(mask: np.ndarray, conj: np.ndarray) -> float:
    """How well a SAM mask captures the reddish conjunctiva region.

    Combines two views so we prefer masks that are BOTH reddish inside and that
    cover most of the reddish pixels (not a tiny sliver, not the whole frame):
        density = reddish px in mask / mask px        (is the mask conjunctiva?)
        recall  = reddish px in mask / all reddish px (did it get the region?)
        score   = density * (0.5 + 0.5 * recall)
    Returns 0 for an empty mask or an image with no reddish pixels at all.
    """
    m = mask.astype(bool)
    msum = int(m.sum())
    csum = int(conj.sum())
    if msum == 0 or csum == 0:
        return 0.0
    inter = int(np.logical_and(m, conj).sum())
    density = inter / msum
    recall = inter / csum
    return float(density * (0.5 + 0.5 * recall))


def _candidate_masks(predictor, arr: np.ndarray):
    """Prompt SAM near the conjunctiva (center, slightly low) and return its
    candidate boolean masks. We use a center + center-bottom point prompt plus a
    generous center box, with multimask_output so SAM returns several scales to
    score. The conjunctiva is the reddish region near center-bottom of an eye."""
    H, W = arr.shape[:2]
    predictor.set_image(arr)

    # Foreground points: image center and a touch below it (where the lower
    # eyelid conjunctiva sits). Labels all 1 = "include this region".
    pts = np.array([[W * 0.5, H * 0.55], [W * 0.5, H * 0.7]], dtype=np.float32)
    labels = np.array([1, 1], dtype=np.int64)
    # Box around the central ~60% of the frame to keep SAM off the far edges.
    box = np.array([W * 0.2, H * 0.3, W * 0.8, H * 0.85], dtype=np.float32)

    masks, _scores, _logits = predictor.predict(
        point_coords=pts,
        point_labels=labels,
        box=box,
        multimask_output=True,
    )
    return [m.astype(bool) for m in masks]


# ---------------------------------------------------------------------------
# Public: the SAM crop entry point
# ---------------------------------------------------------------------------
def sam_crop_image(img, size: int = None):
    """SAM-segment the conjunctiva and crop to it; resize to a `size` square.

    Steps: run SAM with a center / center-bottom prompt, score each candidate
    mask by overlap with the reddish-conjunctiva HSV mask (data._conjunctiva_mask),
    take the best mask's bbox and crop. If SAM is unavailable, returns nothing
    confident (tiny mask or weak reddish overlap), or errors, fall back to the
    rule-based tight_crop_image and flag it.

    Returns (cropped_resized_RGB_image, used_fallback: bool).
    """
    size = config.IMAGE_SIZE if size is None else size
    rgb = img.convert("RGB")

    predictor = _get_predictor()
    if predictor is None:
        cropped, _info = tight_crop_image(rgb, size)
        return cropped, True

    try:
        arr = np.asarray(rgb)
        conj = _conjunctiva_mask(rgb)               # reddish/saturated HSV mask
        masks = _candidate_masks(predictor, arr)
        H, W = arr.shape[:2]
        min_px = max(1, int(config.SAM_MASK_MIN_FRAC * H * W))

        best, best_score = None, -1.0
        for m in masks:
            if int(m.sum()) < min_px:
                continue
            s = _red_overlap_score(m, conj)
            if s > best_score:
                best, best_score = m, s

        if best is None or best_score < config.SAM_MIN_RED_SCORE:
            # SAM found nothing convincingly reddish — trust the rule crop instead.
            cropped, _info = tight_crop_image(rgb, size)
            return cropped, True

        cropped = _bbox_crop(rgb, best, config.SAM_CROP_MARGIN, size)
        return cropped, False
    except Exception as exc:  # noqa: BLE001 - never let a bad image kill the run
        print(f"[sam] WARNING: crop failed ({exc!r}); falling back to rule crop.")
        cropped, _info = tight_crop_image(rgb, size)
        return cropped, True


class SamCrop:
    """Picklable transform wrapper, mirroring data.TightCrop. Holds only scalar
    config (the SAM predictor is a lazily-built, process-local global), so it
    pickles cleanly to DataLoader workers. Returns just the cropped image."""

    def __init__(self, size=None):
        self.size = config.IMAGE_SIZE if size is None else size

    def __call__(self, img):
        return sam_crop_image(img, self.size)[0]


# ---------------------------------------------------------------------------
# Synthetic fixture + offline self-check (safe with no weights / no internet)
# ---------------------------------------------------------------------------
def _synthetic_fixture(size: int = 400):
    """A fake whole-eye photo: black border, a skin-tone band, and a reddish
    conjunctiva blob near center-bottom. Lets us exercise the full code path
    (HSV mask + crop + fallback) offline, without the real dataset or weights."""
    img = Image.new("RGB", (size, size), (8, 8, 10))
    draw = ImageDraw.Draw(img)
    # Lower-face skin band.
    draw.rectangle([0, int(size * 0.55), size, size], fill=(150, 110, 95))
    # Reddish conjunctiva ellipse, center-bottom.
    cx, cy = size * 0.5, size * 0.68
    rx, ry = size * 0.22, size * 0.08
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(200, 40, 45))
    return img


def self_check():
    """Run the synthetic fixture through sam_crop_image once and report. Works
    fully offline: if SAM weights can't load, this exercises the FALLBACK path
    and still succeeds (that's the point — it must never crash)."""
    print("[sam] self-check: SAM import "
          f"{'OK' if _SAM_IMPORT_OK else 'UNAVAILABLE -> ' + _SAM_IMPORT_ERR}")
    img = _synthetic_fixture()
    t0 = time.perf_counter()
    cropped, used_fallback = sam_crop_image(img)
    dt = time.perf_counter() - t0
    assert cropped.size == (config.IMAGE_SIZE, config.IMAGE_SIZE), cropped.size
    frac = foreground_fraction(cropped)
    print(f"[sam] self-check OK: crop={cropped.size} "
          f"fallback={used_fallback} eyelid_frac={frac:.3f} time={dt:.2f}s")
    return cropped, used_fallback


# ---------------------------------------------------------------------------
# Smoke test — eyeball the SAM crops + timing BEFORE committing to a real run
# ---------------------------------------------------------------------------
def smoke_test(df=None, per_source: int = 4, path=None):
    """Run SAM crop on `per_source` images per source, save a labelled grid to
    results/sam_crop_preview.png, and print per-source median eyelid-fraction,
    fallback counts, and average seconds/image.

    This does NOT train anything — it's the cheap look-before-you-leap check so
    you can judge SAM crop quality and per-image timing on Kaggle first.
    """
    import data  # local import: build_dataframe needs the dataset mounted

    if df is None:
        df = data.build_dataframe(verbose=False)

    sources = [s for s in SOURCE_ORDER if (df["source"] == s).any()]
    path = Path(path) if path else (config.RESULTS_DIR / "sam_crop_preview.png")
    cell = config.IMAGE_SIZE
    grid = Image.new("RGB", (per_source * cell, len(sources) * cell), (15, 15, 18))
    draw = ImageDraw.Draw(grid)

    print(f"\n[sam] smoke test: {per_source} imgs/source over {sources} "
          f"(CROP_METHOD='{config.CROP_METHOD}')")
    print(f"  {'source':6} {'n':>3} {'median_eyelid_frac':>18} "
          f"{'fallbacks':>10} {'sec/img':>9}")

    all_times = []
    for r, src in enumerate(sources):
        paths = (df[df["source"] == src].drop_duplicates("patient_id")
                 ["image_path"].tolist()[:per_source])
        fracs, n_fallback = [], 0
        for c, p in enumerate(paths):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:  # noqa: BLE001 - skip unreadable sample
                continue
            t0 = time.perf_counter()
            cropped, used_fallback = sam_crop_image(img)
            all_times.append(time.perf_counter() - t0)
            fracs.append(foreground_fraction(cropped))
            n_fallback += int(used_fallback)

            grid.paste(cropped, (c * cell, r * cell))
            tile = "fallback" if used_fallback else "sam"
            ty = r * cell + cell - 18
            draw.rectangle([c * cell + 2, ty, c * cell + 78, ty + 16], fill=(0, 0, 0))
            draw.text((c * cell + 5, ty + 3), tile, fill=(255, 220, 120))
        # Row source label.
        draw.rectangle([2, r * cell + 2, 86, r * cell + 20], fill=(0, 0, 0))
        draw.text((6, r * cell + 6), src, fill=(255, 255, 255))

        med = float(np.median(fracs)) if fracs else float("nan")
        print(f"  {src:6} {len(fracs):>3} {med:>18.3f} "
              f"{n_fallback:>10} {'':>9}")

    avg = float(np.mean(all_times)) if all_times else float("nan")
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
    print(f"\n[sam] average {avg:.2f} sec/image over {len(all_times)} images")
    print(f"[sam] preview grid saved to {path}")
    print("[sam] smoke test done — NO training was run. Eyeball the preview, then "
          "set config.CROP_METHOD='sam' to use it in a cross-val.")
    return path


if __name__ == "__main__":
    # Default: offline synthetic self-check (safe with no weights/internet).
    # Pass --smoke to run the real per-source preview (needs data + weights).
    if "--smoke" in sys.argv:
        smoke_test()
    else:
        self_check()

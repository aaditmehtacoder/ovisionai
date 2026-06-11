"""
demo.py — minimal Gradio sanity-check demo for the v0 hemoglobin regressor.

Drag in a cropped lower-eyelid (palpebral conjunctiva) image, pick a gender, and
see the model's predicted hemoglobin + a gender-aware anemia flag. This is a
quick sanity check, NOT the product UI.

It reuses the exact evaluation code paths so the demo can't drift from training:
  * the model + checkpoint are loaded via model.load_checkpoint (same as
    evaluate.py), from config.CHECKPOINT_DIR / "ovision_v0_regression.pt".
  * preprocessing is data._build_transforms(train=False) — the IDENTICAL
    resize+normalization used for the val/test set. We do NOT hand-roll a new one.

Run:
    python src/demo.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from data import _build_transforms  # noqa: E402  (reuse the exact test transform)
from model import load_checkpoint  # noqa: E402  (same loader evaluate.py uses)
from utils import get_device  # noqa: E402

from PIL import Image  # noqa: E402

TASK = "regression"
CKPT_PATH = config.CHECKPOINT_DIR / f"ovision_v0_{TASK}.pt"

# The exact val/test transform — same resize + ImageNet normalization as eval.
_TRANSFORM = _build_transforms(train=False)

NOTE = (
    "v0 demo. Upload a cropped lower-eyelid (palpebral conjunctiva) image "
    "similar to the training data — e.g. a `*_palpebral.png` from the dataset. "
    "Full-face selfies will not work yet; segmentation comes in v1. "
    "Research demo only, not for clinical use."
)


def load_model():
    """Load the trained v0 regression checkpoint, same as evaluate.py."""
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CKPT_PATH}. Train the v0 model first:\n"
            f"    python src/train.py --task regression"
        )
    device = get_device()
    model, payload = load_checkpoint(CKPT_PATH, map_location=device)
    model.eval().to(device)
    print(f"[demo] loaded {CKPT_PATH} (task={model.task}, "
          f"epoch={payload.get('epoch')}) on {device}")
    return model, device


# Loaded once at import so the predict closure can reuse them.
MODEL, DEVICE = load_model()


def predict(image, gender):
    """Predict Hgb (g/dL) and a gender-aware anemia flag for one PIL image.

    Returns (hgb_text, anemia_text). On any failure returns a friendly message
    instead of raising, so a bad/corrupt upload can't crash the demo.
    """
    try:
        if image is None:
            return "Upload an image first.", ""
        img = image.convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
        with torch.no_grad():
            hb = float(MODEL(tensor).item())  # regression head outputs Hgb

        cutoff = config.anemia_cutoff(gender)
        anemic = hb < cutoff
        flag = "Anemia: YES" if anemic else "Anemia: NO"
        return (
            f"{hb:.1f} g/dL",
            f"{flag}  (cutoff {cutoff:.0f} g/dL for {gender})",
        )
    except Exception as exc:  # noqa: BLE001 - never crash the UI on a bad image
        return f"Could not process this image: {exc}", ""


def _find_sample_palpebral():
    """First real *_palpebral.png under DATA_ROOT (for the self-check), or None."""
    root = config.DATA_ROOT
    if not root.exists():
        return None
    for p in sorted(root.rglob(f"*{config.PALPEBRAL_SUFFIX}")):
        if any(bad in p.name.lower() for bad in config.PALPEBRAL_EXCLUDE):
            continue
        return p
    return None


def self_check():
    """Run one real palpebral image through predict() before launching."""
    sample = _find_sample_palpebral()
    if sample is None:
        print("[demo] self-check skipped: no *_palpebral.png found under "
              f"{config.DATA_ROOT} (set OVISION_DATA_ROOT to test locally).")
        return
    hgb_text, flag_text = predict(Image.open(sample), "F")
    print(f"[demo] self-check on {sample.name}: predicted {hgb_text}  {flag_text}")


def build_ui():
    import gradio as gr

    with gr.Blocks(title="OVision v0 — Hb sanity demo") as demo:
        gr.Markdown("## OVision v0 — hemoglobin sanity demo")
        gr.Markdown(NOTE)
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", sources=["upload"],
                                    label="Eyelid image (.png / .jpg)")
                gender_in = gr.Dropdown(choices=["F", "M"], value="F",
                                        label="Gender (for anemia cutoff)")
                run = gr.Button("Predict", variant="primary")
            with gr.Column():
                hgb_out = gr.Textbox(label="Predicted hemoglobin")
                flag_out = gr.Textbox(label="Anemia flag")
        run.click(predict, inputs=[image_in, gender_in], outputs=[hgb_out, flag_out])
    return demo


if __name__ == "__main__":
    self_check()
    ui = build_ui()
    # share=True so the public link works from a Kaggle notebook.
    ui.launch(share=True)

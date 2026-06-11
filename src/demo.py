"""
demo.py — Gradio sanity-check demo for the v0 hemoglobin regressor.

Drag in a cropped lower-eyelid (palpebral conjunctiva) image, pick a gender, and
see the model's predicted hemoglobin + a gender-aware anemia flag. This is a
quick sanity check, NOT the product UI.

It reuses the exact evaluation code paths so the demo can't drift from training:
  * the model + checkpoint are loaded via model.load_checkpoint (same as
    evaluate.py), from config.CHECKPOINT_DIR / "ovision_v0_regression.pt".
  * preprocessing is data._build_transforms(train=False) — the IDENTICAL
    resize+normalization used for the val/test set. We do NOT hand-roll a new one.

UI niceties (cosmetic only — they never touch model logic):
  * Soft theme + colored anemia badge.
  * A clickable gallery of real *_palpebral.png samples (a mix of known-anemic and
    known-healthy patients) pulled from the dataset.
  * A language dropdown that translates UI TEXT ONLY (English / हिन्दी / Italiano).

Run:
    python src/demo.py
"""

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import data  # noqa: E402  (build_dataframe for the sample gallery)
from data import _build_transforms  # noqa: E402  (reuse the exact test transform)
from model import load_checkpoint  # noqa: E402  (same loader evaluate.py uses)
from utils import get_device  # noqa: E402

from PIL import Image  # noqa: E402

TASK = "regression"
CKPT_PATH = config.CHECKPOINT_DIR / f"ovision_v0_{TASK}.pt"

# The exact val/test transform — same resize + ImageNet normalization as eval.
_TRANSFORM = _build_transforms(train=False)

DEFAULT_LANG = "English"

# ---------------------------------------------------------------------------
# UI strings per language. TRANSLATION OF DISPLAY TEXT ONLY — no model logic,
# units, or thresholds are localized. Keys are identical across languages.
# ---------------------------------------------------------------------------
UI_STRINGS = {
    "English": {
        "title": "OVision v0 — Hemoglobin Demo",
        "tagline": "Estimate hemoglobin from a lower-eyelid photo.",
        "confidence": "v0 model · typical error ±1.5 g/dL",
        "empty": "Upload an image and press Predict to see the estimate.",
        "subtitle": (
            "Upload a cropped lower-eyelid (palpebral conjunctiva) image like the "
            "training data — e.g. a `*_palpebral.png` from the dataset. Full-face "
            "selfies will not work yet; segmentation comes in v1."
        ),
        "image_label": "Eyelid image (.png / .jpg)",
        "examples_label": "Sample images from the dataset",
        "gender_label": "Gender (for anemia cutoff)",
        "predict": "Predict",
        "hgb_label": "Predicted hemoglobin",
        "flag_label": "Result",
        "language_label": "Language",
        "no_anemia": "No anemia",
        "anemia": "Anemia likely",
        "cutoff_note": "cutoff {cutoff:.0f} g/dL for {gender}",
        "upload_first": "Upload an image first.",
        "error": "Could not process this image",
    },
    "हिन्दी": {
        "title": "OVision v0 — हीमोग्लोबिन डेमो",
        "tagline": "निचली पलक की तस्वीर से हीमोग्लोबिन का अनुमान।",
        "confidence": "v0 मॉडल · सामान्य त्रुटि ±1.5 g/dL",
        "empty": "अनुमान देखने के लिए छवि अपलोड करें और Predict दबाएँ।",
        "subtitle": (
            "प्रशिक्षण डेटा जैसी निचली पलक (पैल्पेब्रल कंजंक्टाइवा) की कटी हुई छवि "
            "अपलोड करें — जैसे डेटासेट का कोई `*_palpebral.png`। पूरे चेहरे की सेल्फ़ी "
            "अभी काम नहीं करेगी; सेग्मेंटेशन v1 में आएगा।"
        ),
        "image_label": "पलक की छवि (.png / .jpg)",
        "examples_label": "डेटासेट से नमूना छवियाँ",
        "gender_label": "लिंग (एनीमिया सीमा हेतु)",
        "predict": "अनुमान लगाएँ",
        "hgb_label": "अनुमानित हीमोग्लोबिन",
        "flag_label": "परिणाम",
        "language_label": "भाषा",
        "no_anemia": "एनीमिया नहीं",
        "anemia": "एनीमिया की संभावना",
        "cutoff_note": "{gender} हेतु सीमा {cutoff:.0f} g/dL",
        "upload_first": "पहले एक छवि अपलोड करें।",
        "error": "यह छवि संसाधित नहीं हो सकी",
    },
    "Italiano": {
        "title": "OVision v0 — Demo dell'emoglobina",
        "tagline": "Stima l'emoglobina da una foto della palpebra inferiore.",
        "confidence": "modello v0 · errore tipico ±1.5 g/dL",
        "empty": "Carica un'immagine e premi Prevedi per vedere la stima.",
        "subtitle": (
            "Carica un'immagine ritagliata della palpebra inferiore (congiuntiva "
            "palpebrale) simile ai dati di addestramento — es. un `*_palpebral.png` "
            "del dataset. I selfie del volto intero non funzioneranno ancora; la "
            "segmentazione arriverà nella v1."
        ),
        "image_label": "Immagine della palpebra (.png / .jpg)",
        "examples_label": "Immagini di esempio dal dataset",
        "gender_label": "Sesso (per la soglia di anemia)",
        "predict": "Prevedi",
        "hgb_label": "Emoglobina prevista",
        "flag_label": "Risultato",
        "language_label": "Lingua",
        "no_anemia": "Nessuna anemia",
        "anemia": "Anemia probabile",
        "cutoff_note": "soglia {cutoff:.0f} g/dL per {gender}",
        "upload_first": "Carica prima un'immagine.",
        "error": "Impossibile elaborare questa immagine",
    },
}

# The clinical disclaimer MUST stay visible in ALL languages, always — so the
# demo never reads as a finished clinical product. Styled as a tasteful footer.
FOOTER_HTML = (
    '<div id="ov-footer">'
    "⚠️ <b>Research demo only — not for clinical use.</b>"
    " &nbsp;·&nbsp; केवल अनुसंधान डेमो — चिकित्सीय उपयोग के लिए नहीं।"
    " &nbsp;·&nbsp; Solo demo di ricerca — non per uso clinico."
    "</div>"
)

# Hero / brand header (the tagline is translated; the brand name is not).
def hero_html(lang: str) -> str:
    t = UI_STRINGS.get(lang, UI_STRINGS[DEFAULT_LANG])
    return (
        '<div id="ov-hero">'
        '<div class="ov-brand"><span class="ov-logo">◐</span>OVision'
        '<span class="ov-tag-v0">v0</span></div>'
        f'<div class="ov-tagline">{t["tagline"]}</div>'
        '</div>'
    )


# Custom CSS for a modern, product-grade look (cosmetic only).
CUSTOM_CSS = """
.gradio-container { background: #eef2f6 !important; max-width: 1100px !important;
  margin: 0 auto !important;
  font-family: 'Inter','SF Pro Display', system-ui, -apple-system, 'Segoe UI',
  Roboto, Helvetica, Arial, sans-serif !important; }
.gradio-container *, .gradio-container input, .gradio-container button { font-family: inherit !important; }

#ov-hero { padding: 6px 2px 2px; }
#ov-hero .ov-brand { display:flex; align-items:center; gap:9px; font-size:27px;
  font-weight:800; color:#0f172a; letter-spacing:-0.025em; }
#ov-hero .ov-logo { color:#0d9488; font-size:26px; }
#ov-hero .ov-tag-v0 { font-size:11px; font-weight:700; color:#0d9488;
  background:#d8f3ef; border-radius:6px; padding:2px 7px; margin-left:2px;
  letter-spacing:.02em; }
#ov-hero .ov-tagline { color:#64748b; font-size:14px; margin-top:3px; }

.ov-card { background:#ffffff !important; border:1px solid #e6ebf1 !important;
  border-radius:18px !important; padding:22px !important;
  box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06) !important; }

.ov-label { font-size:12px !important; font-weight:600 !important; color:#64748b !important;
  text-transform:uppercase; letter-spacing:.05em; margin:2px 0 -2px !important; }
.ov-help, .ov-help p { color:#94a3b8 !important; font-size:12.5px !important; line-height:1.5; }

.ov-predict button { background: linear-gradient(180deg,#12b8aa,#0d9488) !important;
  border:none !important; color:#fff !important; font-weight:700 !important;
  font-size:16px !important; border-radius:12px !important; padding:12px 0 !important;
  box-shadow:0 6px 16px rgba(13,148,136,.32) !important; transition:filter .15s ease; }
.ov-predict button:hover { filter:brightness(1.06); }

.ov-num-wrap { display:flex; align-items:baseline; gap:9px; margin-top:6px; }
.ov-num { font-size:62px; font-weight:800; color:#0f172a; line-height:1; letter-spacing:-0.03em; }
.ov-unit { font-size:19px; font-weight:600; color:#94a3b8; }
.ov-sub { color:#64748b; font-size:13px; margin-top:10px; }
.ov-conf { color:#94a3b8; font-size:12px; margin-top:5px; }
.ov-empty { color:#94a3b8; font-size:14px; padding:22px 0; line-height:1.5; }

.ov-pill { display:inline-flex; align-items:center; gap:9px; padding:11px 20px;
  border-radius:999px; font-weight:700; font-size:18px; margin-top:6px; }
.ov-pill .ov-ic { font-size:18px; line-height:1; }
.ov-pill-good { background:#e7f7ee; color:#067647; border:1px solid #a6e7c3; }
.ov-pill-bad { background:#fdeceb; color:#b42318; border:1px solid #f4b6b0; }

#ov-footer { margin-top:16px; padding:11px 16px; background:#fbfdfe;
  border:1px solid #e6ebf1; border-radius:12px; color:#94a3b8; font-size:12px;
  line-height:1.6; text-align:center; }
#ov-footer b { color:#64748b; font-weight:600; }
"""


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


# ---------------------------------------------------------------------------
# Inference + result rendering
# ---------------------------------------------------------------------------
def _hgb_card(hb: float, cutoff: float, gender: str, lang: str) -> str:
    t = UI_STRINGS[lang]
    note = t["cutoff_note"].format(cutoff=cutoff, gender=gender or "?")
    return (
        f'<div class="ov-num-wrap"><span class="ov-num">{hb:.1f}</span>'
        f'<span class="ov-unit">g/dL</span></div>'
        f'<div class="ov-sub">{note}</div>'
        f'<div class="ov-conf">{t["confidence"]}</div>'
    )


def _flag_badge(anemic: bool, lang: str) -> str:
    s = UI_STRINGS[lang]
    if anemic:
        return f'<div class="ov-pill ov-pill-bad"><span class="ov-ic">⚠</span>{s["anemia"]}</div>'
    return f'<div class="ov-pill ov-pill-good"><span class="ov-ic">✓</span>{s["no_anemia"]}</div>'


def _empty_state(lang: str) -> str:
    return f'<div class="ov-empty">{UI_STRINGS[lang]["empty"]}</div>'


def predict(image, gender, lang=DEFAULT_LANG):
    """Predict Hgb (g/dL) + a gender-aware anemia flag for one PIL image.

    Returns (hgb_html, flag_html). On any failure returns a friendly message
    instead of raising, so a bad/corrupt upload can't crash the demo. `lang`
    only localizes the displayed words — never the model or the thresholds.
    """
    s = UI_STRINGS.get(lang, UI_STRINGS[DEFAULT_LANG])
    try:
        if image is None:
            return f'<div class="ov-empty">{s["upload_first"]}</div>', ""
        img = image.convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
        with torch.no_grad():
            hb = float(MODEL(tensor).item())  # regression head outputs Hgb

        cutoff = config.anemia_cutoff(gender)  # gender-aware: <12 F, <13 M
        anemic = hb < cutoff
        return _hgb_card(hb, cutoff, gender, lang), _flag_badge(anemic, lang)
    except Exception as exc:  # noqa: BLE001 - never crash the UI on a bad image
        return f'<div class="ov-empty">{s["error"]}: {exc}</div>', ""


# ---------------------------------------------------------------------------
# Sample gallery (real images, mix of anemic + healthy)
# ---------------------------------------------------------------------------
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


def build_examples(n_anemic: int = 3, n_healthy: int = 3):
    """Return (examples, labels) for gr.Examples — a mix of known-anemic and
    known-healthy patients read from the spreadsheets, or (None, None) if the
    dataset isn't mounted. Each label shows patient id, gender and true Hgb so
    predictions can be checked against ground truth."""
    if not config.DATA_ROOT.exists():
        return None, None
    try:
        df = data.build_dataframe(verbose=False)
    except Exception as exc:  # noqa: BLE001 - missing/odd data -> just skip gallery
        print(f"[demo] sample gallery skipped: {exc}")
        return None, None

    # One image per patient so the gallery shows distinct patients.
    df = df.drop_duplicates("patient_id")
    anemic = df[df["anemic"] == 1].nsmallest(n_anemic, "hgb")   # clearly low Hgb
    healthy = df[df["anemic"] == 0].nlargest(n_healthy, "hgb")  # clearly normal
    import pandas as pd
    chosen = pd.concat([anemic, healthy])
    if len(chosen) < 2:
        return None, None

    examples, labels = [], []
    for _, r in chosen.iterrows():
        gender = r["gender"] if r["gender"] in ("F", "M") else "F"
        status = "anemic" if r["anemic"] == 1 else "healthy"
        examples.append([r["image_path"], gender])
        labels.append(f'{r["patient_id"]} · {gender} · Hgb {r["hgb"]:.1f} ({status})')
    return examples, labels


def self_check():
    """Run one real palpebral image through predict() before launching."""
    sample = _find_sample_palpebral()
    if sample is None:
        print("[demo] self-check skipped: no *_palpebral.png found under "
              f"{config.DATA_ROOT} (set OVISION_DATA_ROOT to test locally).")
        return
    hgb_html, flag_html = predict(Image.open(sample), "F")
    strip = lambda h: re.sub("<[^>]+>", " ", h).split()  # noqa: E731
    print(f"[demo] self-check on {sample.name}: "
          f"{' '.join(strip(hgb_html))} | {' '.join(strip(flag_html))}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    s = UI_STRINGS[DEFAULT_LANG]
    examples, ex_labels = build_examples()

    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="OVision") as demo:
        # ---- hero header (brand + translated tagline) + language picker ----
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                hero = gr.HTML(hero_html(DEFAULT_LANG))
            with gr.Column(scale=1, min_width=150):
                lang_dd = gr.Dropdown(choices=list(UI_STRINGS.keys()),
                                      value=DEFAULT_LANG, label=s["language_label"])

        # ---- two side-by-side cards: input | results ----
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes="ov-card"):
                image_in = gr.Image(type="pil", sources=["upload"],
                                    label=s["image_label"], height=280)
                subtitle_md = gr.Markdown(s["subtitle"], elem_classes="ov-help")
                gender_in = gr.Dropdown(choices=["F", "M"], value="F",
                                        label=s["gender_label"])
                examples_header = gr.Markdown(s["examples_label"], elem_classes="ov-label")
                if examples:
                    # Clicking an example loads its image (+ gender) into the
                    # inputs above, ready to Predict.
                    gr.Examples(examples=examples, inputs=[image_in, gender_in],
                                example_labels=ex_labels, label=None)
                run = gr.Button(s["predict"], variant="primary", size="lg",
                                elem_classes="ov-predict")
            with gr.Column(scale=1, elem_classes="ov-card"):
                hgb_header = gr.Markdown(s["hgb_label"], elem_classes="ov-label")
                hgb_out = gr.HTML(_empty_state(DEFAULT_LANG))  # smooth empty state
                flag_header = gr.Markdown(s["flag_label"], elem_classes="ov-label")
                flag_out = gr.HTML("")

        gr.HTML(FOOTER_HTML)  # all-languages disclaimer, always visible

        run.click(predict, inputs=[image_in, gender_in, lang_dd],
                  outputs=[hgb_out, flag_out])

        # Language switch: translate DISPLAY TEXT ONLY (no model logic touched).
        def set_language(lang):
            t = UI_STRINGS.get(lang, UI_STRINGS[DEFAULT_LANG])
            return (
                gr.update(value=hero_html(lang)),
                gr.update(value=t["subtitle"]),
                gr.update(label=t["image_label"]),
                gr.update(value=t["examples_label"]),
                gr.update(label=t["gender_label"]),
                gr.update(value=t["predict"]),
                gr.update(value=t["hgb_label"]),
                gr.update(value=t["flag_label"]),
                gr.update(label=t["language_label"]),
            )

        lang_dd.change(
            set_language, inputs=[lang_dd],
            outputs=[hero, subtitle_md, image_in, examples_header, gender_in,
                     run, hgb_header, flag_header, lang_dd],
        )
    return demo


if __name__ == "__main__":
    self_check()
    ui = build_ui()
    # share=True so the public link works from a Kaggle notebook.
    ui.launch(share=True)

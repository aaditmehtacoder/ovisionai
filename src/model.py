"""
model.py — a simple transfer-learning baseline with a switchable head.

One backbone (resnet18 / resnet50 / efficientnet_b0 / efficientnet_b3, pretrained
on ImageNet) feeding a single-logit head used for BOTH tasks:
  * classification: the logit is passed through sigmoid -> P(anemic).
  * regression:     the logit IS the predicted Hb (g/dL), no activation.

Nothing fancy on purpose — this is v0.

# -------------------------------------------------------------------------
# TODO (Phase 2, NOT implemented here): two-stage pipeline
#   Stage A: segment the palpebral conjunctiva from the eye photo
#            (e.g. U-Net) to crop out skin/sclera/background.
#   Stage B: run Hb regression on the segmented conjunctiva only.
# Keep v0 as a whole-image baseline so we have an honest number to beat.
# -------------------------------------------------------------------------
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


class OVisionModel(nn.Module):
    """Pretrained backbone + a single-output linear head."""

    def __init__(self, task: str, backbone: str = None, pretrained: bool = None):
        super().__init__()
        self.task = task
        backbone = backbone or config.BACKBONE
        pretrained = config.PRETRAINED if pretrained is None else pretrained

        self.backbone, in_features = _make_backbone(backbone, pretrained)
        # One output unit serves both tasks (logit for cls, Hb value for reg).
        self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats).squeeze(1)  # shape: (batch,)

    def predict_prob(self, x):
        """P(anemic) — only meaningful for the classification task."""
        return torch.sigmoid(self.forward(x))


def _make_backbone(name: str, pretrained: bool):
    """Return (feature_extractor, num_features) with the classifier stripped.

    Supports resnet18 / resnet50 / efficientnet_b0 / efficientnet_b3. Each loads
    torchvision ImageNet weights (when pretrained), strips the final classifier
    to nn.Identity to expose pooled features, and reports its feature width so the
    shared single-output head in OVisionModel attaches unchanged for every one."""
    name = _canon_backbone(name)
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Identity()  # expose pooled features
        return net, in_features

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Identity()
        return net, in_features

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, in_features

    if name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b3(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, in_features

    raise ValueError(
        f"Unknown backbone '{name}'. Use one of: {', '.join(SUPPORTED_BACKBONES)}."
    )


# Canonical backbone names the sweep + config agree on.
SUPPORTED_BACKBONES = ("resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3")


def _canon_backbone(name: str) -> str:
    """Normalize aliases (hyphens, no-underscore) to a canonical SUPPORTED name."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key.startswith("efficientnet") and "_" not in key[len("efficientnet"):]:
        key = "efficientnet_" + key[len("efficientnet"):]
    return key


def count_parameters(model: nn.Module):
    """(total, trainable) parameter counts for a built model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_loss(task: str):
    """Matching loss for the task."""
    if task == "classification":
        return nn.BCEWithLogitsLoss()  # expects raw logits
    return nn.L1Loss()  # MAE on Hb — directly the metric we report


def save_checkpoint(model: nn.Module, path: Path, extra: dict = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "task": model.task}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model = OVisionModel(task=payload["task"], pretrained=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload

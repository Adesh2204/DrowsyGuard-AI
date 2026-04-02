import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

EYE_LABELS = ["Open", "Closed", "Half"]


class EyeStateClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.4, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = self._build_backbone(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _build_backbone(pretrained: bool = True) -> nn.Module:
        if not pretrained:
            try:
                return models.resnet50(weights=None)
            except TypeError:
                return models.resnet50(pretrained=False)

        try:
            return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except AttributeError:
            return models.resnet50(pretrained=True)
        except Exception as exc:
            warnings.warn(
                f"Failed to load pretrained ResNet-50 weights ({exc}). Falling back to random init.",
                RuntimeWarning,
            )
            try:
                return models.resnet50(weights=None)
            except TypeError:
                return models.resnet50(pretrained=False)

    def freeze_early_layers(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

        for layer_name in ("layer3", "layer4", "fc"):
            layer = getattr(self.backbone, layer_name)
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_eye_model(pretrained: bool = True, freeze_early: bool = True) -> EyeStateClassifier:
    model = EyeStateClassifier(pretrained=pretrained)
    if freeze_early:
        model.freeze_early_layers()
    return model


def load_eye_model_weights(model: nn.Module, weights_path: Path, device: torch.device) -> None:
    if not weights_path.exists() or weights_path.stat().st_size == 0:
        warnings.warn(
            f"Eye model weights not found or empty at {weights_path}. Using initialized weights.",
            RuntimeWarning,
        )
        return

    try:
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            warnings.warn(
                f"Eye model weight mismatch. Missing={len(missing)} Unexpected={len(unexpected)}",
                RuntimeWarning,
            )
    except Exception as exc:
        warnings.warn(
            f"Failed to load eye model weights from {weights_path}: {exc}. Using initialized weights.",
            RuntimeWarning,
        )

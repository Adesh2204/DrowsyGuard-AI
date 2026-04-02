import warnings
from pathlib import Path

import torch
import torch.nn as nn

DROWSINESS_LABELS = ["Awake", "Drowsy", "Distracted", "Microsleep"]


class DrowsinessLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(x)
        last_timestep = sequence_output[:, -1, :]
        return self.classifier(last_timestep)


def load_lstm_weights(model: nn.Module, weights_path: Path, device: torch.device) -> None:
    if not weights_path.exists() or weights_path.stat().st_size == 0:
        warnings.warn(
            f"LSTM model weights not found or empty at {weights_path}. Using initialized weights.",
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
                f"LSTM model weight mismatch. Missing={len(missing)} Unexpected={len(unexpected)}",
                RuntimeWarning,
            )
    except Exception as exc:
        warnings.warn(
            f"Failed to load LSTM weights from {weights_path}: {exc}. Using initialized weights.",
            RuntimeWarning,
        )

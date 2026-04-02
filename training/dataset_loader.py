from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EYE_CLASS_TO_INDEX: Dict[str, int] = {
    "open": 0,
    "closed": 1,
    "half": 2,
}

STATE_CLASS_TO_INDEX: Dict[str, int] = {
    "awake": 0,
    "drowsy": 1,
    "distracted": 2,
    "microsleep": 3,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EyeStateDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        samples: Optional[Sequence[Tuple[Path, int]]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = list(samples) if samples is not None else self._scan_samples()

    def _scan_samples(self) -> List[Tuple[Path, int]]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Eye dataset directory not found: {self.root_dir}")

        samples: List[Tuple[Path, int]] = []
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_key = class_dir.name.lower().strip()
            if class_key not in EYE_CLASS_TO_INDEX:
                continue
            label = EYE_CLASS_TO_INDEX[class_key]

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((image_path, label))

        if not samples:
            raise FileNotFoundError(
                "No eye-state images found. Expected class folders like open/closed/half under the eye dataset root."
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, label_tensor


class EyeSequenceDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        sequence_length: int = 20,
        samples: Optional[Sequence[Tuple[Path, int]]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.samples = list(samples) if samples is not None else self._scan_samples()

    def _scan_samples(self) -> List[Tuple[Path, int]]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Sequence dataset directory not found: {self.root_dir}")

        npz_files = sorted(self.root_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(
                "No .npz sequence files found. Each file should contain `sequence` (20x3) and `label` fields."
            )

        samples: List[Tuple[Path, int]] = []
        for npz_file in npz_files:
            with np.load(npz_file) as payload:
                if "label" in payload:
                    label = int(payload["label"])
                elif "y" in payload:
                    label = int(payload["y"])
                else:
                    guessed = npz_file.stem.split("_")[0].lower()
                    if guessed not in STATE_CLASS_TO_INDEX:
                        raise ValueError(
                            f"Missing label in {npz_file.name}. Provide `label` in npz or prefix filename with class."
                        )
                    label = STATE_CLASS_TO_INDEX[guessed]
            samples.append((npz_file, label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_path, label = self.samples[index]
        with np.load(sample_path) as payload:
            if "sequence" in payload:
                sequence = payload["sequence"]
            elif "x" in payload:
                sequence = payload["x"]
            else:
                raise ValueError(f"Missing `sequence` or `x` in {sample_path.name}")

        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] != 3:
            raise ValueError(f"Sequence in {sample_path.name} must have shape [T, 3], found {sequence.shape}")

        if sequence.shape[0] < self.sequence_length:
            pad_size = self.sequence_length - sequence.shape[0]
            padding = np.zeros((pad_size, sequence.shape[1]), dtype=np.float32)
            sequence = np.concatenate([padding, sequence], axis=0)
        elif sequence.shape[0] > self.sequence_length:
            sequence = sequence[-self.sequence_length :]

        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return sequence_tensor, label_tensor

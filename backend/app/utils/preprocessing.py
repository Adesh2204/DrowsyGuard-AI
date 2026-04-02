import base64

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_eye_transform(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_eye_crop(eye_bgr: np.ndarray, image_size: int = 64) -> torch.Tensor:
    if eye_bgr is None or eye_bgr.size == 0:
        raise ValueError("Eye crop is empty.")

    eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(eye_rgb)
    transform = get_eye_transform(image_size=image_size)
    return transform(pil_image).unsqueeze(0)


def softmax_probs(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy().squeeze(0)


def decode_base64_image(image_data: str) -> np.ndarray:
    if not image_data:
        raise ValueError("Image payload is empty.")

    encoded = image_data.split(",", 1)[1] if "," in image_data else image_data

    try:
        image_bytes = base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError("Invalid base64 payload.") from exc

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Decoded payload is not a valid image.")

    return frame_bgr

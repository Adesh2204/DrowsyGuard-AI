from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from app.utils.mediapipe_utils import MediaPipeEyeExtractor


@dataclass
class ProcessedFrame:
    eye_crops: List[np.ndarray]
    eye_boxes: List[Dict[str, int]]
    face_box: Optional[Dict[str, int]]
    ear: Optional[float]


class VideoProcessor:
    def __init__(self) -> None:
        self.eye_extractor = MediaPipeEyeExtractor()

    def process(self, frame_bgr: np.ndarray) -> ProcessedFrame:
        extraction = self.eye_extractor.extract_eyes(frame_bgr)
        return ProcessedFrame(
            eye_crops=extraction.eye_crops,
            eye_boxes=extraction.eye_boxes,
            face_box=extraction.face_box,
            ear=extraction.ear,
        )

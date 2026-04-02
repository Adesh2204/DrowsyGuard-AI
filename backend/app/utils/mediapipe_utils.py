from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

LEFT_EYE_CONTOUR_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


@dataclass
class EyeExtractionResult:
    eye_crops: List[np.ndarray]
    eye_boxes: List[Dict[str, int]]
    face_box: Optional[Dict[str, int]]
    ear: Optional[float]


class MediaPipeEyeExtractor:
    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.available = mp is not None
        self.face_mesh = None

        if self.available:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    @staticmethod
    def _to_point(landmark, width: int, height: int) -> np.ndarray:
        return np.array([int(landmark.x * width), int(landmark.y * height)], dtype=np.float32)

    def _compute_ear(self, landmarks, width: int, height: int, idx: List[int]) -> float:
        points = [self._to_point(landmarks[i], width, height) for i in idx]
        p1, p2, p3, p4, p5, p6 = points
        vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4) + 1e-6
        return float(vertical / (2.0 * horizontal))

    def _extract_eye_box(self, landmarks, width: int, height: int, idx: List[int]) -> Optional[Dict[str, int]]:
        points = np.array([self._to_point(landmarks[i], width, height) for i in idx], dtype=np.int32)
        x_min = int(points[:, 0].min())
        y_min = int(points[:, 1].min())
        x_max = int(points[:, 0].max())
        y_max = int(points[:, 1].max())

        pad_x = max(int((x_max - x_min) * 0.2), 2)
        pad_y = max(int((y_max - y_min) * 0.3), 2)

        x = max(x_min - pad_x, 0)
        y = max(y_min - pad_y, 0)
        w = min(x_max + pad_x, width - 1) - x
        h = min(y_max + pad_y, height - 1) - y

        if w <= 1 or h <= 1:
            return None

        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    def _extract_face_box(self, landmarks, width: int, height: int) -> Dict[str, int]:
        x_vals = [int(point.x * width) for point in landmarks]
        y_vals = [int(point.y * height) for point in landmarks]

        x_min = max(min(x_vals), 0)
        y_min = max(min(y_vals), 0)
        x_max = min(max(x_vals), width - 1)
        y_max = min(max(y_vals), height - 1)

        return {
            "x": x_min,
            "y": y_min,
            "w": max(x_max - x_min, 1),
            "h": max(y_max - y_min, 1),
        }

    def extract_eyes(self, frame_bgr: np.ndarray) -> EyeExtractionResult:
        if not self.available or self.face_mesh is None:
            return EyeExtractionResult(eye_crops=[], eye_boxes=[], face_box=None, ear=None)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            return EyeExtractionResult(eye_crops=[], eye_boxes=[], face_box=None, ear=None)

        landmarks = result.multi_face_landmarks[0].landmark
        height, width = frame_bgr.shape[:2]

        eye_boxes: List[Dict[str, int]] = []
        eye_crops: List[np.ndarray] = []

        for contour_idx in (LEFT_EYE_CONTOUR_IDX, RIGHT_EYE_CONTOUR_IDX):
            box = self._extract_eye_box(landmarks, width, height, contour_idx)
            if box is None:
                continue
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            crop = frame_bgr[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            eye_boxes.append(box)
            eye_crops.append(crop)

        left_ear = self._compute_ear(landmarks, width, height, LEFT_EYE_EAR_IDX)
        right_ear = self._compute_ear(landmarks, width, height, RIGHT_EYE_EAR_IDX)
        ear = float((left_ear + right_ear) / 2.0)

        face_box = self._extract_face_box(landmarks, width, height)

        return EyeExtractionResult(
            eye_crops=eye_crops,
            eye_boxes=eye_boxes,
            face_box=face_box,
            ear=ear,
        )

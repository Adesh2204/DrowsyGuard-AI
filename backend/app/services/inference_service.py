import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
import torch

from app.config import Settings
from app.models.eye_model import EYE_LABELS, build_eye_model, load_eye_model_weights
from app.models.lstm_model import DROWSINESS_LABELS, DrowsinessLSTM, load_lstm_weights
from app.services.video_processor import VideoProcessor
from app.utils.preprocessing import preprocess_eye_crop, softmax_probs


@dataclass
class SessionState:
    prob_buffer: Deque[np.ndarray]
    fatigue_history: Deque[float]
    blink_timestamps: Deque[float]
    eyes_prev_closed: bool = False
    last_seen_ts: float = field(default_factory=time.time)


class InferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = self._resolve_device(settings.model_device)

        self.eye_model = build_eye_model(pretrained=True, freeze_early=True).to(self.device)
        load_eye_model_weights(self.eye_model, settings.eye_model_weights, self.device)
        self.eye_model.eval()

        self.lstm_model = DrowsinessLSTM(input_size=3, hidden_size=128, num_layers=2, num_classes=4).to(self.device)
        load_lstm_weights(self.lstm_model, settings.lstm_model_weights, self.device)
        self.lstm_model.eval()

        self.video_processor = VideoProcessor()

        self.sessions: Dict[str, SessionState] = {}
        self.max_fatigue_points = max(settings.fatigue_window_seconds * settings.websocket_max_fps, 10)

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name.lower().startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)

    def _get_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(
                prob_buffer=deque(maxlen=self.settings.sequence_length),
                fatigue_history=deque(maxlen=self.max_fatigue_points),
                blink_timestamps=deque(),
            )
        return self.sessions[session_id]

    def close_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def _cleanup_stale_sessions(self, max_idle_seconds: int = 300) -> None:
        now = time.time()
        stale_ids = [
            session_id
            for session_id, session in self.sessions.items()
            if now - session.last_seen_ts > max_idle_seconds
        ]
        for session_id in stale_ids:
            self.close_session(session_id)

    def _infer_eye_probs(self, eye_crops: List[np.ndarray]) -> np.ndarray:
        probabilities: List[np.ndarray] = []
        for eye_crop in eye_crops:
            try:
                eye_tensor = preprocess_eye_crop(eye_crop, image_size=self.settings.eye_image_size).to(self.device)
            except ValueError:
                continue
            with torch.no_grad():
                logits = self.eye_model(eye_tensor)
            probabilities.append(softmax_probs(logits))

        if not probabilities:
            return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)

        return np.mean(np.stack(probabilities, axis=0), axis=0).astype(np.float32)

    def _infer_temporal(self, session: SessionState) -> Optional[Dict[str, object]]:
        if len(session.prob_buffer) < self.settings.sequence_length:
            return None

        sequence = np.stack(list(session.prob_buffer), axis=0).astype(np.float32)
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.lstm_model(sequence_tensor)
        probs = softmax_probs(logits)

        index = int(np.argmax(probs))
        return {
            "label": DROWSINESS_LABELS[index],
            "confidence": float(probs[index]),
            "probs": {label: float(probs[i]) for i, label in enumerate(DROWSINESS_LABELS)},
        }

    def _heuristic_state(self, eye_probs: np.ndarray, face_detected: bool) -> Dict[str, object]:
        if not face_detected:
            return {
                "label": "Distracted",
                "confidence": 0.95,
                "probs": {
                    "Awake": 0.05,
                    "Drowsy": 0.0,
                    "Distracted": 0.95,
                    "Microsleep": 0.0,
                },
            }

        open_prob = float(eye_probs[0])
        closed_prob = float(eye_probs[1])
        half_prob = float(eye_probs[2])

        if closed_prob > 0.78:
            return {
                "label": "Microsleep",
                "confidence": closed_prob,
                "probs": {
                    "Awake": 0.05,
                    "Drowsy": 0.15,
                    "Distracted": 0.0,
                    "Microsleep": 0.8,
                },
            }

        if closed_prob > 0.55 or half_prob > 0.5:
            drowsy_conf = max(closed_prob, half_prob)
            return {
                "label": "Drowsy",
                "confidence": drowsy_conf,
                "probs": {
                    "Awake": max(0.0, 1.0 - drowsy_conf - 0.1),
                    "Drowsy": drowsy_conf,
                    "Distracted": 0.05,
                    "Microsleep": 0.05,
                },
            }

        awake_conf = max(0.55, open_prob)
        return {
            "label": "Awake",
            "confidence": awake_conf,
            "probs": {
                "Awake": awake_conf,
                "Drowsy": max(0.0, 1.0 - awake_conf - 0.1),
                "Distracted": 0.05,
                "Microsleep": 0.05,
            },
        }

    def _update_blink_rate(self, session: SessionState, ear: Optional[float]) -> int:
        now = time.time()
        if ear is not None:
            is_closed = ear < self.settings.ear_threshold
            if session.eyes_prev_closed and not is_closed:
                session.blink_timestamps.append(now)
            session.eyes_prev_closed = is_closed

        window_seconds = self.settings.blink_window_seconds
        while session.blink_timestamps and now - session.blink_timestamps[0] > window_seconds:
            session.blink_timestamps.popleft()

        return int(len(session.blink_timestamps))

    @staticmethod
    def _state_base_attention(label: str) -> float:
        return {
            "Awake": 92.0,
            "Distracted": 58.0,
            "Drowsy": 32.0,
            "Microsleep": 10.0,
        }.get(label, 50.0)

    def _compute_attention_score(
        self,
        label: str,
        confidence: float,
        ear: Optional[float],
        blink_rate: int,
    ) -> float:
        score = self._state_base_attention(label)

        if ear is not None and ear < self.settings.ear_threshold:
            score -= 8.0

        target_blink_rate = 12.0
        blink_deviation = min(abs(blink_rate - target_blink_rate), 25.0)
        score -= blink_deviation * 0.6

        score -= (1.0 - confidence) * 12.0

        if label == "Microsleep":
            score = min(score, 15.0)

        return float(max(0.0, min(score, 100.0)))

    def predict(self, frame_bgr: np.ndarray, session_id: str = "default") -> Dict[str, object]:
        session = self._get_session(session_id)
        session.last_seen_ts = time.time()
        self._cleanup_stale_sessions()

        processed_frame = self.video_processor.process(frame_bgr)

        if processed_frame.eye_crops:
            eye_probs = self._infer_eye_probs(processed_frame.eye_crops)
            session.prob_buffer.append(eye_probs)
        else:
            eye_probs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            session.prob_buffer.clear()

        temporal_result = self._infer_temporal(session)
        if temporal_result is None:
            temporal_result = self._heuristic_state(eye_probs=eye_probs, face_detected=bool(processed_frame.eye_crops))

        label = str(temporal_result["label"])
        confidence = float(temporal_result["confidence"])

        blink_rate = self._update_blink_rate(session, processed_frame.ear)
        attention_score = self._compute_attention_score(label, confidence, processed_frame.ear, blink_rate)

        fatigue_value = 100.0 - attention_score
        session.fatigue_history.append(fatigue_value)

        eye_probabilities = {
            EYE_LABELS[index]: float(eye_probs[index])
            for index in range(len(EYE_LABELS))
        }

        temporal_probabilities = temporal_result.get("probs", {})

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 2),
            "sequence_ready": len(session.prob_buffer) >= self.settings.sequence_length,
            "alarm": label in {"Drowsy", "Microsleep"},
            "eye_probabilities": eye_probabilities,
            "temporal_probabilities": temporal_probabilities,
            "ear": round(processed_frame.ear, 4) if processed_frame.ear is not None else None,
            "blink_rate_per_min": blink_rate,
            "attention_score": round(attention_score, 2),
            "fatigue_trend": [round(value, 2) for value in session.fatigue_history],
            "overlays": {
                "eye_boxes": processed_frame.eye_boxes,
                "face_box": processed_frame.face_box,
            },
            "timestamp": int(time.time() * 1000),
        }

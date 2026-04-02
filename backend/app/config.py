from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE_DIR / "weights"


class Settings(BaseSettings):
    app_name: str = "DrowsyGuard AI API"
    debug: bool = False
    model_device: str = "cpu"

    eye_model_weights: Path = WEIGHTS_DIR / "eye_classifier.pth"
    lstm_model_weights: Path = WEIGHTS_DIR / "drowsiness_lstm.pth"

    eye_image_size: int = 64
    sequence_length: int = 20

    websocket_max_fps: int = 10
    ear_threshold: float = 0.21
    blink_window_seconds: int = 60
    fatigue_window_seconds: int = 60

    allowed_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DG_",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

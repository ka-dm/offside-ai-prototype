"""Shared types and helpers used across pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass
class Point2D:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (float(self.x), float(self.y))

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    label: str = "player"
    track_id: int | None = None
    team_id: int | None = None

    @property
    def center(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2.0, self.y2)

    def crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x1 = max(0, int(self.x1))
        y1 = max(0, int(self.y1))
        x2 = min(w, int(self.x2))
        y2 = min(h, int(self.y2))
        return image[y1:y2, x1:x2]


@dataclass
class DetectionResult:
    boxes: list[BoundingBox] = field(default_factory=list)
    frame_index: int = 0
    fps: float = 25.0
    image_size: tuple[int, int] = (0, 0)  # (width, height)
    backend: str = "unknown"

    def players(self) -> list[BoundingBox]:
        return [b for b in self.boxes if b.label in {"player", "goalkeeper"}]

    def ball(self) -> BoundingBox | None:
        balls = [b for b in self.boxes if b.label == "ball"]
        if not balls:
            return None
        return max(balls, key=lambda b: b.confidence)


@dataclass
class PassEvent:
    """Detected moment of the pass / kick."""

    time_s: float
    frame_index: int
    confidence: float
    source: str = "audio"  # audio | visual | manual | fused
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_video_meta(video_path: str | Path) -> dict[str, Any]:
    """Return fps, frame count, width, height for a video file."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0:
            fps = 25.0
        return {
            "path": str(path),
            "fps": fps,
            "n_frames": n_frames,
            "width": width,
            "height": height,
            "duration_s": n_frames / fps if fps > 0 else 0.0,
        }
    finally:
        cap.release()


def read_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    """Read a single BGR frame by index."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {path}")
        return frame
    finally:
        cap.release()


def time_to_frame(time_s: float, fps: float, offset_s: float = 0.0) -> int:
    """Convert seconds to a frame index with a configurable A/V offset.

    frame = round((t + offset) * fps)
    Positive offset means the audio event appears *later* in the video timeline
    than in the extracted audio (common when containers have stream skew).
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    return int(round((float(time_s) + float(offset_s)) * float(fps)))


def frame_to_time(frame_index: int, fps: float, offset_s: float = 0.0) -> float:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return (float(frame_index) / float(fps)) - float(offset_s)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean_color_bgr(patch: np.ndarray) -> np.ndarray:
    if patch.size == 0:
        return np.zeros(3, dtype=np.float64)
    return patch.reshape(-1, 3).mean(axis=0)


def pairwise(iterable: Sequence[Any]) -> Iterable[tuple[Any, Any]]:
    for i in range(len(iterable) - 1):
        yield iterable[i], iterable[i + 1]

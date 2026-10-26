"""Pitch keypoint detectors (HF / public models when available; mock fallback)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np

from .homography import ManualPitchCalibrator, PitchHomography, pitch_template_points

BackendName = Literal["mock", "hf", "manual"]


class PitchKeypointDetector(ABC):
    name: str

    @abstractmethod
    def detect_keypoints(self, image_bgr: np.ndarray) -> dict[str, tuple[float, float]]:
        """Return mapping of pitch template name -> image (u, v)."""

    def estimate_homography(self, image_bgr: np.ndarray) -> PitchHomography:
        kps = self.detect_keypoints(image_bgr)
        return ManualPitchCalibrator().calibrate_named(kps)


class MockPitchDetector(PitchKeypointDetector):
    """Synthetic keypoints assuming the full pitch roughly fills the frame."""

    name = "mock"

    def detect_keypoints(self, image_bgr: np.ndarray) -> dict[str, tuple[float, float]]:
        h, w = image_bgr.shape[:2]
        # Inset rectangle as a fake pitch quad + halfway
        margin_x, margin_y = 0.08 * w, 0.12 * h
        return {
            "corner_tl": (margin_x, margin_y),
            "corner_tr": (w - margin_x, margin_y),
            "corner_bl": (margin_x, h - margin_y),
            "corner_br": (w - margin_x, h - margin_y),
            "halfway_top": (w / 2.0, margin_y),
            "halfway_bottom": (w / 2.0, h - margin_y),
        }


class HuggingFacePitchDetector(PitchKeypointDetector):
    """Try a Hugging Face keypoint / segmentation model for pitch lines.

    Many sports-pitch models are community-hosted; licenses vary. This wrapper
    fails with a clear message if transformers/weights are unavailable so the
    manual calibrator can be used instead.
    """

    name = "hf"

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or "optional/not-bundled"
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "transformers not installed. pip install transformers "
                "or use --backend mock / manual calibration. "
                f"Original error: {e}"
            ) from e
        raise RuntimeError(
            "No default HF pitch-keypoint model is auto-downloaded in this "
            "prototype (license / size). Pass explicit correspondences via the "
            "manual calibrator, or plug in a model id after reviewing its card "
            f"(attempted id: {self.model_id}). See docs/LICENSES.md and "
            "roboflow/sports for reference implementations."
        )

    def detect_keypoints(self, image_bgr: np.ndarray) -> dict[str, tuple[float, float]]:
        self._load()
        return {}


def create_pitch_detector(backend: BackendName = "mock", **kwargs: Any) -> PitchKeypointDetector:
    if backend == "mock":
        return MockPitchDetector()
    if backend == "hf":
        return HuggingFacePitchDetector(model_id=kwargs.get("model_id"))
    if backend == "manual":
        # Manual path does not auto-detect; caller uses ManualPitchCalibrator.
        return MockPitchDetector()
    raise ValueError(f"Unknown pitch backend: {backend}")


# Re-export template for CLI convenience
AVAILABLE_PITCH_POINTS = sorted(pitch_template_points().keys())

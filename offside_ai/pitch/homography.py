"""Homography between image pixels and a standard FIFA pitch (metres).

Pitch coordinates: origin at bottom-left corner looking toward +x (length)
and +y (width). Standard size 105m x 68m (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from offside_ai.common import Point2D


def pitch_template_points(
    length_m: float = 105.0,
    width_m: float = 68.0,
) -> dict[str, Point2D]:
    """Named keypoints on a rectangular pitch (subset useful for calibration)."""
    L, W = length_m, width_m
    return {
        "corner_bl": Point2D(0.0, 0.0),
        "corner_br": Point2D(L, 0.0),
        "corner_tl": Point2D(0.0, W),
        "corner_tr": Point2D(L, W),
        "halfway_bottom": Point2D(L / 2.0, 0.0),
        "halfway_top": Point2D(L / 2.0, W),
        "center": Point2D(L / 2.0, W / 2.0),
        "penalty_left": Point2D(16.5, W / 2.0),
        "penalty_right": Point2D(L - 16.5, W / 2.0),
        "goal_left_bottom": Point2D(0.0, (W - 7.32) / 2.0),
        "goal_left_top": Point2D(0.0, (W + 7.32) / 2.0),
        "goal_right_bottom": Point2D(L, (W - 7.32) / 2.0),
        "goal_right_top": Point2D(L, (W + 7.32) / 2.0),
    }


@dataclass
class PitchHomography:
    """Pixel <-> pitch metre transforms."""

    H_image_to_pitch: np.ndarray  # 3x3
    length_m: float = 105.0
    width_m: float = 68.0

    @property
    def H_pitch_to_image(self) -> np.ndarray:
        return np.linalg.inv(self.H_image_to_pitch)

    def image_to_pitch(self, u: float, v: float) -> tuple[float, float]:
        vec = np.array([u, v, 1.0], dtype=np.float64)
        out = self.H_image_to_pitch @ vec
        out /= out[2] + 1e-12
        return float(out[0]), float(out[1])

    def pitch_to_image(self, x: float, y: float) -> tuple[float, float]:
        vec = np.array([x, y, 1.0], dtype=np.float64)
        out = self.H_pitch_to_image @ vec
        out /= out[2] + 1e-12
        return float(out[0]), float(out[1])

    @classmethod
    def from_correspondences(
        cls,
        image_points: Sequence[tuple[float, float]],
        pitch_points: Sequence[tuple[float, float]],
        length_m: float = 105.0,
        width_m: float = 68.0,
    ) -> "PitchHomography":
        if len(image_points) < 4 or len(pitch_points) < 4:
            raise ValueError("Need at least 4 point correspondences for homography")
        if len(image_points) != len(pitch_points):
            raise ValueError("image_points and pitch_points length mismatch")
        src = np.array(image_points, dtype=np.float64)
        dst = np.array(pitch_points, dtype=np.float64)
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            raise RuntimeError("cv2.findHomography failed")
        return cls(H_image_to_pitch=H, length_m=length_m, width_m=width_m)


class ManualPitchCalibrator:
    """Fallback: user supplies 4+ named or raw correspondences."""

    def __init__(self, length_m: float = 105.0, width_m: float = 68.0):
        self.length_m = length_m
        self.width_m = width_m
        self.template = pitch_template_points(length_m, width_m)

    def calibrate_named(
        self, named_image_points: dict[str, tuple[float, float]]
    ) -> PitchHomography:
        if len(named_image_points) < 4:
            raise ValueError("Click/provide at least 4 known pitch points")
        img_pts = []
        pitch_pts = []
        for name, uv in named_image_points.items():
            if name not in self.template:
                raise KeyError(
                    f"Unknown pitch point '{name}'. Known: {sorted(self.template)}"
                )
            img_pts.append(uv)
            p = self.template[name]
            pitch_pts.append(p.as_tuple())
        return PitchHomography.from_correspondences(
            img_pts, pitch_pts, self.length_m, self.width_m
        )

    def calibrate_raw(
        self,
        image_points: Sequence[tuple[float, float]],
        pitch_points: Sequence[tuple[float, float]],
    ) -> PitchHomography:
        return PitchHomography.from_correspondences(
            image_points, pitch_points, self.length_m, self.width_m
        )

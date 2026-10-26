"""Optional visual pass cue from ball speed / direction change.

Used when audio is unreliable (crowd noise, no audible kick). Expects a
sequence of ball centres (pixel or pitch metres) over consecutive frames from
the detection stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from offside_ai.common import Point2D


@dataclass
class VisualCrossCheckConfig:
    min_speed_px: float = 2.0
    direction_change_deg: float = 35.0
    speed_jump_ratio: float = 1.8
    smooth_window: int = 3


def _as_xy(points: Sequence[Point2D | tuple[float, float] | None]) -> np.ndarray:
    rows = []
    for p in points:
        if p is None:
            rows.append([np.nan, np.nan])
        elif isinstance(p, Point2D):
            rows.append([p.x, p.y])
        else:
            rows.append([float(p[0]), float(p[1])])
    return np.asarray(rows, dtype=np.float64)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    # nan-safe: fill with nearest valid before filtering
    filled = x.copy()
    nans = np.isnan(filled)
    if nans.any() and (~nans).any():
        idx = np.where(~nans, np.arange(len(filled)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled[nans] = filled[idx[nans]]
    return np.convolve(filled, kernel, mode="same")


def estimate_pass_from_ball_motion(
    ball_centers: Sequence[Point2D | tuple[float, float] | None],
    fps: float,
    frame_start: int = 0,
    cfg: VisualCrossCheckConfig | None = None,
) -> tuple[float | None, float, dict]:
    """Estimate pass time from an abrupt ball velocity / direction change.

    Parameters
    ----------
    ball_centers:
        Per-frame ball centre (None if not detected). Index 0 corresponds to
        ``frame_start``.
    fps:
        Video frames per second.
    frame_start:
        Absolute frame index of the first entry.
    cfg:
        Thresholds.

    Returns
    -------
    (time_s, confidence, extras)
        ``time_s`` is None if no reliable cue is found.
    """
    cfg = cfg or VisualCrossCheckConfig()
    if fps <= 0:
        raise ValueError("fps must be positive")

    xy = _as_xy(ball_centers)
    if len(xy) < 4:
        return None, 0.0, {"reason": "too_few_frames"}

    # Velocities between consecutive frames
    dxy = np.diff(xy, axis=0)
    speed = np.linalg.norm(dxy, axis=1)
    speed_s = _moving_average(speed, cfg.smooth_window)

    # Direction angles (radians); ignore near-zero motion
    angles = np.arctan2(dxy[:, 1], dxy[:, 0])
    d_angle = np.diff(angles)
    # wrap to [-pi, pi]
    d_angle = (d_angle + np.pi) % (2 * np.pi) - np.pi
    d_angle_deg = np.abs(np.degrees(d_angle))

    # Candidate indices on the velocity series (length N-1)
    scores = np.zeros(len(speed_s), dtype=np.float64)
    for i in range(1, len(speed_s)):
        if np.isnan(speed_s[i]) or np.isnan(speed_s[i - 1]):
            continue
        jump = speed_s[i] / (speed_s[i - 1] + 1e-6)
        dir_score = 0.0
        if i - 1 < len(d_angle_deg) and not np.isnan(d_angle_deg[i - 1]):
            if d_angle_deg[i - 1] >= cfg.direction_change_deg:
                dir_score = min(1.0, d_angle_deg[i - 1] / 90.0)
        jump_score = 0.0
        if speed_s[i] >= cfg.min_speed_px and jump >= cfg.speed_jump_ratio:
            jump_score = min(1.0, (jump - 1.0) / 3.0)
        scores[i] = 0.6 * jump_score + 0.4 * dir_score

    best_i = int(np.argmax(scores))
    best_score = float(scores[best_i])
    if best_score < 0.25:
        return None, float(best_score), {
            "reason": "no_strong_motion_change",
            "max_score": best_score,
        }

    # Event at the later of the two frames forming the velocity sample
    frame_idx = frame_start + best_i + 1
    time_s = frame_idx / fps
    return float(time_s), float(np.clip(best_score, 0.0, 1.0)), {
        "frame_index": int(frame_idx),
        "speed_at_event": float(speed_s[best_i]),
        "score": best_score,
    }

"""Draw offside line and verdict on a broadcast frame."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .geometry import OffsideDecision, PlayerPoseOnPitch


HomographyFn = Callable[[float, float], tuple[float, float]]
"""Map pitch metres (x, y) -> image pixel (u, v)."""


def _draw_label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]):
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )


def annotate_offside_frame(
    image_bgr: np.ndarray,
    decision: OffsideDecision,
    players: Sequence[PlayerPoseOnPitch],
    *,
    pitch_to_image: HomographyFn | None = None,
    player_image_points: dict[int, tuple[float, float]] | None = None,
    y_line_span: tuple[float, float] = (0.0, 68.0),
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Annotate a frame with the offside line, player marks, and verdict.

    Provide either ``pitch_to_image`` (homography inverse) to draw the line in
    image space from pitch metres, or ``player_image_points`` only (line drawn
    as a vertical line through the second-last defender's image x — fallback
    when homography is unavailable).
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    color_line = (0, 255, 255) if not decision.is_offside else (0, 0, 255)
    color_def = (255, 180, 0)
    color_atk_on = (0, 220, 0)
    color_atk_off = (0, 0, 255)

    # Draw offside line
    if pitch_to_image is not None:
        y0, y1 = y_line_span
        # Sample along pitch width
        pts = []
        for y in np.linspace(y0, y1, 40):
            u, v = pitch_to_image(decision.offside_line_x, float(y))
            pts.append([int(round(u)), int(round(v))])
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts_arr], isClosed=False, color=color_line, thickness=2)
    elif player_image_points and decision.second_last_defender_id in player_image_points:
        u, _ = player_image_points[decision.second_last_defender_id]
        x = int(round(u))
        cv2.line(out, (x, 0), (x, h - 1), color_line, 2)
    else:
        # Last resort: place line at image centre with a warning label
        x = w // 2
        cv2.line(out, (x, 0), (x, h - 1), color_line, 1)
        _draw_label(out, "line: approx (no homography)", (20, h - 20), (0, 255, 255))

    # Player markers
    img_pts = player_image_points or {}
    for p in players:
        if p.player_id not in img_pts:
            continue
        u, v = img_pts[p.player_id]
        centre = (int(round(u)), int(round(v)))
        if p.team_id == decision.attacking_team_id:
            if p.player_id in decision.offside_attacker_ids:
                color = color_atk_off
            else:
                color = color_atk_on
        else:
            color = color_def
        cv2.circle(out, centre, 8, color, thickness=-1)
        margin = decision.margins_m.get(p.player_id)
        tag = f"#{p.player_id}"
        if margin is not None:
            tag += f" {margin:+.2f}m"
        _draw_label(out, tag, (centre[0] + 10, centre[1] - 10), color)

    verdict = decision.verdict
    if decision.offside_attacker_ids:
        verdict += f" ({len(decision.offside_attacker_ids)} attacker(s))"
    _draw_label(out, verdict, (20, 40), color_line)
    _draw_label(
        out,
        f"line_x={decision.offside_line_x:.2f}m dir={decision.attacking_direction}",
        (20, 70),
        (255, 255, 255),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), out)
    return out

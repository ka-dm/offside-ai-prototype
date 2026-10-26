"""CLI for pitch calibration / homography."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from offside_ai.common import ensure_dir

from .homography import ManualPitchCalibrator
from .keypoints import create_pitch_detector


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="offside-pitch")
    p.add_argument("image", type=str)
    p.add_argument(
        "--backend",
        choices=["mock", "hf", "manual"],
        default="mock",
    )
    p.add_argument(
        "--points-json",
        type=str,
        default=None,
        help='Manual named points JSON: {"corner_bl":[u,v], ...} (4+)',
    )
    p.add_argument("-o", "--output-dir", type=str, default="artifacts/pitch")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    out = ensure_dir(args.output_dir)

    if args.points_json:
        named = {
            k: tuple(v) for k, v in json.loads(Path(args.points_json).read_text()).items()
        }
        homo = ManualPitchCalibrator().calibrate_named(named)
        kps = named
        backend = "manual"
    else:
        det = create_pitch_detector(args.backend)
        kps = det.detect_keypoints(img)
        homo = det.estimate_homography(img)
        backend = det.name

    payload = {
        "backend": backend,
        "keypoints_image": {k: list(v) for k, v in kps.items()},
        "H_image_to_pitch": homo.H_image_to_pitch.tolist(),
        "length_m": homo.length_m,
        "width_m": homo.width_m,
    }
    (out / "homography.json").write_text(json.dumps(payload, indent=2))

    vis = img.copy()
    for name, (u, v) in kps.items():
        cv2.circle(vis, (int(u), int(v)), 5, (0, 255, 255), -1)
        cv2.putText(
            vis, name, (int(u) + 5, int(v) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
        )
    # Draw projected pitch outline
    for (xa, ya), (xb, yb) in [
        ((0, 0), (105, 0)),
        ((105, 0), (105, 68)),
        ((105, 68), (0, 68)),
        ((0, 68), (0, 0)),
        ((52.5, 0), (52.5, 68)),
    ]:
        ua, va = homo.pitch_to_image(xa, ya)
        ub, vb = homo.pitch_to_image(xb, yb)
        cv2.line(vis, (int(ua), int(va)), (int(ub), int(vb)), (255, 0, 0), 2)
    cv2.imwrite(str(out / "pitch_overlay.jpg"), vis)
    print(f"backend={backend} wrote {out / 'homography.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

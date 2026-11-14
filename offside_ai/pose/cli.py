"""CLI for pose estimation on a detection JSON + image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from offside_ai.common import BoundingBox, ensure_dir
from offside_ai.pose.keypoints import (
    LEGAL_OFFSIDE_JOINTS,
    advanced_legal_point,
    create_pose_estimator,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="offside-pose")
    p.add_argument("image", type=str)
    p.add_argument(
        "--detections",
        type=str,
        required=True,
        help="detections.json from offside-detect",
    )
    p.add_argument(
        "--backend",
        choices=["mock", "vitpose", "sam3d"],
        default="mock",
    )
    p.add_argument(
        "--enable-sam3d",
        action="store_true",
        help="Required to instantiate the gated SAM 3D Body backend",
    )
    p.add_argument("--direction", choices=["increasing", "decreasing"], default="increasing")
    p.add_argument("-o", "--output-dir", type=str, default="artifacts/pose")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    data = json.loads(Path(args.detections).read_text())
    boxes = [BoundingBox(**b) for b in data["boxes"]]
    est = create_pose_estimator(args.backend, enable_sam3d=args.enable_sam3d)
    poses = est.estimate(img, boxes)
    out = ensure_dir(args.output_dir)

    payload = []
    vis = img.copy()
    for i, pose in enumerate(poses):
        adv = advanced_legal_point(pose, args.direction)
        payload.append(
            {
                "index": i,
                "advanced_legal_point": [adv.x, adv.y],
                "keypoints": {k: [v.x, v.y] for k, v in pose.keypoints.items()},
                "team_id": pose.bbox.team_id if pose.bbox else None,
            }
        )
        for name, pt in pose.keypoints.items():
            color = (0, 255, 0) if name in LEGAL_OFFSIDE_JOINTS else (128, 128, 128)
            cv2.circle(vis, (int(pt.x), int(pt.y)), 3, color, -1)
        cv2.circle(vis, (int(adv.x), int(adv.y)), 7, (0, 0, 255), 2)

    (out / "poses.json").write_text(json.dumps(payload, indent=2))
    cv2.imwrite(str(out / "poses.jpg"), vis)
    print(f"backend={est.name} n_poses={len(poses)} wrote {out / 'poses.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

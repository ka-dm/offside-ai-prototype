"""CLI for detection + team assignment on a single frame or video frame index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from offside_ai.common import ensure_dir, get_video_meta, read_frame

from .detector import create_detector, detect_frame
from .teams import assign_teams_by_jersey_color


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="offside-detect")
    p.add_argument("source", type=str, help="Image or video path")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument(
        "--detector",
        choices=["rfdetr", "yolo", "mock"],
        default="mock",
        help="Default mock keeps CI/laptop runnable without weights",
    )
    p.add_argument("--model-size", type=str, default="nano", help="RF-DETR size")
    p.add_argument("--yolo-weights", type=str, default="yolov8n.pt")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Manual team override boxIndex:teamId (repeatable)",
    )
    p.add_argument("-o", "--output-dir", type=str, default="artifacts/detection")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src = Path(args.source)
    out = ensure_dir(args.output_dir)

    if src.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
        meta = get_video_meta(src)
        image = read_frame(src, args.frame)
        fps = meta["fps"]
    else:
        image = cv2.imread(str(src))
        if image is None:
            raise RuntimeError(f"Could not read image: {src}")
        fps = 25.0

    kwargs = {}
    if args.detector == "rfdetr":
        kwargs = {"model_size": args.model_size}
    elif args.detector == "yolo":
        kwargs = {"model_name": args.yolo_weights}
    detector = create_detector(args.detector, **kwargs)
    result = detect_frame(image, detector=detector, frame_index=args.frame, fps=fps)

    overrides = {}
    for item in args.override:
        bi, tid = item.split(":")
        overrides[int(bi)] = int(tid)
    assignment = assign_teams_by_jersey_color(image, result, manual_overrides=overrides)
    assignment.apply_to_result(result)

    payload = {
        "backend": result.backend,
        "frame_index": result.frame_index,
        "boxes": [
            {
                "x1": b.x1,
                "y1": b.y1,
                "x2": b.x2,
                "y2": b.y2,
                "confidence": b.confidence,
                "label": b.label,
                "team_id": b.team_id,
            }
            for b in result.boxes
        ],
    }
    (out / "detections.json").write_text(json.dumps(payload, indent=2))

    vis = image.copy()
    for b in result.boxes:
        color = (0, 255, 0) if b.team_id == 0 else (0, 128, 255) if b.team_id == 1 else (200, 200, 200)
        if b.label == "ball":
            color = (0, 255, 255)
        cv2.rectangle(vis, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
        cv2.putText(
            vis,
            f"{b.label} t={b.team_id}",
            (int(b.x1), max(15, int(b.y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    cv2.imwrite(str(out / "detections.jpg"), vis)
    print(f"backend={result.backend} n_boxes={len(result.boxes)}")
    print(f"wrote {out / 'detections.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

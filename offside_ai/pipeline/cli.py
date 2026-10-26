"""CLI for the full offside pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="offside-pipeline")
    p.add_argument("video", type=str)
    p.add_argument("-o", "--output-dir", type=str, default="artifacts/pipeline")
    p.add_argument("--detector", choices=["rfdetr", "yolo", "mock"], default="mock")
    p.add_argument("--pose", choices=["vitpose", "mock", "sam3d"], default="mock")
    p.add_argument("--pitch", choices=["mock", "hf", "manual"], default="mock")
    p.add_argument("--attacking-team", type=int, default=0)
    p.add_argument("--direction", type=str, default="positive_x")
    p.add_argument("--av-offset", type=float, default=0.0)
    p.add_argument("--manual-pass-frame", type=int, default=None)
    p.add_argument(
        "--manual-pitch-json",
        type=str,
        default=None,
        help="Named pitch points JSON for manual homography",
    )
    p.add_argument("--enable-sam3d", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manual_pts = None
    if args.manual_pitch_json:
        raw = json.loads(Path(args.manual_pitch_json).read_text())
        manual_pts = {k: tuple(v) for k, v in raw.items()}
    cfg = PipelineConfig(
        detector_backend=args.detector,
        pose_backend=args.pose,
        pitch_backend=args.pitch,
        enable_sam3d=args.enable_sam3d,
        attacking_team_id=args.attacking_team,
        attacking_direction=args.direction,
        av_offset_s=args.av_offset,
        manual_pass_frame=args.manual_pass_frame,
        manual_pitch_points=manual_pts,
        work_dir=args.output_dir,
    )
    result = run_pipeline(args.video, cfg)
    print(result.decision.verdict)
    print(f"pass frame={result.pass_event.frame_index} source={result.pass_event.source}")
    print(f"annotated={result.annotated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

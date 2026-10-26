"""CLI for offside geometry given a JSON scene at the pass frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from offside_ai.common import ensure_dir

from .annotate import annotate_offside_frame
from .geometry import AttackingDirection, PlayerPoseOnPitch, compute_offside


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="offside-decide",
        description="Compute offside verdict from pitch-metre player positions JSON.",
    )
    p.add_argument(
        "scene_json",
        type=str,
        help=(
            "JSON with keys: attacking_team_id, attacking_direction, players"
            "[{player_id,team_id,x,y,role}], optional ball_x, optional image, "
            "optional player_image_points"
        ),
    )
    p.add_argument("-o", "--output-dir", type=str, default="artifacts/offside")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = json.loads(Path(args.scene_json).read_text())
    players = [PlayerPoseOnPitch(**p) for p in data["players"]]
    decision = compute_offside(
        players,
        attacking_team_id=int(data["attacking_team_id"]),
        attacking_direction=AttackingDirection.from_string(data["attacking_direction"]),
        ball_x=data.get("ball_x"),
    )
    out_dir = ensure_dir(args.output_dir)
    out_json = out_dir / "decision.json"
    out_json.write_text(json.dumps(decision.to_dict(), indent=2))
    print(decision.verdict)
    print(f"wrote {out_json}")

    image_path = data.get("image")
    if image_path:
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        pts = None
        if data.get("player_image_points"):
            pts = {int(k): tuple(v) for k, v in data["player_image_points"].items()}
        ann = out_dir / "annotated.png"
        annotate_offside_frame(
            img, decision, players, player_image_points=pts, output_path=ann
        )
        print(f"wrote {ann}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

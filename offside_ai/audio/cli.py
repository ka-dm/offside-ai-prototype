"""CLI for the audio pass-detection stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offside_ai.common import ensure_dir

from .detect import AudioPassConfig, detect_pass_from_audio, fuse_audio_visual
from .plotting import plot_spectrogram_with_peaks
from .visual_crosscheck import VisualCrossCheckConfig, estimate_pass_from_ball_motion


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="offside-audio",
        description="Detect kick/pass instant from video audio (spectrogram + onsets).",
    )
    p.add_argument("video", type=str, help="Path to input video clip")
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="artifacts/audio",
        help="Directory for WAV, plot, and JSON",
    )
    p.add_argument("--fps", type=float, default=None, help="Override video FPS")
    p.add_argument(
        "--av-offset",
        type=float,
        default=0.0,
        help="Audio/video offset in seconds (frame=(t+offset)*fps)",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--delta", type=float, default=0.15, help="Peak-pick delta")
    p.add_argument(
        "--bandpass",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(200.0, 4000.0),
    )
    p.add_argument(
        "--ball-json",
        type=str,
        default=None,
        help="Optional JSON list of [x,y]|null ball centres per frame for visual cross-check",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = ensure_dir(args.output_dir)

    cfg = AudioPassConfig(
        av_offset_s=args.av_offset,
        top_k=args.top_k,
        delta=args.delta,
        bandpass_hz=(float(args.bandpass[0]), float(args.bandpass[1])),
    )
    result = detect_pass_from_audio(
        args.video, fps=args.fps, cfg=cfg, work_dir=out_dir
    )

    plot_path = out_dir / "spectrogram_peaks.png"
    plot_spectrogram_with_peaks(
        result.spectrogram_db,
        result.spectrogram_times_s,
        result.spectrogram_freqs_hz,
        result.onset_times_s,
        onset_env=result.onset_env,
        onset_times_s=result.onset_env_times_s,
        output_path=plot_path,
    )

    fused = None
    if args.ball_json:
        raw = json.loads(Path(args.ball_json).read_text())
        centres = [None if c is None else tuple(c) for c in raw]
        v_t, v_c, v_extra = estimate_pass_from_ball_motion(
            centres, fps=result.fps, cfg=VisualCrossCheckConfig()
        )
        fused = fuse_audio_visual(result, v_t, v_c)
        fused.extras["visual_extra"] = v_extra

    payload = result.to_dict()
    payload["plot_path"] = str(plot_path)
    if fused is not None:
        payload["fused"] = fused.to_dict()

    json_path = out_dir / "pass_candidates.json"
    json_path.write_text(json.dumps(payload, indent=2))

    best = fused or result.best
    if best is None:
        print("No pass candidates detected.")
        return 1
    print(
        f"best source={best.source} t={best.time_s:.3f}s "
        f"frame={best.frame_index} conf={best.confidence:.3f}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {plot_path}")
    if best.extras.get("fallback"):
        print(f"NOTE: {best.extras['fallback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Cut SoccerNet broadcast windows around Pass and Offside spots.

Videos stay on disk (SoccerNet NDA). This module reads label JSON, cuts a
short clip with the audio track, and writes the paper's annotation CSV.
``gt_margin_m`` is left empty: SoccerNet does not label the offside line.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from offside_ai.common import ensure_dir, get_video_meta, time_to_frame
from offside_ai.eval.metrics import evaluate_annotations, load_annotations_csv
from offside_ai.eval.schema import ANNOTATION_COLUMNS

PASS_LABELS = {"pass", "high pass"}
OFFSIDE_LABELS = {"offside"}
HALF_VIDEO_PATTERNS = (
    "{half}_720p.mkv",
    "{half}_720p.mp4",
    "{half}.mkv",
    "{half}_HQ.mkv",
)
LABEL_FILENAMES = {
    "labels-v2.json",
    "labels-ball.json",
    "labels-ball-action.json",
}
CREDITS = "SoccerNet broadcast (research NDA; do not redistribute or commit)"
VIDEO_FILES = ["1_720p.mkv", "2_720p.mkv", "Labels-v2.json"]


@dataclass(frozen=True)
class SpotEvent:
    label: str  # pass | offside
    raw_label: str
    half: int
    time_s: float
    game_time: str
    team: str
    url_local: str
    label_file: Path
    video_path: Path


def canonical_label(raw: str) -> str | None:
    key = re.sub(r"\s+", " ", str(raw).strip().lower().replace("_", " "))
    if key in OFFSIDE_LABELS:
        return "offside"
    if key in PASS_LABELS:
        return "pass"
    return None


def parse_game_time(game_time: str) -> tuple[int, float]:
    """Parse ``"1 - 06:35"`` or ``"2 - 1:06:35"`` into (half, seconds)."""
    match = re.match(
        r"^\s*(\d+)\s*-\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s*$",
        str(game_time),
    )
    if match is None:
        raise ValueError(f"Unrecognized SoccerNet gameTime: {game_time!r}")
    half = int(match.group(1))
    if match.group(2) is None:
        minutes = int(match.group(3))
        seconds = float(match.group(4))
    else:
        minutes = int(match.group(2)) * 60 + int(match.group(3))
        seconds = float(match.group(4))
    return half, minutes * 60.0 + seconds


def annotation_time_s(annotation: dict) -> float:
    """Spot time inside the half, in seconds. ``position`` is milliseconds."""
    position = annotation.get("position", "")
    if str(position).strip() != "":
        return float(position) / 1000.0
    _, seconds = parse_game_time(str(annotation["gameTime"]))
    return seconds


def slug(text: str, limit: int = 60) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return (cleaned or "game")[:limit]


def load_label_file(path: Path) -> tuple[str | None, list[dict]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return None, [item for item in data if isinstance(item, dict)]
    url = data.get("UrlLocal") or data.get("urlLocal")
    annotations = data.get("annotations") or data.get("Annotations") or []
    return (str(url) if url else None), list(annotations)


def find_label_files(root: Path) -> list[Path]:
    found: list[Path] = []
    if not root.exists():
        return found
    for path in root.rglob("*.json"):
        if path.name.lower() in LABEL_FILENAMES:
            found.append(path)
    return sorted(found)


def resolve_half_video(
    root: Path,
    url_local: str | None,
    label_file: Path,
    half: int,
) -> Path | None:
    names = [pattern.format(half=half) for pattern in HALF_VIDEO_PATTERNS]
    bases: list[Path] = [label_file.parent]
    if url_local:
        rel = Path(url_local)
        bases.append(root / rel)
        for prefix in (
            Path("spotting-ball-2024"),
            Path("spotting-ball-2024") / "train",
            Path("spotting-ball-2024") / "valid",
            Path("spotting-ball-2024") / "test",
            Path("spotting-ball-2023"),
            Path("spotting-ball-2023") / "train",
        ):
            bases.append(root / prefix / rel)
        for parent in label_file.parents:
            bases.append(parent / rel)
            if parent == root:
                break
    seen: set[Path] = set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        for name in names:
            path = base / name
            if path.is_file():
                return path
    return None


def collect_events(
    root: Path,
    label_files: list[Path] | None = None,
    *,
    max_events: int | None = None,
) -> tuple[list[SpotEvent], list[str]]:
    """Return Pass/Offside spots whose half video is on disk, plus warnings."""
    files = label_files if label_files is not None else find_label_files(root)
    events: list[SpotEvent] = []
    warnings: list[str] = []
    seen: set[tuple[str, int, str, int]] = set()
    missing_video: set[tuple[str, int]] = set()
    for label_file in files:
        url_local, annotations = load_label_file(label_file)
        for annotation in annotations:
            raw = str(annotation.get("label", ""))
            kind = canonical_label(raw)
            if kind is None:
                continue
            game_time = str(annotation.get("gameTime", ""))
            try:
                half, _ = parse_game_time(game_time) if game_time else (1, 0.0)
                time_s = annotation_time_s(annotation)
            except (KeyError, TypeError, ValueError) as exc:
                warnings.append(f"{label_file}: skip {raw!r} ({exc})")
                continue
            if not game_time:
                half = 1
            video = resolve_half_video(root, url_local, label_file, half)
            if video is None:
                missing_key = (url_local or str(label_file.parent), half)
                if missing_key not in missing_video:
                    missing_video.add(missing_key)
                    warnings.append(
                        f"no video for half {half} ({missing_key[0]}). "
                        "Expected 1_720p.mkv next to the labels or under --root."
                    )
                continue
            game_key = url_local or str(label_file.parent)
            dedup_key = (game_key, half, kind, int(round(time_s * 5)))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            events.append(
                SpotEvent(
                    label=kind,
                    raw_label=raw,
                    half=half,
                    time_s=time_s,
                    game_time=game_time,
                    team=str(annotation.get("team", "")),
                    url_local=game_key,
                    label_file=label_file,
                    video_path=video,
                )
            )
    events.sort(key=lambda event: (event.url_local, event.half, event.time_s, event.label))
    if max_events is not None:
        events = events[: max(0, max_events)]
    return events, warnings


def clip_id_for(event: SpotEvent) -> str:
    ms = int(round(event.time_s * 1000))
    return f"{slug(event.url_local)}_h{event.half}_{event.label}_{ms}"


def window_bounds(
    time_s: float,
    duration_s: float,
    pre_s: float,
    post_s: float,
) -> tuple[float, float] | None:
    start = max(0.0, time_s - pre_s)
    end = min(duration_s, time_s + post_s)
    if end - start < 0.2:
        return None
    return start, end


def cut_window(src: Path, dst: Path, start_s: float, duration_s: float, fps: float) -> None:
    """Frame-accurate cut. Audio is re-encoded so the clip keeps a playable track."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-r",
        f"{fps:.6f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        tail = (proc.stderr or "")[-800:]
        raise RuntimeError(f"ffmpeg failed cutting {src} -> {dst}\n{tail}")


def cut_event_windows(
    root: Path,
    output_dir: Path,
    *,
    label_files: list[Path] | None = None,
    pre_s: float = 2.0,
    post_s: float = 3.0,
    max_events: int | None = None,
) -> tuple[Path, list[str]]:
    """Cut clips and write ``annotations.csv``. Returns the CSV path and warnings."""
    events, warnings = collect_events(root, label_files, max_events=max_events)
    if not events:
        raise FileNotFoundError(
            "No Pass/Offside events with a local half video. "
            "Download one match first (see docs/soccernet.md)."
        )
    out = ensure_dir(output_dir)
    clips = ensure_dir(out / "clips")
    rows: list[dict[str, object]] = []
    for event in events:
        meta = get_video_meta(event.video_path)
        bounds = window_bounds(event.time_s, float(meta["duration_s"]), pre_s, post_s)
        if bounds is None:
            warnings.append(f"skip {event.raw_label} at {event.time_s:.2f}s (outside video)")
            continue
        start_s, end_s = bounds
        fps = float(meta["fps"])
        clip_id = clip_id_for(event)
        dst = clips / f"{clip_id}.mp4"
        cut_window(event.video_path, dst, start_s, end_s - start_s, fps)
        gt_frame = time_to_frame(event.time_s - start_s, fps)
        decision = "offside" if event.label == "offside" else ""
        notes = (
            f"label={event.raw_label}; team={event.team}; gameTime={event.game_time}; "
            f"half={event.half}; spot_s={event.time_s:.3f}; window={start_s:.3f}-{end_s:.3f}. "
            "gt_margin_m unknown (not in SoccerNet). "
            "Attacking direction is not in the label; set it before trusting the line."
        )
        rows.append(
            {
                "clip_id": clip_id,
                "clip_path": str(dst.relative_to(out)),
                "fps": fps,
                "gt_pass_frame": gt_frame,
                "gt_decision": decision,
                "gt_margin_m": "",
                "attacking_team_id": "",
                "attacking_direction": "",
                "source": "soccernet",
                "credits": CREDITS,
                "notes": notes,
            }
        )
    if not rows:
        raise RuntimeError("Events were listed but every window was skipped")
    csv_path = out / "annotations.csv"
    pd.DataFrame(rows, columns=ANNOTATION_COLUMNS).to_csv(csv_path, index=False)
    return csv_path, warnings


def _require_soccernet():
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
        from SoccerNet.utils import getListGames
    except ImportError as exc:
        raise SystemExit(
            "SoccerNet is not installed. For the download step only:\n"
            "  pip install -U SoccerNet huggingface_hub\n"
            "  hf auth login\n"
            "Accept the NDA on https://huggingface.co/datasets/SoccerNet/SoccerNet_raw_HQ\n"
            "Cutting clips does not need that package if the files are already on disk."
        ) from exc
    return SoccerNetDownloader, getListGames


def list_games(split: str = "train", limit: int = 10) -> list[str]:
    _, get_list_games = _require_soccernet()
    games = list(get_list_games(split))
    return games[:limit] if limit else games


def download_one_game(
    local_dir: Path,
    game: str,
    split: str = "train",
    *,
    ball_labels: bool = False,
) -> None:
    """Download one match at 720p plus Labels-v2. Optionally the ball-action zip."""
    downloader_cls, _ = _require_soccernet()
    local_dir = ensure_dir(local_dir)
    downloader = downloader_cls(LocalDirectory=str(local_dir))
    downloader.downloadGame(game=game, files=VIDEO_FILES, spl=split)
    if not ball_labels:
        return
    print(
        f"Downloading spotting-ball-2024/{split}.zip (labels for the whole split, not 500 videos)."
    )
    downloader.downloadDataTask(task="spotting-ball-2024", split=[split])
    zpath = local_dir / "spotting-ball-2024" / f"{split}.zip"
    if zpath.is_file():
        dest = local_dir / "spotting-ball-2024" / split
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as archive:
            archive.extractall(dest)
        print(f"extracted {zpath} -> {dest}")


def _resolve_clip(annotations_csv: Path, clip_path: str) -> Path:
    path = Path(clip_path)
    if not path.is_absolute():
        path = annotations_csv.parent / path
    return path


def contrast_clips(
    annotations_csv: Path,
    output_dir: Path,
    *,
    detector: str = "mock",
    pose: str = "mock",
    pitch: str = "mock",
    av_offset_s: float = 0.0,
) -> Path:
    """Run audio onset and the full pipeline, then score the fused pass frame.

    ``gt_decision`` is scored only when the label was Offside. Pass spots stay
    in the pass-frame error and are marked unlabeled for the verdict.
    """
    from offside_ai.audio.detect import AudioPassConfig, detect_pass_from_audio
    from offside_ai.audio.plotting import plot_spectrogram_with_peaks
    from offside_ai.pipeline.run import PipelineConfig, run_pipeline

    annotations_csv = Path(annotations_csv)
    frame = load_annotations_csv(annotations_csv)
    out = ensure_dir(output_dir)
    pred_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for _, row in frame.iterrows():
        clip_id = str(row["clip_id"])
        clip = _resolve_clip(annotations_csv, str(row["clip_path"]))
        if not clip.is_file():
            failures.append(f"{clip_id}: missing {clip}")
            continue
        audio_dir = ensure_dir(out / "audio" / clip_id)
        pipe_dir = ensure_dir(out / "pipeline" / clip_id)
        try:
            audio = detect_pass_from_audio(
                clip,
                fps=float(row["fps"]) if not _blank_fps(row["fps"]) else None,
                cfg=AudioPassConfig(av_offset_s=av_offset_s),
                work_dir=audio_dir,
            )
            plot_spectrogram_with_peaks(
                audio.spectrogram_db,
                audio.spectrogram_times_s,
                audio.spectrogram_freqs_hz,
                audio.onset_times_s,
                onset_env=audio.onset_env,
                onset_times_s=audio.onset_env_times_s,
                output_path=audio_dir / "spectrogram_peaks.png",
            )
            direction = str(row.get("attacking_direction", "") or "")
            if direction not in {"positive_x", "negative_x"}:
                direction = "positive_x"
            team_raw = row.get("attacking_team_id", "")
            try:
                team_id = int(team_raw)
            except (TypeError, ValueError):
                team_id = 0
            result = run_pipeline(
                clip,
                PipelineConfig(
                    detector_backend=detector,  # type: ignore[arg-type]
                    pose_backend=pose,  # type: ignore[arg-type]
                    pitch_backend=pitch,  # type: ignore[arg-type]
                    attacking_team_id=team_id,
                    attacking_direction=direction,
                    av_offset_s=av_offset_s,
                    work_dir=str(pipe_dir),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{clip_id}: {exc}")
            continue
        margins = result.decision.margins_m
        visual_time = result.pass_event.extras.get("visual_time_s", "")
        pred_rows.append(
            {
                "clip_id": clip_id,
                "pred_pass_frame": int(result.pass_event.frame_index),
                "pred_decision": result.decision.verdict.lower(),
                "pred_margin_m": max(margins.values()) if margins else "",
                "audio_confidence": (
                    float(audio.best.confidence) if audio.best is not None else ""
                ),
                "audio_pass_frame": (
                    int(audio.best.frame_index) if audio.best is not None else ""
                ),
                "visual_time_s": visual_time,
                "pass_source": result.pass_event.source,
            }
        )
    if failures:
        (out / "failures.txt").write_text("\n".join(failures) + "\n")
    if not pred_rows:
        raise RuntimeError("No clip produced a fused pass frame:\n" + "\n".join(failures))
    pred_path = out / "predictions.csv"
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
    scored = frame[frame["clip_id"].isin([row["clip_id"] for row in pred_rows])]
    report = evaluate_annotations(scored, pd.read_csv(pred_path))
    (out / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    pd.DataFrame([err.__dict__ for err in report.error_sources]).to_csv(
        out / "error_sources.csv", index=False
    )
    pd.DataFrame(report.per_clip).to_csv(out / "per_clip.csv", index=False)
    return pred_path


def _blank_fps(value: object) -> bool:
    try:
        return float(value) <= 0
    except (TypeError, ValueError):
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="offside-soccernet",
        description="Download one SoccerNet match and cut Pass/Offside windows with audio.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    games = sub.add_parser("games", help="Print a few game ids (needs the SoccerNet package)")
    games.add_argument("--split", default="train")
    games.add_argument("--limit", type=int, default=10)

    download = sub.add_parser("download", help="Download one 720p match and Labels-v2.json")
    download.add_argument("--local-dir", type=Path, default=Path("artifacts/soccernet"))
    download.add_argument("--game", required=True, help="Path from `offside-soccernet games`")
    download.add_argument("--split", default="train")
    download.add_argument(
        "--ball-labels",
        action="store_true",
        help="Also fetch spotting-ball-2024/<split>.zip (Pass spots for the whole split)",
    )

    cut = sub.add_parser("cut", help="Cut short windows around Pass and Offside")
    cut.add_argument("--root", type=Path, required=True, help="Folder that contains the match")
    cut.add_argument("--labels", type=Path, nargs="*", default=None)
    cut.add_argument("--out", type=Path, default=Path("artifacts/soccernet/clips"))
    cut.add_argument("--pre", type=float, default=2.0, help="Seconds before the spot")
    cut.add_argument("--post", type=float, default=3.0, help="Seconds after the spot")
    cut.add_argument("--max-events", type=int, default=None)

    contrast = sub.add_parser(
        "contrast",
        help="Audio onset vs ball motion, then score the fused frame",
    )
    contrast.add_argument("--annotations", type=Path, required=True)
    contrast.add_argument("--out", type=Path, default=Path("artifacts/soccernet/contrast"))
    contrast.add_argument("--detector", choices=["rfdetr", "yolo", "mock"], default="mock")
    contrast.add_argument("--pose", choices=["vitpose", "mock", "sam3d"], default="mock")
    contrast.add_argument("--pitch", choices=["mock", "hf", "manual"], default="mock")
    contrast.add_argument("--av-offset", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "games":
        for game in list_games(args.split, args.limit):
            print(game)
        return 0
    if args.cmd == "download":
        download_one_game(
            args.local_dir, args.game, args.split, ball_labels=args.ball_labels
        )
        print(f"files under {args.local_dir}")
        return 0
    if args.cmd == "cut":
        csv_path, warnings = cut_event_windows(
            args.root,
            args.out,
            label_files=args.labels,
            pre_s=args.pre,
            post_s=args.post,
            max_events=args.max_events,
        )
        for warning in warnings:
            print(f"warning: {warning}")
        print(f"wrote {csv_path}")
        return 0
    pred = contrast_clips(
        args.annotations,
        args.out,
        detector=args.detector,
        pose=args.pose,
        pitch=args.pitch,
        av_offset_s=args.av_offset,
    )
    print(f"wrote {pred}")
    print(f"report {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

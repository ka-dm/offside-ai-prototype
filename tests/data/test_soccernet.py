"""SoccerNet window cutting and audio/visual contrast (no broadcast footage)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from offside_ai.data.soccernet import (
    annotation_time_s,
    canonical_label,
    collect_events,
    cut_event_windows,
    parse_game_time,
    window_bounds,
)
from offside_ai.eval.cli import main as eval_main
from offside_ai.eval.metrics import evaluate_annotations


def test_parse_game_time_and_labels():
    assert parse_game_time("1 - 06:35") == (1, 6 * 60 + 35)
    assert parse_game_time("2 - 1:06:35") == (2, 66 * 60 + 35)
    assert canonical_label("PASS") == "pass"
    assert canonical_label("High Pass") == "pass"
    assert canonical_label("Offside") == "offside"
    assert canonical_label("Shot") is None
    assert annotation_time_s({"position": "395000", "gameTime": "1 - 06:35"}) == 395.0


def test_window_clamps_to_video_start():
    assert window_bounds(0.4, duration_s=10.0, pre_s=2.0, post_s=3.0) == (0.0, 3.4)


def test_collect_events_reads_both_label_files(tmp_path: Path):
    game = tmp_path / "league" / "2014" / "Game"
    game.mkdir(parents=True)
    (game / "1_720p.mkv").write_bytes(b"not-a-video")
    url = "league/2014/Game"
    (game / "Labels-v2.json").write_text(
        json.dumps(
            {
                "UrlLocal": url,
                "annotations": [
                    {
                        "gameTime": "1 - 00:03",
                        "label": "Offside",
                        "position": "3000",
                        "team": "home",
                    },
                    {
                        "gameTime": "1 - 00:10",
                        "label": "Foul",
                        "position": "10000",
                    },
                ],
            }
        )
    )
    (game / "Labels-ball.json").write_text(
        json.dumps(
            {
                "UrlLocal": url,
                "annotations": [
                    {
                        "gameTime": "1 - 00:05",
                        "label": "PASS",
                        "position": "5500",
                        "team": "away",
                    }
                ],
            }
        )
    )
    events, warnings = collect_events(tmp_path)
    assert warnings == []
    assert [event.label for event in events] == ["offside", "pass"]
    assert events[0].video_path.name == "1_720p.mkv"


def test_blank_decision_scores_pass_frame_only():
    ann = pd.DataFrame(
        {
            "clip_id": ["a", "b"],
            "gt_pass_frame": [10, 20],
            "gt_decision": ["offside", ""],
            "gt_margin_m": ["", ""],
            "fps": [25.0, 25.0],
            "notes": ["", ""],
        }
    )
    pred = pd.DataFrame(
        {
            "clip_id": ["a", "b"],
            "pred_pass_frame": [10, 22],
            "pred_decision": ["offside", "onside"],
            "pred_margin_m": ["", ""],
            "audio_confidence": [0.8, 0.2],
        }
    )
    report = evaluate_annotations(ann, pred)
    assert report.pass_frame.n == 2
    assert report.decision.n == 1
    assert report.decision.accuracy == 1.0
    assert any(err.error_type == "decision_unlabeled" for err in report.error_sources)


def _write_kicks(path: Path, duration_s: float, kicks: tuple[float, ...]) -> None:
    sr = 22050
    n = int(sr * duration_s)
    y = 0.01 * np.random.default_rng(0).standard_normal(n)
    burst_len = int(0.025 * sr)
    for kick_t in kicks:
        i0 = int(kick_t * sr)
        i1 = min(n, i0 + burst_len)
        tb = np.arange(i1 - i0) / sr
        y[i0:i1] += 0.9 * np.sin(2 * np.pi * 1200 * tb) * np.exp(-tb * 80)
    sf.write(str(path), y, sr)


def test_cut_and_contrast_synthetic_match(tmp_path: Path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    game = tmp_path / "league" / "2014" / "Game"
    game.mkdir(parents=True)
    wav = tmp_path / "kicks.wav"
    _write_kicks(wav, duration_s=8.0, kicks=(3.0, 5.5))
    video = game / "1_720p.mp4"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=160x120:r=25:d=8",
            "-i",
            str(wav),
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"could not mux test video: {proc.stderr[-400:]}")

    url = "league/2014/Game"
    (game / "Labels-v2.json").write_text(
        json.dumps(
            {
                "UrlLocal": url,
                "annotations": [
                    {
                        "gameTime": "1 - 00:03",
                        "label": "Offside",
                        "position": "3000",
                        "team": "home",
                        "visibility": "visible",
                    }
                ],
            }
        )
    )
    (game / "Labels-ball.json").write_text(
        json.dumps(
            {
                "UrlLocal": url,
                "annotations": [
                    {
                        "gameTime": "1 - 00:05",
                        "label": "PASS",
                        "position": "5500",
                        "team": "away",
                        "visibility": "visible",
                    }
                ],
            }
        )
    )
    out = tmp_path / "windows"
    csv_path, warnings = cut_event_windows(
        tmp_path, out, pre_s=2.0, post_s=1.5, max_events=4
    )
    assert warnings == []
    ann = pd.read_csv(csv_path)
    assert len(ann) == 2
    assert set(ann["gt_decision"].fillna("")) == {"offside", ""}
    assert ann["gt_margin_m"].isna().all()
    assert ann["source"].eq("soccernet").all()
    for rel in ann["clip_path"]:
        assert (out / rel).is_file()
    # Spot sits 2 s into each window at 25 fps.
    assert ann["gt_pass_frame"].between(48, 52).all()

    from offside_ai.data.soccernet import contrast_clips

    pred_path = contrast_clips(csv_path, tmp_path / "contrast")
    pred = pd.read_csv(pred_path)
    assert set(pred["pass_source"]).issubset({"audio", "visual", "fused"})
    assert "audio_pass_frame" in pred.columns
    code = eval_main(
        ["score", str(csv_path), str(pred_path), "-o", str(tmp_path / "eval")]
    )
    assert code == 0
    report = json.loads((tmp_path / "eval" / "report.json").read_text())
    assert report["pass_frame"]["n"] == 2
    assert report["decision"]["n"] == 1
    assert report["pass_frame"]["mean_abs_error_frames"] < 8

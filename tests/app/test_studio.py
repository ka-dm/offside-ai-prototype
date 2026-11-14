"""Light studio page: events and spectrogram payload."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from offside_ai.app.studio import HTML_PATH, analyze_clip
from offside_ai.eval.synthetic import synthesize_kick_wav


def test_page_is_light_and_names_events():
    html = HTML_PATH.read_text()
    assert "Eventos" in html
    assert "#ffffff" in html
    assert "Espectrograma" in html


def test_analyze_clip_returns_events_and_spectrogram(tmp_path: Path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    wav = synthesize_kick_wav(tmp_path / "kick.wav", duration_s=2.0, kick_time_s=0.8)
    video = tmp_path / "clip.mp4"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=160x120:r=25:d=2",
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
        pytest.skip(proc.stderr[-300:])
    payload = analyze_clip(video, run_geometry=True)
    labels = [event["label"] for event in payload["events"]]
    assert any(label.startswith("Audio") or label == "Onset" for label in labels)
    assert any(label.startswith("Pase") for label in labels)
    assert any(label in {"Fuera de juego", "En juego"} for label in labels)
    assert payload["spectrogram"].startswith("iVBOR")
    assert payload["duration_s"] > 0
    labels_seen = {box["label"] for track in payload["tracks"] for box in track["boxes"]}
    assert {"player", "referee", "ball"} <= labels_seen
    assert any(track["line"] for track in payload["tracks"])

"""Unit tests for the audio pass-detection stage (synthetic signals)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from offside_ai.audio.detect import (
    AudioPassConfig,
    bandpass_filter,
    detect_pass_from_audio,
    fuse_audio_visual,
    pick_onset_peaks,
    compute_onset_envelope,
)
from offside_ai.audio.extract import extract_audio
from offside_ai.audio.plotting import plot_spectrogram_with_peaks
from offside_ai.audio.visual_crosscheck import estimate_pass_from_ball_motion
from offside_ai.common import time_to_frame
from offside_ai.eval.synthetic import synthesize_kick_wav


def test_time_to_frame_offset():
    assert time_to_frame(1.0, fps=25.0, offset_s=0.0) == 25
    assert time_to_frame(1.0, fps=25.0, offset_s=0.04) == 26


def test_bandpass_attenuates_out_of_band():
    sr = 22050
    t = np.arange(sr) / sr
    low = np.sin(2 * np.pi * 50 * t)
    mid = np.sin(2 * np.pi * 1000 * t)
    y = low + mid
    filtered = bandpass_filter(y, sr, 200.0, 4000.0)
    # Mid-band energy should dominate after filtering
    assert np.std(filtered) > 0.1
    # Correlation with mid > low
    corr_mid = abs(np.corrcoef(filtered, mid)[0, 1])
    corr_low = abs(np.corrcoef(filtered, low)[0, 1])
    assert corr_mid > corr_low


def test_detect_synthetic_kick(tmp_path: Path):
    kick_t = 0.75
    fps = 25.0
    wav = synthesize_kick_wav(tmp_path / "kick.wav", kick_time_s=kick_t)
    cfg = AudioPassConfig(av_offset_s=0.0, top_k=3, delta=0.05, min_relative_peak=0.2)
    result = detect_pass_from_audio(wav_path=wav, fps=fps, cfg=cfg)
    assert result.best is not None
    # Within ~3 frames (120 ms) for a clean synthetic burst
    assert abs(result.best.frame_index - int(round(kick_t * fps))) <= 3
    assert result.best.confidence > 0.2
    assert result.spectrogram_db.ndim == 2


def test_plot_spectrogram(tmp_path: Path):
    kick_t = 0.5
    wav = synthesize_kick_wav(tmp_path / "kick.wav", kick_time_s=kick_t)
    result = detect_pass_from_audio(wav_path=wav, fps=25.0)
    out = tmp_path / "spec.png"
    path = plot_spectrogram_with_peaks(
        result.spectrogram_db,
        result.spectrogram_times_s,
        result.spectrogram_freqs_hz,
        result.onset_times_s,
        onset_env=result.onset_env,
        onset_times_s=result.onset_env_times_s,
        output_path=out,
    )
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_fuse_prefers_visual_when_audio_weak():
    from offside_ai.audio.detect import AudioPassResult
    from offside_ai.common import PassEvent

    weak = PassEvent(1.0, 25, 0.2, source="audio")
    audio = AudioPassResult(
        candidates=[weak],
        best=weak,
        fps=25.0,
        sample_rate=22050,
        onset_times_s=np.array([1.0]),
        onset_env=np.ones(10),
        onset_env_times_s=np.linspace(0, 1, 10),
        spectrogram_db=np.zeros((8, 10)),
        spectrogram_times_s=np.linspace(0, 1, 10),
        spectrogram_freqs_hz=np.linspace(0, 1000, 8),
        meta={"config": {"av_offset_s": 0.0}},
    )
    fused = fuse_audio_visual(audio, visual_time_s=1.05, visual_confidence=0.8)
    assert fused.source == "visual"


def test_visual_ball_motion_speed_jump():
    # Ball slowly moving then suddenly accelerates
    centres = [(float(i), 0.0) for i in range(10)]
    centres += [(10 + 5 * (i + 1), 0.0) for i in range(5)]
    t, conf, extra = estimate_pass_from_ball_motion(centres, fps=25.0)
    assert t is not None
    assert conf >= 0.25


def test_extract_audio_from_video_if_ffmpeg(tmp_path: Path):
    """Create a tiny video with ffmpeg+audio and extract it back."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    wav = synthesize_kick_wav(tmp_path / "a.wav", duration_s=1.0, kick_time_s=0.4)
    video = tmp_path / "clip.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=green:s=160x120:d=1",
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
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"could not mux test video: {proc.stderr[-500:]}")
    out = extract_audio(video, output_wav=tmp_path / "out.wav", sample_rate=22050)
    assert out.exists()
    result = detect_pass_from_audio(video_path=video, work_dir=tmp_path)
    assert result.best is not None


def test_pick_onset_peaks_empty_safe():
    cfg = AudioPassConfig()
    times, vals, idx = pick_onset_peaks(
        np.array([], dtype=np.float64), np.array([], dtype=np.float64), cfg
    )
    assert len(times) == 0

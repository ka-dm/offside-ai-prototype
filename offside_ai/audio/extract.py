"""Extract mono PCM audio from a video file using ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def extract_audio(
    video_path: str | Path,
    output_wav: str | Path | None = None,
    sample_rate: int = 22050,
    mono: bool = True,
) -> Path:
    """Extract audio track to WAV via ffmpeg.

    Parameters
    ----------
    video_path:
        Input video (any container ffmpeg can read).
    output_wav:
        Destination WAV path. Defaults to `<video_stem>_audio.wav` beside the video.
    sample_rate:
        Target sample rate in Hz.
    mono:
        If True, downmix to a single channel.

    Returns
    -------
    Path to the written WAV file.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg (apt/brew) or use Colab's system ffmpeg."
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_wav is None:
        output_wav = video_path.with_name(f"{video_path.stem}_audio.wav")
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(int(sample_rate)),
    ]
    if mono:
        cmd.extend(["-ac", "1"])
    cmd.append(str(output_wav))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg audio extraction failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr: {proc.stderr[-2000:]}"
        )
    if not output_wav.exists() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty audio file: {output_wav}")
    return output_wav

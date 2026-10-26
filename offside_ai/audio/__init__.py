"""Pass/kick instant detection from broadcast audio (+ optional visual cross-check)."""

from .detect import (
    AudioPassConfig,
    AudioPassResult,
    detect_pass_from_audio,
    fuse_audio_visual,
)
from .extract import extract_audio
from .plotting import plot_spectrogram_with_peaks
from .visual_crosscheck import VisualCrossCheckConfig, estimate_pass_from_ball_motion

__all__ = [
    "AudioPassConfig",
    "AudioPassResult",
    "detect_pass_from_audio",
    "fuse_audio_visual",
    "extract_audio",
    "plot_spectrogram_with_peaks",
    "VisualCrossCheckConfig",
    "estimate_pass_from_ball_motion",
]

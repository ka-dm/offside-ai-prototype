"""Spectrogram + band-pass onset detection for kick/pass instants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from offside_ai.common import PassEvent, get_video_meta, time_to_frame

from .extract import extract_audio


@dataclass
class AudioPassConfig:
    """Tunable parameters for audio-based pass detection."""

    sample_rate: int = 22050
    # Football kick / ball impact energy is often concentrated mid-high.
    bandpass_hz: tuple[float, float] = (200.0, 4000.0)
    n_fft: int = 2048
    hop_length: int = 512
    # Onset strength -> peak picking
    pre_max: int = 3
    post_max: int = 3
    pre_avg: int = 3
    post_avg: int = 5
    delta: float = 0.15
    wait: int = 10
    # Relative threshold on onset envelope (fraction of max)
    min_relative_peak: float = 0.35
    # Audio/video sync: frame = (t_audio + av_offset_s) * fps
    av_offset_s: float = 0.0
    # How many top candidates to keep
    top_k: int = 5
    # Confidence heuristics
    snr_floor_db: float = 3.0
    # If peak prominence is weak vs noise floor, flag low confidence
    low_confidence_below: float = 0.45
    # Drop energy that stays put (chants, whistle). A kick is a short burst.
    suppress_crowd: bool = True
    crowd_sustain_s: float = 0.18


@dataclass
class AudioPassResult:
    candidates: list[PassEvent]
    best: PassEvent | None
    fps: float
    sample_rate: int
    onset_times_s: np.ndarray
    onset_env: np.ndarray
    onset_env_times_s: np.ndarray
    spectrogram_db: np.ndarray
    spectrogram_times_s: np.ndarray
    spectrogram_freqs_hz: np.ndarray
    wav_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "best": self.best.to_dict() if self.best else None,
            "fps": self.fps,
            "sample_rate": self.sample_rate,
            "wav_path": self.wav_path,
            "meta": self.meta,
        }


def _butter_bandpass_sos(
    low_hz: float, high_hz: float, sr: int, order: int = 4
) -> np.ndarray:
    nyq = 0.5 * sr
    low = max(low_hz / nyq, 1e-5)
    high = min(high_hz / nyq, 0.999)
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid band-pass range for sr={sr}: {(low_hz, high_hz)}")
    return butter(order, [low, high], btype="band", output="sos")


def bandpass_filter(
    y: np.ndarray, sr: int, low_hz: float, high_hz: float
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    sos = _butter_bandpass_sos(low_hz, high_hz, sr)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    # sosfiltfilt needs sufficient length
    if len(y) < 32:
        return y.astype(np.float64)
    return sosfiltfilt(sos, y.astype(np.float64))


def attenuate_sustained(
    y: np.ndarray,
    sr: int,
    sustain_s: float = 0.18,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Turn down sounds that last longer than a kick, such as a chant.

    A median over ``sustain_s`` keeps the steady part of each frequency.
    Subtracting it leaves the short impacts.
    """
    if y.size < sr // 5:
        return y.astype(np.float64)
    spectrum = librosa.stft(y.astype(np.float64), n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(spectrum)
    width = max(3, int(round(sustain_s * sr / hop_length)))
    if width % 2 == 0:
        width += 1
    if width >= magnitude.shape[1]:
        return y.astype(np.float64)
    from scipy.ndimage import median_filter

    sustained = median_filter(magnitude, size=(1, width))
    impact = np.maximum(magnitude - sustained, 0.0)
    cleaned = spectrum * (impact / (magnitude + 1e-8))
    out = librosa.istft(cleaned, hop_length=hop_length, length=len(y))
    return np.asarray(out, dtype=np.float64)


def prepare_impact_audio(y: np.ndarray, sr: int, cfg: AudioPassConfig) -> np.ndarray:
    """Band-limit, then quiet the chant before onset detection."""
    filtered = bandpass_filter(y, sr, cfg.bandpass_hz[0], cfg.bandpass_hz[1])
    if not cfg.suppress_crowd:
        return filtered
    return attenuate_sustained(filtered, sr, sustain_s=cfg.crowd_sustain_s)


def compute_spectrogram_db(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (S_db [freq, time], times_s, freqs_hz)."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)
    times = librosa.frames_to_time(
        np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S_db, times, freqs


def compute_onset_envelope(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (onset_env, times_s)."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=sr, hop_length=hop_length
    )
    return onset_env.astype(np.float64), times.astype(np.float64)


def pick_onset_peaks(
    onset_env: np.ndarray,
    times_s: np.ndarray,
    cfg: AudioPassConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Peak-pick onsets; return (peak_times_s, peak_values, peak_indices)."""
    if len(onset_env) == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
        )

    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=cfg.pre_max,
        post_max=cfg.post_max,
        pre_avg=cfg.pre_avg,
        post_avg=cfg.post_avg,
        delta=cfg.delta,
        wait=cfg.wait,
    )
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size == 0:
        # Fallback: global max (still useful for short clips with one kick)
        idx = int(np.argmax(onset_env))
        peaks = np.array([idx], dtype=np.int64)

    values = onset_env[peaks]
    vmax = float(np.max(onset_env) + 1e-12)
    keep = values >= (cfg.min_relative_peak * vmax)
    peaks = peaks[keep]
    values = values[keep]
    if peaks.size == 0:
        idx = int(np.argmax(onset_env))
        peaks = np.array([idx], dtype=np.int64)
        values = onset_env[peaks]

    # Rank by strength and keep top_k
    order = np.argsort(-values)[: cfg.top_k]
    peaks = peaks[order]
    values = values[order]
    times = times_s[peaks]
    return times.astype(np.float64), values.astype(np.float64), peaks


def _estimate_confidence(
    peak_value: float,
    onset_env: np.ndarray,
    cfg: AudioPassConfig,
) -> float:
    """Map peak prominence / SNR-like ratio into [0, 1]."""
    if onset_env.size == 0:
        return 0.0
    noise = float(np.median(onset_env) + 1e-12)
    peak = float(peak_value)
    ratio = peak / noise
    # Soft saturation: ratio ~1 -> low, ratio >= 8 -> high
    conf = (np.log1p(ratio) / np.log1p(8.0)).item()
    conf = float(np.clip(conf, 0.0, 1.0))
    # Penalize if absolute peak is weak vs global max
    gmax = float(np.max(onset_env) + 1e-12)
    conf *= float(np.clip(peak / gmax, 0.0, 1.0))
    return conf


def detect_pass_from_audio(
    video_path: str | Path | None = None,
    *,
    wav_path: str | Path | None = None,
    fps: float | None = None,
    cfg: AudioPassConfig | None = None,
    work_dir: str | Path | None = None,
) -> AudioPassResult:
    """Detect candidate kick/pass times from audio.

    Provide either ``video_path`` (audio extracted with ffmpeg) or an existing
    ``wav_path``. Frame indices use ``frame = round((t + av_offset_s) * fps)``.
    """
    cfg = cfg or AudioPassConfig()
    extracted_wav: Path | None = None

    if wav_path is None:
        if video_path is None:
            raise ValueError("Provide video_path or wav_path")
        out = None
        if work_dir is not None:
            out = Path(work_dir) / f"{Path(video_path).stem}_audio.wav"
        extracted_wav = extract_audio(
            video_path, output_wav=out, sample_rate=cfg.sample_rate
        )
        wav_path = extracted_wav
    else:
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(wav_path)

    if fps is None:
        if video_path is not None:
            fps = float(get_video_meta(video_path)["fps"])
        else:
            fps = 25.0

    y, sr = librosa.load(str(wav_path), sr=cfg.sample_rate, mono=True)
    y_bp = prepare_impact_audio(y, sr, cfg)

    S_db, S_times, S_freqs = compute_spectrogram_db(
        y_bp, sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    onset_env, onset_times = compute_onset_envelope(
        y_bp, sr, hop_length=cfg.hop_length
    )
    peak_times, peak_values, _ = pick_onset_peaks(onset_env, onset_times, cfg)

    # SNR-like diagnostic over the band-passed signal
    rms = float(np.sqrt(np.mean(y_bp**2) + 1e-12))
    peak_abs = float(np.max(np.abs(y_bp)) + 1e-12)
    crest_db = 20.0 * np.log10(peak_abs / rms)

    candidates: list[PassEvent] = []
    for t, v in zip(peak_times.tolist(), peak_values.tolist()):
        conf = _estimate_confidence(v, onset_env, cfg)
        # Crowd / weak impact heuristic
        if crest_db < cfg.snr_floor_db:
            conf *= 0.5
        frame_idx = time_to_frame(t, fps, cfg.av_offset_s)
        candidates.append(
            PassEvent(
                time_s=float(t),
                frame_index=int(frame_idx),
                confidence=float(conf),
                source="audio",
                extras={
                    "onset_strength": float(v),
                    "crest_db": float(crest_db),
                    "av_offset_s": float(cfg.av_offset_s),
                },
            )
        )

    best = candidates[0] if candidates else None
    if best is not None and best.confidence < cfg.low_confidence_below:
        best.extras["low_confidence"] = True
        best.extras["advice"] = (
            "Audio peak is weak (crowd noise or missing ball sound). "
            "Prefer visual cross-check or manual frame selection."
        )

    return AudioPassResult(
        candidates=candidates,
        best=best,
        fps=float(fps),
        sample_rate=int(sr),
        onset_times_s=peak_times,
        onset_env=onset_env,
        onset_env_times_s=onset_times,
        spectrogram_db=S_db,
        spectrogram_times_s=S_times,
        spectrogram_freqs_hz=S_freqs,
        wav_path=str(wav_path),
        meta={
            "config": asdict(cfg),
            "crest_db": float(crest_db),
            "duration_s": float(len(y) / sr),
            "n_candidates": len(candidates),
        },
    )


def fuse_audio_visual(
    audio: AudioPassResult,
    visual_time_s: float | None,
    visual_confidence: float = 0.0,
    max_gap_s: float = 0.12,
) -> PassEvent:
    """Combine audio best candidate with an optional visual ball-motion cue.

    If audio confidence is low and visual is available within ``max_gap_s``,
    prefer a fused / visual timestamp and report reduced confidence when they
    disagree beyond the gap.
    """
    if audio.best is None and visual_time_s is None:
        raise ValueError("No audio or visual pass candidate available")

    if audio.best is None:
        assert visual_time_s is not None
        return PassEvent(
            time_s=float(visual_time_s),
            frame_index=time_to_frame(
                visual_time_s, audio.fps, audio.meta.get("config", {}).get("av_offset_s", 0.0)
            ),
            confidence=float(visual_confidence),
            source="visual",
            extras={"reason": "no_audio_candidate"},
        )

    audio_event = audio.best
    if visual_time_s is None:
        out = PassEvent(
            time_s=audio_event.time_s,
            frame_index=audio_event.frame_index,
            confidence=audio_event.confidence,
            source="audio",
            extras={**audio_event.extras, "visual": None},
        )
        if audio_event.confidence < 0.45:
            out.extras["fallback"] = "manual_frame_selection_recommended"
        return out

    gap = abs(float(visual_time_s) - float(audio_event.time_s))
    offset = float(audio.meta.get("config", {}).get("av_offset_s", 0.0))

    if audio_event.confidence >= 0.45 and gap <= max_gap_s:
        # Agree — slight confidence boost
        t = 0.7 * audio_event.time_s + 0.3 * float(visual_time_s)
        return PassEvent(
            time_s=float(t),
            frame_index=time_to_frame(t, audio.fps, offset),
            confidence=float(min(1.0, audio_event.confidence + 0.1)),
            source="fused",
            extras={"gap_s": gap, "visual_confidence": visual_confidence},
        )

    if audio_event.confidence < 0.45 and visual_confidence >= 0.4:
        return PassEvent(
            time_s=float(visual_time_s),
            frame_index=time_to_frame(visual_time_s, audio.fps, offset),
            confidence=float(visual_confidence * 0.9),
            source="visual",
            extras={
                "gap_s": gap,
                "audio_confidence": audio_event.confidence,
                "fallback": "audio_unreliable",
            },
        )

    # Disagree or weak visual: keep audio but flag
    return PassEvent(
        time_s=audio_event.time_s,
        frame_index=audio_event.frame_index,
        confidence=float(audio_event.confidence * (0.7 if gap > max_gap_s else 1.0)),
        source="audio",
        extras={
            "gap_s": gap,
            "visual_time_s": float(visual_time_s),
            "visual_confidence": visual_confidence,
            "fallback": "manual_frame_selection_recommended"
            if gap > max_gap_s or audio_event.confidence < 0.45
            else None,
        },
    )

"""Listen to a file or stream URL the way a live broadcast arrives.

Audio is kept in a short rolling window. A pass event is emitted only when a
new onset appears at the leading edge of that window. The offside verdict
runs on that one frame, not on every frame of the match.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from offside_ai.audio.detect import (
    AudioPassConfig,
    _estimate_confidence,
    compute_onset_envelope,
    compute_spectrogram_db,
    prepare_impact_audio,
)
from offside_ai.common import read_frame
from offside_ai.pipeline.run import PipelineConfig

SR = 22050
WINDOW_S = 8.0
HOP_S = 0.5


def _event(event_id: str, time_s: float, kind: str, label: str, detail: str) -> dict:
    return {
        "id": event_id,
        "time_s": round(float(time_s), 3),
        "kind": kind,
        "label": label,
        "detail": detail,
    }


@dataclass
class RollingEar:
    """Onset detector over a rolling audio window."""

    sr: int = SR
    window_s: float = WINDOW_S
    hop_s: float = HOP_S
    cfg: AudioPassConfig = field(
        default_factory=lambda: AudioPassConfig(delta=0.08, min_relative_peak=0.30)
    )
    samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    origin: int = 0
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    last_event_s: float = -10.0
    seq: int = 0

    def push(self, chunk: np.ndarray) -> list[dict]:
        """Append samples. Return events whose onset just reached the present."""
        if chunk.size == 0:
            return []
        self.pending = np.concatenate([self.pending, np.asarray(chunk, dtype=np.float64)])
        hop = int(self.hop_s * self.sr)
        found: list[dict] = []
        while self.pending.size >= hop:
            piece = self.pending[:hop]
            self.pending = self.pending[hop:]
            self.samples = np.concatenate([self.samples, piece])
            max_n = int(self.window_s * self.sr)
            if self.samples.size > max_n:
                drop = self.samples.size - max_n
                self.samples = self.samples[drop:]
                self.origin += drop
            found.extend(self._scan())
        return found

    def _scan(self) -> list[dict]:
        y = prepare_impact_audio(self.samples, self.sr, self.cfg)
        env, times = compute_onset_envelope(y, self.sr, hop_length=self.cfg.hop_length)
        if env.size == 0 or times.size == 0:
            return []
        now = (self.origin + self.samples.size) / self.sr
        if now < 1.0:
            return []
        window_start = self.origin / self.sr
        newest = (self.samples.size / self.sr) - self.hop_s
        fresh = times >= (newest - 0.02)
        if not np.any(fresh):
            return []
        local = env[fresh]
        local_times = times[fresh]
        best = int(np.argmax(local))
        strength = float(local[best])
        floor = float(np.median(env) + 1e-12) * 4.0
        if strength < floor:
            return []
        absolute = window_start + float(local_times[best])
        if absolute < self.last_event_s + 0.8:
            return []
        confidence = _estimate_confidence(strength, env, self.cfg)
        if confidence < 0.35:
            return []
        self.last_event_s = absolute
        self.seq += 1
        event = _event(
            f"live-audio-{self.seq}",
            absolute,
            "audio",
            "Audio",
            f"confianza {confidence:.2f}",
        )
        event["confidence"] = round(confidence, 3)
        return [event]

    def spectrogram(self) -> tuple[np.ndarray, np.ndarray]:
        if self.samples.size < self.sr // 4:
            return np.zeros((1, 1)), np.array([0.0])
        y = prepare_impact_audio(self.samples, self.sr, self.cfg)
        spec, _, freqs = compute_spectrogram_db(y, self.sr, hop_length=self.cfg.hop_length)
        return spec, freqs


def _ffmpeg_audio(source: str, pace: float) -> subprocess.Popen:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg no está instalado")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
    if pace <= 1.01:
        cmd.append("-re")
    cmd += ["-i", source, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def iter_pcm(source: str, pace: float = 1.0):
    """Yield float chunks of ``hop`` seconds from a file or stream URL."""
    proc = _ffmpeg_audio(source, pace)
    hop = int(HOP_S * SR) * 2
    assert proc.stdout is not None
    try:
        while True:
            raw = proc.stdout.read(hop)
            if not raw:
                break
            yield np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    finally:
        proc.kill()
        proc.wait(timeout=2)


def verdict_at(
    source: str,
    time_s: float,
    fps: float,
    *,
    attacking_team_id: int = 0,
    attacking_direction: str = "positive_x",
) -> dict | None:
    """Run detection, pose and the offside line on the single frame at ``time_s``."""
    if not Path(source).is_file():
        return None
    frame_index = max(0, int(round(time_s * fps)))
    try:
        frame = read_frame(source, frame_index)
        from offside_ai.pipeline.run import describe_frame

        described = describe_frame(
            frame,
            frame_index,
            fps,
            PipelineConfig(
                detector_backend="broadcast",
                attacking_team_id=attacking_team_id,
                attacking_direction=attacking_direction,
            ),
        )
        decision = described["decision"]
        overlay = described["overlay"]
    except Exception:
        return None
    offside = bool(decision.is_offside)
    margin = ""
    if decision.margins_m:
        margin = f" · margen {max(decision.margins_m.values()):.2f} m"
    event = _event(
        f"live-verdict-{frame_index}",
        time_s,
        "offside" if offside else "onside",
        "Fuera de juego" if offside else "En juego",
        f"línea {decision.offside_line_x:.2f} m{margin}",
    )
    overlay["time_s"] = round(float(time_s), 3)
    event["overlay"] = overlay
    return event


@dataclass
class LiveSession:
    source: str
    pace: float = 1.0
    fps: float = 25.0
    judge: bool = True
    attacking_team_id: int = 0
    attacking_direction: str = "positive_x"
    messages: deque = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    done: threading.Event = field(default_factory=threading.Event)
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _push(self, message: dict) -> None:
        with self.lock:
            self.messages.append(message)

    def drain(self) -> list[dict]:
        with self.lock:
            found = list(self.messages)
            self.messages.clear()
            return found

    def _run(self) -> None:
        ear = RollingEar()
        heard = 0
        try:
            for chunk in iter_pcm(self.source, self.pace):
                if self.stop.is_set():
                    break
                heard += chunk.size
                now = heard / SR
                for event in ear.push(chunk):
                    self._push({"type": "event", "event": {k: v for k, v in event.items() if k != "confidence"}})
                    confidence = float(event.get("confidence", 1))
                    if self.judge and confidence >= 0.45 and Path(self.source).is_file():
                        verdict = verdict_at(
                            self.source,
                            event["time_s"],
                            self.fps,
                            attacking_team_id=self.attacking_team_id,
                            attacking_direction=self.attacking_direction,
                        )
                        if verdict is not None:
                            self._push({"type": "event", "event": verdict})
                spec, freqs = ear.spectrogram()
                from offside_ai.app.studio import _strip_png

                self._push(
                    {
                        "type": "tick",
                        "time_s": round(now, 3),
                        "window_s": WINDOW_S,
                        "spectrogram": _strip_png(spec, freqs),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            self._push({"type": "error", "message": str(exc)})
        finally:
            self._push({"type": "done", "time_s": round(heard / SR, 3)})
            self.done.set()

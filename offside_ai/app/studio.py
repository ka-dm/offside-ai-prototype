"""Light timeline page: video, spectrogram, and named events."""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import shutil
import subprocess
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

from offside_ai.app.live import WINDOW_S, LiveSession
from offside_ai.audio.detect import AudioPassConfig, detect_pass_from_audio
from offside_ai.common import ensure_dir, get_video_meta, read_frame
from offside_ai.pipeline.run import PipelineConfig, run_pipeline

HTML_PATH = Path(__file__).with_name("studio.html")
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"}
ROOT = Path(__file__).resolve().parents[2]
MEDIA: dict[str, Path] = {}
LIVE: dict[str, LiveSession] = {}
MAX_GEOMETRY_S = 25.0


def _strip_png(spectrogram_db: np.ndarray, freqs_hz: np.ndarray) -> str:
    """Magma spectrogram strip, low frequencies at the bottom."""
    if spectrogram_db.size == 0:
        return ""
    mask = freqs_hz <= 5000.0
    image = spectrogram_db[mask] if np.any(mask) else spectrogram_db
    cols = image.shape[1]
    if cols > 960:
        index = np.linspace(0, cols - 1, 960).astype(int)
        image = image[:, index]
    lo, hi = np.percentile(image, [5, 99])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    stops = np.array(
        [
            [0, 0, 4],
            [28, 16, 68],
            [79, 18, 123],
            [182, 54, 121],
            [251, 135, 97],
            [252, 253, 191],
        ],
        dtype=np.float64,
    )
    scaled = norm * (len(stops) - 1)
    left = np.floor(scaled).astype(int)
    left = np.clip(left, 0, len(stops) - 2)
    frac = (scaled - left)[..., None]
    rgb = stops[left] * (1.0 - frac) + stops[left + 1] * frac
    rgb = np.flipud(rgb).astype(np.uint8)
    picture = Image.fromarray(rgb, mode="RGB")
    picture = picture.resize((max(rgb.shape[1], 1), 140), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _event(event_id: str, time_s: float, kind: str, label: str, detail: str) -> dict:
    return {
        "id": event_id,
        "time_s": round(float(time_s), 3),
        "kind": kind,
        "label": label,
        "detail": detail,
    }


def analyze_clip(video_path: str | Path, *, run_geometry: bool | None = None) -> dict:
    """Audio onsets, optional image/pass/verdict, and a spectrogram strip."""
    video_path = Path(video_path)
    meta = get_video_meta(video_path)
    duration = float(meta["duration_s"])
    work = ensure_dir(ROOT / "artifacts" / "studio" / "work" / uuid.uuid4().hex[:8])
    audio = detect_pass_from_audio(
        video_path,
        fps=float(meta["fps"]),
        cfg=AudioPassConfig(),
        work_dir=work,
    )
    events: list[dict] = []
    for index, candidate in enumerate(audio.candidates):
        events.append(
            _event(
                f"audio-{index}",
                candidate.time_s,
                "audio",
                "Audio" if index == 0 else "Onset",
                f"confianza {candidate.confidence:.2f} · frame {candidate.frame_index}",
            )
        )

    note = ""
    do_geometry = duration <= MAX_GEOMETRY_S if run_geometry is None else run_geometry
    if not do_geometry:
        note = "Clip largo: eventos de audio. Recorta una ventana corta para el veredicto."
    else:
        try:
            result = run_pipeline(
                video_path,
                PipelineConfig(work_dir=str(work / "pipeline")),
            )
        except Exception as exc:  # noqa: BLE001
            note = f"Audio listo. Geometría no disponible: {exc}"
        else:
            chosen = result.pass_event
            visual_t = chosen.extras.get("visual_time_s")
            if chosen.source == "visual":
                visual_t = chosen.time_s
            if visual_t is not None:
                events.append(
                    _event(
                        "image",
                        float(visual_t),
                        "image",
                        "Imagen",
                        "movimiento del balón",
                    )
                )
            source_label = {
                "audio": "Pase por audio",
                "visual": "Pase por imagen",
                "fused": "Pase audio + imagen",
                "manual": "Pase manual",
            }.get(chosen.source, "Pase")
            events.append(
                _event(
                    "pass",
                    chosen.time_s,
                    "pass",
                    source_label,
                    f"confianza {chosen.confidence:.2f} · frame {chosen.frame_index}",
                )
            )
            offside = result.decision.is_offside
            margin = ""
            if result.decision.margins_m:
                margin = f" · margen {max(result.decision.margins_m.values()):.2f} m"
            events.append(
                _event(
                    "verdict",
                    chosen.time_s,
                    "offside" if offside else "onside",
                    "Fuera de juego" if offside else "En juego",
                    f"línea {result.decision.offside_line_x:.2f} m{margin}",
                )
            )
            note = "Audio contrastado con la imagen"

    events.sort(key=lambda item: (item["time_s"], item["id"]))
    tracks = _sample_tracks(video_path, float(meta["fps"]), duration) if do_geometry else []
    return {
        "name": video_path.name,
        "duration_s": duration,
        "fps": float(meta["fps"]),
        "events": events,
        "tracks": tracks,
        "spectrogram": _strip_png(audio.spectrogram_db, audio.spectrogram_freqs_hz),
        "note": note,
    }


def _sample_tracks(video_path: Path, fps: float, duration: float) -> list[dict]:
    """Overlay samples so the page can follow players across a short clip."""
    from offside_ai.pipeline.run import describe_frame

    if duration <= 0 or fps <= 0:
        return []
    step = 0.4
    times = []
    cursor = 0.0
    while cursor < duration - 0.05 and len(times) < 24:
        times.append(cursor)
        cursor += step
    tracks = []
    for time_s in times:
        frame_index = max(0, int(round(time_s * fps)))
        try:
            frame = read_frame(video_path, frame_index)
            overlay = describe_frame(frame, frame_index, fps)["overlay"]
        except Exception:
            continue
        overlay["time_s"] = round(time_s, 3)
        tracks.append(overlay)
    return tracks


def playback_file(src: Path) -> Path:
    """Browsers play MP4. Remux other containers without re-encoding."""
    if src.suffix.lower() in {".mp4", ".m4v", ".webm"}:
        return src
    dest = ensure_dir(ROOT / "artifacts" / "studio" / "playback") / f"{src.stem}.mp4"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if shutil.which("ffmpeg") is None:
        return src
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not dest.exists():
        return src
    return dest


def register_media(path: Path) -> str:
    media_id = uuid.uuid4().hex[:12]
    MEDIA[media_id] = path
    return media_id


def local_clips() -> list[dict[str, str]]:
    root = ROOT / "artifacts" / "soccernet"
    if not root.exists():
        return []
    found = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_SUFFIXES or path.name.startswith("."):
            continue
        if "playback" in path.parts or "windows" in path.parts:
            continue
        half = "parte 1" if path.name.startswith("1") else "parte 2" if path.name.startswith("2") else path.stem
        parent = path.parent.name
        match = parent.split(" ", 3)[-1] if parent[:4].isdigit() else parent
        found.append({"label": f"{match} · {half}", "path": str(path)})
    return found[:8]


def build_demo_clip(directory: Path) -> Path:
    from offside_ai.eval.synthetic import synthesize_kick_wav

    directory.mkdir(parents=True, exist_ok=True)
    wav = synthesize_kick_wav(directory / "demo.wav", duration_s=3.0, kick_time_s=1.2)
    video = directory / "demo.mp4"
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg no está instalado")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x8fbf8f:s=640x360:r=25:d=3",
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
        raise RuntimeError(proc.stderr[-400:])
    return video


def _analysis_payload(src: Path) -> dict:
    play = playback_file(src)
    media_id = register_media(play)
    payload = analyze_clip(src)
    payload["video_url"] = f"/media/{media_id}"
    payload["media_id"] = media_id
    payload["name"] = src.name
    return payload


def _is_stream_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "rtmp://", "rtsp://"))


def _play_choices(raw: dict) -> tuple[int, str]:
    try:
        team = int(raw.get("attacking_team_id", 0))
    except (TypeError, ValueError):
        team = 0
    if team not in {0, 1}:
        team = 0
    direction = str(raw.get("attacking_direction") or "positive_x")
    if direction not in {"positive_x", "negative_x"}:
        direction = "positive_x"
    return team, direction


def start_live(
    source: str,
    pace: float = 1.0,
    *,
    attacking_team_id: int = 0,
    attacking_direction: str = "positive_x",
) -> dict:
    fps = 25.0
    video_url = ""
    media_id = ""
    name = source
    if _is_stream_url(source):
        name = "Directo"
    else:
        path = Path(source).expanduser()
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            raise FileNotFoundError("Elige un vídeo local o una URL http, rtmp o rtsp")
        meta = get_video_meta(path)
        fps = float(meta["fps"])
        play = playback_file(path)
        media_id = register_media(play)
        video_url = f"/media/{media_id}"
        name = path.name
        source = str(path)
    session = LiveSession(
        source=source,
        pace=max(0.25, pace),
        fps=fps,
        judge=not _is_stream_url(source),
        attacking_team_id=attacking_team_id,
        attacking_direction=attacking_direction,
    )
    live_id = uuid.uuid4().hex[:12]
    LIVE[live_id] = session
    session.start()
    return {
        "id": live_id,
        "name": name,
        "media_id": media_id,
        "video_url": video_url,
        "window_s": WINDOW_S,
        "pace": session.pace,
    }


def inspect_media(
    media_id: str,
    time_s: float,
    *,
    attacking_team_id: int = 0,
    attacking_direction: str = "positive_x",
) -> dict:
    """Person boxes, teams and the offside line for one frame of a loaded clip."""
    path = MEDIA.get(media_id)
    if path is None or not path.is_file():
        raise FileNotFoundError("Ese clip ya no está en memoria")
    meta = get_video_meta(path)
    fps = float(meta["fps"]) or 25.0
    frame_index = max(0, int(round(float(time_s) * fps)))
    frame = read_frame(path, frame_index)
    from offside_ai.pipeline.run import PipelineConfig, describe_frame

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
    overlay = described["overlay"]
    overlay["time_s"] = round(float(time_s), 3)
    return {"overlay": overlay, "players": sum(1 for box in overlay["boxes"] if box["label"] != "ball")}


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "OpenSAOT/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[studio] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: object) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _error(self, code: int, message: str) -> None:
        self._send(code, message.encode(), "text/plain; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/clips":
            self._json(200, local_clips())
            return
        if path.startswith("/api/live/"):
            self._stream_live(unquote(path[len("/api/live/") :]))
            return
        if path.startswith("/media/"):
            self._send_media(unquote(path[len("/media/") :]))
            return
        self._error(404, "No encontrado")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/demo":
                demo = build_demo_clip(ROOT / "artifacts" / "studio" / "demo")
                self._json(200, _analysis_payload(demo))
                return
            if path == "/api/open":
                length = int(self.headers.get("Content-Length", "0"))
                raw = json.loads(self.rfile.read(length) or b"{}")
                src = Path(str(raw.get("path", ""))).expanduser()
                if not src.is_file() or src.suffix.lower() not in VIDEO_SUFFIXES:
                    self._error(400, "Elige un vídeo local (.mp4, .mkv, .mov)")
                    return
                self._json(200, _analysis_payload(src))
                return
            if path == "/api/live":
                length = int(self.headers.get("Content-Length", "0"))
                raw = json.loads(self.rfile.read(length) or b"{}")
                if raw.get("media"):
                    src = MEDIA.get(str(raw["media"]))
                    if src is None:
                        self._error(404, "Ese clip ya no está en memoria")
                        return
                    source = str(src)
                else:
                    source = str(raw.get("url") or raw.get("path") or "")
                pace = float(raw.get("pace") or 1)
                team, direction = _play_choices(raw)
                self._json(200, start_live(source, pace, attacking_team_id=team, attacking_direction=direction))
                return
            if path == "/api/inspect":
                length = int(self.headers.get("Content-Length", "0"))
                raw = json.loads(self.rfile.read(length) or b"{}")
                team, direction = _play_choices(raw)
                self._json(
                    200,
                    inspect_media(
                        str(raw.get("media", "")),
                        float(raw.get("time_s") or 0),
                        attacking_team_id=team,
                        attacking_direction=direction,
                    ),
                )
                return
            if path == "/api/live-stop":
                length = int(self.headers.get("Content-Length", "0"))
                raw = json.loads(self.rfile.read(length) or b"{}")
                session = LIVE.get(str(raw.get("id", "")))
                if session is not None:
                    session.stop.set()
                self._json(200, {"ok": True})
                return
            if path == "/api/analyze":
                name = Path(self.headers.get("X-Filename", "clip.mp4")).name
                if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
                    name += ".mp4"
                dest = ensure_dir(ROOT / "artifacts" / "studio" / "uploads") / f"{uuid.uuid4().hex[:8]}_{name}"
                length = int(self.headers.get("Content-Length", "0"))
                remaining = length
                with dest.open("wb") as handle:
                    while remaining > 0:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        handle.write(chunk)
                        remaining -= len(chunk)
                self._json(200, _analysis_payload(dest))
                return
        except Exception as exc:  # noqa: BLE001
            self._error(500, str(exc))
            return
        self._error(404, "No encontrado")

    def _stream_live(self, live_id: str) -> None:
        session = LIVE.get(live_id)
        if session is None:
            self._error(404, "La sesión en vivo no existe")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                messages = session.drain()
                finished = False
                for message in messages:
                    payload = json.dumps(message).encode()
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    self.wfile.flush()
                    if message.get("type") in {"done", "error"}:
                        finished = True
                if finished:
                    break
                if session.done.wait(0.2) and not session.messages:
                    break
        except (BrokenPipeError, ConnectionResetError):
            session.stop.set()

    def _send_media(self, media_id: str) -> None:
        path = MEDIA.get(media_id)
        if path is None or not path.is_file():
            self._error(404, "Vídeo no disponible")
            return
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        status = 200
        if range_header and range_header.startswith("bytes="):
            spec = range_header.removeprefix("bytes=").split(",", 1)[0]
            start_s, _, end_s = spec.partition("-")
            if start_s:
                start = int(start_s)
            if end_s:
                end = int(end_s)
            end = min(end, size - 1)
            if start > end or start < 0:
                self._error(416, "Rango inválido")
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def make_server(host: str = "127.0.0.1", port: int = 7861) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), StudioHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="offside-studio", description="OpenSAOT, semiautomatic VAR")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port)
    print(f"OpenSAOT http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

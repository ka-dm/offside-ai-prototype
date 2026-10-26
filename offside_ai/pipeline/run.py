"""Wire audio → detection → pitch → pose → offside with graceful fallbacks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from offside_ai.audio.detect import AudioPassConfig, detect_pass_from_audio, fuse_audio_visual
from offside_ai.audio.plotting import plot_spectrogram_with_peaks
from offside_ai.audio.visual_crosscheck import estimate_pass_from_ball_motion
from offside_ai.common import PassEvent, ensure_dir, get_video_meta, read_frame
from offside_ai.detection.detector import create_detector, detect_frame
from offside_ai.detection.teams import assign_teams_by_jersey_color
from offside_ai.offside.annotate import annotate_offside_frame
from offside_ai.offside.geometry import (
    AttackingDirection,
    OffsideDecision,
    PlayerPoseOnPitch,
    compute_offside,
)
from offside_ai.pitch.homography import ManualPitchCalibrator, PitchHomography
from offside_ai.pitch.keypoints import create_pitch_detector
from offside_ai.pose.keypoints import advanced_legal_point, create_pose_estimator


@dataclass
class PipelineConfig:
    detector_backend: Literal["rfdetr", "yolo", "mock"] = "mock"
    pose_backend: Literal["vitpose", "mock", "sam3d"] = "mock"
    pitch_backend: Literal["mock", "hf", "manual"] = "mock"
    enable_sam3d: bool = False
    attacking_team_id: int = 0
    attacking_direction: str = "positive_x"
    av_offset_s: float = 0.0
    manual_pass_frame: int | None = None
    manual_pitch_points: dict[str, tuple[float, float]] | None = None
    team_overrides: dict[int, int] = field(default_factory=dict)
    work_dir: str = "artifacts/pipeline"


@dataclass
class PipelineResult:
    pass_event: PassEvent
    decision: OffsideDecision
    annotated_path: str
    spectrogram_path: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_event": self.pass_event.to_dict(),
            "decision": self.decision.to_dict(),
            "annotated_path": self.annotated_path,
            "spectrogram_path": self.spectrogram_path,
            "meta": self.meta,
        }


def run_pipeline(video_path: str | Path, cfg: PipelineConfig | None = None) -> PipelineResult:
    cfg = cfg or PipelineConfig()
    video_path = Path(video_path)
    out = ensure_dir(cfg.work_dir)
    meta = get_video_meta(video_path)

    # --- 1) Audio pass detection ---
    audio_cfg = AudioPassConfig(av_offset_s=cfg.av_offset_s)
    audio = detect_pass_from_audio(
        video_path, fps=meta["fps"], cfg=audio_cfg, work_dir=out
    )
    spec_path = out / "spectrogram_peaks.png"
    plot_spectrogram_with_peaks(
        audio.spectrogram_db,
        audio.spectrogram_times_s,
        audio.spectrogram_freqs_hz,
        audio.onset_times_s,
        onset_env=audio.onset_env,
        onset_times_s=audio.onset_env_times_s,
        output_path=spec_path,
    )

    if cfg.manual_pass_frame is not None:
        pass_event = PassEvent(
            time_s=cfg.manual_pass_frame / meta["fps"],
            frame_index=int(cfg.manual_pass_frame),
            confidence=1.0,
            source="manual",
        )
    else:
        # Optional visual cross-check using mock/real detections on a window
        pass_event = audio.best or PassEvent(0.0, 0, 0.0, "audio")
        # Try ball motion around candidates when detector available
        try:
            det = create_detector(cfg.detector_backend)
            centres = []
            start = max(0, pass_event.frame_index - 8)
            end = min(meta["n_frames"] - 1, pass_event.frame_index + 8)
            for fi in range(start, end + 1):
                frame = read_frame(video_path, fi)
                det_res = detect_frame(frame, detector=det, frame_index=fi, fps=meta["fps"])
                ball = det_res.ball()
                centres.append(None if ball is None else ball.center.as_tuple())
            v_t, v_c, _ = estimate_pass_from_ball_motion(
                centres, fps=meta["fps"], frame_start=start
            )
            pass_event = fuse_audio_visual(audio, v_t, v_c)
        except Exception as e:  # noqa: BLE001
            if audio.best is None:
                raise RuntimeError(
                    f"Audio found no pass and visual fallback failed: {e}. "
                    "Set manual_pass_frame."
                ) from e
            pass_event = audio.best
            pass_event.extras["visual_error"] = str(e)

    frame = read_frame(video_path, pass_event.frame_index)

    # --- 2) Detection + teams ---
    detector = create_detector(cfg.detector_backend)
    det_res = detect_frame(
        frame, detector=detector, frame_index=pass_event.frame_index, fps=meta["fps"]
    )
    assign_teams_by_jersey_color(
        frame, det_res, manual_overrides=cfg.team_overrides
    ).apply_to_result(det_res)

    # --- 3) Pitch homography ---
    if cfg.manual_pitch_points:
        homo = ManualPitchCalibrator().calibrate_named(cfg.manual_pitch_points)
        pitch_backend = "manual"
    else:
        pitch_det = create_pitch_detector(cfg.pitch_backend)
        try:
            homo = pitch_det.estimate_homography(frame)
            pitch_backend = pitch_det.name
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Pitch calibration failed ({e}). Provide manual_pitch_points "
                "(4+ named corners/keypoints)."
            ) from e

    # --- 4) Pose + advanced legal points ---
    pose_est = create_pose_estimator(
        cfg.pose_backend, enable_sam3d=cfg.enable_sam3d
    )
    poses = pose_est.estimate(frame, det_res.boxes)

    direction = AttackingDirection.from_string(cfg.attacking_direction)
    img_dir = (
        "increasing"
        if direction is AttackingDirection.POSITIVE_X
        else "decreasing"
    )

    players: list[PlayerPoseOnPitch] = []
    image_points: dict[int, tuple[float, float]] = {}
    player_boxes = [b for b in det_res.boxes if b.label in {"player", "goalkeeper"}]
    for i, pose in enumerate(poses):
        box = pose.bbox or (player_boxes[i] if i < len(player_boxes) else None)
        if box is None:
            continue
        adv_img = advanced_legal_point(pose, img_dir)
        x_m, y_m = homo.image_to_pitch(adv_img.x, adv_img.y)
        pid = i
        team = 0 if box.team_id is None else int(box.team_id)
        role = "goalkeeper" if box.label == "goalkeeper" else "player"
        players.append(
            PlayerPoseOnPitch(
                player_id=pid, team_id=team, x=x_m, y=y_m, role=role
            )
        )
        image_points[pid] = (adv_img.x, adv_img.y)

    if len(players) < 2:
        raise RuntimeError(
            f"Need at least 2 players with poses; got {len(players)}. "
            "Check detector backend / frame."
        )

    ball = det_res.ball()
    ball_x = None
    if ball is not None:
        ball_x, _ = homo.image_to_pitch(ball.center.x, ball.center.y)

    # Infer attacking team if all detections share one cluster oddly
    attacking_team_id = cfg.attacking_team_id

    decision = compute_offside(
        players,
        attacking_team_id=attacking_team_id,
        attacking_direction=direction,
        ball_x=ball_x,
    )

    ann_path = out / "annotated.jpg"
    annotate_offside_frame(
        frame,
        decision,
        players,
        pitch_to_image=homo.pitch_to_image,
        player_image_points=image_points,
        output_path=ann_path,
    )

    result = PipelineResult(
        pass_event=pass_event,
        decision=decision,
        annotated_path=str(ann_path),
        spectrogram_path=str(spec_path),
        meta={
            "video": str(video_path),
            "fps": meta["fps"],
            "detector": detector.name,
            "pose": pose_est.name,
            "pitch": pitch_backend,
            "n_players": len(players),
            "audio_candidates": [c.to_dict() for c in audio.candidates],
        },
    )
    (out / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result

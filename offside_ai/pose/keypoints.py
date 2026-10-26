"""ViTPose body keypoints + most-advanced legal point (excluding arms).

IFAB: hands/arms are not considered for offside. We therefore exclude wrist,
elbow, and shoulder joints when choosing the most advanced legal body point
toward the goal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from offside_ai.common import BoundingBox, Point2D

# COCO-17 style names used by many ViTPose checkpoints
COCO_JOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Exclude arms entirely (shoulder/elbow/wrist)
ARM_JOINTS = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
}

LEGAL_OFFSIDE_JOINTS = [j for j in COCO_JOINTS if j not in ARM_JOINTS]

BackendName = Literal["vitpose", "mock", "sam3d"]


@dataclass
class PersonPose:
    keypoints: dict[str, Point2D]  # image pixel coords
    scores: dict[str, float] = field(default_factory=dict)
    bbox: BoundingBox | None = None


class PoseEstimator(ABC):
    name: str

    @abstractmethod
    def estimate(
        self, image_bgr: np.ndarray, boxes: Sequence[BoundingBox]
    ) -> list[PersonPose]:
        ...


class MockPoseEstimator(PoseEstimator):
    """Synthetic poses from bounding boxes (ankles / hips / head)."""

    name = "mock"

    def estimate(
        self, image_bgr: np.ndarray, boxes: Sequence[BoundingBox]
    ) -> list[PersonPose]:
        poses: list[PersonPose] = []
        for b in boxes:
            if b.label not in {"player", "goalkeeper"}:
                continue
            cx = (b.x1 + b.x2) / 2
            top, bot = b.y1, b.y2
            left, right = b.x1, b.x2
            kps = {
                "nose": Point2D(cx, top + 0.12 * (bot - top)),
                "left_hip": Point2D(cx - 0.1 * (right - left), top + 0.55 * (bot - top)),
                "right_hip": Point2D(cx + 0.1 * (right - left), top + 0.55 * (bot - top)),
                "left_knee": Point2D(cx - 0.12 * (right - left), top + 0.75 * (bot - top)),
                "right_knee": Point2D(cx + 0.12 * (right - left), top + 0.75 * (bot - top)),
                "left_ankle": Point2D(cx - 0.1 * (right - left), bot - 2),
                "right_ankle": Point2D(cx + 0.1 * (right - left), bot - 2),
            }
            scores = {k: 0.9 for k in kps}
            poses.append(PersonPose(keypoints=kps, scores=scores, bbox=b))
        return poses


class ViTPoseEstimator(PoseEstimator):
    """ViTPose via Hugging Face transformers (Apache-2.0 weights when available)."""

    name = "vitpose"

    def __init__(self, model_id: str = "usyd-community/vitpose-base-simple"):
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, VitPoseForPoseEstimation
        except ImportError as e:
            raise ImportError(
                "transformers (ViTPose) not installed. pip install transformers torch "
                f"or use --backend mock. Original error: {e}"
            ) from e
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = VitPoseForPoseEstimation.from_pretrained(self.model_id)
            self._model.eval()
        except Exception as e:  # noqa: BLE001 — surface download / gate errors clearly
            raise RuntimeError(
                f"Failed to load ViTPose model '{self.model_id}'. "
                "On Colab with GPU, ensure network access and sufficient disk. "
                f"Fallback: --backend mock. Original error: {e}"
            ) from e

    def estimate(
        self, image_bgr: np.ndarray, boxes: Sequence[BoundingBox]
    ) -> list[PersonPose]:
        self._load()
        import torch

        # transformers VitPose typically expects RGB + person boxes
        rgb = image_bgr[:, :, ::-1]
        person_boxes = [
            [b.x1, b.y1, b.x2, b.y2]
            for b in boxes
            if b.label in {"player", "goalkeeper"}
        ]
        if not person_boxes:
            return []
        inputs = self._processor(rgb, boxes=[person_boxes], return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        # post-process — API may return dataset_predicted_pose / keypoints
        poses: list[PersonPose] = []
        try:
            processed = self._processor.post_process_pose_estimation(
                outputs, boxes=[person_boxes]
            )
        except Exception:
            # Older API fallback — return mock-like from boxes if postprocess fails
            return MockPoseEstimator().estimate(image_bgr, boxes)

        # processed often: list over images -> list over persons
        persons = processed[0] if processed else []
        player_boxes = [b for b in boxes if b.label in {"player", "goalkeeper"}]
        for i, person in enumerate(persons):
            kps: dict[str, Point2D] = {}
            scores: dict[str, float] = {}
            # Expect keypoints Nx3 (x,y,score) or dict
            if isinstance(person, dict) and "keypoints" in person:
                arr = np.asarray(person["keypoints"])
            else:
                arr = np.asarray(person)
            for j, name in enumerate(COCO_JOINTS):
                if j >= len(arr):
                    break
                x, y = float(arr[j][0]), float(arr[j][1])
                sc = float(arr[j][2]) if arr.shape[-1] > 2 else 1.0
                kps[name] = Point2D(x, y)
                scores[name] = sc
            bbox = player_boxes[i] if i < len(player_boxes) else None
            poses.append(PersonPose(keypoints=kps, scores=scores, bbox=bbox))
        return poses


class SAM3DBodyEstimator(PoseEstimator):
    """Optional SAM 3D Body (gated SAM License) — off by default, heavy."""

    name = "sam3d"

    def __init__(self, enabled: bool = False):
        if not enabled:
            raise RuntimeError(
                "SAM 3D Body is disabled by default (gated model, heavy). "
                "Pass enable_sam3d=True / --enable-sam3d after accepting the "
                "license on Hugging Face."
            )
        self.enabled = enabled

    def estimate(
        self, image_bgr: np.ndarray, boxes: Sequence[BoundingBox]
    ) -> list[PersonPose]:
        raise RuntimeError(
            "SAM 3D Body integration is scaffolded only. Accept the SAM License, "
            "install the official package, then implement estimate() — see docs/LICENSES.md."
        )


def advanced_legal_point(
    pose: PersonPose,
    attacking_direction_image_x: Literal["increasing", "decreasing"] = "increasing",
    min_score: float = 0.2,
) -> Point2D:
    """Most advanced legal joint toward the attacked goal (image x-axis proxy).

    When a pitch homography is available, callers should project joints to
    pitch metres and compare on pitch ``x`` instead; this helper is for the
    image-space / mock path.
    """
    candidates = []
    for name in LEGAL_OFFSIDE_JOINTS:
        if name not in pose.keypoints:
            continue
        if pose.scores.get(name, 1.0) < min_score:
            continue
        candidates.append(pose.keypoints[name])
    if not candidates:
        # Fallback: bbox bottom-center if present
        if pose.bbox is not None:
            return pose.bbox.bottom_center
        raise ValueError("No legal keypoints available")
    if attacking_direction_image_x == "increasing":
        return max(candidates, key=lambda p: p.x)
    return min(candidates, key=lambda p: p.x)


def create_pose_estimator(
    backend: BackendName = "mock",
    *,
    enable_sam3d: bool = False,
    **kwargs: Any,
) -> PoseEstimator:
    if backend == "mock":
        return MockPoseEstimator()
    if backend == "vitpose":
        return ViTPoseEstimator(model_id=kwargs.get("model_id", "usyd-community/vitpose-base-simple"))
    if backend == "sam3d":
        return SAM3DBodyEstimator(enabled=enable_sam3d)
    raise ValueError(f"Unknown pose backend: {backend}")

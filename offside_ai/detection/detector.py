"""Swappable detectors: RF-DETR (default) or Ultralytics YOLO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from offside_ai.common import BoundingBox, DetectionResult

BackendName = Literal["rfdetr", "yolo", "mock", "broadcast"]


@dataclass
class DetectorBackend(ABC):
    name: str

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[BoundingBox]:
        ...


class MockDetector(DetectorBackend):
    """Deterministic synthetic detections for tests / offline demos."""

    def __init__(self):
        super().__init__(name="mock")

    def detect(self, image_bgr: np.ndarray) -> list[BoundingBox]:
        h, w = image_bgr.shape[:2]
        # Two teams + ball — purely geometric placeholders
        boxes = [
            BoundingBox(w * 0.20, h * 0.40, w * 0.28, h * 0.70, 0.9, "player", team_id=0),
            BoundingBox(w * 0.35, h * 0.42, w * 0.43, h * 0.72, 0.9, "player", team_id=0),
            BoundingBox(w * 0.55, h * 0.40, w * 0.63, h * 0.70, 0.9, "player", team_id=1),
            BoundingBox(w * 0.70, h * 0.38, w * 0.78, h * 0.68, 0.9, "goalkeeper", team_id=1),
            BoundingBox(w * 0.08, h * 0.36, w * 0.14, h * 0.66, 0.85, "referee"),
            BoundingBox(w * 0.48, h * 0.55, w * 0.52, h * 0.60, 0.8, "ball"),
        ]
        return boxes


class RFDETRDetector(DetectorBackend):
    """Roboflow RF-DETR (Apache-2.0). Requires optional ``rfdetr`` + weights."""

    def __init__(self, model_size: str = "nano", weights: str | None = None):
        super().__init__(name=f"rfdetr-{model_size}")
        self.model_size = model_size
        self.weights = weights
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import rfdetr  # type: ignore
        except ImportError as e:
            raise ImportError(
                "RF-DETR is not installed. On Colab: pip install rfdetr "
                "or use --detector yolo / --detector mock. "
                f"Original error: {e}"
            ) from e
        # API surface varies by package version; try common constructors.
        size = self.model_size.lower()
        ctor = None
        for name in (f"RFDETR{size.capitalize()}", "RFDETRBase", "RFDETR"):
            ctor = getattr(rfdetr, name, None)
            if ctor is not None:
                break
        if ctor is None:
            raise RuntimeError(
                "Could not find an RF-DETR constructor in the installed package. "
                "Check rfdetr docs / version, or switch to --detector yolo."
            )
        self._model = ctor() if self.weights is None else ctor(pretrain_weights=self.weights)

    def detect(self, image_bgr: np.ndarray) -> list[BoundingBox]:
        self._load()
        # Prefer a predict API if present; otherwise fail clearly.
        model = self._model
        if hasattr(model, "predict"):
            preds = model.predict(image_bgr, threshold=0.3)
        else:
            raise RuntimeError(
                "Installed RF-DETR build has no predict(); use --detector yolo."
            )
        return _parse_rfdetr_predictions(preds, image_bgr.shape)


class YOLODetector(DetectorBackend):
    """Ultralytics YOLO alternative (AGPL-3.0 — see docs/LICENSES.md)."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        super().__init__(name=f"yolo:{model_name}")
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. pip install ultralytics "
                f"or use --detector mock. Original error: {e}"
            ) from e
        self._model = YOLO(self.model_name)

    def detect(self, image_bgr: np.ndarray) -> list[BoundingBox]:
        self._load()
        results = self._model.predict(image_bgr, verbose=False)[0]
        boxes: list[BoundingBox] = []
        names = results.names or {}
        if results.boxes is None:
            return boxes
        for b in results.boxes:
            cls_id = int(b.cls.item())
            label = str(names.get(cls_id, cls_id)).lower()
            # Map COCO sports ball / person
            if label in {"sports ball", "ball"}:
                mapped = "ball"
            elif label in {"person"}:
                mapped = "player"
            else:
                mapped = label
            xyxy = b.xyxy[0].tolist()
            conf = float(b.conf.item())
            boxes.append(
                BoundingBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, mapped)
            )
        return boxes


def _parse_rfdetr_predictions(preds: Any, shape: tuple) -> list[BoundingBox]:
    """Best-effort parse of RF-DETR prediction objects across versions."""
    boxes: list[BoundingBox] = []
    # supervision.Detections-like
    if hasattr(preds, "xyxy"):
        xyxy = np.asarray(preds.xyxy)
        conf = np.asarray(getattr(preds, "confidence", np.ones(len(xyxy))))
        cls = getattr(preds, "class_id", None)
        class_names = getattr(preds, "data", {}).get("class_name", None) if hasattr(preds, "data") else None
        for i in range(len(xyxy)):
            label = "player"
            if class_names is not None:
                label = str(class_names[i]).lower()
            elif cls is not None:
                label = str(cls[i])
            if "ball" in label:
                label = "ball"
            elif "goal" in label:
                label = "goalkeeper"
            elif "ref" in label:
                label = "referee"
            else:
                label = "player"
            boxes.append(
                BoundingBox(
                    float(xyxy[i][0]),
                    float(xyxy[i][1]),
                    float(xyxy[i][2]),
                    float(xyxy[i][3]),
                    float(conf[i]),
                    label,
                )
            )
        return boxes
    if isinstance(preds, list):
        for p in preds:
            if isinstance(p, dict) and "bbox" in p:
                x1, y1, x2, y2 = p["bbox"]
                boxes.append(
                    BoundingBox(
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        float(p.get("confidence", 1.0)),
                        str(p.get("label", "player")).lower(),
                    )
                )
        return boxes
    raise RuntimeError(
        f"Unrecognized RF-DETR prediction type: {type(preds)}. Use --detector yolo."
    )


class BroadcastDetector(DetectorBackend):
    """People and the ball on a broadcast frame, via the OpenCV YOLOX model.

    The weights live under artifacts/ and are not part of the repo. Boxes are
    the detector output, not the fixed mock rectangles.
    """

    def __init__(self, weights: str | None = None):
        super().__init__(name="broadcast")
        self.weights = weights

    def detect(self, image_bgr: np.ndarray) -> list[BoundingBox]:
        height, width = image_bgr.shape[:2]
        boxes = _yolox_boxes(image_bgr, self.weights)
        people = [
            box
            for box in boxes
            if box.label == "player"
            and box.y2 > height * 0.36
            and (box.y2 - box.y1) >= height * 0.035
            and (box.x2 - box.x1) < width * 0.22
        ]
        people.sort(key=lambda box: box.confidence, reverse=True)
        people = people[:14]
        _mark_referee(image_bgr, people)
        ball = [box for box in boxes if box.label == "ball"]
        if ball:
            people.append(max(ball, key=lambda box: box.confidence))
        return people


_YOLOX_NET = None
_YOLOX_LOCK = __import__("threading").Lock()
_YOLOX_GRIDS: np.ndarray | None = None
_YOLOX_STRIDES: np.ndarray | None = None


def _yolox_path(weights: str | None) -> Path:
    if weights:
        return Path(weights)
    return Path(__file__).resolve().parents[2] / "artifacts" / "models" / "yolox.onnx"


def _yolox_net(weights: str | None):
    global _YOLOX_NET, _YOLOX_GRIDS, _YOLOX_STRIDES
    with _YOLOX_LOCK:
        if _YOLOX_NET is None:
            path = _yolox_path(weights)
            if not path.is_file():
                raise FileNotFoundError(
                    "Falta el detector de personas en artifacts/models/yolox.onnx"
                )
            _YOLOX_NET = cv2.dnn.readNet(str(path))
            grids = []
            strides = []
            for stride in (8, 16, 32):
                cells = 640 // stride
                xv, yv = np.meshgrid(np.arange(cells), np.arange(cells))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                strides.append(np.full((1, grid.shape[1], 1), stride))
            _YOLOX_GRIDS = np.concatenate(grids, 1)[0]
            _YOLOX_STRIDES = np.concatenate(strides, 1)[0]
        return _YOLOX_NET, _YOLOX_GRIDS, _YOLOX_STRIDES


def _yolox_boxes(image_bgr: np.ndarray, weights: str | None) -> list[BoundingBox]:
    net, grids, strides = _yolox_net(weights)
    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    ratio = min(640 / height, 640 / width)
    resized = cv2.resize(rgb, (int(width * ratio), int(height * ratio)))
    padded = np.full((640, 640, 3), 114, dtype=np.float32)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    blob = np.transpose(padded, (2, 0, 1))[np.newaxis]
    with _YOLOX_LOCK:
        net.setInput(blob)
        dets = net.forward()[0].copy()
    dets[:, :2] = (dets[:, :2] + grids) * strides
    dets[:, 2:4] = np.exp(dets[:, 2:4]) * strides
    xywh = np.empty_like(dets[:, :4])
    xywh[:, 0] = dets[:, 0] - dets[:, 2] / 2
    xywh[:, 1] = dets[:, 1] - dets[:, 3] / 2
    xywh[:, 2] = dets[:, 2]
    xywh[:, 3] = dets[:, 3]
    class_scores = dets[:, 4:5] * dets[:, 5:]
    # person = 0, sports ball = 32
    chosen = np.maximum(class_scores[:, 0], class_scores[:, 32])
    class_id = np.where(class_scores[:, 32] > class_scores[:, 0], 32, 0)
    person = class_id == 0
    ball = class_id == 32
    keep_mask = (person & (chosen >= 0.45)) | (ball & (chosen >= 0.25))
    if not np.any(keep_mask):
        return []
    xywh = xywh[keep_mask]
    chosen = chosen[keep_mask]
    class_id = class_id[keep_mask]
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), chosen.tolist(), 0.25, 0.5)
    if keep is None or len(keep) == 0:
        return []
    found: list[BoundingBox] = []
    for index in np.array(keep).reshape(-1):
        x, y, w, h = xywh[int(index)] / ratio
        x1 = float(np.clip(x, 0, width - 1))
        y1 = float(np.clip(y, 0, height - 1))
        x2 = float(np.clip(x + w, 0, width - 1))
        y2 = float(np.clip(y + h, 0, height - 1))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        label = "ball" if int(class_id[int(index)]) == 32 else "player"
        found.append(BoundingBox(x1, y1, x2, y2, float(chosen[int(index)]), label))
    return found


def _mark_referee(image_bgr: np.ndarray, people: list[BoundingBox]) -> None:
    """The referee is the person whose shirt is black and yellow, not a team kit."""
    scores = []
    for box in people:
        patch = box.crop(image_bgr)
        if patch.size == 0:
            scores.append(0.0)
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        yellow = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] >= 70)
        black = hsv[:, :, 2] <= 50
        scores.append(float((yellow | black).mean()))
    if not scores:
        return
    best = int(np.argmax(scores))
    second = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0.0
    if scores[best] >= 0.45 and scores[best] >= second + 0.12:
        people[best].label = "referee"
        people[best].team_id = None


def create_detector(
    backend: BackendName = "rfdetr",
    **kwargs: Any,
) -> DetectorBackend:
    if backend == "rfdetr":
        return RFDETRDetector(
            model_size=kwargs.get("model_size", "nano"),
            weights=kwargs.get("weights"),
        )
    if backend == "yolo":
        return YOLODetector(model_name=kwargs.get("model_name", "yolov8n.pt"))
    if backend == "mock":
        return MockDetector()
    if backend == "broadcast":
        return BroadcastDetector(weights=kwargs.get("weights"))
    raise ValueError(f"Unknown detector backend: {backend}")


def detect_frame(
    image_bgr: np.ndarray,
    detector: DetectorBackend | None = None,
    backend: BackendName = "mock",
    frame_index: int = 0,
    fps: float = 25.0,
) -> DetectionResult:
    det = detector or create_detector(backend)
    h, w = image_bgr.shape[:2]
    boxes = det.detect(image_bgr)
    return DetectionResult(
        boxes=boxes,
        frame_index=frame_index,
        fps=fps,
        image_size=(w, h),
        backend=det.name,
    )

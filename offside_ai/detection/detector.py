"""Swappable detectors: RF-DETR (default) or Ultralytics YOLO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from offside_ai.common import BoundingBox, DetectionResult

BackendName = Literal["rfdetr", "yolo", "mock"]


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

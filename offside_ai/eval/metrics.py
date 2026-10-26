"""Metrics: pass-frame error, decision accuracy/P/R, margin error, error sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .schema import ANNOTATION_COLUMNS, DECISION_NEGATIVE, DECISION_POSITIVE


@dataclass
class PassFrameMetrics:
    n: int
    mean_abs_error_frames: float
    mean_abs_error_ms: float
    median_abs_error_frames: float
    median_abs_error_ms: float
    within_2_frames: float
    within_100_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionMetrics:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    mean_abs_margin_error_m: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorSourceRow:
    clip_id: str
    error_type: str
    detail: str


@dataclass
class EvalReport:
    pass_frame: PassFrameMetrics
    decision: DecisionMetrics
    error_sources: list[ErrorSourceRow] = field(default_factory=list)
    per_clip: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_frame": self.pass_frame.to_dict(),
            "decision": self.decision.to_dict(),
            "error_sources": [asdict(e) for e in self.error_sources],
            "per_clip": self.per_clip,
        }


def load_annotations_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ANNOTATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Annotation CSV missing columns: {missing}")
    return df


def _norm_decision(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"offside", "off", "1", "true", "yes"}:
        return DECISION_POSITIVE
    if v in {"onside", "on", "0", "false", "no"}:
        return DECISION_NEGATIVE
    raise ValueError(f"Unknown decision label: {value}")


def compute_pass_frame_metrics(
    gt_frames: Iterable[int],
    pred_frames: Iterable[int],
    fps_list: Iterable[float],
) -> PassFrameMetrics:
    gt = np.asarray(list(gt_frames), dtype=np.float64)
    pred = np.asarray(list(pred_frames), dtype=np.float64)
    fps = np.asarray(list(fps_list), dtype=np.float64)
    if len(gt) == 0:
        return PassFrameMetrics(0, 0, 0, 0, 0, 0, 0)
    err_f = np.abs(pred - gt)
    err_ms = err_f / np.maximum(fps, 1e-6) * 1000.0
    return PassFrameMetrics(
        n=int(len(gt)),
        mean_abs_error_frames=float(err_f.mean()),
        mean_abs_error_ms=float(err_ms.mean()),
        median_abs_error_frames=float(np.median(err_f)),
        median_abs_error_ms=float(np.median(err_ms)),
        within_2_frames=float(np.mean(err_f <= 2)),
        within_100_ms=float(np.mean(err_ms <= 100.0)),
    )


def compute_decision_metrics(
    gt_labels: Iterable[str],
    pred_labels: Iterable[str],
    gt_margins: Iterable[float | None] | None = None,
    pred_margins: Iterable[float | None] | None = None,
) -> DecisionMetrics:
    gt = [_norm_decision(x) for x in gt_labels]
    pred = [_norm_decision(x) for x in pred_labels]
    tp = fp = tn = fn = 0
    for g, p in zip(gt, pred):
        if g == DECISION_POSITIVE and p == DECISION_POSITIVE:
            tp += 1
        elif g == DECISION_NEGATIVE and p == DECISION_POSITIVE:
            fp += 1
        elif g == DECISION_NEGATIVE and p == DECISION_NEGATIVE:
            tn += 1
        else:
            fn += 1
    n = len(gt)
    acc = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    margin_err = None
    if gt_margins is not None and pred_margins is not None:
        errs = []
        for g, p in zip(gt_margins, pred_margins):
            if g is None or p is None or (isinstance(g, float) and np.isnan(g)):
                continue
            try:
                if pd.isna(g) or pd.isna(p):
                    continue
            except Exception:
                pass
            errs.append(abs(float(p) - float(g)))
        if errs:
            margin_err = float(np.mean(errs))

    return DecisionMetrics(
        n=n,
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        mean_abs_margin_error_m=margin_err,
    )


def classify_error_sources(
    clip_id: str,
    gt_frame: int,
    pred_frame: int | None,
    gt_decision: str,
    pred_decision: str | None,
    fps: float,
    audio_confidence: float | None = None,
    notes: str = "",
) -> list[ErrorSourceRow]:
    """Heuristic error taxonomy for the paper's qualitative table."""
    rows: list[ErrorSourceRow] = []
    gt_d = _norm_decision(gt_decision)
    if pred_frame is None:
        rows.append(ErrorSourceRow(clip_id, "pass_detection_miss", "no predicted pass frame"))
    else:
        err_ms = abs(pred_frame - gt_frame) / max(fps, 1e-6) * 1000.0
        if err_ms > 100:
            rows.append(
                ErrorSourceRow(
                    clip_id,
                    "pass_timing_error",
                    f"|Δ|={err_ms:.1f}ms ({abs(pred_frame - gt_frame)} frames)",
                )
            )
            if audio_confidence is not None and audio_confidence < 0.45:
                rows.append(
                    ErrorSourceRow(
                        clip_id,
                        "audio_low_confidence",
                        f"conf={audio_confidence:.2f} (crowd / weak impact?)",
                    )
                )
    if pred_decision is None:
        rows.append(ErrorSourceRow(clip_id, "decision_missing", "no predicted decision"))
    else:
        pred_d = _norm_decision(pred_decision)
        if pred_d != gt_d:
            rows.append(
                ErrorSourceRow(
                    clip_id,
                    "decision_mismatch",
                    f"gt={gt_d} pred={pred_d}",
                )
            )
            # Suggest likely geometric/source causes for paper discussion
            rows.append(
                ErrorSourceRow(
                    clip_id,
                    "possible_cause",
                    "homography / pose joint / team assignment / pass-frame skew",
                )
            )
    if notes:
        rows.append(ErrorSourceRow(clip_id, "annotator_note", notes))
    return rows


PredPassFn = Callable[[dict[str, Any]], dict[str, Any]]
"""Callable receiving a row dict; returns pred_pass_frame, optional confidence, etc."""


def evaluate_annotations(
    df: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    *,
    pred_pass_col: str = "pred_pass_frame",
    pred_decision_col: str = "pred_decision",
    pred_margin_col: str = "pred_margin_m",
    audio_conf_col: str = "audio_confidence",
) -> EvalReport:
    """Evaluate a joined annotations+predictions table.

    If ``predictions`` is provided, merge on ``clip_id``.
    """
    data = df.copy()
    if predictions is not None:
        data = data.merge(predictions, on="clip_id", how="left", suffixes=("", "_pred"))

    required_pred = [pred_pass_col, pred_decision_col]
    for c in required_pred:
        if c not in data.columns:
            raise ValueError(f"Missing prediction column: {c}")

    pass_metrics = compute_pass_frame_metrics(
        data["gt_pass_frame"].astype(int),
        data[pred_pass_col].astype(int),
        data["fps"].astype(float),
    )

    gt_margins = []
    pred_margins = []
    for _, row in data.iterrows():
        g = row.get("gt_margin_m", np.nan)
        p = row.get(pred_margin_col, np.nan)
        gt_margins.append(None if pd.isna(g) else float(g))
        pred_margins.append(None if pd.isna(p) else float(p))

    decision_metrics = compute_decision_metrics(
        data["gt_decision"],
        data[pred_decision_col],
        gt_margins,
        pred_margins,
    )

    errors: list[ErrorSourceRow] = []
    per_clip = []
    for _, row in data.iterrows():
        conf = row.get(audio_conf_col, None)
        if conf is not None and not pd.isna(conf):
            conf_f = float(conf)
        else:
            conf_f = None
        errs = classify_error_sources(
            str(row["clip_id"]),
            int(row["gt_pass_frame"]),
            int(row[pred_pass_col]) if not pd.isna(row[pred_pass_col]) else None,
            str(row["gt_decision"]),
            str(row[pred_decision_col]) if not pd.isna(row[pred_decision_col]) else None,
            float(row["fps"]),
            audio_confidence=conf_f,
            notes=str(row.get("notes", "") or ""),
        )
        errors.extend(errs)
        per_clip.append(
            {
                "clip_id": row["clip_id"],
                "gt_pass_frame": int(row["gt_pass_frame"]),
                "pred_pass_frame": int(row[pred_pass_col]),
                "frame_error": int(row[pred_pass_col]) - int(row["gt_pass_frame"]),
                "gt_decision": _norm_decision(row["gt_decision"]),
                "pred_decision": _norm_decision(row[pred_decision_col]),
            }
        )

    return EvalReport(
        pass_frame=pass_metrics,
        decision=decision_metrics,
        error_sources=errors,
        per_clip=per_clip,
    )

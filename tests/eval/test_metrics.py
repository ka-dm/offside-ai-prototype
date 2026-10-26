"""Tests for evaluation metrics and synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from offside_ai.eval.metrics import (
    compute_decision_metrics,
    compute_pass_frame_metrics,
    evaluate_annotations,
)
from offside_ai.eval.synthetic import build_synthetic_eval_dir


def test_pass_frame_metrics_units():
    # 2 frames error at 25 fps = 80 ms
    m = compute_pass_frame_metrics([10, 20], [12, 20], [25.0, 25.0])
    assert m.n == 2
    assert abs(m.mean_abs_error_frames - 1.0) < 1e-9
    assert abs(m.mean_abs_error_ms - 40.0) < 1e-6
    assert m.within_2_frames == 1.0


def test_decision_metrics_prf():
    m = compute_decision_metrics(
        ["offside", "offside", "onside", "onside"],
        ["offside", "onside", "onside", "offside"],
    )
    assert m.tp == 1 and m.fn == 1 and m.tn == 1 and m.fp == 1
    assert abs(m.precision - 0.5) < 1e-9
    assert abs(m.recall - 0.5) < 1e-9


def test_synthetic_eval_end_to_end(tmp_path: Path):
    root = build_synthetic_eval_dir(tmp_path / "syn")
    assert (root / "annotations.csv").exists()
    assert (root / "predictions.csv").exists()
    report = evaluate_annotations(
        pd.read_csv(root / "annotations.csv"),
        pd.read_csv(root / "predictions.csv"),
    )
    assert report.pass_frame.n == 3
    # Clean synthetic kicks should be within a few frames on average
    assert report.pass_frame.mean_abs_error_frames < 5
    assert report.decision.n == 3
    assert report.decision.accuracy == 1.0

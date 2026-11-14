"""Homography round-trip tests."""

from __future__ import annotations

import numpy as np

from offside_ai.pitch.homography import ManualPitchCalibrator, PitchHomography


def test_homography_roundtrip():
    # Fake camera: simple affine-ish mapping
    pitch = [(0, 0), (105, 0), (105, 68), (0, 68)]
    image = [(50, 400), (600, 410), (580, 80), (70, 90)]
    H = PitchHomography.from_correspondences(image, pitch)
    for (u, v), (x, y) in zip(image, pitch):
        x2, y2 = H.image_to_pitch(u, v)
        assert abs(x2 - x) < 1.5
        assert abs(y2 - y) < 1.5
        u2, v2 = H.pitch_to_image(x, y)
        assert abs(u2 - u) < 2.0


def test_manual_named_points():
    cal = ManualPitchCalibrator()
    named = {
        "corner_bl": (10.0, 200.0),
        "corner_br": (300.0, 200.0),
        "corner_tr": (290.0, 20.0),
        "corner_tl": (20.0, 25.0),
    }
    H = cal.calibrate_named(named)
    x, y = H.image_to_pitch(10.0, 200.0)
    assert abs(x) < 2 and abs(y) < 2

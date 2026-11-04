"""Rolling live ear hears a kick without loading a whole match."""

from __future__ import annotations

import numpy as np

from offside_ai.app.live import RollingEar


def test_rolling_ear_emits_one_kick():
    sr = 22050
    duration = 2.4
    kick_t = 1.2
    rng = np.random.default_rng(1)
    y = 0.01 * rng.standard_normal(int(sr * duration))
    burst_len = int(0.025 * sr)
    i0 = int(kick_t * sr)
    tb = np.arange(burst_len) / sr
    y[i0 : i0 + burst_len] += 0.9 * np.sin(2 * np.pi * 1200 * tb) * np.exp(-tb * 80)

    ear = RollingEar(sr=sr, window_s=4.0, hop_s=0.5)
    found = []
    hop = int(0.5 * sr)
    for start in range(0, len(y), hop):
        found.extend(ear.push(y[start : start + hop]))
    assert found, "expected an audio event"
    assert abs(found[0]["time_s"] - kick_t) < 0.35
    assert all(found[0]["kind"] == "audio" for _ in found[:1])
    assert len(found) == 1

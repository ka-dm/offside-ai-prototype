"""Team assignment clustering smoke tests."""

from __future__ import annotations

import numpy as np

from offside_ai.common import BoundingBox, DetectionResult
from offside_ai.detection.detector import MockDetector, detect_frame
from offside_ai.detection.teams import assign_teams_by_jersey_color


def test_mock_detector_and_team_override():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Paint two jersey colours
    img[190:336, 128:179] = (0, 0, 200)  # red-ish BGR
    img[190:336, 224:275] = (0, 0, 200)
    img[190:336, 352:403] = (200, 0, 0)  # blue-ish
    img[182:326, 448:499] = (200, 0, 0)

    result = detect_frame(img, detector=MockDetector())
    assignment = assign_teams_by_jersey_color(img, result, manual_overrides={0: 1})
    assignment.apply_to_result(result)
    assert result.boxes[0].team_id == 1
    assert any(b.team_id is not None for b in result.players())

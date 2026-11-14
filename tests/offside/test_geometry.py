"""Unit tests for offside geometry (synthetic pitch metres)."""

from __future__ import annotations

import numpy as np

from offside_ai.offside.annotate import annotate_offside_frame
from offside_ai.offside.geometry import (
    AttackingDirection,
    PlayerPoseOnPitch,
    compute_offside,
)


def _scene_offside_positive_x():
    # Defenders: GK at 40, others at 58 and 60 → second-last = 58
    # Attacker at 59.2 → offside by 1.2 m
    return [
        PlayerPoseOnPitch(1, 1, 40.0, role="goalkeeper"),
        PlayerPoseOnPitch(2, 1, 60.0),
        PlayerPoseOnPitch(3, 1, 58.0),
        PlayerPoseOnPitch(10, 0, 59.2),
        PlayerPoseOnPitch(11, 0, 50.0),
    ]


def test_offside_positive_x():
    players = _scene_offside_positive_x()
    d = compute_offside(players, attacking_team_id=0, attacking_direction="positive_x")
    assert d.is_offside
    assert d.verdict == "OFFSIDE"
    assert d.offside_line_x == 58.0
    assert d.second_last_defender_id == 3
    assert 10 in d.offside_attacker_ids
    assert 11 not in d.offside_attacker_ids
    assert abs(d.margins_m[10] - 1.2) < 1e-9


def test_onside_positive_x():
    players = _scene_offside_positive_x()
    players = [p for p in players if p.player_id != 10]
    players.append(PlayerPoseOnPitch(10, 0, 57.5))
    d = compute_offside(players, 0, AttackingDirection.POSITIVE_X)
    assert not d.is_offside
    assert d.margins_m[10] < 0


def test_offside_negative_x():
    # Attacking toward decreasing x: most advanced defender = smallest x (45),
    # second-last = next (50). Attacker nearer the goal than second-last ⇒ offside.
    players = [
        PlayerPoseOnPitch(1, 1, 70.0, role="goalkeeper"),
        PlayerPoseOnPitch(2, 1, 45.0),
        PlayerPoseOnPitch(3, 1, 50.0),
        PlayerPoseOnPitch(10, 0, 44.0),
    ]
    d = compute_offside(players, 0, "negative_x")
    assert d.is_offside
    assert d.offside_line_x == 50.0
    assert d.second_last_defender_id == 3
    assert d.margins_m[10] > 0


def test_level_within_epsilon_is_onside():
    players = [
        PlayerPoseOnPitch(1, 1, 40.0, role="goalkeeper"),
        PlayerPoseOnPitch(2, 1, 60.0),
        PlayerPoseOnPitch(3, 1, 58.0),
        PlayerPoseOnPitch(10, 0, 58.005),
    ]
    d = compute_offside(players, 0, "positive_x", margin_epsilon_m=0.01)
    assert not d.is_offside


def test_ball_moves_line_when_more_advanced():
    players = [
        PlayerPoseOnPitch(1, 1, 40.0, role="goalkeeper"),
        PlayerPoseOnPitch(2, 1, 60.0),
        PlayerPoseOnPitch(3, 1, 58.0),
        PlayerPoseOnPitch(10, 0, 59.0),
    ]
    # Ball at 59.5 → line becomes 59.5; attacker at 59.0 is onside
    d = compute_offside(players, 0, "positive_x", ball_x=59.5)
    assert d.details["line_source"] == "ball"
    assert abs(d.offside_line_x - 59.5) < 1e-9
    assert not d.is_offside


def test_annotate_draws_without_homography():
    players = _scene_offside_positive_x()
    d = compute_offside(players, 0, "positive_x")
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    pts = {p.player_id: (50.0 + 20 * p.player_id, 120.0) for p in players}
    out = annotate_offside_frame(img, d, players, player_image_points=pts)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_direction_aliases():
    assert AttackingDirection.from_string("right") is AttackingDirection.POSITIVE_X
    assert AttackingDirection.from_string("left") is AttackingDirection.NEGATIVE_X

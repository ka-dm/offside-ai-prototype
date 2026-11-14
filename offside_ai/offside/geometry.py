"""IFAB-inspired offside geometry in pitch metres.

Simplifications for the research prototype (documented in README/docs):
- Single-camera 2D projection via homography (no multi-view triangulation).
- "Most advanced legal body point" is supplied by the pose stage (excl. arms).
- Goalkeeper is identified by role/label when available; otherwise the deepest
  defending player toward the defended goal is treated as a defender.
- The offside line is the second-last defender's advanced point (or the ball
  if nearer to the goal line — not fully modelled here; optional ball_x).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Sequence


class AttackingDirection(str, Enum):
    """Attacking team plays toward increasing (+x) or decreasing (-x) pitch metres."""

    POSITIVE_X = "positive_x"
    NEGATIVE_X = "negative_x"

    @classmethod
    def from_string(cls, value: str) -> "AttackingDirection":
        v = value.strip().lower().replace("-", "_")
        aliases = {
            "positive_x": cls.POSITIVE_X,
            "pos": cls.POSITIVE_X,
            "+x": cls.POSITIVE_X,
            "right": cls.POSITIVE_X,
            "negative_x": cls.NEGATIVE_X,
            "neg": cls.NEGATIVE_X,
            "-x": cls.NEGATIVE_X,
            "left": cls.NEGATIVE_X,
        }
        if v not in aliases:
            raise ValueError(f"Unknown attacking direction: {value}")
        return aliases[v]


@dataclass
class PlayerPoseOnPitch:
    """One player at the pass instant, positions in pitch metres."""

    player_id: int
    team_id: int  # attacking team id vs defending
    x: float  # advanced legal body point along length of pitch
    y: float = 0.0
    role: Literal["player", "goalkeeper", "referee"] = "player"
    label: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class OffsideDecision:
    is_offside: bool
    offside_line_x: float
    second_last_defender_id: int | None
    offside_attacker_ids: list[int]
    margins_m: dict[int, float]  # attacker_id -> signed margin (positive = offside)
    attacking_direction: str
    attacking_team_id: int
    defending_team_id: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def verdict(self) -> str:
        return "OFFSIDE" if self.is_offside else "ONSIDE"


def _advanced_key(direction: AttackingDirection):
    """Sort key: larger = more advanced toward the attacked goal."""
    if direction is AttackingDirection.POSITIVE_X:
        return lambda p: p.x
    return lambda p: -p.x


def _signed_margin(
    attacker_x: float, line_x: float, direction: AttackingDirection
) -> float:
    """Positive margin means the attacker is beyond the offside line (offside)."""
    if direction is AttackingDirection.POSITIVE_X:
        return attacker_x - line_x
    return line_x - attacker_x


def compute_offside(
    players: Sequence[PlayerPoseOnPitch],
    attacking_team_id: int,
    attacking_direction: AttackingDirection | str,
    *,
    ball_x: float | None = None,
    margin_epsilon_m: float = 0.01,
    include_keepers_as_defenders: bool = True,
) -> OffsideDecision:
    """Compute offside line and verdict at the pass instant.

    Parameters
    ----------
    players:
        Defenders and attackers with pitch-metre ``x`` of their most advanced
        *legal* body point (excluding arms/hands).
    attacking_team_id:
        Team id of the attacking side.
    attacking_direction:
        Toward which goal the attackers play.
    ball_x:
        Optional ball x; IFAB also considers the ball position. When provided,
        the offside line is the more advanced of (second-last defender, ball)
        from the defending team's perspective (i.e. closer to the halfway /
        less advanced toward the attacked goal wins for the attacker...).
        Actually: a player is offline if nearer the opponent's goal line than
        both the ball and the second-last opponent. So the relevant line is
        min(second_last, ball) in the "advanced" sense — the less advanced of
        the two thresholds? No: nearer to goal line than BOTH means beyond the
        *more advanced* of (ball, second-last). Wait:

        IFAB: "nearer to the opponents' goal line than both the ball and the
        second-last opponent". So attacker is offside if advanced past BOTH.
        That means the offside threshold is the *less advanced* of the two
        (the one closer to halfway): attacker must be beyond the max of the
        two "distance-from-goal" constraints... 

        If attacking +x: nearer goal line => larger x.
        Offside if attacker_x > ball_x AND attacker_x > second_last_x
        <=> attacker_x > max(ball_x, second_last_x).

        So the offside line is max(ball, second_last) when attacking +x
        (the more advanced of the two). We'll use that.
    margin_epsilon_m:
        Tolerances below this (metres) are treated as level / onside.
    """
    if isinstance(attacking_direction, str):
        attacking_direction = AttackingDirection.from_string(attacking_direction)

    if not players:
        raise ValueError("players must be non-empty")

    team_ids = {p.team_id for p in players if p.role != "referee"}
    if attacking_team_id not in team_ids and len(team_ids) >= 1:
        # Still allow; defending team inferred from others
        pass

    defenders = [
        p
        for p in players
        if p.team_id != attacking_team_id and p.role != "referee"
    ]
    if not include_keepers_as_defenders:
        defenders = [p for p in defenders if p.role != "goalkeeper"]

    attackers = [
        p
        for p in players
        if p.team_id == attacking_team_id and p.role != "referee"
    ]

    defending_team_ids = {p.team_id for p in defenders}
    defending_team_id = next(iter(defending_team_ids), -1)

    if len(defenders) < 1:
        # No defenders: every attacker beyond halfway would be offside in law,
        # but for the prototype treat as no line / not offside with a flag.
        return OffsideDecision(
            is_offside=False,
            offside_line_x=0.0,
            second_last_defender_id=None,
            offside_attacker_ids=[],
            margins_m={},
            attacking_direction=attacking_direction.value,
            attacking_team_id=attacking_team_id,
            defending_team_id=defending_team_id,
            details={"warning": "no_defenders"},
        )

    key = _advanced_key(attacking_direction)
    ranked = sorted(defenders, key=key, reverse=True)
    # Last defender = most advanced; second-last = next
    last = ranked[0]
    second_last = ranked[1] if len(ranked) > 1 else ranked[0]

    line_x = float(second_last.x)
    line_source = "second_last_defender"
    if ball_x is not None:
        # Offside line = more advanced of (second-last, ball)
        if attacking_direction is AttackingDirection.POSITIVE_X:
            if float(ball_x) > line_x:
                line_x = float(ball_x)
                line_source = "ball"
        else:
            if float(ball_x) < line_x:
                line_x = float(ball_x)
                line_source = "ball"

    margins: dict[int, float] = {}
    offside_ids: list[int] = []
    for a in attackers:
        m = _signed_margin(a.x, line_x, attacking_direction)
        margins[a.player_id] = float(m)
        if m > margin_epsilon_m:
            offside_ids.append(a.player_id)

    return OffsideDecision(
        is_offside=len(offside_ids) > 0,
        offside_line_x=line_x,
        second_last_defender_id=second_last.player_id,
        offside_attacker_ids=offside_ids,
        margins_m=margins,
        attacking_direction=attacking_direction.value,
        attacking_team_id=attacking_team_id,
        defending_team_id=defending_team_id,
        details={
            "last_defender_id": last.player_id,
            "last_defender_x": last.x,
            "second_last_x": second_last.x,
            "line_source": line_source,
            "n_defenders": len(defenders),
            "n_attackers": len(attackers),
            "margin_epsilon_m": margin_epsilon_m,
            "ball_x": ball_x,
        },
    )

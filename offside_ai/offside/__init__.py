"""Offside line geometry, verdict, and frame annotation."""

from .geometry import (
    AttackingDirection,
    OffsideDecision,
    PlayerPoseOnPitch,
    compute_offside,
)
from .annotate import annotate_offside_frame

__all__ = [
    "AttackingDirection",
    "OffsideDecision",
    "PlayerPoseOnPitch",
    "compute_offside",
    "annotate_offside_frame",
]

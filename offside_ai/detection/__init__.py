"""Player / ball detection and jersey-colour team assignment."""

from .detector import DetectorBackend, create_detector, detect_frame
from .teams import TeamAssignment, assign_teams_by_jersey_color

__all__ = [
    "DetectorBackend",
    "create_detector",
    "detect_frame",
    "TeamAssignment",
    "assign_teams_by_jersey_color",
]

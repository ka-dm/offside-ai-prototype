"""CSV schema for annotated plays (paper dataset)."""

from __future__ import annotations

# Columns expected in eval annotation CSVs. Do NOT commit copyrighted footage;
# store clips locally and reference paths / public URLs with credits.
ANNOTATION_COLUMNS = [
    "clip_id",
    "clip_path",
    "fps",
    "gt_pass_frame",
    "gt_decision",  # offside | onside
    "gt_margin_m",  # optional; empty if unknown
    "attacking_team_id",
    "attacking_direction",  # positive_x | negative_x
    "source",
    "credits",
    "notes",
]

DECISION_POSITIVE = "offside"
DECISION_NEGATIVE = "onside"

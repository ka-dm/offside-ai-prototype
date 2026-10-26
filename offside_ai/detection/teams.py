"""Jersey-colour clustering for automatic team assignment."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans

from offside_ai.common import BoundingBox, DetectionResult, mean_color_bgr


@dataclass
class TeamAssignment:
    team_of_box_index: list[int | None]
    team_centroids_bgr: list[np.ndarray]
    manual_overrides: dict[int, int]

    def apply_to_result(self, result: DetectionResult) -> DetectionResult:
        for i, box in enumerate(result.boxes):
            if box.label in {"player", "goalkeeper"}:
                tid = self.team_of_box_index[i]
                if tid is not None:
                    box.team_id = int(tid)
        return result


def _jersey_patch(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Upper-torso crop approximate jersey region."""
    h = box.y2 - box.y1
    w = box.x2 - box.x1
    # Focus on upper 55% excluding head-ish top 15%
    y1 = box.y1 + 0.15 * h
    y2 = box.y1 + 0.55 * h
    x1 = box.x1 + 0.2 * w
    x2 = box.x2 - 0.2 * w
    tmp = BoundingBox(x1, y1, x2, y2)
    return tmp.crop(image)


def assign_teams_by_jersey_color(
    image_bgr: np.ndarray,
    result: DetectionResult,
    n_teams: int = 2,
    manual_overrides: dict[int, int] | None = None,
    random_state: int = 0,
) -> TeamAssignment:
    """Cluster player jersey colours into ``n_teams`` (default 2).

    ``manual_overrides`` maps detection box index -> team_id and is applied
    after clustering (and those points are still used as soft seeds when enough
    overrides exist).
    """
    manual_overrides = manual_overrides or {}
    indices = [
        i
        for i, b in enumerate(result.boxes)
        if b.label in {"player", "goalkeeper"}
    ]
    team_of = [None] * len(result.boxes)
    centroids: list[np.ndarray] = []

    if not indices:
        return TeamAssignment(team_of, centroids, manual_overrides)

    features = []
    for i in indices:
        patch = _jersey_patch(image_bgr, result.boxes[i])
        if patch.size == 0:
            features.append(np.zeros(3))
            continue
        # Use HSV hue+sat+val mean for lighting robustness
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        features.append(hsv.reshape(-1, 3).mean(axis=0))
    X = np.asarray(features, dtype=np.float64)

    k = min(n_teams, len(indices))
    if k <= 1:
        for i in indices:
            team_of[i] = 0
        centroids = [mean_color_bgr(_jersey_patch(image_bgr, result.boxes[indices[0]]))]
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        for idx_in_feat, box_i in enumerate(indices):
            team_of[box_i] = int(labels[idx_in_feat])
        # Centroids back to approx BGR via mean patch of members
        centroids = []
        for t in range(k):
            members = [indices[j] for j, lab in enumerate(labels) if lab == t]
            if not members:
                centroids.append(np.zeros(3))
            else:
                patches = [_jersey_patch(image_bgr, result.boxes[m]) for m in members]
                patches = [p for p in patches if p.size]
                if patches:
                    centroids.append(
                        np.mean([mean_color_bgr(p) for p in patches], axis=0)
                    )
                else:
                    centroids.append(np.zeros(3))

    for box_i, team_id in manual_overrides.items():
        if 0 <= box_i < len(team_of):
            team_of[box_i] = int(team_id)

    return TeamAssignment(team_of, centroids, manual_overrides)

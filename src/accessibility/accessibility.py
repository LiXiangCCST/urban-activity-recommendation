"""Personalized accessibility scoring."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.profiles.individual_profiles import IndividualProfile


def travel_time_column(profile: IndividualProfile) -> str:
    return f"{profile.primary_mode()}_time"


def compute_accessibility(
    area_id: str,
    profile: IndividualProfile,
    resources: pd.DataFrame,
    travel_times: pd.DataFrame,
    theta_by_category: dict[str, float],
) -> dict[str, Any]:
    """Compute personalized accessibility for one area and profile."""

    area_travel = travel_times[travel_times["area_id"] == area_id]
    if "profile_id" in area_travel.columns:
        area_travel = area_travel[area_travel["profile_id"] == profile.profile_id]
    merged = area_travel.merge(resources, on=["resource_id", "category"])
    mode_col = travel_time_column(profile)
    by_category: dict[str, float] = {}
    total = 0.0

    for category, weight in profile.activity_weights.items():
        category_rows = merged[merged["category"] == category]
        theta = float(theta_by_category.get(category, 20.0)) / max(profile.time_sensitivity, 0.1)
        category_score = 0.0
        for _, row in category_rows.iterrows():
            category_score += float(row["quality"]) * math.exp(-float(row[mode_col]) / theta)
        weighted_score = float(weight) * category_score
        by_category[category] = weighted_score
        total += weighted_score

    return {"total": float(total), "by_category": by_category}


def compute_accessibility_for_all(
    profile: IndividualProfile,
    areas: pd.DataFrame,
    resources: pd.DataFrame,
    travel_times: pd.DataFrame,
    theta_by_category: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for area_id in areas["area_id"]:
        result = compute_accessibility(area_id, profile, resources, travel_times, theta_by_category)
        rows.append({"area_id": area_id, "accessibility": result["total"], **{f"access_{k}": v for k, v in result["by_category"].items()}})
    return pd.DataFrame(rows)

"""Cost-first baseline with accessibility tie-break."""

from __future__ import annotations

import pandas as pd

from src.accessibility.accessibility import compute_accessibility_for_all
from src.profiles.individual_profiles import IndividualProfile


def recommend(
    profile: IndividualProfile,
    areas: pd.DataFrame,
    resources: pd.DataFrame,
    travel_times: pd.DataFrame,
    theta_by_category: dict[str, float],
) -> dict:
    affordable = areas[areas["housing_cost"] <= profile.max_affordable_housing_cost].copy()
    if affordable.empty:
        affordable = areas.nsmallest(5, "housing_cost").copy()

    access = compute_accessibility_for_all(profile, affordable, resources, travel_times, theta_by_category)
    ranked = affordable.merge(access[["area_id", "accessibility"]], on="area_id")
    ranked["cost_score"] = -ranked["housing_cost"] + ranked["accessibility"] * 120.0
    best = ranked.sort_values("cost_score", ascending=False).iloc[0]
    return {
        "method": "cost_first",
        "area_id": best["area_id"],
        "score": float(best["cost_score"]),
        "explanation": {
            "housing_cost": float(best["housing_cost"]),
            "affordable": bool(best["housing_cost"] <= profile.max_affordable_housing_cost),
            "accessibility": float(best["accessibility"]),
        },
    }

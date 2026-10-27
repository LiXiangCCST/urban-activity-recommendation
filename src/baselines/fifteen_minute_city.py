"""Fifteen-minute coverage baseline."""

from __future__ import annotations

import pandas as pd

from src.accessibility.accessibility import travel_time_column
from src.profiles.individual_profiles import IndividualProfile


def recommend(
    profile: IndividualProfile,
    areas: pd.DataFrame,
    resources: pd.DataFrame,
    travel_times: pd.DataFrame,
    threshold: float = 15.0,
) -> dict:
    mode_col = travel_time_column(profile)
    profile_times = travel_times
    if "profile_id" in profile_times.columns:
        profile_times = profile_times[profile_times["profile_id"] == profile.profile_id]
    reachable = profile_times[profile_times[mode_col] <= threshold].copy()
    reachable["weighted_coverage"] = reachable["category"].map(profile.activity_weights).fillna(0.0)
    scores = reachable.groupby("area_id")["weighted_coverage"].sum().reset_index()
    scores = areas[["area_id"]].merge(scores, on="area_id", how="left").fillna({"weighted_coverage": 0.0})
    best = scores.sort_values("weighted_coverage", ascending=False).iloc[0]
    category_count = reachable[reachable["area_id"] == best["area_id"]]["category"].nunique()
    return {
        "method": "fifteen_minute_city",
        "area_id": best["area_id"],
        "score": float(best["weighted_coverage"]),
        "explanation": {"threshold_minutes": threshold, "covered_categories": int(category_count)},
    }

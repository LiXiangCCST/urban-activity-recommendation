"""Nearest important-resource baseline."""

from __future__ import annotations

import pandas as pd

from src.accessibility.accessibility import travel_time_column
from src.profiles.individual_profiles import IndividualProfile


def recommend(profile: IndividualProfile, areas: pd.DataFrame, resources: pd.DataFrame, travel_times: pd.DataFrame) -> dict:
    mode_col = travel_time_column(profile)
    top_categories = sorted(profile.activity_weights.items(), key=lambda item: item[1], reverse=True)[:3]
    selected_categories = [category for category, _ in top_categories]
    profile_times = travel_times
    if "profile_id" in profile_times.columns:
        profile_times = profile_times[profile_times["profile_id"] == profile.profile_id]
    subset = profile_times[profile_times["category"].isin(selected_categories)]
    used_categories = selected_categories
    if subset.empty:
        subset = profile_times.copy()
        used_categories = sorted(subset["category"].dropna().unique().tolist())
    scores = subset.groupby("area_id")[mode_col].min().reset_index(name="nearest_time")
    if scores.empty:
        raise ValueError("No travel-time rows available for nearest-resource baseline.")
    best = scores.sort_values("nearest_time", ascending=True).iloc[0]
    return {
        "method": "nearest_resource",
        "area_id": best["area_id"],
        "score": -float(best["nearest_time"]),
        "explanation": {"nearest_time": float(best["nearest_time"]), "categories": used_categories},
    }

"""Evaluation metrics for recommendation outputs."""

from __future__ import annotations

import pandas as pd

from src.accessibility.accessibility import travel_time_column
from src.profiles.individual_profiles import IndividualProfile


def average_accessibility(results: pd.DataFrame) -> float:
    return float(results["accessibility"].mean()) if not results.empty else 0.0


def average_reward(results: pd.DataFrame) -> float:
    return float(results["reward"].mean()) if not results.empty else 0.0


def average_travel_time_proxy(profile: IndividualProfile, recommendations: pd.DataFrame, travel_times: pd.DataFrame) -> float:
    mode_col = travel_time_column(profile)
    selected_times = travel_times[travel_times["area_id"].isin(recommendations["area_id"])]
    if "profile_id" in selected_times.columns:
        selected_times = selected_times[selected_times["profile_id"] == profile.profile_id]
    return float(selected_times[mode_col].mean()) if not selected_times.empty else 0.0


def activity_coverage_within_15(profile: IndividualProfile, area_id: str, travel_times: pd.DataFrame, threshold: float = 15.0) -> float:
    mode_col = travel_time_column(profile)
    profile_times = travel_times
    if "profile_id" in profile_times.columns:
        profile_times = profile_times[profile_times["profile_id"] == profile.profile_id]
    reachable = profile_times[(profile_times["area_id"] == area_id) & (profile_times[mode_col] <= threshold)]
    if not profile.activity_weights:
        return 0.0
    covered_weight = sum(profile.activity_weights.get(category, 0.0) for category in reachable["category"].unique())
    return float(covered_weight / sum(profile.activity_weights.values()))


def recommendation_comparison_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["profile_id", "method"]).reset_index(drop=True)

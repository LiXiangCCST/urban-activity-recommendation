"""Reward function for personalized area recommendation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.accessibility.accessibility import compute_accessibility, travel_time_column
from src.profiles.individual_profiles import IndividualProfile


def _normalize_positive(value: float, upper: float) -> float:
    return max(0.0, min(1.0, value / upper))


def compute_reward(
    area_id: str,
    profile: IndividualProfile,
    areas: pd.DataFrame,
    resources: pd.DataFrame,
    travel_times: pd.DataFrame,
    theta_by_category: dict[str, float],
    reward_weights: dict[str, float],
) -> dict[str, Any]:
    area = areas.loc[areas["area_id"] == area_id].iloc[0]
    access = compute_accessibility(area_id, profile, resources, travel_times, theta_by_category)
    mode_col = travel_time_column(profile)
    area_times = travel_times[travel_times["area_id"] == area_id]
    if "profile_id" in area_times.columns:
        area_times = area_times[area_times["profile_id"] == profile.profile_id]

    activity_fit = min(1.0, access["total"] / 3.5)
    mobility = float(area["transit_accessibility"]) if not profile.has_car else min(1.0, float(area["transit_accessibility"]) + 0.2)
    affordability_margin = profile.max_affordable_housing_cost - float(area["housing_cost"])
    equity = 1.0 if profile.income_level.startswith("low") and affordability_margin >= 0 else max(0.0, affordability_margin / profile.max_affordable_housing_cost)
    cost = _normalize_positive(max(0.0, float(area["housing_cost"]) - profile.max_affordable_housing_cost), profile.max_affordable_housing_cost)
    commute_rows = area_times[area_times["category"] == "employment"]
    commute = float(commute_rows[mode_col].mean()) if not commute_rows.empty else float(area_times[mode_col].mean())
    commute_penalty = _normalize_positive(commute * profile.commute_weight, 60.0)
    risk_penalty = float(area["risk_score"]) * profile.risk_sensitivity

    reward = (
        reward_weights["alpha_access"] * access["total"]
        + reward_weights["alpha_activity_fit"] * activity_fit
        + reward_weights["alpha_mobility"] * mobility
        + reward_weights["alpha_equity"] * equity
        - reward_weights["beta_cost"] * cost
        - reward_weights["beta_commute"] * commute_penalty
        - reward_weights["beta_risk"] * risk_penalty
    )

    return {
        "area_id": area_id,
        "reward": float(reward),
        "accessibility": float(access["total"]),
        "activity_fit": float(activity_fit),
        "mobility": float(mobility),
        "equity": float(equity),
        "cost_penalty": float(cost),
        "commute_penalty": float(commute_penalty),
        "risk_penalty": float(risk_penalty),
        "accessibility_by_category": access["by_category"],
    }


def compute_rewards_for_all(
    profile: IndividualProfile,
    areas: pd.DataFrame,
    resources: pd.DataFrame,
    travel_times: pd.DataFrame,
    theta_by_category: dict[str, float],
    reward_weights: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            compute_reward(area_id, profile, areas, resources, travel_times, theta_by_category, reward_weights)
            for area_id in areas["area_id"]
        ]
    ).drop(columns=["accessibility_by_category"])

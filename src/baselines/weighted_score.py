"""Weighted reward-score baseline."""

from __future__ import annotations

import pandas as pd

from src.accessibility.reward import compute_rewards_for_all
from src.profiles.individual_profiles import IndividualProfile


def recommend(
    profile: IndividualProfile,
    areas: pd.DataFrame,
    resources: pd.DataFrame,
    travel_times: pd.DataFrame,
    theta_by_category: dict[str, float],
    reward_weights: dict[str, float],
) -> dict:
    scores = compute_rewards_for_all(profile, areas, resources, travel_times, theta_by_category, reward_weights)
    best = scores.sort_values("reward", ascending=False).iloc[0]
    return {
        "method": "weighted_score",
        "area_id": best["area_id"],
        "score": float(best["reward"]),
        "explanation": {
            "accessibility": float(best["accessibility"]),
            "mobility": float(best["mobility"]),
            "cost_penalty": float(best["cost_penalty"]),
            "risk_penalty": float(best["risk_penalty"]),
        },
    }

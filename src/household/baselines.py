"""Deterministic household recommendation baselines."""

from __future__ import annotations

import pandas as pd


def _best(df: pd.DataFrame, score_col: str, method: str, ascending: bool = False) -> dict:
    ranked = df.sort_values(score_col, ascending=ascending).iloc[0]
    return {
        "method": method,
        "household_id": ranked["household_id"],
        "area_id": ranked["area_id"],
        "score": float(ranked[score_col]),
        "reward": float(ranked["final_reward"]),
        "accessibility": float(ranked["accessibility_score"]),
        "mean_commute_time": float(ranked["mean_commute_time"]),
        "max_commute_time": float(ranked["max_commute_time"]),
        "housing_cost": float(ranked["housing_cost"]),
    }


def household_weighted_score(reward_surface: pd.DataFrame) -> dict:
    return _best(reward_surface, "final_reward", "household_weighted_score")


def commute_first(reward_surface: pd.DataFrame) -> dict:
    df = reward_surface.copy()
    df["commute_score"] = -(0.65 * df["mean_commute_time"] + 0.35 * df["max_commute_time"] + 12.0 * df["workers_over_tolerance_count"])
    return _best(df, "commute_score", "commute_first")


def affordability_first(reward_surface: pd.DataFrame) -> dict:
    df = reward_surface.copy()
    df["affordability_score"] = -(df["housing_cost_penalty"] + 0.4 * df["rent_income_ratio"]) + 0.08 * df["accessibility_score"]
    return _best(df, "affordability_score", "affordability_first")


def child_elderly_service_first(reward_surface: pd.DataFrame) -> dict:
    df = reward_surface.copy()
    df["care_service_score"] = df["child_accessibility"] + df["elderly_medical_accessibility"] + 0.4 * df["daily_service_accessibility"] - 0.3 * df["care_coordination_penalty"]
    return _best(df, "care_service_score", "child_elderly_service_first")


def balanced_household_score(reward_surface: pd.DataFrame) -> dict:
    df = reward_surface.copy()
    df["balanced_score"] = (
        0.45 * df["final_reward"]
        + 0.20 * df["accessibility_score"]
        - 0.15 * df["mean_commute_penalty"]
        - 0.10 * df["housing_cost_penalty"]
        - 0.10 * df["commute_inequality_penalty"]
    )
    return _best(df, "balanced_score", "balanced_household_score")


def run_household_baselines(reward_surface: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for household_id, df in reward_surface.groupby("household_id"):
        for fn in [household_weighted_score, commute_first, affordability_first, child_elderly_service_first, balanced_household_score]:
            rows.append(fn(df))
    return pd.DataFrame(rows)

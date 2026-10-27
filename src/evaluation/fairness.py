"""Basic fairness summaries across profile groups."""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_reward_gap(results: pd.DataFrame, group_col: str = "profile_id", reward_col: str = "reward") -> float:
    grouped = results.groupby(group_col)[reward_col].mean()
    return float(grouped.max() - grouped.min()) if not grouped.empty else 0.0


def minimum_group_reward(results: pd.DataFrame, group_col: str = "profile_id", reward_col: str = "reward") -> float:
    grouped = results.groupby(group_col)[reward_col].mean()
    return float(grouped.min()) if not grouped.empty else 0.0


def gini_coefficient(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    if np.min(array) < 0:
        array = array - np.min(array)
    if np.allclose(array.sum(), 0):
        return 0.0
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * array)) / (n * np.sum(array)) - (n + 1) / n)


def fairness_summary(results: pd.DataFrame) -> dict[str, float]:
    return {
        "group_reward_gap": group_reward_gap(results),
        "minimum_group_reward": minimum_group_reward(results),
        "gini_reward": gini_coefficient(results["reward"].to_numpy() if "reward" in results else []),
    }

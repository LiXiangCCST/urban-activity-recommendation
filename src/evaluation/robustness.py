"""Placeholder robustness checks."""

from __future__ import annotations

import pandas as pd


def recommendation_stability(base_results: pd.DataFrame, perturbed_results: pd.DataFrame) -> float:
    """Share of profile/method recommendations that remain unchanged."""

    if base_results.empty or perturbed_results.empty:
        return 0.0
    merged = base_results.merge(
        perturbed_results,
        on=["profile_id", "method"],
        suffixes=("_base", "_perturbed"),
    )
    if merged.empty:
        return 0.0
    return float((merged["area_id_base"] == merged["area_id_perturbed"]).mean())

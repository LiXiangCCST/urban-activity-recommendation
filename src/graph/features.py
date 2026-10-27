"""Area node feature engineering."""

from __future__ import annotations

import pandas as pd


AREA_FEATURE_COLUMNS = [
    "housing_cost_norm",
    "risk_score",
    "transit_accessibility",
    "employment_density",
    "resource_count_15_min",
]


def compute_area_features(areas: pd.DataFrame, travel_times: pd.DataFrame) -> pd.DataFrame:
    features = areas.copy()
    if "transit_accessibility" not in features.columns and "transit_score" in features.columns:
        features["transit_accessibility"] = features["transit_score"]
    if "employment_density" not in features.columns:
        features["employment_density"] = 0.5
    max_cost = max(float(features["housing_cost"].max()), 1.0)
    features["housing_cost_norm"] = features["housing_cost"] / max_cost
    time_col = "walk_time" if "walk_time" in travel_times.columns else "travel_time_minutes"
    coverage = (
        travel_times[travel_times[time_col] <= 15.0]
        .groupby("area_id")["resource_id"]
        .count()
        .rename("resource_count_15_min")
    )
    features = features.merge(coverage, on="area_id", how="left")
    features["resource_count_15_min"] = features["resource_count_15_min"].fillna(0)
    max_count = max(float(features["resource_count_15_min"].max()), 1.0)
    features["resource_count_15_min"] = features["resource_count_15_min"] / max_count
    return features[["area_id", *AREA_FEATURE_COLUMNS]]


def feature_vector(area_id: str, area_features: pd.DataFrame) -> list[float]:
    row = area_features.loc[area_features["area_id"] == area_id].iloc[0]
    return [float(row[col]) for col in AREA_FEATURE_COLUMNS]

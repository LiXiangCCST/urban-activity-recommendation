"""Walking travel-time and 15-minute walkability layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.travel.utils import ensure_dir, haversine_meters, load_standard_areas, load_standard_resources


def build_walking_travel_times(
    areas: pd.DataFrame,
    resources: pd.DataFrame,
    processed_dir: str | Path = "data/processed/nyc/real",
    detour_factor: float = 1.25,
    walking_speed_m_per_min: float = 80.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = ensure_dir(processed_dir)
    rows = []
    for _, area in areas.iterrows():
        for _, resource in resources.iterrows():
            distance = haversine_meters(area["lon"], area["lat"], resource["lon"], resource["lat"])
            minutes = distance * detour_factor / walking_speed_m_per_min
            rows.append(
                {
                    "area_id": area["area_id"],
                    "resource_id": resource["resource_id"],
                    "activity_type": resource["activity_type"],
                    "distance_meters": round(distance, 3),
                    "walking_time_minutes": round(minutes, 3),
                    "is_walkable_15min": bool(minutes <= 15.0),
                    "source": "WALK_HAVERSINE_DETOUR_APPROX",
                }
            )
    walking = pd.DataFrame(rows)
    walking.to_csv(output / "walking_travel_times.csv", index=False)

    summary = (
        walking.groupby(["area_id", "activity_type"], as_index=False)
        .agg(walkable_resource_count=("is_walkable_15min", "sum"), total_resource_count=("resource_id", "count"))
    )
    summary["walkable_15min_coverage"] = summary["walkable_resource_count"] / summary["total_resource_count"].replace(0, pd.NA)
    summary["source"] = "WALK_HAVERSINE_DETOUR_APPROX"
    summary.to_csv(output / "walkability_15min_summary.csv", index=False)
    return walking, summary


def build_from_files(
    areas_path: str | Path = "data/processed/nyc/real/candidate_areas.csv",
    resources_path: str | Path = "data/processed/nyc/real/resources.csv",
    processed_dir: str | Path = "data/processed/nyc/real",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_walking_travel_times(load_standard_areas(areas_path), load_standard_resources(resources_path), processed_dir)

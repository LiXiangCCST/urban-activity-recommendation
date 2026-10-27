"""Taxi-zone mapping for area/resource points."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from src.travel.utils import ensure_dir


def can_read_taxi_geometry() -> bool:
    return bool(importlib.util.find_spec("geopandas") or importlib.util.find_spec("shapefile"))


def build_area_resource_taxi_zone_mapping(areas: pd.DataFrame, resources: pd.DataFrame, raw_travel_dir: str | Path, processed_dir: str | Path, output_dir: str | Path = "outputs") -> tuple[pd.DataFrame, dict]:
    output = ensure_dir(processed_dir)
    report_dir = ensure_dir(output_dir)
    taxi_zip = Path(raw_travel_dir) / "taxi_zones.zip"
    if not taxi_zip.exists() or not can_read_taxi_geometry():
        result = pd.DataFrame(
            [
                {
                    "area_id": area["area_id"],
                    "area_taxi_zone_id": pd.NA,
                    "resource_id": resource["resource_id"],
                    "resource_taxi_zone_id": pd.NA,
                    "mapping_method": "unavailable_geometry_reader",
                    "source": "TAXI_ZONE_MAPPING_BLOCKED",
                }
                for _, area in areas.iterrows()
                for _, resource in resources.iterrows()
            ]
        )
        result.to_csv(output / "area_resource_taxi_zone_mapping.csv", index=False)
        lines = ["# Taxi zone mapping blocker", "", f"- taxi_zones.zip exists: {taxi_zip.exists()}", f"- geospatial reader available: {can_read_taxi_geometry()}", "- Mapping method: blocked; area-resource road times use distance fallback.", "- Needed file/dependency: a readable taxi zone polygon layer via GeoPandas, pyshp, or precomputed zone centroids/polygon assignments."]
        (report_dir / "taxi_zone_mapping_blocker.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return result, {"status": "blocked", "rows": len(result), "geometry_available": False}
    # Placeholder for future geopandas/pyshp implementation.
    result = pd.DataFrame()
    result.to_csv(output / "area_resource_taxi_zone_mapping.csv", index=False)
    return result, {"status": "not_implemented", "rows": 0, "geometry_available": True}

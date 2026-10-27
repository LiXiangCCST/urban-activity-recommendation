"""Build all multimodal travel-time layers for NYC real mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.travel.effective_travel_time import build_effective_travel_times
from src.travel.road_time_tlc import aggregate_tlc_road_times, build_area_resource_road_fallback, inspect_travel_files
from src.travel.taxi_zone_mapping import build_area_resource_taxi_zone_mapping
from src.travel.transit_time_gtfs import build_transit_travel_times
from src.travel.walking_time import build_walking_travel_times


def build_multimodal_travel_layer(
    areas: pd.DataFrame,
    resources: pd.DataFrame,
    raw_dir: str | Path = "data/raw/nyc/real",
    processed_dir: str | Path = "data/processed/nyc/real",
    output_dir: str | Path = "outputs",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    config = config or {}
    travel_cfg = config.get("travel_time", {}) if isinstance(config, dict) else {}

    area_std = areas[["area_id", "lon", "lat"]].copy()
    resource_std = resources[["resource_id", "category", "lon", "lat"]].rename(columns={"category": "activity_type"}).copy()

    inspection = inspect_travel_files(raw_path / "travel", output_path)
    walking, walk_summary = build_walking_travel_times(
        area_std,
        resource_std,
        processed_path,
        detour_factor=float(travel_cfg.get("walking_detour_factor", 1.25)),
        walking_speed_m_per_min=float(travel_cfg.get("walking_speed_m_per_min", 80.0)),
    )
    existing_road_area_path = processed_path / "area_resource_road_travel_times.csv"
    existing_road_od_path = processed_path / "road_travel_times.csv"
    road_od = pd.DataFrame()
    road_summary = {"status": "not_run"}
    taxi_mapping = pd.DataFrame()
    taxi_summary = {"status": "not_run"}
    road_area = pd.DataFrame()
    if existing_road_area_path.exists():
        candidate = pd.read_csv(existing_road_area_path)
        if not candidate.empty and candidate["source"].astype(str).str.contains("TLC", na=False).any():
            road_area = candidate
            if existing_road_od_path.exists():
                road_od = pd.read_csv(existing_road_od_path)
            road_summary = {"status": "reused_existing_observed_or_tlc_fallback", "od_period_records": len(road_od)}
            taxi_summary = {"status": "reused_existing_mapping"}
    if road_area.empty:
        road_od, road_summary = aggregate_tlc_road_times(raw_path / "travel", processed_path, output_path, max_trip_files=travel_cfg.get("max_trip_files"))
        taxi_mapping, taxi_summary = build_area_resource_taxi_zone_mapping(area_std, resource_std, raw_path / "travel", processed_path, output_path)
        road_area = build_area_resource_road_fallback(area_std, resource_std, processed_path, car_speed_kmh=float(travel_cfg.get("road_fallback_speed_kmh", 22.0)))
    transit = build_transit_travel_times(
        area_std,
        resource_std,
        raw_path / "gtfs.zip",
        processed_path,
        walking_speed_m_per_min=float(travel_cfg.get("walking_speed_m_per_min", 80.0)),
        transit_speed_kmh=float(travel_cfg.get("gtfs_transit_speed_kmh", 22.0)),
    )
    effective, compat = build_effective_travel_times(walking, transit, road_area, processed_path)

    observed_road = int((road_area["source"].str.contains("TLC", na=False)).sum()) if not road_area.empty else 0
    fallback_road = int((road_area["source"] == "ROAD_DISTANCE_FALLBACK").sum()) if not road_area.empty else 0
    summary = {
        "walking_records": len(walking),
        "walkability_summary_records": len(walk_summary),
        "road_od_records": len(road_od),
        "road_area_resource_records": len(road_area),
        "transit_records": len(transit),
        "effective_records": len(effective),
        "observed_tlc_road_records": observed_road,
        "road_distance_fallback_records": fallback_road,
        "gtfs_approx_records": int((transit["source"] == "GTFS_APPROX_NEAREST_STOP").sum()) if not transit.empty else 0,
        "walking_selected_records": int((effective["selected_mode"] == "walk").sum()) if not effective.empty else 0,
        "transit_selected_records": int((effective["selected_mode"] == "transit").sum()) if not effective.empty else 0,
        "road_selected_records": int(effective["selected_mode"].isin(["car", "ridehail"]).sum()) if not effective.empty else 0,
        "parquet_supported": inspection.get("parquet_supported"),
        "taxi_zone_mapping_status": taxi_summary.get("status"),
        "road_od_status": road_summary.get("status"),
    }
    _write_report(output_path, summary, effective)
    return {"summary": summary, "effective_wide": compat, "effective": effective}


def _write_report(output_dir: Path, summary: dict[str, Any], effective: pd.DataFrame) -> None:
    lines = ["# Step 11 multimodal travel time report", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Modes available by profile", ""])
    if not effective.empty:
        table = effective.groupby(["profile_id", "selected_mode"]).size().reset_index(name="records")
        for _, row in table.iterrows():
            lines.append(f"- {row['profile_id']} / {row['selected_mode']}: {row['records']}")
    lines.extend(["", "## Fallback rates", "", f"- Road distance fallback records: {summary.get('road_distance_fallback_records', 0)}", f"- Observed TLC road records: {summary.get('observed_tlc_road_records', 0)}", f"- GTFS approximation records: {summary.get('gtfs_approx_records', 0)}", "", "## Remaining limitations", "", "- GTFS times are nearest-stop approximations, not OpenTripPlanner routed itineraries.", "- TLC observed road times were not used when Parquet support or taxi-zone geometry is unavailable.", "- Road fallback uses haversine distance, detour factor, and period multipliers."])
    (output_dir / "step11_multimodal_travel_time_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

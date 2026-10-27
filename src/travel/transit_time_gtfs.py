"""GTFS-based approximate public-transit travel times."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from src.travel.utils import ensure_dir, haversine_meters


def load_gtfs_stops(gtfs_zip: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(gtfs_zip) as zf:
        with zf.open("stops.txt") as file:
            stops = pd.read_csv(file)
    stops = stops.rename(columns={"stop_lon": "lon", "stop_lat": "lat", "stop_name": "name"})
    stops["lon"] = pd.to_numeric(stops["lon"], errors="coerce")
    stops["lat"] = pd.to_numeric(stops["lat"], errors="coerce")
    return stops.dropna(subset=["lon", "lat"])[["stop_id", "name", "lon", "lat"]]


def _nearest_stop(point_lon: float, point_lat: float, stops: pd.DataFrame) -> tuple[str, float, float, float]:
    best = None
    for _, stop in stops.iterrows():
        d = haversine_meters(point_lon, point_lat, stop["lon"], stop["lat"])
        if best is None or d < best[3]:
            best = (str(stop["stop_id"]), float(stop["lon"]), float(stop["lat"]), float(d))
    return best if best else ("", point_lon, point_lat, 0.0)


def build_transit_travel_times(
    areas: pd.DataFrame,
    resources: pd.DataFrame,
    gtfs_zip: str | Path,
    processed_dir: str | Path,
    walking_speed_m_per_min: float = 80.0,
    transit_speed_kmh: float = 22.0,
) -> pd.DataFrame:
    output = ensure_dir(processed_dir)
    if not Path(gtfs_zip).exists():
        result = pd.DataFrame(columns=["area_id", "resource_id", "activity_type", "departure_period", "transit_time_minutes", "access_walk_minutes", "egress_walk_minutes", "wait_time_minutes", "in_vehicle_proxy_minutes", "transfer_penalty_minutes", "source"])
        result.to_csv(output / "transit_travel_times.csv", index=False)
        return result
    stops = load_gtfs_stops(gtfs_zip)
    area_stop = {}
    for _, area in areas.iterrows():
        area_stop[area["area_id"]] = _nearest_stop(area["lon"], area["lat"], stops)
    resource_stop = {}
    for _, resource in resources.iterrows():
        resource_stop[resource["resource_id"]] = _nearest_stop(resource["lon"], resource["lat"], stops)
    rows = []
    for _, area in areas.iterrows():
        a_stop, a_lon, a_lat, a_dist = area_stop[area["area_id"]]
        for _, resource in resources.iterrows():
            r_stop, r_lon, r_lat, r_dist = resource_stop[resource["resource_id"]]
            stop_distance = haversine_meters(a_lon, a_lat, r_lon, r_lat)
            access = a_dist / walking_speed_m_per_min
            egress = r_dist / walking_speed_m_per_min
            in_vehicle = (stop_distance / 1000.0) / transit_speed_kmh * 60.0
            for period, wait in [("morning_peak", 6.0), ("midday", 6.0), ("evening_peak", 6.0), ("night", 12.0), ("other", 8.0)]:
                transfer = 0.0 if a_stop == r_stop else 5.0
                total = access + egress + in_vehicle + wait + transfer
                rows.append({"area_id": area["area_id"], "resource_id": resource["resource_id"], "activity_type": resource["activity_type"], "departure_period": period, "transit_time_minutes": round(total, 3), "access_walk_minutes": round(access, 3), "egress_walk_minutes": round(egress, 3), "wait_time_minutes": wait, "in_vehicle_proxy_minutes": round(in_vehicle, 3), "transfer_penalty_minutes": transfer, "source": "GTFS_APPROX_NEAREST_STOP"})
    result = pd.DataFrame(rows)
    result.to_csv(output / "transit_travel_times.csv", index=False)
    return result

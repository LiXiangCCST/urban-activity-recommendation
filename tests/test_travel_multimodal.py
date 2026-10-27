import pandas as pd

from src.profiles.individual_profiles import get_profiles
from src.travel.effective_travel_time import build_effective_travel_times
from src.travel.road_time_tlc import departure_period
from src.travel.taxi_zone_mapping import build_area_resource_taxi_zone_mapping
from src.travel.walking_time import build_walking_travel_times


def test_walking_time_calculation(tmp_path):
    areas = pd.DataFrame([{"area_id": "a1", "lon": -73.0, "lat": 40.0}])
    resources = pd.DataFrame([{"resource_id": "r1", "activity_type": "grocery", "lon": -73.0, "lat": 40.0}])
    walking, summary = build_walking_travel_times(areas, resources, tmp_path)
    assert walking.iloc[0]["walking_time_minutes"] == 0
    assert bool(walking.iloc[0]["is_walkable_15min"])
    assert summary.iloc[0]["walkable_resource_count"] == 1


def test_departure_period_assignment():
    assert departure_period(pd.Timestamp("2026-01-01 08:00")) == "morning_peak"
    assert departure_period(pd.Timestamp("2026-01-01 12:00")) == "midday"
    assert departure_period(pd.Timestamp("2026-01-01 17:00")) == "evening_peak"
    assert departure_period(pd.Timestamp("2026-01-01 23:00")) == "night"


def test_taxi_zone_mapping_fallback(tmp_path):
    areas = pd.DataFrame([{"area_id": "a1", "lon": -73.0, "lat": 40.0}])
    resources = pd.DataFrame([{"resource_id": "r1", "activity_type": "grocery", "lon": -73.0, "lat": 40.0}])
    mapping, summary = build_area_resource_taxi_zone_mapping(areas, resources, tmp_path / "missing", tmp_path, tmp_path)
    assert summary["status"] == "blocked"
    assert mapping.iloc[0]["mapping_method"] == "unavailable_geometry_reader"


def test_effective_travel_time_mode_selection(tmp_path):
    walking = pd.DataFrame([{"area_id": "a1", "resource_id": "r1", "activity_type": "employment", "walking_time_minutes": 40.0}])
    transit = pd.DataFrame([{"area_id": "a1", "resource_id": "r1", "activity_type": "employment", "departure_period": "morning_peak", "transit_time_minutes": 20.0, "source": "GTFS_APPROX"}])
    road = pd.DataFrame([{"area_id": "a1", "resource_id": "r1", "activity_type": "employment", "departure_period": "morning_peak", "travel_time_minutes": 10.0, "source": "ROAD_DISTANCE_FALLBACK"}])
    effective, wide = build_effective_travel_times(walking, transit, road, tmp_path, {"low_income_worker_without_car": get_profiles()["low_income_worker_without_car"]})
    assert effective.iloc[0]["selected_mode"] == "transit"
    assert "profile_id" in wide.columns

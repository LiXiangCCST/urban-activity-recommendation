from argparse import Namespace

import pandas as pd
import pytest

from src.data.census_loader import standardize_candidate_areas
from src.data.nyc_loader import NYCDataUnavailable, load_first_available, standardize_facility_table
from src.data.preprocess_nyc import build_travel_time_tables, estimate_minutes, process_nyc_data
from src.main import load_dataset_for_mode, load_config


def test_nyc_loader_graceful_failure_when_missing(tmp_path):
    with pytest.raises(NYCDataUnavailable):
        load_first_available([tmp_path / "missing.csv"], "facilities")


def test_schema_standardization_for_local_area_and_resource_files():
    raw_areas = pd.DataFrame(
        {
            "GEOID": ["36061000100"],
            "NAME": ["Example tract"],
            "longitude": [-73.98],
            "latitude": [40.75],
            "median_rent": [1800],
            "B19013_001E": [72000],
            "B01003_001E": [4100],
        }
    )
    areas = standardize_candidate_areas(raw_areas)
    assert {"area_id", "name", "centroid_x", "centroid_y", "housing_cost", "transit_accessibility"}.issubset(areas.columns)

    raw_resources = pd.DataFrame(
        {
            "facility_name": ["Example school"],
            "facility_type": ["school"],
            "longitude": [-73.99],
            "latitude": [40.76],
        }
    )
    resources = standardize_facility_table(raw_resources, "facilities", {"school": "school"})
    assert resources.iloc[0]["category"] == "school"
    assert "quality_score" in resources.columns
    assert "quality" in resources.columns


def test_travel_time_estimation_and_matrix_schema():
    assert estimate_minutes(1000, 5) == pytest.approx(12.0)
    areas = pd.DataFrame(
        [
            {
                "area_id": "a1",
                "centroid_x": -73.98,
                "centroid_y": 40.75,
                "x": -73.98,
                "y": 40.75,
                "housing_cost": 1800,
                "risk_score": 0.2,
                "transit_accessibility": 0.7,
                "employment_density": 0.5,
            }
        ]
    )
    resources = pd.DataFrame(
        [
            {
                "resource_id": "r1",
                "category": "hospital",
                "name": "Hospital",
                "x": -73.99,
                "y": 40.76,
                "quality_score": 0.9,
                "quality": 0.9,
                "source": "test",
            }
        ]
    )
    wide, long = build_travel_time_tables(areas, resources, {"walk": 4.8, "bike": 14, "car": 25, "transit": 18})
    assert {"walk_time", "bike_time", "car_time", "transit_time", "distance_meters"}.issubset(wide.columns)
    assert {"area_id", "resource_id", "mode", "travel_time_minutes", "distance_meters"}.issubset(long.columns)
    assert wide.iloc[0]["walk_time"] > 0


def test_process_nyc_data_missing_resources_fails_gracefully(tmp_path):
    config = load_config()
    config["bounding_box"] = {"min_lon": -74.1, "min_lat": 40.6, "max_lon": -73.8, "max_lat": 40.9}
    config["nyc"]["raw_paths"] = {
        "candidate_areas": str(tmp_path / "missing_areas.csv"),
        "facilities": str(tmp_path / "missing_facilities.csv"),
        "schools": str(tmp_path / "missing_schools.csv"),
        "hospitals": str(tmp_path / "missing_hospitals.csv"),
        "parks": str(tmp_path / "missing_parks.csv"),
        "libraries": str(tmp_path / "missing_libraries.csv"),
        "gtfs_zip": str(tmp_path / "missing_gtfs.zip"),
    }
    config["nyc"]["processed_paths"] = {
        "areas": str(tmp_path / "processed" / "areas.csv"),
        "resources": str(tmp_path / "processed" / "resources.csv"),
        "transit_stops": str(tmp_path / "processed" / "transit_stops.csv"),
        "travel_times": str(tmp_path / "processed" / "travel_times.csv"),
        "travel_times_long": str(tmp_path / "processed" / "travel_times_long.csv"),
        "metadata": str(tmp_path / "processed" / "metadata.yaml"),
    }
    with pytest.raises(NYCDataUnavailable):
        process_nyc_data(config)


def test_main_cli_synthetic_mode_still_loads_data(tmp_path):
    config = load_config()
    config["paths"]["synthetic_sample"] = str(tmp_path / "sample")
    args = Namespace(data_mode="synthetic", generate_sample=True, no_fallback=False, max_areas=None, max_resources=None)
    areas, resources, travel_times, metadata = load_dataset_for_mode(args, config)
    assert metadata["data_mode"] == "synthetic"
    assert len(areas) > 0
    assert len(resources) > 0
    assert len(travel_times) > 0

from argparse import Namespace
from pathlib import Path

import pandas as pd

from src.data.schema_validation import (
    validate_coordinate_columns,
    validate_dataframe_columns,
    validate_nyc_raw_file,
    validate_numeric_columns,
)
from src.main import apply_nyc_raw_dir, load_config, validate_nyc_raw_data


def test_schema_validation_success(tmp_path):
    path = tmp_path / "facilities.csv"
    path.write_text(
        "facility_id,name,facility_type,longitude,latitude,quality_score\n"
        "fac_1,Example Facility,public_service,-73.98,40.75,0.8\n"
    )
    result = validate_nyc_raw_file(path, "facilities")
    assert result["valid"] is True
    assert result["row_count"] == 1


def test_schema_validation_missing_columns():
    df = pd.DataFrame({"name": ["Missing coords"]})
    result = validate_dataframe_columns(df, required_columns=["name", "longitude", "latitude"])
    assert result["valid"] is False
    assert "longitude" in result["missing_required_columns"]
    assert "latitude" in result["missing_required_columns"]


def test_coordinate_validation_failure():
    df = pd.DataFrame({"longitude": [-73.9, -500.0], "latitude": [40.7, 40.8]})
    result = validate_coordinate_columns(df, "longitude", "latitude")
    assert result["valid"] is False
    assert result["invalid_coordinate_rows"] == [1]


def test_numeric_validation_failure():
    df = pd.DataFrame({"housing_cost": [1800, "not_number"]})
    result = validate_numeric_columns(df, ["housing_cost"])
    assert result["valid"] is False
    assert "housing_cost" in result["invalid_numeric_columns"]


def test_validate_data_nyc_command_path(tmp_path):
    config = apply_nyc_raw_dir(load_config(), "data/raw/nyc/example_minimal")
    results, summary = validate_nyc_raw_data(config, tmp_path)
    assert (tmp_path / "nyc_data_validation_report.csv").exists()
    assert (tmp_path / "nyc_data_validation_report.json").exists()
    assert summary["existing_files"] >= 5
    assert any(result["dataset_type"] == "census_areas" and result["valid"] for result in results)
    assert any(result["dataset_type"] == "facilities" and result["valid"] for result in results)


def test_minimal_nyc_example_files_validate():
    base = Path("data/raw/nyc/example_minimal")
    for dataset_type, filename in [
        ("census_areas", "census_areas.csv"),
        ("facilities", "facilities.csv"),
        ("transit_stops", "transit_stops.csv"),
        ("housing_cost", "housing_cost.csv"),
        ("risk", "risk.csv"),
    ]:
        result = validate_nyc_raw_file(base / filename, dataset_type)
        assert result["valid"], result


from argparse import Namespace
from pathlib import Path

import pandas as pd

from src.data.download_nyc_public_data import download_nyc_public_data, load_dataset_manifest
from src.data.standardize_nyc_real import standardize_resource_file
from src.main import apply_nyc_raw_dir, load_config, load_dataset_for_mode


def test_nyc_datasets_manifest_parses():
    manifest = load_dataset_manifest("config/nyc_datasets.yaml")
    assert "datasets" in manifest
    assert any(item["dataset_id"] == "mta_subway_gtfs" for item in manifest["datasets"])
    assert all("manual_download_instructions" in item for item in manifest["datasets"])


def test_downloader_skips_no_direct_url(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "datasets:\n"
        "  - dataset_id: no_direct\n"
        "    name: No Direct\n"
        "    source_platform: Test\n"
        "    url: https://example.invalid\n"
        "    direct_download_url:\n"
        "    expected_local_filename: test.csv\n"
        "    dataset_type: facilities\n"
        "    required_columns_after_standardization: []\n"
        "    optional_columns: []\n"
        "    notes: test\n"
        "    manual_download_instructions: download manually\n"
    )
    report = download_nyc_public_data(manifest_path=manifest, raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    assert report["results"][0]["status"] == "skipped_no_direct_url"
    assert (tmp_path / "out" / "nyc_download_report.json").exists()


def test_standardization_flexible_columns(tmp_path):
    raw = tmp_path / "schools.csv"
    raw.write_text("school_id,school_name,facility_type,lng,lat,score\ns1,Example School,school,-73.98,40.75,85\n")
    config = load_config()
    table, report = standardize_resource_file(raw, "schools", config["nyc"]["category_mapping"])
    assert report["status"] == "loaded"
    assert table.iloc[0]["name"] == "Example School"
    assert table.iloc[0]["category"] == "school"
    assert table.iloc[0]["lon"] == -73.98
    assert table.iloc[0]["quality_score"] == 0.85


def test_real_mode_missing_files_fails_gracefully(tmp_path):
    config = apply_nyc_raw_dir(load_config(), str(tmp_path / "missing_real"))
    args = Namespace(data_mode="nyc", nyc_data_variant="real", generate_sample=False, no_fallback=False, max_areas=None, max_resources=None)
    areas, resources, travel_times, metadata = load_dataset_for_mode(args, config, output_dir=tmp_path / "out")
    assert metadata["data_provenance"] == "synthetic"
    assert "fallback_reason" in metadata
    assert len(areas) > 0


def test_auto_variant_falls_back_to_example_minimal(tmp_path):
    config = load_config()
    args = Namespace(data_mode="nyc", nyc_data_variant="auto", generate_sample=False, no_fallback=False, max_areas=None, max_resources=None)
    areas, resources, travel_times, metadata = load_dataset_for_mode(args, config, output_dir=tmp_path / "auto")
    assert metadata["data_provenance"] in {"partial_real", "partial_real_placeholder_housing", "partial_real_acs_housing", "partial_real_acs_housing_multimodal_time", "partial_real_multi_resource", "real_public_data", "real_public_data_candidate", "fictional_minimal", "synthetic"}
    assert len(areas) > 0
    assert (tmp_path / "auto" / "real_nyc_data_quality_report.md").exists()


def test_provenance_labels_for_example_minimal(tmp_path):
    config = apply_nyc_raw_dir(load_config(), "data/raw/nyc/example_minimal")
    args = Namespace(data_mode="nyc", nyc_data_variant="example_minimal", generate_sample=False, no_fallback=True, max_areas=None, max_resources=None)
    _, _, _, metadata = load_dataset_for_mode(args, config, output_dir=tmp_path / "example")
    assert metadata["nyc_data_variant"] == "example_minimal"
    assert metadata["data_provenance"] == "fictional_minimal"

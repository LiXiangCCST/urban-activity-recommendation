from src.accessibility.accessibility import compute_accessibility
from src.data.generate_sample_data import generate_sample_data
from src.profiles.individual_profiles import get_profiles


def test_accessibility_score_is_numeric_and_non_negative(tmp_path):
    areas, resources, travel_times = generate_sample_data(output_dir=tmp_path / "sample")
    profile = get_profiles()["low_income_worker_without_car"]
    theta = {"employment": 20, "school": 20, "hospital": 20, "grocery": 20, "park": 20, "transit": 20, "leisure": 20, "public_service": 20}

    result = compute_accessibility(areas.iloc[0]["area_id"], profile, resources, travel_times, theta)

    assert isinstance(result["total"], float)
    assert result["total"] >= 0
    assert "employment" in result["by_category"]

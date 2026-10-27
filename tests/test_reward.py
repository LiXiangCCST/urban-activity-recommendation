from src.accessibility.reward import compute_reward
from src.data.generate_sample_data import generate_sample_data
from src.profiles.individual_profiles import get_profiles


def test_reward_score_is_numeric(tmp_path):
    areas, resources, travel_times = generate_sample_data(output_dir=tmp_path / "sample")
    profile = get_profiles()["family_with_children"]
    theta = {"employment": 20, "school": 20, "hospital": 20, "grocery": 20, "park": 20, "transit": 20, "leisure": 20, "public_service": 20}
    weights = {
        "alpha_access": 1.0,
        "alpha_activity_fit": 0.3,
        "alpha_mobility": 0.3,
        "alpha_equity": 0.2,
        "beta_cost": 0.4,
        "beta_commute": 0.3,
        "beta_risk": 0.4,
    }

    result = compute_reward(areas.iloc[0]["area_id"], profile, areas, resources, travel_times, theta, weights)

    assert isinstance(result["reward"], float)
    assert isinstance(result["accessibility"], float)

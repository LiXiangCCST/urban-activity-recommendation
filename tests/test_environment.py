from src.data.generate_sample_data import generate_sample_data
from src.graph.build_graph import build_urban_graph
from src.profiles.individual_profiles import get_profiles
from src.rl.environment import UrbanActivityRecommendationEnv


def _make_env(tmp_path):
    areas, resources, travel_times = generate_sample_data(output_dir=tmp_path / "sample")
    graph = build_urban_graph(areas, resources, travel_times)
    profile = get_profiles()["transit_dependent_resident"]
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
    return UrbanActivityRecommendationEnv(graph, profile, areas, resources, travel_times, theta, weights, start_area=areas.iloc[0]["area_id"])


def test_environment_reset_and_step_work(tmp_path):
    env = _make_env(tmp_path)
    state = env.reset()
    assert state["current_area"]
    next_state, reward, done, info = env.step("stay")
    assert next_state["current_area"] == state["current_area"]
    assert isinstance(reward, float)
    assert done is False
    assert "selected_area" in info


def test_selecting_current_area_terminates(tmp_path):
    env = _make_env(tmp_path)
    env.reset()
    _, _, done, info = env.step("select_current_area")
    assert done is True
    assert info["selected_area"] == env.current_area


from argparse import Namespace

from src.data.generate_sample_data import generate_sample_data
from src.graph.build_graph import build_urban_graph
from src.main import load_config, run_recommendation_workflow
from src.reporting.research_reports import (
    build_area_diagnostics,
    build_baseline_ranking,
    build_reward_decomposition,
    write_profile_explanations,
)


def _small_run(tmp_path):
    config = load_config()
    config["sample_data"] = dict(config["sample_data"])
    config["sample_data"]["n_areas"] = 9
    config["sample_data"]["grid_width"] = 3
    config["sample_data"]["resources_per_category"] = 2
    areas, resources, travel_times = generate_sample_data(output_dir=tmp_path / "sample", config_path="config/config.yaml")
    # Keep tests lightweight by truncating generated data after creation.
    areas = areas.head(9)
    resources = resources.groupby("category").head(2).reset_index(drop=True)
    travel_times = travel_times[travel_times["area_id"].isin(areas["area_id"]) & travel_times["resource_id"].isin(resources["resource_id"])]
    recommendations, rl_results = run_recommendation_workflow(
        areas,
        resources,
        travel_times,
        config,
        tmp_path / "outputs",
        {"data_mode": "test", "case_city": "Test City", "messages": []},
    )
    return config, areas, resources, travel_times, recommendations, rl_results


def test_reward_decomposition_fields(tmp_path):
    config, areas, resources, travel_times, recommendations, _ = _small_run(tmp_path)
    decomp = build_reward_decomposition(recommendations, areas, resources, travel_times, config)
    expected = {
        "profile_id",
        "area_id",
        "accessibility_contribution",
        "activity_fit_contribution",
        "mobility_contribution",
        "equity_contribution",
        "cost_penalty",
        "commute_penalty",
        "risk_penalty",
        "final_reward",
    }
    assert expected.issubset(decomp.columns)
    assert len(decomp) > 0


def test_baseline_ranking_output(tmp_path):
    _, _, _, _, recommendations, rl_results = _small_run(tmp_path)
    ranking = build_baseline_ranking(recommendations, rl_results)
    assert {"profile_id", "rank", "method", "recommended_area", "reward", "accessibility"}.issubset(ranking.columns)
    assert "rl_demo" in set(ranking["method"])
    assert ranking.groupby("profile_id")["rank"].min().eq(1).all()


def test_profile_explanation_generation(tmp_path):
    config, areas, resources, travel_times, recommendations, rl_results = _small_run(tmp_path)
    decomp = build_reward_decomposition(recommendations, areas, resources, travel_times, config)
    ranking = build_baseline_ranking(recommendations, rl_results)
    path = write_profile_explanations(tmp_path, recommendations, decomp, ranking, {"data_mode": "test"})
    text = path.read_text()
    assert "Profile recommendation explanations" in text
    assert "Recommended area" in text
    assert "Strongest positive factors" in text


def test_area_diagnostics_generation(tmp_path):
    config, areas, resources, travel_times, recommendations, _ = _small_run(tmp_path)
    diagnostics = build_area_diagnostics(areas, resources, travel_times, recommendations, config)
    assert {"area_id", "average_reward_across_profiles", "average_accessibility", "number_of_times_selected"}.issubset(diagnostics.columns)
    assert len(diagnostics) == len(areas)

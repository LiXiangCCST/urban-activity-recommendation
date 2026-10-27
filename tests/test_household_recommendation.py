from argparse import Namespace
from pathlib import Path

import pandas as pd

from src.main import apply_nyc_raw_dir, load_config, load_dataset_for_mode
from src.household.experiment import run_household_experiment
from src.profiles.household_profiles import load_household_profiles, validate_household_inputs
from src.reward.household_reward import compute_household_reward, compute_member_commutes


def small_data():
    areas = pd.DataFrame(
        [
            {"area_id": "A1", "name": "Area 1", "lon": -73.99, "lat": 40.72, "housing_cost": 1500, "risk_score": 0.2, "transit_score": 0.8},
            {"area_id": "A2", "name": "Area 2", "lon": -73.90, "lat": 40.80, "housing_cost": 1100, "risk_score": 0.4, "transit_score": 0.5},
        ]
    )
    resources = pd.DataFrame(
        [
            {"resource_id": "R1", "category": "grocery", "name": "Grocery", "lon": -73.98, "lat": 40.72, "quality_score": 1.0},
            {"resource_id": "R2", "category": "school", "name": "School", "lon": -73.97, "lat": 40.73, "quality_score": 1.0},
            {"resource_id": "R3", "category": "hospital", "name": "Hospital", "lon": -73.96, "lat": 40.74, "quality_score": 1.0},
            {"resource_id": "R4", "category": "employment", "name": "Jobs", "lon": -73.95, "lat": 40.75, "quality_score": 1.0},
        ]
    )
    rows = []
    for area_id in areas["area_id"]:
        for _, res in resources.iterrows():
            rows.append({"area_id": area_id, "resource_id": res["resource_id"], "category": res["category"], "walk_time": 12, "bike_time": 8, "car_time": 10, "transit_time": 14, "distance_meters": 1000})
    return areas, resources, pd.DataFrame(rows)


def write_profiles(tmp_path: Path):
    households = tmp_path / "households.csv"
    members = tmp_path / "members.csv"
    households.write_text(
        "household_id,household_type,household_income,housing_budget,car_count,num_adults,num_workers,num_children,num_elderly,preferred_modes,commute_tolerance_minutes,affordability_sensitivity,care_responsibility_level\n"
        "hh1,test,80000,1800,1,2,1,1,0,car|transit|walk,45,1.0,0.7\n"
    )
    members.write_text(
        "household_id,member_id,role,employed,workplace_area_id,workplace_lon,workplace_lat,work_schedule,mode_preference,commute_weight,activity_needs\n"
        "hh1,m1,adult_worker,true,A2,,,day,car,1.0,employment|grocery\n"
        "hh1,m2,child,false,,,,day,walk,0.0,school|park\n"
    )
    return households, members


def test_household_profile_loading_and_workplace_mapping(tmp_path):
    areas, _, _ = small_data()
    households_path, members_path = write_profiles(tmp_path)
    profiles = load_household_profiles(households_path, members_path, areas)
    household = profiles["hh1"]
    assert household.num_children == 1
    assert household.members[0].workplace_lon == areas.loc[areas["area_id"] == "A2", "lon"].iloc[0]
    assert household.workplace_assumptions


def test_household_member_validation_missing_columns():
    households = pd.DataFrame({"household_id": ["hh1"]})
    members = pd.DataFrame({"household_id": ["hh1"]})
    errors = validate_household_inputs(households, members)
    assert errors


def test_household_commute_calculation(tmp_path):
    areas, _, _ = small_data()
    households_path, members_path = write_profiles(tmp_path)
    household = load_household_profiles(households_path, members_path, areas)["hh1"]
    commutes = compute_member_commutes(areas.iloc[0], household)
    assert len(commutes) == 1
    assert commutes.iloc[0]["commute_time_minutes"] >= 0


def test_household_reward_decomposition(tmp_path):
    areas, resources, travel_times = small_data()
    households_path, members_path = write_profiles(tmp_path)
    household = load_household_profiles(households_path, members_path, areas)["hh1"]
    reward = compute_household_reward("A1", household, areas, resources, travel_times, {"grocery": 20, "school": 20, "hospital": 20, "employment": 20})
    assert isinstance(reward["final_reward"], float)
    assert "housing_cost_penalty" in reward
    assert "mean_commute_time" in reward


def test_household_experiment_smoke(tmp_path):
    areas, resources, travel_times = small_data()
    households_path, members_path = write_profiles(tmp_path)
    config = {"sample_data": {"theta_by_category": {"grocery": 20, "school": 20, "hospital": 20, "employment": 20, "park": 20, "public_service": 20, "transit": 20, "leisure": 20}}}
    results = run_household_experiment(areas, resources, travel_times, config, tmp_path / "out", households_path, members_path, metadata={"data_provenance": "test", "travel_time_source": "test"})
    assert not results["recommendations"].empty
    assert (tmp_path / "out" / "household_recommendations.csv").exists()
    assert (tmp_path / "out" / "household_executive_summary.md").exists()

from src.reward.household_reward import compute_housing_cost_penalty, HOUSING_PENALTY_SCENARIOS


def test_strengthened_housing_penalty_increases_low_income_penalty(tmp_path):
    areas, _, _ = small_data()
    households_path, members_path = write_profiles(tmp_path)
    household = load_household_profiles(households_path, members_path, areas)["hh1"]
    object.__setattr__(household, "household_income", 40000.0)
    base = compute_housing_cost_penalty(2200, household, areas, HOUSING_PENALTY_SCENARIOS["baseline_step14_like"])
    strong = compute_housing_cost_penalty(2200, household, areas, HOUSING_PENALTY_SCENARIOS["strong_affordability_penalty"])
    assert strong["housing_cost_penalty"] > base["housing_cost_penalty"]


def test_zero_housing_cost_is_treated_as_missing(tmp_path):
    areas, _, _ = small_data()
    areas.loc[0, "housing_cost"] = 0
    households_path, members_path = write_profiles(tmp_path)
    household = load_household_profiles(households_path, members_path, areas)["hh1"]
    penalty = compute_housing_cost_penalty(0, household, areas, HOUSING_PENALTY_SCENARIOS["medium_affordability_penalty"])
    assert penalty["housing_cost_missing"] is True
    assert penalty["effective_housing_cost"] > 0
    assert penalty["housing_cost_penalty"] > 0


def test_expanded_step15_profiles_load():
    areas, _, _ = small_data()
    profiles = load_household_profiles("data/profiles/household_profiles_step15.csv", "data/profiles/household_members_step15.csv", areas)
    assert len(profiles) >= 18
    assert any(profile.household_type == "night_shift_worker_family" for profile in profiles.values())


def test_household_sensitivity_outputs(tmp_path):
    areas, resources, travel_times = small_data()
    households_path, members_path = write_profiles(tmp_path)
    config = {"sample_data": {"theta_by_category": {"grocery": 20, "school": 20, "hospital": 20, "employment": 20, "park": 20, "public_service": 20, "transit": 20, "leisure": 20}}}
    results = run_household_experiment(areas, resources, travel_times, config, tmp_path / "out", households_path, members_path, metadata={"data_provenance": "test", "travel_time_source": "test"}, housing_penalty_scenario="all", sensitivity_run=True)
    assert "all_recommendations" in results
    assert (tmp_path / "out" / "household_affordability_sensitivity_comparison.csv").exists()
    assert (tmp_path / "out" / "household_recommendation_shift_summary.csv").exists()
    comparison = pd.read_csv(tmp_path / "out" / "household_affordability_sensitivity_comparison.csv")
    assert set(comparison["housing_penalty_scenario"]) >= {"baseline_step14_like", "medium_affordability_penalty", "strong_affordability_penalty"}

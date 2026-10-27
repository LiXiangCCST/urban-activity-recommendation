"""End-to-end household recommendation experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.household.baselines import run_household_baselines
from src.household.reports import markdown_table, write_household_reports
from src.profiles.household_profiles import load_household_profiles
from src.reward.household_reward import HOUSING_PENALTY_SCENARIOS, build_household_accessibility_cache, compute_household_rewards_for_all, compute_member_commutes, summarize_commutes, housing_penalty_config

SENSITIVITY_ORDER = ["baseline_step14_like", "medium_affordability_penalty", "strong_affordability_penalty"]


def _household_area_diagnostics(reward_surface: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    selected = recommendations[recommendations["method"] == "household_weighted_score"].groupby("area_id")["household_id"].apply(lambda s: ";".join(sorted(s))).reset_index(name="selected_by_households")
    counts = recommendations[recommendations["method"] == "household_weighted_score"]["area_id"].value_counts().rename_axis("area_id").reset_index(name="number_of_times_selected")
    diag = reward_surface.groupby("area_id").agg(
        average_household_reward=("final_reward", "mean"),
        average_accessibility=("accessibility_score", "mean"),
        mean_housing_cost_penalty=("housing_cost_penalty", "mean"),
        mean_commute_penalty=("mean_commute_penalty", "mean"),
        mean_max_commute_penalty=("max_commute_penalty", "mean"),
        mean_care_coordination_penalty=("care_coordination_penalty", "mean"),
        mean_risk_penalty=("risk_penalty", "mean"),
    ).reset_index()
    diag = diag.merge(counts, on="area_id", how="left").merge(selected, on="area_id", how="left")
    diag["number_of_times_selected"] = diag["number_of_times_selected"].fillna(0).astype(int)
    diag["selected_by_households"] = diag["selected_by_households"].fillna("")
    return diag.sort_values(["number_of_times_selected", "average_household_reward"], ascending=False)


def _recommendation_table(baselines: pd.DataFrame, reward_surface: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lookup_cols = ["household_id", "area_id"]
    if "housing_penalty_scenario" in reward_surface.columns:
        lookup_cols = ["housing_penalty_scenario", *lookup_cols]
    lookup = reward_surface.set_index(lookup_cols)
    for _, row in baselines.iterrows():
        key = tuple(row[col] for col in lookup_cols)
        surface = lookup.loc[key]
        out = row.to_dict()
        for col in ["child_accessibility", "elderly_medical_accessibility", "daily_service_accessibility", "workers_over_tolerance_count", "commute_inequality", "rent_income_ratio", "effective_housing_cost", "housing_cost_missing", "housing_budget_overage", "housing_cost_penalty", "rent_burden_penalty_component", "missing_housing_cost_penalty_component"]:
            out[col] = surface[col]
        rows.append(out)
    return pd.DataFrame(rows)


def _compute_outputs_for_scenario(households, areas, resources, travel_times, theta, scenario_name: str, config: dict[str, Any] | None = None, access_cache=None):
    surface_rows = []
    commute_rows = []
    member_rows = []
    area_by_id = {str(row["area_id"]): row for _, row in areas.iterrows()}
    scenario_config = housing_penalty_config(scenario_name, config)
    for household in households.values():
        surface = compute_household_rewards_for_all(household, areas, resources, travel_times, theta, housing_config=scenario_config, access_cache=access_cache)
        surface["housing_penalty_scenario"] = scenario_name
        surface_rows.extend(surface.drop(columns=["accessibility_by_category"], errors="ignore").to_dict("records"))
        for area_id, area_row in area_by_id.items():
            member_commutes = compute_member_commutes(area_row, household)
            if not member_commutes.empty:
                temp = member_commutes.copy()
                temp["housing_penalty_scenario"] = scenario_name
                member_rows.extend(temp.to_dict("records"))
            summary = summarize_commutes(member_commutes, household, area_id)
            summary["housing_penalty_scenario"] = scenario_name
            commute_rows.append(summary)
    reward_surface = pd.DataFrame(surface_rows)
    commute_summary = pd.DataFrame(commute_rows)
    member_commutes = pd.DataFrame(member_rows)
    baselines = run_household_baselines(reward_surface)
    baselines["housing_penalty_scenario"] = scenario_name
    recommendations = _recommendation_table(baselines, reward_surface)
    return reward_surface, commute_summary, member_commutes, recommendations


def _profile_summary(households: dict) -> pd.DataFrame:
    rows = []
    for household in households.values():
        rows.append({
            "household_id": household.household_id,
            "household_type": household.household_type,
            "household_income": household.household_income,
            "housing_budget": household.housing_budget,
            "car_count": household.car_count,
            "num_workers": household.num_workers,
            "num_children": household.num_children,
            "num_elderly": household.num_elderly,
            "care_responsibility_level": household.care_responsibility_level,
            "preferred_modes": "|".join(household.preferred_modes),
            "workplace_assumptions": "; ".join(household.workplace_assumptions),
        })
    return pd.DataFrame(rows)


def _shift_summary(step14_path: Path, all_recommendations: pd.DataFrame) -> pd.DataFrame:
    weighted = all_recommendations[all_recommendations["method"] == "household_weighted_score"].copy()
    rows = []
    step14 = pd.DataFrame()
    if step14_path.exists():
        step14 = pd.read_csv(step14_path)
        step14 = step14[step14["method"] == "household_weighted_score"]
    for _, row in weighted.iterrows():
        base_type = str(row["household_id"])
        for suffix in ["_bk", "_mn", "_qn", "_bk2", "_one_car_mn_bk", "_no_car_mn_qn", "_one_car_bk_bk", "_no_car_bk_mn", "_mn_qn", "_bk_mn", "_bk_bk", "_qn_bk", "_mn_low", "_bk_mid", "_qn_mid", "_low_bk", "_mid_mn", "_mid_bk", "_low_qn", "_bk_ridehail", "_mn_ridehail", "_qn_transit", "_bk_car"]:
            if base_type.endswith(suffix):
                base_type = base_type[: -len(suffix)]
                break
        old_area = None
        if not step14.empty:
            match = step14[step14["household_id"].astype(str).str.startswith(base_type)]
            if not match.empty:
                old_area = match.iloc[0]["area_id"]
        rows.append({
            "housing_penalty_scenario": row["housing_penalty_scenario"],
            "household_id": row["household_id"],
            "household_type_inferred": base_type,
            "step14_reference_area": old_area,
            "step15_area": row["area_id"],
            "changed_from_step14_reference": bool(old_area is not None and old_area != row["area_id"]),
            "reward": row["reward"],
            "rent_income_ratio": row.get("rent_income_ratio"),
            "housing_cost": row.get("housing_cost"),
            "effective_housing_cost": row.get("effective_housing_cost"),
        })
    return pd.DataFrame(rows)


def _write_step15_reports(output: Path, households: dict, all_recommendations: pd.DataFrame, all_decomposition: pd.DataFrame, shift: pd.DataFrame, profile_summary: pd.DataFrame, step14_reference: pd.DataFrame) -> None:
    weighted = all_recommendations[all_recommendations["method"] == "household_weighted_score"]
    selection_counts = weighted.groupby("housing_penalty_scenario")["area_id"].nunique().reset_index(name="unique_selected_areas")
    low_income = weighted[weighted["household_id"].str.contains("low_income|single_parent|elderly_care", regex=True)]
    strong = weighted[weighted["housing_penalty_scenario"] == "strong_affordability_penalty"]
    old_all_bk0202 = False
    if not step14_reference.empty:
        old_weighted = step14_reference[step14_reference["method"] == "household_weighted_score"]
        old_all_bk0202 = bool((old_weighted["area_id"] == "BK0202").all()) if not old_weighted.empty else False
    reports = {
        "household_step15_executive_summary.md": f"""# Step 15 household affordability sensitivity executive summary

Step 15 expands the household experiment from 6 to {len(households)} representative scenario households and evaluates three housing-penalty settings: baseline-like, medium, and strong. The expanded profiles vary income, housing budget, workplace area, car ownership, care burden, children, elderly members, and night-shift constraints.

- Step 14 reference all weighted households selected BK0202: {old_all_bk0202}.
- Unique selected areas by scenario:

{markdown_table(selection_counts)}

These are deterministic household-aware sensitivity results, not trained household-level RL/PPO results.
""",
        "household_affordability_sensitivity_analysis.md": f"""# Household affordability sensitivity analysis

The housing penalty now separates budget overage, rent-income burden, missing housing-cost handling, and low-income extra penalty. Zero housing cost is treated as missing and imputed to the nonzero median with an added missing-data penalty, so affordability-first cannot treat missing cost as free housing.

## Housing penalty scenarios

{markdown_table(pd.DataFrame([{'scenario': k, **v} for k, v in HOUSING_PENALTY_SCENARIOS.items()]))}

## Low-income and care-sensitive weighted recommendations

{markdown_table(low_income[['housing_penalty_scenario','household_id','area_id','reward','rent_income_ratio','housing_cost','housing_cost_penalty']].head(80))}
""",
        "household_recommendation_shift_analysis.md": f"""# Household recommendation shift analysis

This report compares Step 15 weighted recommendations against the Step 14 household weighted-score reference where a matching household type exists. It also reports shifts across housing penalty scenarios.

{markdown_table(shift.head(120))}
""",
        "household_scenario_expansion_methodology.md": f"""# Household scenario expansion methodology

The Step 15 profiles are representative scenario samples, not observed household microdata. Each Step 14 household category was expanded into multiple variants with different income, housing budget, car ownership, workplace area assumptions, schedules, and care burdens.

{markdown_table(profile_summary)}

Generated workplace assumptions are recorded in `household_expanded_profiles_summary.csv` and, where applicable, in `household_workplace_assumptions.csv`.
""",
        "household_low_income_case_analysis.md": f"""# Low-income household case analysis

Low-income and transit-dependent households receive stronger rent-burden penalties under the medium and strong scenarios. This makes high rent-income-ratio areas less attractive unless their accessibility and commute advantages are large enough to compensate.

{markdown_table(low_income[['housing_penalty_scenario','household_id','area_id','reward','rent_income_ratio','housing_cost_penalty','mean_commute_time','accessibility']].head(100))}
""",
        "household_limitations_step15.md": """# Step 15 limitations

- Expanded households are representative scenarios, not observed household microdata.
- Some workplace areas are generated assumptions; Manhattan/Queens workplace IDs may be mapped to available candidate-area centroids when not present in the current candidate subset.
- Area-to-workplace commute remains centroid-based, not full routed travel time.
- Shared resource accessibility inherits Step 12 GTFS approximation and partial TLC/HVFHV observed road-time coverage.
- Stronger housing penalties are sensitivity settings, not calibrated behavioral estimates.
- Household-level RL/PPO remains future work; Step 15 is deterministic household-aware sensitivity analysis.
""",
    }
    for filename, text in reports.items():
        (output / filename).write_text(text, encoding="utf-8")


def _write_step15_figures(output: Path, all_recommendations: pd.DataFrame, all_decomposition: pd.DataFrame, shift: pd.DataFrame) -> None:
    figs = output
    weighted = all_recommendations[all_recommendations["method"] == "household_weighted_score"].copy()
    if weighted.empty:
        return
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 8})
    def save(name):
        plt.tight_layout(); plt.savefig(figs / name, bbox_inches="tight"); plt.close()
    pivot = weighted.pivot_table(index="household_id", columns="housing_penalty_scenario", values="area_id", aggfunc="first")
    change_count = pivot.nunique(axis=1).sort_values()
    plt.figure(figsize=(9, 7)); plt.barh(change_count.index, change_count.values, color="#4C78A8"); plt.xlabel("Unique recommended areas across scenarios"); plt.title("Step 15 recommendation shift by housing penalty"); save("step15_recommendation_shift_by_penalty.png")

    type_reward = weighted.assign(household_type=weighted["household_id"].str.replace(r"_(bk|mn|qn|bk2|one_car_mn_bk|no_car_mn_qn|one_car_bk_bk|no_car_bk_mn|mn_qn|bk_mn|bk_bk|qn_bk|mn_low|bk_mid|qn_mid|low_bk|mid_mn|mid_bk|low_qn|bk_ridehail|mn_ridehail|qn_transit|bk_car|low|mid)$", "", regex=True)).groupby(["housing_penalty_scenario","household_type"])["reward"].mean().reset_index()
    plt.figure(figsize=(11, 6));
    for scenario, df in type_reward.groupby("housing_penalty_scenario"):
        plt.plot(df["household_type"], df["reward"], marker="o", label=scenario)
    plt.xticks(rotation=35, ha="right"); plt.ylabel("Mean reward"); plt.title("Step 15 reward by household type"); plt.legend(fontsize=7); save("step15_reward_by_household_type.png")

    plt.figure(figsize=(8, 5));
    for scenario, df in weighted.groupby("housing_penalty_scenario"):
        plt.scatter(df["rent_income_ratio"], df["reward"], label=scenario, alpha=0.75)
    plt.xlabel("Rent-income ratio"); plt.ylabel("Reward"); plt.title("Rent burden vs reward"); plt.legend(fontsize=7); save("step15_rent_burden_vs_reward.png")

    penalty = all_decomposition[all_decomposition["area_id"].isin(weighted["area_id"])].groupby(["housing_penalty_scenario","household_id"])["housing_cost_penalty"].mean().reset_index()
    plt.figure(figsize=(10, 7));
    for scenario, df in penalty.groupby("housing_penalty_scenario"):
        plt.plot(df["household_id"], df["housing_cost_penalty"], marker="o", label=scenario)
    plt.xticks(rotation=75, ha="right"); plt.ylabel("Housing cost penalty"); plt.title("Affordability penalty effect"); plt.legend(fontsize=7); save("step15_affordability_penalty_effect.png")

    freq = weighted.groupby(["housing_penalty_scenario","area_id"]).size().reset_index(name="count")
    plt.figure(figsize=(9, 5));
    for scenario, df in freq.groupby("housing_penalty_scenario"):
        plt.bar(df["area_id"].astype(str) + "\n" + scenario.split("_")[0], df["count"], alpha=0.75, label=scenario)
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Selections"); plt.title("Step 15 area selection frequency"); save("step15_area_selection_frequency.png")

    low = weighted[weighted["household_id"].str.contains("low_income|single_parent", regex=True)]
    plt.figure(figsize=(9, 5));
    for scenario, df in low.groupby("housing_penalty_scenario"):
        plt.scatter(df["rent_income_ratio"], df["housing_cost_penalty"], label=scenario, alpha=0.8)
    plt.xlabel("Rent-income ratio"); plt.ylabel("Housing penalty"); plt.title("Low-income recommendation affordability response"); plt.legend(fontsize=7); save("step15_low_income_recommendation_change.png")


def _write_step15_outputs(output: Path, all_recommendations: pd.DataFrame, all_decomposition: pd.DataFrame, all_commutes: pd.DataFrame, all_member_commutes: pd.DataFrame, households: dict) -> None:
    weighted = all_recommendations[all_recommendations["method"] == "household_weighted_score"].copy()
    baseline_like = weighted[weighted["housing_penalty_scenario"] == "baseline_step14_like"][["household_id", "area_id", "reward", "rent_income_ratio", "housing_cost_penalty"]].rename(columns={"area_id": "baseline_area", "reward": "baseline_reward", "rent_income_ratio": "baseline_rent_income_ratio", "housing_cost_penalty": "baseline_housing_penalty"})
    comparison_rows = []
    for scenario in SENSITIVITY_ORDER:
        sc = weighted[weighted["housing_penalty_scenario"] == scenario][["household_id", "area_id", "reward", "rent_income_ratio", "housing_cost_penalty"]].rename(columns={"area_id": "scenario_area", "reward": "scenario_reward", "rent_income_ratio": "scenario_rent_income_ratio", "housing_cost_penalty": "scenario_housing_penalty"})
        comp = baseline_like.merge(sc, on="household_id", how="right")
        comp["housing_penalty_scenario"] = scenario
        comp["area_changed_from_baseline_like"] = comp["baseline_area"] != comp["scenario_area"]
        comparison_rows.append(comp)
    affordability_comparison = pd.concat(comparison_rows, ignore_index=True)
    shift = _shift_summary(Path("outputs/real_step14_household/household_recommendations.csv"), all_recommendations)
    profile_summary = _profile_summary(households)
    penalty_by_household = weighted[["housing_penalty_scenario", "household_id", "area_id", "reward", "housing_cost", "effective_housing_cost", "rent_income_ratio", "housing_cost_penalty", "housing_budget_overage"]].copy()

    outputs = {
        "household_recommendations_step15.csv": all_recommendations,
        "household_reward_decomposition_step15.csv": all_decomposition,
        "household_affordability_sensitivity_comparison.csv": affordability_comparison,
        "household_recommendation_shift_summary.csv": shift,
        "household_expanded_profiles_summary.csv": profile_summary,
        "household_member_commute_times_step15.csv": all_member_commutes,
        "household_baseline_comparison_step15.csv": all_recommendations,
        "housing_penalty_sensitivity_by_household.csv": penalty_by_household,
    }
    for filename, df in outputs.items():
        df.to_csv(output / filename, index=False)
    _write_step15_reports(output, households, all_recommendations, all_decomposition, shift, profile_summary, pd.read_csv("outputs/real_step14_household/household_recommendations.csv") if Path("outputs/real_step14_household/household_recommendations.csv").exists() else pd.DataFrame())
    _write_step15_figures(output, all_recommendations, all_decomposition, shift)


def run_household_experiment(
    areas: pd.DataFrame,
    resources: pd.DataFrame,
    travel_times: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
    household_profiles_path: str | Path = "data/profiles/household_profiles.csv",
    household_members_path: str | Path = "data/profiles/household_members.csv",
    metadata: dict[str, Any] | None = None,
    housing_penalty_scenario: str = "baseline_step14_like",
    sensitivity_run: bool = False,
) -> dict[str, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    households = load_household_profiles(household_profiles_path, household_members_path, candidate_areas=areas)
    theta = config.get("sample_data", {}).get("theta_by_category", {})
    scenario_names = SENSITIVITY_ORDER if sensitivity_run or housing_penalty_scenario == "all" else [housing_penalty_scenario]
    access_cache = build_household_accessibility_cache(resources, travel_times, theta)

    surfaces = []; commutes = []; member_tables = []; recommendations = []
    for scenario in scenario_names:
        surface, commute_summary, member_commutes, recs = _compute_outputs_for_scenario(households, areas, resources, travel_times, theta, scenario, access_cache=access_cache)
        surfaces.append(surface); commutes.append(commute_summary); member_tables.append(member_commutes); recommendations.append(recs)

    all_decomposition = pd.concat(surfaces, ignore_index=True)
    all_commutes = pd.concat(commutes, ignore_index=True)
    all_member_commutes = pd.concat(member_tables, ignore_index=True) if member_tables else pd.DataFrame()
    all_recommendations = pd.concat(recommendations, ignore_index=True)

    primary_scenario = "medium_affordability_penalty" if "medium_affordability_penalty" in scenario_names else scenario_names[0]
    decomposition = all_decomposition[all_decomposition["housing_penalty_scenario"] == primary_scenario].copy()
    commute_summary = all_commutes[all_commutes["housing_penalty_scenario"] == primary_scenario].copy()
    member_commutes = all_member_commutes[all_member_commutes["housing_penalty_scenario"] == primary_scenario].copy() if not all_member_commutes.empty else all_member_commutes
    recommendations_primary = all_recommendations[all_recommendations["housing_penalty_scenario"] == primary_scenario].copy()
    area_diagnostics = _household_area_diagnostics(decomposition, recommendations_primary)

    for filename, df in {
        "household_recommendations.csv": recommendations_primary,
        "household_reward_decomposition.csv": decomposition,
        "household_commute_summary.csv": commute_summary,
        "household_baseline_comparison.csv": recommendations_primary,
        "household_area_diagnostics.csv": area_diagnostics,
        "household_member_commute_times.csv": member_commutes,
    }.items():
        df.to_csv(output / filename, index=False)

    assumptions = []
    for household in households.values():
        for assumption in household.workplace_assumptions:
            assumptions.append({"household_id": household.household_id, "assumption": assumption})
    if assumptions:
        pd.DataFrame(assumptions).to_csv(output / "household_workplace_assumptions.csv", index=False)

    write_household_reports(output, households, recommendations_primary, decomposition, commute_summary, recommendations_primary, area_diagnostics, member_commutes, metadata=metadata)
    if len(scenario_names) > 1:
        _write_step15_outputs(output, all_recommendations, all_decomposition, all_commutes, all_member_commutes, households)

    print("\nHousehold-aware Activity Area Recommendation")
    print(f"Households: {len(households)} | Candidate areas: {len(areas)} | Resources: {len(resources)} | Housing scenario: {primary_scenario}")
    weighted = recommendations_primary[recommendations_primary["method"] == "household_weighted_score"]
    print(weighted[["household_id", "area_id", "reward", "accessibility", "rent_income_ratio", "housing_cost_penalty"]].to_string(index=False))
    print(f"Household outputs saved to {output}/")

    return {
        "recommendations": recommendations_primary,
        "reward_decomposition": decomposition,
        "commute_summary": commute_summary,
        "baseline_comparison": recommendations_primary,
        "area_diagnostics": area_diagnostics,
        "member_commutes": member_commutes,
        "all_recommendations": all_recommendations,
        "all_decomposition": all_decomposition,
    }

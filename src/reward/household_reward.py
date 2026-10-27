"""Household-aware reward and commute calculations."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.profiles.household_profiles import HouseholdMember, HouseholdProfile

EARTH_RADIUS_M = 6371000.0

DEFAULT_HOUSING_PENALTY_CONFIG = {
    "housing_penalty_multiplier": 1.0,
    "rent_burden_soft_threshold": 0.30,
    "rent_burden_hard_threshold": 0.50,
    "low_income_extra_penalty_multiplier": 1.0,
    "zero_housing_cost_missing_policy": "impute_median_plus_penalty",
    "missing_housing_cost_penalty": 0.25,
}

HOUSING_PENALTY_SCENARIOS = {
    "baseline_step14_like": {
        "housing_penalty_multiplier": 1.0,
        "rent_burden_soft_threshold": 0.30,
        "rent_burden_hard_threshold": 0.50,
        "low_income_extra_penalty_multiplier": 1.0,
        "zero_housing_cost_missing_policy": "impute_median_plus_penalty",
        "missing_housing_cost_penalty": 0.15,
    },
    "medium_affordability_penalty": {
        "housing_penalty_multiplier": 2.2,
        "rent_burden_soft_threshold": 0.30,
        "rent_burden_hard_threshold": 0.50,
        "low_income_extra_penalty_multiplier": 1.6,
        "zero_housing_cost_missing_policy": "impute_median_plus_penalty",
        "missing_housing_cost_penalty": 0.35,
    },
    "strong_affordability_penalty": {
        "housing_penalty_multiplier": 4.0,
        "rent_burden_soft_threshold": 0.30,
        "rent_burden_hard_threshold": 0.50,
        "low_income_extra_penalty_multiplier": 2.4,
        "zero_housing_cost_missing_policy": "impute_median_plus_penalty",
        "missing_housing_cost_penalty": 0.60,
    },
}


def housing_penalty_config(name: str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_HOUSING_PENALTY_CONFIG)
    if name:
        config.update(HOUSING_PENALTY_SCENARIOS.get(name, {}))
    if overrides:
        config.update(overrides)
    return config


def haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _area_lon_lat(area: pd.Series) -> tuple[float, float]:
    lon_col = "lon" if "lon" in area.index else "centroid_x"
    lat_col = "lat" if "lat" in area.index else "centroid_y"
    return float(area[lon_col]), float(area[lat_col])


def _mode_time(distance_meters: float, mode: str, schedule: str = "day") -> float:
    detour = 1.25
    mode = (mode or "transit").lower()
    if mode == "walk":
        return distance_meters * detour / 80.0
    if mode == "bike":
        return distance_meters * 1.20 / 220.0
    if mode in {"car", "ridehail"}:
        minutes = distance_meters * 1.35 / 420.0
        if mode == "ridehail":
            minutes += 6.0 if schedule != "night" else 8.0
        return minutes
    wait = 6.0 if schedule != "night" else 12.0
    return distance_meters * 1.35 / 365.0 + wait + 5.0


def _member_mode(member: HouseholdMember, household: HouseholdProfile) -> tuple[str, float]:
    preferred = (member.mode_preference or "transit").lower()
    schedule = (member.work_schedule or "day").lower()
    if preferred == "car" and household.car_count <= 0:
        return "transit", 12.0
    if preferred == "ridehail":
        if household.household_income >= 90000 or schedule == "night" or household.num_elderly:
            return "ridehail", 4.0 if household.household_income >= 150000 else 12.0
        return "transit", 10.0
    if preferred not in household.preferred_modes and preferred not in {"walk", "transit"}:
        return "transit", 6.0
    return preferred, 0.0


def compute_member_commutes(area: pd.Series, household: HouseholdProfile) -> pd.DataFrame:
    origin_lon, origin_lat = _area_lon_lat(area)
    rows: list[dict[str, Any]] = []
    for member in household.members:
        if not member.employed:
            continue
        if member.workplace_lon is None or member.workplace_lat is None:
            continue
        mode, mode_penalty = _member_mode(member, household)
        distance = haversine_meters(origin_lon, origin_lat, member.workplace_lon, member.workplace_lat)
        raw_time = _mode_time(distance, mode, member.work_schedule)
        over_tolerance = raw_time + mode_penalty > household.commute_tolerance_minutes
        rows.append(
            {
                "household_id": household.household_id,
                "member_id": member.member_id,
                "role": member.role,
                "area_id": area["area_id"],
                "workplace_area_id": member.workplace_area_id,
                "work_schedule": member.work_schedule,
                "selected_mode": mode,
                "distance_meters": round(distance, 3),
                "raw_commute_time_minutes": round(raw_time, 3),
                "mode_penalty_minutes": round(mode_penalty, 3),
                "commute_time_minutes": round(raw_time + mode_penalty, 3),
                "commute_weight": member.commute_weight,
                "over_tolerance": bool(over_tolerance),
                "source": "HOUSEHOLD_AREA_TO_WORKPLACE_CENTROID_APPROX",
            }
        )
    return pd.DataFrame(rows)


def summarize_commutes(commutes: pd.DataFrame, household: HouseholdProfile, area_id: str) -> dict[str, float | int | str]:
    if commutes.empty:
        return {"household_id": household.household_id, "area_id": area_id, "mean_commute_time": 0.0, "max_commute_time": 0.0, "weighted_commute_time": 0.0, "commute_inequality": 0.0, "workers_over_tolerance_count": 0}
    weights = pd.to_numeric(commutes["commute_weight"], errors="coerce").fillna(1.0).clip(lower=0.01)
    times = pd.to_numeric(commutes["commute_time_minutes"], errors="coerce").fillna(0.0)
    return {
        "household_id": household.household_id,
        "area_id": area_id,
        "mean_commute_time": float(times.mean()),
        "max_commute_time": float(times.max()),
        "weighted_commute_time": float((times * weights).sum() / weights.sum()),
        "commute_inequality": float(times.std(ddof=0)) if len(times) > 1 else 0.0,
        "workers_over_tolerance_count": int(commutes["over_tolerance"].sum()),
    }


def _travel_time_for_area(travel_times: pd.DataFrame, area_id: str) -> pd.DataFrame:
    rows = travel_times[travel_times["area_id"] == area_id].copy()
    if rows.empty:
        return rows
    time_cols = [col for col in ["transit_time", "car_time", "walk_time", "bike_time"] if col in rows.columns]
    if not time_cols:
        return rows
    rows["household_time"] = rows[time_cols].apply(pd.to_numeric, errors="coerce").min(axis=1)
    keep_cols = ["area_id", "resource_id", "category", "household_time"]
    return rows[keep_cols].groupby(["area_id", "resource_id", "category"], as_index=False)["household_time"].min()



def build_household_accessibility_cache(resources: pd.DataFrame, travel_times: pd.DataFrame, theta_by_category: dict[str, float]) -> dict[str, dict[str, float]]:
    res = resources.copy()
    quality_col = "quality" if "quality" in res.columns else "quality_score"
    if quality_col not in res.columns:
        res[quality_col] = 1.0
    cache: dict[str, dict[str, float]] = {}
    for area_id in sorted(travel_times["area_id"].dropna().astype(str).unique()):
        area_times = _travel_time_for_area(travel_times, area_id)
        if area_times.empty:
            cache[area_id] = {}
            continue
        merged = area_times.merge(res, on=["resource_id", "category"], how="inner")
        scores: dict[str, float] = {}
        for category, rows in merged.groupby("category"):
            theta = float(theta_by_category.get(category, 20.0))
            score = 0.0
            for _, row in rows.iterrows():
                score += float(row.get(quality_col, 1.0)) * math.exp(-float(row["household_time"]) / theta)
            scores[str(category)] = float(score)
        cache[area_id] = scores
    return cache


def compute_household_resource_accessibility(area_id: str, household: HouseholdProfile, resources: pd.DataFrame, travel_times: pd.DataFrame, theta_by_category: dict[str, float], access_cache: dict[str, dict[str, float]] | None = None) -> tuple[float, dict[str, float]]:
    weights = household.activity_weights()
    raw_scores = access_cache.get(str(area_id), {}) if access_cache is not None else None
    if raw_scores is None:
        raw_scores = build_household_accessibility_cache(resources, travel_times[travel_times["area_id"] == area_id], theta_by_category).get(str(area_id), {})
    by_category: dict[str, float] = {}
    total = 0.0
    for category, weight in weights.items():
        value = float(weight) * float(raw_scores.get(category, 0.0))
        by_category[category] = value
        total += value
    return float(total), by_category


def _normalize(value: float, upper: float) -> float:
    if upper <= 0:
        return 0.0
    return max(0.0, min(1.0, value / upper))


def _median_nonzero_housing_cost(areas: pd.DataFrame) -> float:
    if "housing_cost" not in areas.columns:
        return 0.0
    costs = pd.to_numeric(areas["housing_cost"], errors="coerce")
    costs = costs[costs > 0]
    return float(costs.median()) if not costs.empty else 0.0


def _is_low_income(household: HouseholdProfile) -> bool:
    return household.household_income < 70000 or household.housing_budget < 1700


def compute_housing_cost_penalty(raw_housing_cost: float, household: HouseholdProfile, areas: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, float | bool]:
    cfg = housing_penalty_config(overrides=config)
    housing_cost_missing = raw_housing_cost <= 0
    effective_housing_cost = raw_housing_cost
    missing_penalty = 0.0
    if housing_cost_missing:
        policy = str(cfg.get("zero_housing_cost_missing_policy", "impute_median_plus_penalty"))
        if policy == "impute_median_plus_penalty":
            effective_housing_cost = _median_nonzero_housing_cost(areas)
            missing_penalty = float(cfg.get("missing_housing_cost_penalty", 0.25))
        elif policy == "penalty_only":
            missing_penalty = float(cfg.get("missing_housing_cost_penalty", 0.25))
        else:
            effective_housing_cost = _median_nonzero_housing_cost(areas)
    rent_income_ratio = (effective_housing_cost * 12.0) / max(household.household_income, 1.0)
    budget_overage = max(0.0, effective_housing_cost - household.housing_budget) / max(household.housing_budget, 1.0)
    soft = float(cfg.get("rent_burden_soft_threshold", 0.30))
    hard = float(cfg.get("rent_burden_hard_threshold", 0.50))
    if rent_income_ratio <= soft:
        rent_penalty = 0.20 * rent_income_ratio / max(soft, 0.01)
    elif rent_income_ratio <= hard:
        rent_penalty = 0.20 + 2.0 * ((rent_income_ratio - soft) / max(hard - soft, 0.01)) ** 1.4
    else:
        rent_penalty = 2.20 + 5.0 * ((rent_income_ratio - hard) / max(hard, 0.01)) ** 1.2
    base_penalty = budget_overage + rent_penalty + missing_penalty
    multiplier = float(cfg.get("housing_penalty_multiplier", 1.0))
    if _is_low_income(household):
        multiplier *= float(cfg.get("low_income_extra_penalty_multiplier", 1.0))
    penalty = household.affordability_sensitivity * multiplier * base_penalty
    return {
        "housing_cost": float(raw_housing_cost),
        "effective_housing_cost": float(effective_housing_cost),
        "housing_cost_missing": bool(housing_cost_missing),
        "rent_income_ratio": float(rent_income_ratio),
        "housing_budget_overage": float(budget_overage),
        "rent_burden_penalty_component": float(rent_penalty),
        "missing_housing_cost_penalty_component": float(missing_penalty),
        "housing_cost_penalty": float(penalty),
    }


def compute_household_reward(area_id: str, household: HouseholdProfile, areas: pd.DataFrame, resources: pd.DataFrame, travel_times: pd.DataFrame, theta_by_category: dict[str, float], housing_config: dict[str, Any] | None = None, access_cache: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    area = areas[areas["area_id"] == area_id].iloc[0]
    raw_housing_cost = float(area.get("housing_cost", 0.0) or 0.0)
    risk_score = float(area.get("risk_score", 0.0) or 0.0)
    transit_score = float(area.get("transit_score", area.get("transit_accessibility", 0.5)) or 0.5)
    access_total, access_by_category = compute_household_resource_accessibility(area_id, household, resources, travel_times, theta_by_category, access_cache=access_cache)
    member_commutes = compute_member_commutes(area, household)
    commute = summarize_commutes(member_commutes, household, area_id)

    child_access = access_by_category.get("school", 0.0) + 0.7 * access_by_category.get("park", 0.0) if household.num_children else 0.0
    elderly_access = access_by_category.get("hospital", 0.0) + 0.5 * access_by_category.get("public_service", 0.0) if household.num_elderly else 0.0
    daily_access = access_by_category.get("grocery", 0.0) + 0.5 * access_by_category.get("public_service", 0.0)
    employment_access = access_by_category.get("employment", 0.0) * max(1, household.num_workers)
    mobility = min(1.0, transit_score + 0.12 * household.car_count + (0.08 if "ridehail" in household.preferred_modes else 0.0))
    housing = compute_housing_cost_penalty(raw_housing_cost, household, areas, housing_config)
    care_coordination = household.care_responsibility_level * (0.6 * household.num_children + 0.8 * household.num_elderly) * _normalize(commute["max_commute_time"], 90.0)
    mode_infeasible = 0.0
    if household.num_workers > household.car_count and "car" in household.preferred_modes:
        mode_infeasible += _normalize(household.num_workers - household.car_count, max(1, household.num_workers)) * 0.4
    if household.car_count == 0 and commute["mean_commute_time"] > household.commute_tolerance_minutes:
        mode_infeasible += 0.4

    resource_accessibility_benefit = access_total
    household_activity_fit = min(1.5, child_access + elderly_access + daily_access)
    mobility_feasibility = mobility
    equity_adjustment = 0.25 if _is_low_income(household) and raw_housing_cost > 0 and raw_housing_cost <= household.housing_budget else 0.0
    mean_commute_penalty = _normalize(commute["mean_commute_time"], 75.0)
    max_commute_penalty = _normalize(commute["max_commute_time"], 90.0)
    commute_inequality_penalty = _normalize(commute["commute_inequality"], 35.0)
    risk_penalty = risk_score * (1.0 + 0.15 * household.num_children + 0.2 * household.num_elderly)

    final_reward = resource_accessibility_benefit + household_activity_fit + mobility_feasibility + equity_adjustment - float(housing["housing_cost_penalty"]) - mean_commute_penalty - max_commute_penalty - commute_inequality_penalty - care_coordination - mode_infeasible - risk_penalty
    return {
        "household_id": household.household_id,
        "area_id": area_id,
        "resource_accessibility_benefit": float(resource_accessibility_benefit),
        "household_activity_fit": float(household_activity_fit),
        "child_accessibility": float(child_access),
        "elderly_medical_accessibility": float(elderly_access),
        "daily_service_accessibility": float(daily_access),
        "employment_accessibility": float(employment_access),
        "mobility_feasibility": float(mobility_feasibility),
        "equity_adjustment": float(equity_adjustment),
        "housing_cost_penalty": float(housing["housing_cost_penalty"]),
        "mean_commute_penalty": float(mean_commute_penalty),
        "max_commute_penalty": float(max_commute_penalty),
        "commute_inequality_penalty": float(commute_inequality_penalty),
        "care_coordination_penalty": float(care_coordination),
        "mode_infeasibility_penalty": float(mode_infeasible),
        "risk_penalty": float(risk_penalty),
        "final_reward": float(final_reward),
        "accessibility_score": float(access_total),
        **housing,
        **commute,
        "accessibility_by_category": access_by_category,
    }


def compute_household_rewards_for_all(household: HouseholdProfile, areas: pd.DataFrame, resources: pd.DataFrame, travel_times: pd.DataFrame, theta_by_category: dict[str, float], housing_config: dict[str, Any] | None = None, access_cache: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    rows = [compute_household_reward(area_id, household, areas, resources, travel_times, theta_by_category, housing_config=housing_config, access_cache=access_cache) for area_id in areas["area_id"]]
    return pd.DataFrame(rows)

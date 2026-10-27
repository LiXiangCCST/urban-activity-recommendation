"""Profile-based effective multimodal travel times."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.profiles.individual_profiles import IndividualProfile, get_profiles
from src.travel.utils import ensure_dir, period_for_profile


def _income_bucket(profile: IndividualProfile) -> str:
    if profile.income_level.startswith("low"):
        return "low"
    if profile.income_level == "high":
        return "high"
    return "middle"


def _ridehail_allowed(profile: IndividualProfile, activity_type: str) -> bool:
    if profile.profile_id == "transit_dependent_resident":
        return False
    if profile.income_level == "high":
        return True
    if profile.profile_id == "night_shift_worker":
        return True
    if profile.profile_id == "elderly_medical_needs" and activity_type == "hospital":
        return True
    return False


def _cost_penalty(profile: IndividualProfile, mode: str) -> float:
    bucket = _income_bucket(profile)
    if mode == "ridehail":
        return {"low": 25.0, "middle": 12.0, "high": 4.0}[bucket]
    if mode == "car":
        return 3.0 if profile.has_car else 999.0
    return 0.0


def _availability_penalty(profile: IndividualProfile, mode: str, walking_time: float, activity_type: str) -> float:
    if mode == "walk" and walking_time > 15.0:
        if profile.profile_id == "elderly_medical_needs":
            return 20.0
        return 10.0
    if mode == "transit" and "transit" not in profile.preferred_modes:
        return 8.0
    if mode == "ridehail" and not _ridehail_allowed(profile, activity_type):
        return 999.0
    return 0.0


def build_effective_travel_times(
    walking: pd.DataFrame,
    transit: pd.DataFrame,
    road: pd.DataFrame,
    processed_dir: str | Path,
    profiles: dict[str, IndividualProfile] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = ensure_dir(processed_dir)
    profiles = profiles or get_profiles()
    base = walking[["area_id", "resource_id", "activity_type", "walking_time_minutes"]].copy()
    rows = []
    for profile in profiles.values():
        period = period_for_profile(profile.profile_id)
        t = transit[transit["departure_period"] == period][["area_id", "resource_id", "transit_time_minutes", "source"]].rename(columns={"source": "transit_source"}) if not transit.empty else pd.DataFrame(columns=["area_id", "resource_id", "transit_time_minutes", "transit_source"])
        r = road[road["departure_period"] == period][["area_id", "resource_id", "travel_time_minutes", "source"]].rename(columns={"travel_time_minutes": "road_time_minutes", "source": "road_source"}) if not road.empty else pd.DataFrame(columns=["area_id", "resource_id", "road_time_minutes", "road_source"])
        merged = base.merge(t, on=["area_id", "resource_id"], how="left").merge(r, on=["area_id", "resource_id"], how="left")
        merged["transit_time_minutes"] = pd.to_numeric(merged["transit_time_minutes"], errors="coerce")
        merged["road_time_minutes"] = pd.to_numeric(merged["road_time_minutes"], errors="coerce")
        for _, row in merged.iterrows():
            choices = []
            walk = float(row["walking_time_minutes"])
            choices.append(("walk", walk, _cost_penalty(profile, "walk"), _availability_penalty(profile, "walk", walk, row["activity_type"]), "WALK_HAVERSINE_DETOUR_APPROX"))
            if pd.notna(row["transit_time_minutes"]):
                tr = float(row["transit_time_minutes"])
                choices.append(("transit", tr, _cost_penalty(profile, "transit"), _availability_penalty(profile, "transit", walk, row["activity_type"]), str(row.get("transit_source") or "GTFS_APPROX")))
            if pd.notna(row["road_time_minutes"]):
                rd = float(row["road_time_minutes"])
                if profile.has_car:
                    choices.append(("car", rd, _cost_penalty(profile, "car"), _availability_penalty(profile, "car", walk, row["activity_type"]), str(row.get("road_source") or "ROAD_DISTANCE_FALLBACK")))
                choices.append(("ridehail", rd, _cost_penalty(profile, "ridehail"), _availability_penalty(profile, "ridehail", walk, row["activity_type"]), str(row.get("road_source") or "ROAD_DISTANCE_FALLBACK")))
            feasible = [(mode, raw, cost, avail, source) for mode, raw, cost, avail, source in choices if avail < 900 and cost < 900]
            if not feasible:
                feasible = choices[:1]
            mode, raw, cost, avail, source = min(feasible, key=lambda item: item[1] + item[2] + item[3])
            rows.append({"profile_id": profile.profile_id, "area_id": row["area_id"], "resource_id": row["resource_id"], "activity_type": row["activity_type"], "departure_period": period, "selected_mode": mode, "effective_travel_time_minutes": round(raw + cost + avail, 3), "raw_travel_time_minutes": round(raw, 3), "walking_time_minutes": round(walk, 3), "transit_time_minutes": round(float(row["transit_time_minutes"]), 3) if pd.notna(row["transit_time_minutes"]) else pd.NA, "road_time_minutes": round(float(row["road_time_minutes"]), 3) if pd.notna(row["road_time_minutes"]) else pd.NA, "cost_penalty_minutes": cost, "availability_penalty_minutes": avail, "source": f"EFFECTIVE_MULTIMODAL::{source}"})
    effective = pd.DataFrame(rows)
    effective.to_csv(output / "effective_travel_times.csv", index=False)

    # Existing recommendation code expects walk_time/transit_time/car_time columns. Duplicate the selected effective time across modal columns and preserve profile_id for profile-specific filtering.
    compat = effective[["profile_id", "area_id", "resource_id", "activity_type", "effective_travel_time_minutes"]].rename(columns={"activity_type": "category"}).copy()
    compat["walk_time"] = compat["effective_travel_time_minutes"]
    compat["transit_time"] = compat["effective_travel_time_minutes"]
    compat["car_time"] = compat["effective_travel_time_minutes"]
    compat["bike_time"] = compat["effective_travel_time_minutes"]
    compat["distance_meters"] = pd.NA
    compat = compat[["profile_id", "area_id", "resource_id", "category", "walk_time", "bike_time", "car_time", "transit_time", "distance_meters"]]
    compat.to_csv(output / "effective_travel_times_wide.csv", index=False)
    return effective, compat

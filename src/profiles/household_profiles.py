"""Household profile loading and validation for household-level recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

HOUSEHOLD_REQUIRED_COLUMNS = {
    "household_id",
    "household_type",
    "household_income",
    "housing_budget",
    "car_count",
    "num_adults",
    "num_workers",
    "num_children",
    "num_elderly",
    "preferred_modes",
    "commute_tolerance_minutes",
    "affordability_sensitivity",
    "care_responsibility_level",
}

MEMBER_REQUIRED_COLUMNS = {
    "household_id",
    "member_id",
    "role",
    "employed",
    "workplace_area_id",
    "workplace_lon",
    "workplace_lat",
    "work_schedule",
    "mode_preference",
    "commute_weight",
    "activity_needs",
}


@dataclass(frozen=True)
class HouseholdMember:
    household_id: str
    member_id: str
    role: str
    employed: bool
    workplace_area_id: str | None
    workplace_lon: float | None
    workplace_lat: float | None
    work_schedule: str
    mode_preference: str
    commute_weight: float
    activity_needs: list[str]


@dataclass(frozen=True)
class HouseholdProfile:
    household_id: str
    household_type: str
    household_income: float
    housing_budget: float
    car_count: int
    num_adults: int
    num_workers: int
    num_children: int
    num_elderly: int
    preferred_modes: list[str]
    commute_tolerance_minutes: float
    affordability_sensitivity: float
    care_responsibility_level: float
    members: list[HouseholdMember] = field(default_factory=list)
    workplace_assumptions: list[str] = field(default_factory=list)

    @property
    def has_car(self) -> bool:
        return self.car_count > 0

    @property
    def household_size(self) -> int:
        return max(1, self.num_adults + self.num_children + self.num_elderly)

    @property
    def workers(self) -> list[HouseholdMember]:
        return [member for member in self.members if member.employed]

    def activity_weights(self) -> dict[str, float]:
        weights = {
            "grocery": 0.18,
            "public_service": 0.10,
            "transit": 0.10,
            "employment": 0.12 if self.num_workers else 0.04,
            "school": 0.06,
            "park": 0.08,
            "hospital": 0.08,
            "leisure": 0.08,
        }
        if self.num_children:
            weights["school"] += 0.16
            weights["park"] += 0.10
            weights["public_service"] += 0.04
        if self.num_elderly:
            weights["hospital"] += 0.20
            weights["public_service"] += 0.06
        if self.num_workers:
            weights["employment"] += 0.10 + 0.04 * min(self.num_workers, 3)
        total = sum(weights.values()) or 1.0
        return {key: value / total for key, value in weights.items()}


def _split_pipe(value: Any) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _optional_float(value: Any) -> float | None:
    if pd.isna(value) or value == "":
        return None
    return float(value)


def validate_household_inputs(households: pd.DataFrame, members: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing_households = sorted(HOUSEHOLD_REQUIRED_COLUMNS - set(households.columns))
    missing_members = sorted(MEMBER_REQUIRED_COLUMNS - set(members.columns))
    if missing_households:
        errors.append(f"household_profiles.csv missing columns: {missing_households}")
    if missing_members:
        errors.append(f"household_members.csv missing columns: {missing_members}")
    if not errors:
        household_ids = set(households["household_id"].astype(str))
        member_ids = set(members["household_id"].astype(str))
        unknown = sorted(member_ids - household_ids)
        if unknown:
            errors.append(f"household_members.csv has unknown household_id values: {unknown}")
        if households["household_id"].duplicated().any():
            errors.append("household_profiles.csv contains duplicate household_id values")
    return errors


def _fallback_workplace_area(areas: pd.DataFrame, index: int) -> str | None:
    if areas.empty:
        return None
    employment_like = areas[areas["area_id"].astype(str).str.startswith(("MN", "BK", "QN"), na=False)]
    pool = employment_like if not employment_like.empty else areas
    return str(pool.iloc[index % len(pool)]["area_id"])


def _map_workplace(row: pd.Series, areas_by_id: dict[str, tuple[float, float]], fallback_area: str | None) -> tuple[str | None, float | None, float | None, str | None]:
    area_id = str(row.get("workplace_area_id") or "").strip() or None
    lon = _optional_float(row.get("workplace_lon"))
    lat = _optional_float(row.get("workplace_lat"))
    assumption = None
    if lon is not None and lat is not None:
        return area_id, lon, lat, None
    if area_id and area_id in areas_by_id:
        lon, lat = areas_by_id[area_id]
        return area_id, lon, lat, "workplace_area_id mapped to candidate area centroid"
    if _bool(row.get("employed")) and fallback_area and fallback_area in areas_by_id:
        lon, lat = areas_by_id[fallback_area]
        return fallback_area, lon, lat, "generated workplace_area_id from candidate areas because no valid workplace coordinate was provided"
    return area_id, lon, lat, None


def load_household_profiles(
    household_profiles_path: str | Path = "data/profiles/household_profiles.csv",
    household_members_path: str | Path = "data/profiles/household_members.csv",
    candidate_areas: pd.DataFrame | None = None,
) -> dict[str, HouseholdProfile]:
    households_df = pd.read_csv(household_profiles_path)
    members_df = pd.read_csv(household_members_path)
    errors = validate_household_inputs(households_df, members_df)
    if errors:
        raise ValueError("; ".join(errors))

    areas = candidate_areas if candidate_areas is not None else pd.DataFrame()
    areas_by_id = {
        str(row["area_id"]): (float(row["lon"] if "lon" in row else row.get("centroid_x", 0.0)), float(row["lat"] if "lat" in row else row.get("centroid_y", 0.0)))
        for _, row in areas.iterrows()
        if pd.notna(row.get("area_id"))
    }

    member_objects: dict[str, list[HouseholdMember]] = {str(hid): [] for hid in households_df["household_id"]}
    assumptions: dict[str, list[str]] = {str(hid): [] for hid in households_df["household_id"]}
    for idx, row in members_df.iterrows():
        household_id = str(row["household_id"])
        fallback_area = _fallback_workplace_area(areas, idx)
        workplace_area_id, lon, lat, assumption = _map_workplace(row, areas_by_id, fallback_area)
        if assumption:
            assumptions[household_id].append(f"{row['member_id']}: {assumption}")
        member_objects[household_id].append(
            HouseholdMember(
                household_id=household_id,
                member_id=str(row["member_id"]),
                role=str(row["role"]),
                employed=_bool(row["employed"]),
                workplace_area_id=workplace_area_id,
                workplace_lon=lon,
                workplace_lat=lat,
                work_schedule=str(row.get("work_schedule") or "day"),
                mode_preference=str(row.get("mode_preference") or "transit"),
                commute_weight=float(row.get("commute_weight") or 0.0),
                activity_needs=_split_pipe(row.get("activity_needs")),
            )
        )

    profiles: dict[str, HouseholdProfile] = {}
    for _, row in households_df.iterrows():
        household_id = str(row["household_id"])
        profiles[household_id] = HouseholdProfile(
            household_id=household_id,
            household_type=str(row["household_type"]),
            household_income=float(row["household_income"]),
            housing_budget=float(row["housing_budget"]),
            car_count=int(row["car_count"]),
            num_adults=int(row["num_adults"]),
            num_workers=int(row["num_workers"]),
            num_children=int(row["num_children"]),
            num_elderly=int(row["num_elderly"]),
            preferred_modes=_split_pipe(row["preferred_modes"]),
            commute_tolerance_minutes=float(row["commute_tolerance_minutes"]),
            affordability_sensitivity=float(row["affordability_sensitivity"]),
            care_responsibility_level=float(row["care_responsibility_level"]),
            members=member_objects.get(household_id, []),
            workplace_assumptions=assumptions.get(household_id, []),
        )
    return profiles


def household_feature_vector(household: HouseholdProfile) -> dict[str, float]:
    workers = max(1, household.num_workers)
    return {
        "household_income": household.household_income,
        "housing_budget": household.housing_budget,
        "car_count": float(household.car_count),
        "cars_per_worker": household.car_count / workers,
        "num_workers": float(household.num_workers),
        "num_children": float(household.num_children),
        "num_elderly": float(household.num_elderly),
        "care_burden_index": household.care_responsibility_level * (household.num_children + household.num_elderly),
        "commute_tolerance_minutes": household.commute_tolerance_minutes,
        "affordability_sensitivity": household.affordability_sensitivity,
    }

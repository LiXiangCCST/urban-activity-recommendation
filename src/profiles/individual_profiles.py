"""Representative individual profiles used by the demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndividualProfile:
    profile_id: str
    income_level: str
    has_car: bool
    preferred_modes: list[str]
    activity_weights: dict[str, float]
    time_sensitivity: float
    max_affordable_housing_cost: float
    commute_weight: float
    risk_sensitivity: float

    def primary_mode(self) -> str:
        if self.has_car and "car" in self.preferred_modes:
            return "car"
        if "transit" in self.preferred_modes:
            return "transit"
        return "walk"


def get_profiles() -> dict[str, IndividualProfile]:
    return {
        "low_income_worker_without_car": IndividualProfile(
            profile_id="low_income_worker_without_car",
            income_level="low",
            has_car=False,
            preferred_modes=["transit", "walk"],
            activity_weights={"employment": 0.34, "transit": 0.22, "grocery": 0.14, "public_service": 0.10, "hospital": 0.08, "park": 0.05, "school": 0.03, "leisure": 0.04},
            time_sensitivity=1.25,
            max_affordable_housing_cost=1200,
            commute_weight=1.35,
            risk_sensitivity=0.8,
        ),
        "family_with_children": IndividualProfile(
            profile_id="family_with_children",
            income_level="middle",
            has_car=True,
            preferred_modes=["car", "walk", "transit"],
            activity_weights={"school": 0.25, "grocery": 0.18, "park": 0.16, "hospital": 0.12, "employment": 0.12, "transit": 0.07, "public_service": 0.06, "leisure": 0.04},
            time_sensitivity=1.0,
            max_affordable_housing_cost=1850,
            commute_weight=0.85,
            risk_sensitivity=1.2,
        ),
        "elderly_medical_needs": IndividualProfile(
            profile_id="elderly_medical_needs",
            income_level="low_middle",
            has_car=False,
            preferred_modes=["walk", "transit"],
            activity_weights={"hospital": 0.30, "grocery": 0.18, "park": 0.15, "public_service": 0.14, "transit": 0.11, "leisure": 0.05, "school": 0.02, "employment": 0.05},
            time_sensitivity=1.4,
            max_affordable_housing_cost=1450,
            commute_weight=0.45,
            risk_sensitivity=1.5,
        ),
        "high_income_commuter": IndividualProfile(
            profile_id="high_income_commuter",
            income_level="high",
            has_car=True,
            preferred_modes=["car", "transit"],
            activity_weights={"employment": 0.32, "transit": 0.18, "leisure": 0.15, "grocery": 0.10, "park": 0.10, "hospital": 0.05, "school": 0.04, "public_service": 0.06},
            time_sensitivity=1.15,
            max_affordable_housing_cost=2600,
            commute_weight=1.45,
            risk_sensitivity=0.6,
        ),
        "night_shift_worker": IndividualProfile(
            profile_id="night_shift_worker",
            income_level="middle",
            has_car=False,
            preferred_modes=["transit", "walk"],
            activity_weights={"employment": 0.30, "transit": 0.25, "grocery": 0.12, "hospital": 0.10, "public_service": 0.08, "leisure": 0.06, "park": 0.05, "school": 0.04},
            time_sensitivity=1.5,
            max_affordable_housing_cost=1550,
            commute_weight=1.5,
            risk_sensitivity=1.35,
        ),
        "transit_dependent_resident": IndividualProfile(
            profile_id="transit_dependent_resident",
            income_level="middle",
            has_car=False,
            preferred_modes=["transit", "walk"],
            activity_weights={"transit": 0.28, "employment": 0.20, "grocery": 0.15, "hospital": 0.10, "public_service": 0.10, "park": 0.08, "leisure": 0.06, "school": 0.03},
            time_sensitivity=1.3,
            max_affordable_housing_cost=1650,
            commute_weight=1.15,
            risk_sensitivity=0.9,
        ),
    }

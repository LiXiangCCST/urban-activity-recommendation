"""Gymnasium-style environment skeleton for area recommendation."""

from __future__ import annotations

import random
from typing import Any

import networkx as nx
import pandas as pd

from src.accessibility.reward import compute_reward
from src.graph.build_graph import area_neighbors
from src.graph.features import compute_area_features, feature_vector
from src.profiles.individual_profiles import IndividualProfile


class UrbanActivityRecommendationEnv:
    """Small environment that searches over candidate areas for one profile."""

    ACTION_STAY = "stay"
    ACTION_SELECT = "select_current_area"
    ACTION_JUMP_BEST = "jump_to_best_candidate"

    def __init__(
        self,
        graph: nx.Graph,
        profile: IndividualProfile,
        areas: pd.DataFrame,
        resources: pd.DataFrame,
        travel_times: pd.DataFrame,
        theta_by_category: dict[str, float],
        reward_weights: dict[str, float],
        max_steps: int = 8,
        start_area: str = "random",
    ) -> None:
        self.graph = graph
        self.profile = profile
        self.areas = areas
        self.resources = resources
        self.travel_times = travel_times
        self.theta_by_category = theta_by_category
        self.reward_weights = reward_weights
        self.max_steps = max_steps
        self.start_area = start_area
        self.area_features = compute_area_features(areas, travel_times)
        self.current_area = str(areas.iloc[0]["area_id"])
        self.visited: set[str] = set()
        self.steps = 0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            random.seed(seed)
        if self.start_area == "random":
            self.current_area = random.choice(list(self.areas["area_id"]))
        else:
            self.current_area = self.start_area
        self.visited = {self.current_area}
        self.steps = 0
        return self._state()

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        previous_reward = self._current_reward()["reward"]
        done = False

        if action.startswith("move_to_neighbor:"):
            target = action.split(":", 1)[1]
            if target in area_neighbors(self.graph, self.current_area):
                self.current_area = target
        elif action == self.ACTION_JUMP_BEST:
            self.current_area = self._best_candidate_area()
        elif action == self.ACTION_SELECT:
            done = True
        elif action == self.ACTION_STAY:
            pass
        else:
            raise ValueError(f"Unknown action: {action}")

        self.steps += 1
        self.visited.add(self.current_area)
        current_reward = self._current_reward()["reward"]
        if self.steps >= self.max_steps:
            done = True
        transition_reward = current_reward if done else current_reward - previous_reward
        return self._state(), float(transition_reward), done, {"selected_area": self.current_area if done else None}

    def get_valid_actions(self) -> list[str]:
        neighbor_actions = [f"move_to_neighbor:{neighbor}" for neighbor in area_neighbors(self.graph, self.current_area)]
        return [self.ACTION_STAY, self.ACTION_SELECT, self.ACTION_JUMP_BEST, *neighbor_actions]

    def render(self) -> str:
        return f"step={self.steps} current_area={self.current_area} visited={sorted(self.visited)}"

    def _state(self) -> dict[str, Any]:
        reward_parts = self._current_reward()
        return {
            "current_area": self.current_area,
            "profile_id": self.profile.profile_id,
            "profile_vector": self._profile_vector(),
            "area_feature_vector": feature_vector(self.current_area, self.area_features),
            "visited_areas": sorted(self.visited),
            "current_accessibility_score": reward_parts["accessibility"],
            "current_reward": reward_parts["reward"],
        }

    def _profile_vector(self) -> list[float]:
        return [
            1.0 if self.profile.has_car else 0.0,
            float(self.profile.time_sensitivity),
            float(self.profile.max_affordable_housing_cost) / 3000.0,
            float(self.profile.commute_weight),
            float(self.profile.risk_sensitivity),
        ]

    def _current_reward(self) -> dict[str, Any]:
        return compute_reward(
            self.current_area,
            self.profile,
            self.areas,
            self.resources,
            self.travel_times,
            self.theta_by_category,
            self.reward_weights,
        )

    def _best_candidate_area(self) -> str:
        rewards = [
            compute_reward(
                area_id,
                self.profile,
                self.areas,
                self.resources,
                self.travel_times,
                self.theta_by_category,
                self.reward_weights,
            )
            for area_id in self.areas["area_id"]
        ]
        return max(rewards, key=lambda row: row["reward"])["area_id"]

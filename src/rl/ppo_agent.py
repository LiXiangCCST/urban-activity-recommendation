"""PPO agent skeleton with a runnable heuristic policy."""

from __future__ import annotations

import random
from typing import Any


class PPOAgent:
    """Small policy wrapper matching a future PPO interface.

    TODO: Add actor/critic networks, rollout storage, advantage estimation, and
    gradient updates. For now the agent can choose random or simple heuristic
    actions so the environment loop is executable.
    """

    def __init__(self, policy: str = "heuristic", random_seed: int = 42) -> None:
        self.policy = policy
        self.rng = random.Random(random_seed)

    def act(self, state: dict[str, Any], valid_actions: list[str]) -> str:
        if self.policy == "random":
            return self.rng.choice(valid_actions)
        if state["current_reward"] > 1.0 and "select_current_area" in valid_actions:
            return "select_current_area"
        if "jump_to_best_candidate" in valid_actions:
            return "jump_to_best_candidate"
        return self.rng.choice(valid_actions)

    def update(self, trajectories: list[dict[str, Any]]) -> dict[str, float]:
        return {"loss": 0.0, "num_transitions": float(len(trajectories))}

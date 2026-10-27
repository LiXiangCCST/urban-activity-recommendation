"""Training-loop skeleton for the RL prototype."""

from __future__ import annotations

from typing import Any

from src.rl.environment import UrbanActivityRecommendationEnv
from src.rl.ppo_agent import PPOAgent


def run_demo_episode(env: UrbanActivityRecommendationEnv, agent: PPOAgent) -> dict[str, Any]:
    state = env.reset()
    trajectory = []
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(state, env.get_valid_actions())
        next_state, reward, done, info = env.step(action)
        trajectory.append({"state": state, "action": action, "reward": reward, "next_state": next_state, "done": done})
        total_reward += reward
        state = next_state

    agent.update(trajectory)
    return {
        "profile_id": env.profile.profile_id,
        "selected_area": info.get("selected_area", env.current_area),
        "steps": env.steps,
        "total_episode_reward": float(total_reward),
        "final_reward": float(state["current_reward"]),
        "final_accessibility": float(state["current_accessibility_score"]),
    }

"""Simple matplotlib visualizations for demo outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_recommended_area_by_profile(results: pd.DataFrame, output_dir: str | Path = "outputs") -> Path:
    path = _prepare_output_dir(output_dir) / "recommended_area_by_profile.png"
    weighted = results[results["method"] == "weighted_score"].sort_values("reward", ascending=True)
    labels = [f"{row.profile_id}\n{row.area_id}" for row in weighted.itertuples()]
    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = range(len(weighted))
    bars = ax.barh(y_pos, weighted["reward"], color="#4C78A8")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Reward")
    ax.set_title("Recommended area by profile")
    for bar, row in zip(bars, weighted.itertuples()):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  reward={row.reward:.2f}, access={row.accessibility:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_baseline_comparison(results: pd.DataFrame, output_dir: str | Path = "outputs") -> Path:
    path = _prepare_output_dir(output_dir) / "baseline_comparison.png"
    pivot = results.pivot_table(index="profile_id", columns="method", values="reward", aggfunc="mean")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Reward of recommended area")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_accessibility_comparison(results: pd.DataFrame, output_dir: str | Path = "outputs") -> Path:
    path = _prepare_output_dir(output_dir) / "accessibility_comparison.png"
    weighted = results[results["method"] == "weighted_score"]
    plt.figure(figsize=(9, 4))
    plt.bar(weighted["profile_id"], weighted["accessibility"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Accessibility")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_reward_comparison(results: pd.DataFrame, output_dir: str | Path = "outputs") -> Path:
    path = _prepare_output_dir(output_dir) / "reward_comparison.png"
    weighted = results[results["method"] == "weighted_score"]
    plt.figure(figsize=(9, 4))
    plt.bar(weighted["profile_id"], weighted["reward"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

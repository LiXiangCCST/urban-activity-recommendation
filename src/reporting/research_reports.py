
from __future__ import annotations

from pathlib import Path
from typing import Any
import textwrap

import matplotlib.pyplot as plt
import pandas as pd

from src.accessibility.reward import compute_reward
from src.evaluation.fairness import fairness_summary
from src.evaluation.metrics import activity_coverage_within_15
from src.profiles.individual_profiles import get_profiles

METHOD_LABELS = {
    "weighted_score": "Weighted score",
    "nearest_resource": "Nearest resource",
    "fifteen_minute_city": "15-minute city",
    "cost_first": "Cost first",
    "rl_demo": "RL demo",
}


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def wrap_labels(labels: list[str], width: int = 22) -> list[str]:
    return ["\n".join(textwrap.wrap(str(label), width=width, break_long_words=False)) for label in labels]


def weighted_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    return recommendations[recommendations["method"] == "weighted_score"].copy()


def build_reward_decomposition(recommendations, areas, resources, travel_times, config):
    profiles = get_profiles()
    weights = config["reward_weights"]
    rows = []
    for _, rec in weighted_recommendations(recommendations).iterrows():
        profile = profiles[rec["profile_id"]]
        reward = compute_reward(rec["area_id"], profile, areas, resources, travel_times, config["sample_data"]["theta_by_category"], weights)
        rows.append({
            "profile_id": profile.profile_id,
            "area_id": rec["area_id"],
            "accessibility_contribution": weights["alpha_access"] * reward["accessibility"],
            "activity_fit_contribution": weights["alpha_activity_fit"] * reward["activity_fit"],
            "mobility_contribution": weights["alpha_mobility"] * reward["mobility"],
            "equity_contribution": weights["alpha_equity"] * reward["equity"],
            "cost_penalty": -weights["beta_cost"] * reward["cost_penalty"],
            "commute_penalty": -weights["beta_commute"] * reward["commute_penalty"],
            "risk_penalty": -weights["beta_risk"] * reward["risk_penalty"],
            "final_reward": reward["reward"],
            "accessibility_score": reward["accessibility"],
            "coverage_15_min": activity_coverage_within_15(profile, rec["area_id"], travel_times),
        })
    return pd.DataFrame(rows)


def build_baseline_ranking(recommendations, rl_results):
    rows = []
    for _, row in recommendations.iterrows():
        rows.append({
            "profile_id": row["profile_id"],
            "method": row["method"],
            "recommended_area": row["area_id"],
            "reward": row["reward"],
            "accessibility": row["accessibility"],
            "coverage_15_min": row["coverage_15_min"],
        })
    for _, row in rl_results.iterrows():
        rows.append({
            "profile_id": row["profile_id"],
            "method": "rl_demo",
            "recommended_area": row["selected_area"],
            "reward": row["final_reward"],
            "accessibility": row["final_accessibility"],
            "coverage_15_min": None,
        })
    ranking = pd.DataFrame(rows)
    ranking["rank"] = ranking.groupby("profile_id")["reward"].rank(method="first", ascending=False).astype(int)
    return ranking.sort_values(["profile_id", "rank"])[["profile_id", "rank", "method", "recommended_area", "reward", "accessibility", "coverage_15_min"]].reset_index(drop=True)


def build_area_diagnostics(areas, resources, travel_times, recommendations, config):
    profiles = get_profiles()
    weighted = weighted_recommendations(recommendations)
    selected = weighted.groupby("area_id")["profile_id"].apply(lambda x: ",".join(sorted(x))).rename("selected_by_profiles")
    counts = weighted.groupby("area_id")["profile_id"].count().rename("number_of_times_selected")
    rows = []
    for area_id in areas["area_id"]:
        values = []
        for profile in profiles.values():
            reward = compute_reward(area_id, profile, areas, resources, travel_times, config["sample_data"]["theta_by_category"], config["reward_weights"])
            reward["coverage_15_min"] = activity_coverage_within_15(profile, area_id, travel_times)
            values.append(reward)
        frame = pd.DataFrame(values)
        rows.append({
            "area_id": area_id,
            "average_reward_across_profiles": frame["reward"].mean(),
            "average_accessibility": frame["accessibility"].mean(),
            "number_of_times_selected": int(counts.get(area_id, 0)),
            "selected_by_profiles": selected.get(area_id, ""),
            "mean_cost_penalty": frame["cost_penalty"].mean(),
            "mean_risk_penalty": frame["risk_penalty"].mean(),
            "mean_mobility_score": frame["mobility"].mean(),
            "mean_coverage_15_min": frame["coverage_15_min"].mean(),
        })
    return pd.DataFrame(rows).sort_values("average_reward_across_profiles", ascending=False).reset_index(drop=True)


def write_profile_explanations(output_dir, recommendations, decomp, ranking, metadata):
    output_dir = Path(output_dir)
    lines = ["# Profile recommendation explanations", ""]
    mode = metadata.get("data_mode", "unknown") if metadata else "unknown"
    variant = metadata.get("nyc_data_variant", "none") if metadata else "none"
    provenance = metadata.get("data_provenance", "unknown") if metadata else "unknown"
    lines.append(f"Data mode: `{mode}`; NYC data variant: `{variant}`; data provenance: `{provenance}`. Explanations are generated from computed reward components and baseline outputs.")
    cols = ["accessibility_contribution", "activity_fit_contribution", "mobility_contribution", "equity_contribution", "cost_penalty", "commute_penalty", "risk_penalty"]
    for _, row in decomp.sort_values("profile_id").iterrows():
        profile_id = row["profile_id"]
        profile_rank = ranking[ranking["profile_id"] == profile_id].sort_values("rank")
        weighted_rank = profile_rank[profile_rank["method"] == "weighted_score"].iloc[0]
        comparison = profile_rank[profile_rank["method"] != "weighted_score"].iloc[0]
        positives = sorted([(c, row[c]) for c in cols if row[c] > 0], key=lambda x: x[1], reverse=True)[:3]
        negatives = sorted([(c, row[c]) for c in cols if row[c] < 0], key=lambda x: x[1])[:3]
        pos_text = ", ".join(f"{c.replace('_', ' ')} ({v:.3f})" for c, v in positives) or "none"
        neg_text = ", ".join(f"{c.replace('_', ' ')} ({v:.3f})" for c, v in negatives) or "none"
        lines.extend([
            "", f"## {profile_id}", "",
            f"- Recommended area: `{row['area_id']}`",
            f"- Final reward: {row['final_reward']:.3f}",
            f"- Accessibility score: {row['accessibility_score']:.3f}",
            f"- 15-minute coverage: {row['coverage_15_min']:.3f}",
            f"- Strongest positive factors: {pos_text}.",
            f"- Strongest negative constraints: {neg_text}.",
            f"- Baseline comparison: weighted-score ranked {int(weighted_rank['rank'])}; {METHOD_LABELS.get(comparison['method'], comparison['method'])} recommended `{comparison['recommended_area']}` with reward {comparison['reward']:.3f}.",
            f"- Suitability interpretation: `{row['area_id']}` is suitable for this individual because the weighted accessibility and fit benefits exceed the modeled cost, commute, and risk constraints for this profile.",
        ])
    path = output_dir / "profile_recommendation_explanations.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_baseline_ranking_markdown(output_dir, ranking):
    output_dir = Path(output_dir)
    lines = ["# Baseline ranking by profile", ""]
    for profile_id in sorted(ranking["profile_id"].unique()):
        table = ranking[ranking["profile_id"] == profile_id].copy()
        table["method"] = table["method"].map(lambda value: METHOD_LABELS.get(value, value))
        lines.extend([f"## {profile_id}", "", dataframe_to_markdown(table), ""])
    path = output_dir / "baseline_ranking_by_profile.md"
    path.write_text("\n".join(lines))
    return path


def write_fairness_interpretation(output_dir, weighted, metadata):
    output_dir = Path(output_dir)
    summary = fairness_summary(weighted)
    highest = weighted.sort_values("reward", ascending=False).iloc[0]
    lowest = weighted.sort_values("reward", ascending=True).iloc[0]
    gap = summary["group_reward_gap"]
    if gap < 0.25:
        interp = "Reward inequality is low in this run, although profile differences remain visible."
    elif gap < 1.0:
        interp = "Reward inequality is moderate; some profiles receive materially stronger matches than others."
    else:
        interp = "Reward inequality is high; weaker groups should be examined before making substantive claims."
    variant = metadata.get("nyc_data_variant", "none") if metadata else "none"
    provenance = metadata.get("data_provenance", "unknown") if metadata else "unknown"
    if provenance == "partial_real":
        caution = "Caution: this is a partial-real run. Some public datasets are used, but placeholder, estimated, or approximate inputs remain, so fairness results should be interpreted as pipeline diagnostics rather than real-world equity evidence."
    elif provenance == "real_public_data":
        caution = "Caution: these results use public datasets but still depend on model assumptions, proxy variables, and approximate travel times; they should not be interpreted as final causal or policy evidence."
    elif provenance == "fictional_minimal":
        caution = "Caution: these results are based on fictional minimal NYC-like data and should be interpreted only as pipeline diagnostics."
    else:
        caution = "Caution: these results should be interpreted as prototype diagnostics, not final real-world equity evidence."
    lines = [
        "# Fairness interpretation", "",
        f"- NYC data variant: {variant}",
        f"- Data provenance: {provenance}",
        f"- Group reward gap: {summary['group_reward_gap']:.3f}",
        f"- Minimum group reward: {summary['minimum_group_reward']:.3f}",
        f"- Gini reward: {summary['gini_reward']:.3f}",
        f"- Highest reward profile: `{highest['profile_id']}` in `{highest['area_id']}` ({highest['reward']:.3f})",
        f"- Lowest reward profile: `{lowest['profile_id']}` in `{lowest['area_id']}` ({lowest['reward']:.3f})",
        "", interp, "",
        f"The weakest group in this run is `{lowest['profile_id']}`. It is disadvantaged relative to the strongest group by {gap:.3f} reward units under the current reward design and available data.",
        "", caution,
    ]
    path = output_dir / "fairness_interpretation.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_method_comparison_summary(output_dir, ranking, metadata=None):
    output_dir = Path(output_dir)
    best_counts = ranking[ranking["rank"] == 1]["method"].value_counts()
    best_method = best_counts.idxmax() if not best_counts.empty else "none"
    cost = ranking[ranking["method"] == "cost_first"]
    weighted = ranking[ranking["method"] == "weighted_score"]
    nearest = ranking[ranking["method"] == "nearest_resource"]
    fifteen = ranking[ranking["method"] == "fifteen_minute_city"]
    delta = 0.0
    if not cost.empty and not weighted.empty:
        merged = cost.merge(weighted, on="profile_id", suffixes=("_cost", "_weighted"))
        delta = (merged["accessibility_cost"] - merged["accessibility_weighted"]).mean()
    variant = metadata.get("nyc_data_variant", "none") if metadata else "none"
    provenance = metadata.get("data_provenance", "unknown") if metadata else "unknown"
    lines = [
        "# Method comparison summary", "",
        f"- NYC data variant: {variant}",
        f"- Data provenance: {provenance}",
        f"- Method ranked first most often: `{METHOD_LABELS.get(best_method, best_method)}` ({int(best_counts.max()) if not best_counts.empty else 0} profiles).",
        f"- Cost-first accessibility difference relative to weighted-score: {delta:.3f} on average.",
        f"- Nearest-resource average reward: {nearest['reward'].mean():.3f}.",
        f"- Fifteen-minute-city average reward: {fifteen['reward'].mean():.3f}.",
        f"- Weighted-score average reward: {weighted['reward'].mean():.3f}.",
        "", "The cost-first baseline is useful for affordability sensitivity, but it can sacrifice accessibility when lower-cost areas are farther from high-weight resources.",
        "", "The nearest-resource baseline is intentionally narrow: it can find close resources, but it does not balance multiple activity categories, affordability, risk, or mobility constraints.",
        "", "The fifteen-minute-city baseline captures local coverage well, but it is less personalized because it does not fully encode each profile's budget, commute, and risk tradeoffs.",
        "", "The weighted-score method currently acts as the strongest deterministic baseline because it directly implements the configured personalized reward function.",
        "", "The RL demo should be interpreted as environment validation only. It uses a heuristic policy and should not be presented as trained DRL performance or evidence of PPO superiority.",
    ]
    path = output_dir / "method_comparison_summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def plot_reward_decomposition(decomp, output_dir, filename="reward_decomposition_by_profile.png"):
    output_dir = Path(output_dir)
    cols = ["accessibility_contribution", "activity_fit_contribution", "mobility_contribution", "equity_contribution", "cost_penalty", "commute_penalty", "risk_penalty"]
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#ECA82C", "#F58518", "#E45756", "#B279A2"]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = list(range(len(decomp)))
    pos = [0.0] * len(decomp)
    neg = [0.0] * len(decomp)
    for col, color in zip(cols, colors):
        values = decomp[col].astype(float).tolist()
        bottoms = [pos[i] if values[i] >= 0 else neg[i] for i in range(len(values))]
        ax.bar(x, values, bottom=bottoms, label=col.replace("_", " "), color=color)
        for i, value in enumerate(values):
            if value >= 0:
                pos[i] += value
            else:
                neg[i] += value
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(decomp["profile_id"].tolist(), 16), rotation=0, ha="center")
    ax.set_ylabel("Reward contribution")
    ax.set_title("Reward decomposition by profile")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_profile_recommendations(weighted, output_dir, filename="fig_profile_recommendations.png"):
    output_dir = Path(output_dir)
    ordered = weighted.sort_values("reward", ascending=True)
    labels = [f"{p}\n{a}" for p, a in zip(ordered["profile_id"], ordered["area_id"])]
    fig, ax = plt.subplots(figsize=(11, 6))
    y = list(range(len(ordered)))
    bars = ax.barh(y, ordered["reward"], color="#4C78A8")
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(labels, 28), fontsize=8)
    ax.set_xlabel("Reward")
    ax.set_title("Recommended area by profile (weighted-score baseline)")
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  area={row['area_id']} | access={row['accessibility']:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_baseline_reward_comparison(ranking, output_dir):
    output_dir = Path(output_dir)
    data = ranking.copy()
    data["method_label"] = data["method"].map(lambda v: METHOD_LABELS.get(v, v))
    pivot = data.pivot_table(index="profile_id", columns="method_label", values="reward", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Reward")
    ax.set_xlabel("Profile")
    ax.set_title("Reward comparison across methods")
    ax.set_xticklabels(wrap_labels(list(pivot.index), 16), rotation=0, ha="center")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    path = output_dir / "fig_baseline_reward_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_accessibility_by_profile(weighted, output_dir):
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(weighted["profile_id"], weighted["accessibility"], color="#72B7B2")
    ax.set_ylabel("Accessibility")
    ax.set_title("Accessibility of weighted-score recommendations")
    ax.set_xticks(range(len(weighted)))
    ax.set_xticklabels(wrap_labels(weighted["profile_id"].tolist(), 14), rotation=0, ha="center")
    fig.tight_layout()
    path = output_dir / "fig_accessibility_by_profile.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_fairness_summary(weighted, output_dir):
    output_dir = Path(output_dir)
    ordered = weighted.sort_values("reward", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = list(range(len(ordered)))
    ax.barh(y_pos, ordered["reward"], color="#54A24B")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wrap_labels(ordered["profile_id"].tolist(), 22), fontsize=8)
    ax.set_xlabel("Reward")
    ax.set_title("Reward distribution across profiles")
    fig.tight_layout()
    path = output_dir / "fig_fairness_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_area_selection_frequency(area_diagnostics, output_dir):
    output_dir = Path(output_dir)
    selected = area_diagnostics[area_diagnostics["number_of_times_selected"] > 0]
    if selected.empty:
        selected = area_diagnostics.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = list(range(len(selected)))
    ax.barh(y_pos, selected["number_of_times_selected"], color="#F58518")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wrap_labels(selected["area_id"].tolist(), 24), fontsize=8)
    ax.set_xlabel("Number of weighted-score selections")
    ax.set_title("Area selection frequency")
    fig.tight_layout()
    path = output_dir / "fig_area_selection_frequency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_results_narrative(metadata, weighted, ranking):
    mode = metadata.get("data_mode", "unknown") if metadata else "unknown"
    city = metadata.get("case_city", "N/A") if metadata else "N/A"
    variant = metadata.get("nyc_data_variant", "none") if metadata else "none"
    provenance = metadata.get("data_provenance", "unknown") if metadata else "unknown"
    best_counts = ranking[ranking["rank"] == 1]["method"].value_counts()
    best_method = best_counts.idxmax() if not best_counts.empty else "none"
    distinct_areas = weighted["area_id"].nunique()
    lines = [
        "# Results narrative draft", "",
        f"This draft summarizes a `{mode}` run for `{city}` with NYC data variant `{variant}` and data provenance `{provenance}`. If the run uses the bundled minimal NYC-like data, the data are fictional and intended to validate the pipeline rather than support empirical claims about New York City. If the run is partial-real, some public datasets were used but missing components were filled by local fallbacks or approximations.",
        "", "## Result 1: The framework produces individualized area recommendations.", "",
        f"The pipeline generated recommendations for {len(weighted)} representative profiles. The weighted-score baseline selected {distinct_areas} distinct areas, indicating that the scoring function can produce profile-specific outcomes rather than a single universal best location.",
        "", "## Result 2: Personalized weighted scoring outperforms simpler non-personalized baselines in the prototype.", "",
        f"Across the current run, `{METHOD_LABELS.get(best_method, best_method)}` ranked first most often. Because the weighted-score method directly implements the configured reward, it should be interpreted as the strongest deterministic baseline, not as a learned model.",
        "", "## Result 3: Different profiles prioritize different urban areas.", "",
        "Differences in income limits, mobility assumptions, commute sensitivity, activity weights, and risk sensitivity shift the recommended area across profiles. The profile explanation report identifies the positive and negative reward components behind each recommendation.",
        "", "## Result 4: Fairness metrics reveal remaining differences among groups.", "",
        "The fairness report summarizes reward gaps, minimum group reward, and reward Gini. These metrics are diagnostic and do not establish real-world equity impacts without validated public data and stakeholder review.",
        "", "## Result 5: The RL environment is operational but requires trained PPO for final claims.", "",
        "The current RL module uses a heuristic policy to validate environment mechanics, state construction, action handling, and reward feedback. No claim should be made that DRL or PPO outperforms deterministic baselines until a trained policy is implemented and evaluated under controlled experiments.",
    ]
    path = Path("docs/results_narrative_draft.md")
    path.write_text("\n".join(lines) + "\n")
    return path


def generate_research_reports(output_dir, recommendations, rl_results, areas, resources, travel_times, config, metadata=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paper_dir = output_path / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    weighted = weighted_recommendations(recommendations)
    decomp = build_reward_decomposition(recommendations, areas, resources, travel_times, config)
    ranking = build_baseline_ranking(recommendations, rl_results)
    diagnostics = build_area_diagnostics(areas, resources, travel_times, recommendations, config)
    for filename, df in [("reward_decomposition.csv", decomp), ("baseline_ranking_by_profile.csv", ranking), ("area_diagnostics.csv", diagnostics)]:
        df.to_csv(output_path / filename, index=False)
        generated.append(output_path / filename)
    generated.append(write_profile_explanations(output_path, recommendations, decomp, ranking, metadata))
    generated.append(write_baseline_ranking_markdown(output_path, ranking))
    generated.append(write_fairness_interpretation(output_path, weighted, metadata))
    generated.append(write_method_comparison_summary(output_path, ranking, metadata))
    generated.append(write_results_narrative(metadata, weighted, ranking))
    generated.append(plot_reward_decomposition(decomp, output_path, "reward_decomposition_by_profile.png"))
    generated.append(plot_profile_recommendations(weighted, paper_dir, "fig_profile_recommendations.png"))
    generated.append(plot_baseline_reward_comparison(ranking, paper_dir))
    generated.append(plot_accessibility_by_profile(weighted, paper_dir))
    generated.append(plot_reward_decomposition(decomp, paper_dir, "fig_reward_decomposition.png"))
    generated.append(plot_fairness_summary(weighted, paper_dir))
    generated.append(plot_area_selection_frequency(diagnostics, paper_dir))
    return generated

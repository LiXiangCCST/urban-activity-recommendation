"""Reporting utilities for the household-level experiment."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No data available._\n"
    data = df.head(max_rows).copy() if max_rows else df.copy()
    cols = list(data.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in data.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value).replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _wrap(labels, width=24):
    return ["\n".join(textwrap.wrap(str(label), width=width)) for label in labels]


def write_household_reports(
    output_dir: str | Path,
    households: dict,
    recommendations: pd.DataFrame,
    decomposition: pd.DataFrame,
    commute_summary: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    area_diagnostics: pd.DataFrame,
    member_commutes: pd.DataFrame,
    metadata: dict | None = None,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    weighted = recommendations[recommendations["method"] == "household_weighted_score"].copy()
    fairness_gap = float(weighted["reward"].max() - weighted["reward"].min()) if not weighted.empty else 0.0
    weakest = weighted.sort_values("reward").iloc[0] if not weighted.empty else None
    strongest = weighted.sort_values("reward", ascending=False).iloc[0] if not weighted.empty else None

    reports = {
        "household_executive_summary.md": [
            "# Household-aware activity-area recommendation executive summary",
            "",
            f"- Households evaluated: {len(households)}",
            f"- Candidate areas evaluated per household: {decomposition['area_id'].nunique() if not decomposition.empty else 0}",
            f"- Data provenance: {(metadata or {}).get('data_provenance', 'unknown')}",
            f"- Travel-time source: {(metadata or {}).get('travel_time_source', 'unknown')}",
            "",
            "The household experiment extends the individual recommendation framework by balancing multiple workers' commutes, child and elderly care needs, household housing budget, car availability, shared daily resources, and risk. The outputs are deterministic household-aware baselines, not trained household-level reinforcement learning results.",
            "",
            "## Top household recommendations",
            "",
            markdown_table(weighted[["household_id", "area_id", "reward", "accessibility", "mean_commute_time", "max_commute_time", "housing_cost"]] if not weighted.empty else weighted),
        ],
        "household_methodology.md": [
            "# Household methodology",
            "",
            "Households are represented by household-level income, budget, vehicle availability, household composition, commute tolerance, affordability sensitivity, and care responsibility. Members carry roles, employment status, workplace area or coordinates, schedule, mode preference, commute weight, and activity needs.",
            "",
            "Workplace destinations in the example profiles are scenario assumptions. When a member has a workplace_area_id, it is mapped to the loaded candidate-area centroid. If no valid workplace is provided for an employed member, the loader assigns a candidate-area workplace and records that assumption.",
            "",
            "For each residential candidate area, the experiment computes member commute time, mean commute, maximum commute, commute inequality, workers over tolerance, shared resource accessibility, child/elderly care access, grocery/public-service access, housing burden, mode feasibility, and risk penalties.",
        ],
        "household_recommendation_analysis.md": [
            "# Household recommendation analysis",
            "",
            markdown_table(weighted[["household_id", "area_id", "reward", "accessibility", "mean_commute_time", "max_commute_time", "workers_over_tolerance_count"]] if not weighted.empty else weighted),
            "",
            "Household recommendations differ because the reward combines shared accessibility with household-specific composition and commute constraints. This supports the thesis claim that a residential activity area cannot be evaluated only through a single individual profile.",
        ],
        "household_commute_tradeoff_analysis.md": [
            "# Household commute tradeoff analysis",
            "",
            markdown_table(commute_summary.sort_values(["household_id", "weighted_commute_time"]).groupby("household_id").head(3) if not commute_summary.empty else commute_summary),
            "",
            "Mean commute captures aggregate burden, maximum commute captures the worst-member burden, and commute inequality captures uneven household sacrifice. The commute-first baseline is included to expose how much reward is lost when commute minimization dominates other needs.",
        ],
        "household_fairness_analysis.md": [
            "# Household fairness analysis",
            "",
            f"- Household reward gap: {fairness_gap:.3f}",
            f"- Highest reward household: {strongest['household_id'] if strongest is not None else 'n/a'}",
            f"- Lowest reward household: {weakest['household_id'] if weakest is not None else 'n/a'}",
            "",
            "Lower-income, no-car, single-parent, elderly-care, and night-shift households should be interpreted carefully because their feasible mobility set is narrower and their care or schedule constraints are more binding. The fairness interpretation remains provisional because commute times use centroid approximations and partial observed TLC road-time support.",
        ],
        "household_limitations.md": [
            "# Household experiment limitations",
            "",
            "- Example household profiles are scenario inputs, not observed household microdata.",
            "- Workplace destinations are generated or area-based assumptions unless explicit workplace coordinates are supplied.",
            "- Area-to-workplace commute times use centroid and speed assumptions rather than full network routing.",
            "- Shared resource accessibility uses the existing processed travel-time matrix, which includes GTFS approximation and partial TLC/HVFHV observed road-time integration.",
            "- Household-level RL is not implemented; this step uses deterministic household-aware baselines.",
        ],
    }
    for filename, lines in reports.items():
        path = out / filename
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})

    def savefig(name: str):
        path = out / name
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        paths.append(path)

    if not weighted.empty:
        d = weighted.sort_values("reward")
        plt.figure(figsize=(9, 5))
        plt.barh(_wrap(d["household_id"]), d["reward"], color="#4C78A8")
        plt.xlabel("Reward")
        plt.title("Household weighted-score recommendations")
        savefig("household_recommendations.png")

        d = decomposition[decomposition["area_id"].isin(weighted["area_id"])].merge(weighted[["household_id", "area_id"]], on=["household_id", "area_id"], how="inner")
        comps = ["resource_accessibility_benefit", "household_activity_fit", "mobility_feasibility", "equity_adjustment", "housing_cost_penalty", "mean_commute_penalty", "max_commute_penalty", "commute_inequality_penalty", "care_coordination_penalty", "mode_infeasibility_penalty", "risk_penalty"]
        plot_df = d.set_index("household_id")[comps]
        plt.figure(figsize=(11, 6))
        x = np.arange(len(plot_df))
        pos = np.zeros(len(plot_df)); neg = np.zeros(len(plot_df))
        for col in comps:
            vals = plot_df[col].to_numpy()
            signed_vals = -vals if col.endswith("penalty") else vals
            bottoms = np.where(signed_vals >= 0, pos, neg)
            plt.bar(x, signed_vals, bottom=bottoms, label=col.replace("_", " "))
            pos += np.where(signed_vals >= 0, signed_vals, 0)
            neg += np.where(signed_vals < 0, signed_vals, 0)
        plt.xticks(x, _wrap(plot_df.index, 14))
        plt.ylabel("Reward contribution")
        plt.title("Household reward decomposition")
        plt.legend(fontsize=6, ncol=2)
        savefig("household_reward_decomposition.png")

        plt.figure(figsize=(9, 5))
        plt.barh(_wrap(weighted["household_id"]), weighted["mean_commute_time"], label="mean", color="#59A14F")
        plt.scatter(weighted["max_commute_time"], _wrap(weighted["household_id"]), label="max", color="#E15759")
        plt.xlabel("Minutes")
        plt.title("Household commute burden")
        plt.legend()
        savefig("household_commute_burden.png")

        plt.figure(figsize=(8, 5))
        plt.scatter(weighted["housing_cost"], weighted["accessibility"], s=80, color="#4C78A8")
        for _, row in weighted.iterrows():
            plt.text(row["housing_cost"], row["accessibility"], row["household_id"], fontsize=7)
        plt.xlabel("Housing cost")
        plt.ylabel("Accessibility")
        plt.title("Affordability vs accessibility")
        savefig("household_affordability_vs_accessibility.png")

        plt.figure(figsize=(9, 5))
        plt.barh(_wrap(weighted["household_id"]), weighted["commute_inequality"], color="#F28E2B")
        plt.xlabel("Commute-time standard deviation")
        plt.title("Household member commute inequality")
        savefig("household_member_commute_inequality.png")

    if not baseline_comparison.empty:
        pivot = baseline_comparison.pivot(index="household_id", columns="method", values="reward")
        plt.figure(figsize=(11, 6))
        pivot.plot(kind="bar", ax=plt.gca(), width=0.82)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Reward")
        plt.title("Household baseline comparison")
        plt.legend(fontsize=7, ncol=2)
        savefig("household_baseline_comparison.png")

    return paths

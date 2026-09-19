"""Generate revised paper figures from frozen structured experiment results only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POLICY_LABELS = {
    "strict_local_target": "Strict Local",
    "remote_target_without_speculation": "Remote Target",
    "best_feasible_legacy_activation_split": "Legacy Split",
    "token_speculation_without_pipeline": "Token, no pipeline",
    "global_static_deepflow": "Global Static",
    "heuristic_deepflow": "Frozen Heuristic",
    "ppo_deepflow": "PPO",
    "per_scenario_oracle": "Per-scenario Oracle",
}
POLICY_COLORS = {
    "strict_local_target": "#6b7280",
    "remote_target_without_speculation": "#64748b",
    "best_feasible_legacy_activation_split": "#b45309",
    "token_speculation_without_pipeline": "#8b5cf6",
    "global_static_deepflow": "#0284c7",
    "heuristic_deepflow": "#059669",
    "ppo_deepflow": "#dc2626",
    "per_scenario_oracle": "#374151",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    layout_rect: tuple[float, float, float, float] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=layout_rect)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def _configure_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.32, linewidth=0.8)
    ax.set_axisbelow(True)


def _independent_summary(results_dir: Path) -> Mapping[str, Any]:
    return _load_json(results_dir / "independent_test" / "three_seed_summary.json")


def _independent_seed_payloads(results_dir: Path) -> List[Mapping[str, Any]]:
    payloads = []
    for seed in (42, 20260912, 20261127):
        path = results_dir / "independent_test" / f"seed_{seed}" / f"independent_test_seed_{seed}.json"
        payloads.append(_load_json(path))
    return payloads


def _plot_overall_comparison(results_dir: Path, output_dir: Path) -> None:
    summary = _independent_summary(results_dir)["aggregate_by_policy"]
    policies = [
        "strict_local_target",
        "remote_target_without_speculation",
        "best_feasible_legacy_activation_split",
        "token_speculation_without_pipeline",
        "global_static_deepflow",
        "heuristic_deepflow",
        "ppo_deepflow",
        "per_scenario_oracle",
    ]
    values = [summary[policy]["mean_throughput_tok_s"]["mean"] for policy in policies]
    errors = [summary[policy]["mean_throughput_tok_s"]["sample_std"] for policy in policies]
    labels = [POLICY_LABELS[policy] for policy in policies]

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    positions = np.arange(len(policies))
    bars = ax.bar(
        positions,
        values,
        yerr=errors,
        capsize=3,
        color=[POLICY_COLORS[policy] for policy in policies],
        edgecolor="black",
        linewidth=0.7,
    )
    for bar, value, error in zip(bars, values, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + 0.65,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(positions, labels, rotation=27, ha="right")
    ax.set_ylabel("Mean throughput (tok/s)")
    ax.set_title("Independent-test comparison across three PPO seeds")
    ax.set_ylim(0, max(values) * 1.20)
    _configure_axes(ax)
    _save_figure(fig, output_dir, "fig8_overall_comparison")


def _plot_bandwidth_strata(results_dir: Path, output_dir: Path) -> None:
    policies = [
        "best_feasible_legacy_activation_split",
        "global_static_deepflow",
        "heuristic_deepflow",
        "ppo_deepflow",
        "per_scenario_oracle",
    ]
    grouped: Dict[str, Dict[float, List[float]]] = {
        policy: defaultdict(list) for policy in policies
    }
    for payload in _independent_seed_payloads(results_dir):
        for record in payload["records"]:
            policy = str(record["policy_name"])
            if policy in grouped:
                grouped[policy][float(record["bandwidth_mbps"])].append(float(record["throughput_tok_s"]))
    bandwidths = sorted({bandwidth for policy_data in grouped.values() for bandwidth in policy_data})

    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    for policy in policies:
        values = [float(np.mean(grouped[policy][bandwidth])) for bandwidth in bandwidths]
        ax.plot(
            bandwidths,
            values,
            marker="o" if policy != "ppo_deepflow" else "^",
            linewidth=2.0,
            markersize=6,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
    ax.set_xscale("log")
    ax.set_xticks(bandwidths, [f"{value:g}" for value in bandwidths])
    ax.set_xlabel("Bandwidth stratum (Mbps, log scale)")
    ax.set_ylabel("Mean throughput (tok/s)")
    ax.set_title("Independent-test throughput by bandwidth stratum")
    ax.legend(frameon=False, fontsize=8, loc="best")
    _configure_axes(ax)
    _save_figure(fig, output_dir, "fig9_bandwidth_sensitivity")


def _plot_stress_robustness(results_dir: Path, output_dir: Path) -> None:
    payload = _load_json(results_dir / "stress_experiments" / "three_seed_stress_summary.json")
    aggregate = payload["aggregate_by_profile_policy"]
    profiles = ["base", "mild", "severe"]
    policies = [
        "best_feasible_legacy_activation_split",
        "global_static_deepflow",
        "heuristic_deepflow",
        "ppo_deepflow",
        "per_profile_oracle",
    ]
    labels = {
        **POLICY_LABELS,
        "per_profile_oracle": "Profile Oracle",
    }
    colors = {
        **POLICY_COLORS,
        "per_profile_oracle": "#374151",
    }
    x = np.arange(len(profiles))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for index, policy in enumerate(policies):
        values = [aggregate[profile][policy]["mean_throughput_tok_s"]["mean"] for profile in profiles]
        errors = [aggregate[profile][policy]["mean_throughput_tok_s"]["sample_std"] for profile in profiles]
        ax.bar(
            x + (index - 2) * width,
            values,
            width,
            yerr=errors if policy == "ppo_deepflow" else None,
            capsize=3 if policy == "ppo_deepflow" else 0,
            color=colors[policy],
            edgecolor="black",
            linewidth=0.55,
            label=labels[policy],
        )
    ax.set_xticks(x, [profile.capitalize() for profile in profiles])
    ax.set_ylabel("Mean throughput (tok/s)")
    ax.set_title("Robustness under common-random network and resource stress")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _configure_axes(ax)
    _save_figure(fig, output_dir, "fig10_combined_policy_map")


def _context_rows(results_dir: Path) -> List[Mapping[str, Any]]:
    return _load_json(results_dir / "sensitivity_experiments" / "context_boundary_audit.json")["rows"]


def _plot_context_feasibility(results_dir: Path, output_dir: Path) -> None:
    rows = _context_rows(results_dir)
    prompts = [2048, 4096, 8192]
    targets = ["7b", "13b"]
    row_map = {(str(row["target_model_key"]), int(row["prompt_len"])): row for row in rows}
    x = np.arange(len(prompts))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for offset, target in [(-width / 2, "7b"), (width / 2, "13b")]:
        values = [100.0 * float(row_map[(target, prompt)]["feasible_rate"]) for prompt in prompts]
        ax.bar(
            x + offset,
            values,
            width,
            color="#0284c7" if target == "7b" else "#7c3aed",
            edgecolor="black",
            linewidth=0.65,
            label=f"LLaMA-style {target.upper()}",
        )
        for position, value in zip(x + offset, values):
            ax.text(position, value + 1.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [str(prompt) for prompt in prompts])
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Feasible action ratio (%)")
    ax.set_ylim(0, 18)
    ax.set_title("Position and memory constrained action space")
    ax.legend(frameon=False)
    _configure_axes(ax)
    _save_figure(fig, output_dir, "fig11_feasible_ratio_vs_prompt")


def _plot_context_rejections(results_dir: Path, output_dir: Path) -> None:
    rows = _context_rows(results_dir)
    rows = sorted(rows, key=lambda row: (str(row["target_model_key"]), int(row["prompt_len"])))
    labels = [f"{row['target_model_key'].upper()}\n{row['prompt_len']}" for row in rows]
    components = [
        ("Position limit", "position_rejections", "#ef4444"),
        ("Edge OOM", "edge_oom_rejections", "#f59e0b"),
        ("Cloud OOM", "cloud_oom_rejections", "#06b6d4"),
        ("Both OOM", "both_oom_rejections", "#7c3aed"),
    ]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    fig, ax = plt.subplots(figsize=(8.1, 5.0))
    for label, field, color in components:
        values = np.asarray([int(row[field]) for row in rows])
        ax.bar(x, values, bottom=bottom, color=color, edgecolor="black", linewidth=0.5, label=label)
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_ylabel("Rejected actions")
    ax.set_title("Context-boundary rejection composition")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _configure_axes(ax)
    _save_figure(fig, output_dir, "fig12_oom_distribution")


def _plot_scale_acceptance(results_dir: Path, output_dir: Path) -> None:
    payload = _load_json(results_dir / "sensitivity_experiments" / "sensitivity_protocol_summary.json")
    rows = payload["summaries"]
    profiles = ["conservative", "default", "favorable"]
    policies = [
        "best_feasible_legacy_activation_split",
        "global_static_deepflow",
        "heuristic_deepflow",
        "per_scenario_oracle",
    ]
    row_map = {
        (str(row["target_model_key"]), str(row["acceptance_profile"]), str(row["policy_name"])): row
        for row in rows
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
    for ax, target in zip(axes, ("7b", "13b")):
        for policy in policies:
            values = [
                float(row_map[(target, profile, policy)]["mean_throughput_tok_s"])
                for profile in profiles
            ]
            color = POLICY_COLORS.get(policy, "#374151")
            ax.plot(
                profiles,
                values,
                marker="o",
                linewidth=2.0,
                color=color,
                label=POLICY_LABELS.get(policy, "Oracle"),
            )
        ax.set_title(f"LLaMA-style {target.upper()} target")
        ax.set_xlabel("Acceptance profile")
        _configure_axes(ax)
    axes[0].set_ylabel("Mean throughput (tok/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Protocol sensitivity to draft acceptance and target scale", y=0.995)
    _save_figure(fig, output_dir, "fig13_throughput_vs_prompt", layout_rect=(0.0, 0.0, 1.0, 0.80))


def generate_figures(results_dir: Path, output_dir: Path) -> None:
    _plot_overall_comparison(results_dir, output_dir)
    _plot_bandwidth_strata(results_dir, output_dir)
    _plot_stress_robustness(results_dir, output_dir)
    _plot_context_feasibility(results_dir, output_dir)
    _plot_context_rejections(results_dir, output_dir)
    _plot_scale_acceptance(results_dir, output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="paper")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    generate_figures(Path(args.results_dir), Path(args.output_dir))
    print(f"Generated revised Fig. 8-13 in {Path(args.output_dir)}")


if __name__ == "__main__":
    main()

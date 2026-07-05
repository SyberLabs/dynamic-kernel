"""
Mechanism sensitivity analysis for Phase 2 influence-game thresholds.

The dense budget frontier found a local robust-safe threshold at 0.25. This
script asks why: is the threshold created by defender action granularity,
dynamic immune-response bootstrapping, or a genuine topology-repair minimum?

Usage:
    .venv\\Scripts\\python.exe influence_mechanism_sensitivity.py
    .venv\\Scripts\\python.exe influence_mechanism_sensitivity.py --quick
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from influence_game import GameConfig
from influence_qlearn import train_q_adversary


DEFAULT_BUDGETS = [0.0, 0.25, 0.5, 1.0, 2.0]
DEFAULT_ACTION_UNITS = [0.25, 0.5, 1.0, 2.0]


@dataclass(frozen=True)
class ImmuneRegime:
    name: str
    dynamic_defense: bool
    gain: float
    cap: float
    max_extra: float


IMMUNE_REGIMES = [
    ImmuneRegime("none", False, 0.0, 0.0, 0.0),
    ImmuneRegime("weak", True, 0.25, 0.75, 2.0),
    ImmuneRegime("default", True, 0.75, 3.0, 5.0),
    ImmuneRegime("strong", True, 1.25, 5.0, 8.0),
]


def _mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    array = np.array(values, dtype=np.float64)
    mean = float(array.mean())
    half_width = float(z * array.std(ddof=1) / np.sqrt(len(array)))
    return mean - half_width, mean + half_width


def _game_config(
    seed: int,
    defender_budget: float,
    action_unit: float,
    immune: ImmuneRegime,
    agents: int,
    rounds: int,
    steps_per_round: int,
    adversary_budget: float,
) -> GameConfig:
    return GameConfig(
        rounds=rounds,
        agents=agents,
        steps_per_round=steps_per_round,
        adversary_budget=adversary_budget,
        defender_budget=defender_budget,
        action_unit=action_unit,
        seed=seed,
        dynamic_defense=immune.dynamic_defense,
        immune_response_gain=immune.gain,
        immune_response_cap=immune.cap,
        max_defender_budget=max(defender_budget, defender_budget + immune.max_extra),
        trust_hysteresis=True,
    )


def _row_from_training(
    axis: str,
    axis_value: str | float,
    seed: int,
    budget: float,
    action_unit: float,
    immune: ImmuneRegime,
    episodes: int,
    summary: dict,
) -> dict:
    return {
        "axis": axis,
        "axis_value": axis_value,
        "seed": seed,
        "defender_budget": budget,
        "action_unit": action_unit,
        "immune_regime": immune.name,
        "dynamic_defense": immune.dynamic_defense,
        "immune_gain": immune.gain,
        "immune_cap": immune.cap,
        "episodes": episodes,
        "q_eval_peak_risk": summary["q_eval_peak_risk"],
        "q_eval_final_risk": summary["q_eval_final_risk"],
        "baseline_peak_risk": summary["baseline_peak_risk"],
        "baseline_final_risk": summary["baseline_final_risk"],
        "q_minus_baseline_final_risk": summary["q_minus_baseline_final_risk"],
        "q_threshold_crossed": summary["q_threshold_crossed"],
        "q_pause_rate": summary["q_pause_rate"],
    }


def _run_cell(
    seed: int,
    budget: float,
    action_unit: float,
    immune: ImmuneRegime,
    episodes: int,
    agents: int,
    rounds: int,
    steps_per_round: int,
    adversary_budget: float,
) -> dict:
    config = _game_config(
        seed=seed,
        defender_budget=budget,
        action_unit=action_unit,
        immune=immune,
        agents=agents,
        rounds=rounds,
        steps_per_round=steps_per_round,
        adversary_budget=adversary_budget,
    )
    return train_q_adversary(config, episodes=episodes)["summary"]


def run_action_unit_sensitivity(
    action_units: list[float] | None = None,
    budgets: list[float] | None = None,
    seeds: int = 3,
    episodes: int = 5,
    agents: int = 64,
    rounds: int = 6,
    steps_per_round: int = 8,
    adversary_budget: float = 10.0,
    seed_base: int = 20260610,
) -> dict:
    action_units = action_units or DEFAULT_ACTION_UNITS
    budgets = budgets or DEFAULT_BUDGETS
    immune = IMMUNE_REGIMES[2]
    rows = []
    for unit in action_units:
        for seed_index in range(seeds):
            seed = seed_base + seed_index
            for budget in budgets:
                summary = _run_cell(
                    seed=seed,
                    budget=budget,
                    action_unit=unit,
                    immune=immune,
                    episodes=episodes,
                    agents=agents,
                    rounds=rounds,
                    steps_per_round=steps_per_round,
                    adversary_budget=adversary_budget,
                )
                rows.append(_row_from_training(
                    "action_unit",
                    unit,
                    seed,
                    budget,
                    unit,
                    immune,
                    episodes,
                    summary,
                ))
    return {
        "summary": summarize_axis(rows, group_key="action_unit"),
        "rows": rows,
        "config": {
            "action_units": action_units,
            "budgets": budgets,
            "seeds": seeds,
            "episodes": episodes,
            "agents": agents,
            "rounds": rounds,
            "steps_per_round": steps_per_round,
            "adversary_budget": adversary_budget,
            "seed_base": seed_base,
        },
    }


def run_immune_sensitivity(
    regimes: list[ImmuneRegime] | None = None,
    budgets: list[float] | None = None,
    action_unit: float = 1.0,
    seeds: int = 3,
    episodes: int = 5,
    agents: int = 64,
    rounds: int = 6,
    steps_per_round: int = 8,
    adversary_budget: float = 10.0,
    seed_base: int = 20260610,
) -> dict:
    regimes = regimes or IMMUNE_REGIMES
    budgets = budgets or DEFAULT_BUDGETS
    rows = []
    for regime_index, immune in enumerate(regimes):
        for seed_index in range(seeds):
            seed = seed_base + seed_index
            for budget in budgets:
                summary = _run_cell(
                    seed=seed,
                    budget=budget,
                    action_unit=action_unit,
                    immune=immune,
                    episodes=episodes,
                    agents=agents,
                    rounds=rounds,
                    steps_per_round=steps_per_round,
                    adversary_budget=adversary_budget,
                )
                rows.append(_row_from_training(
                    "immune_regime",
                    immune.name,
                    seed,
                    budget,
                    action_unit,
                    immune,
                    episodes,
                    summary,
                ))
    return {
        "summary": summarize_axis(rows, group_key="immune_regime"),
        "rows": rows,
        "config": {
            "regimes": [regime.__dict__ for regime in regimes],
            "budgets": budgets,
            "action_unit": action_unit,
            "seeds": seeds,
            "episodes": episodes,
            "agents": agents,
            "rounds": rounds,
            "steps_per_round": steps_per_round,
            "adversary_budget": adversary_budget,
            "seed_base": seed_base,
        },
    }


def summarize_axis(rows: list[dict], group_key: str, risk_threshold: float = 0.18) -> dict:
    by_group_budget: dict[tuple[str, float], list[dict]] = defaultdict(list)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group = str(row[group_key])
        by_group_budget[(group, row["defender_budget"])].append(row)
        by_group[group].append(row)

    group_summaries = {}
    for group, group_rows in sorted(by_group.items()):
        budgets = sorted({row["defender_budget"] for row in group_rows})
        budget_summary = {}
        all_seed_safe_budget = None
        empirical_robust_safe_budget = None
        conservative_mean_ci_safe_budget = None
        for budget in budgets:
            items = by_group_budget[(group, budget)]
            peaks = [row["q_eval_peak_risk"] for row in items]
            peak_low, peak_high = _mean_ci(peaks)
            crossed = sum(row["q_threshold_crossed"] for row in items)
            budget_summary[f"{budget:.2f}"] = {
                "runs": len(items),
                "mean_q_peak_risk": float(np.mean(peaks)),
                "mean_q_peak_risk_ci95": [peak_low, peak_high],
                "max_q_peak_risk": float(np.max(peaks)),
                "mean_q_final_risk": float(np.mean([row["q_eval_final_risk"] for row in items])),
                "threshold_crossing_rate": float(crossed / max(len(items), 1)),
                "mean_q_pause_rate": float(np.mean([row["q_pause_rate"] for row in items])),
            }
        for budget in budgets:
            if all(not row["q_threshold_crossed"] for row in by_group_budget[(group, budget)]):
                all_seed_safe_budget = budget
                break
        for idx, budget in enumerate(budgets):
            future = budgets[idx:]
            if all(
                not row["q_threshold_crossed"]
                for future_budget in future
                for row in by_group_budget[(group, future_budget)]
            ):
                empirical_robust_safe_budget = budget
                break
        for idx, budget in enumerate(budgets):
            future = budgets[idx:]
            if all(budget_summary[f"{future_budget:.2f}"]["mean_q_peak_risk_ci95"][1] < risk_threshold for future_budget in future):
                conservative_mean_ci_safe_budget = budget
                break

        monotonicity_violations = []
        for prev, curr in zip(budgets, budgets[1:]):
            prev_mean = budget_summary[f"{prev:.2f}"]["mean_q_peak_risk"]
            curr_mean = budget_summary[f"{curr:.2f}"]["mean_q_peak_risk"]
            if curr_mean > prev_mean + 0.005:
                monotonicity_violations.append({
                    "from_budget": prev,
                    "to_budget": curr,
                    "previous_mean_peak": prev_mean,
                    "current_mean_peak": curr_mean,
                })

        group_summaries[group] = {
            "budgets_tested": budgets,
            "runs": len(group_rows),
            "all_seed_safe_budget": all_seed_safe_budget,
            "empirical_robust_safe_budget": empirical_robust_safe_budget,
            "conservative_mean_ci_safe_budget": conservative_mean_ci_safe_budget,
            "monotonicity_violation_count": len(monotonicity_violations),
            "monotonicity_violations": monotonicity_violations,
            "budget_summary": budget_summary,
        }

    return {
        "axis": group_key,
        "risk_threshold": risk_threshold,
        "groups": group_summaries,
        "runs": len(rows),
        "seeds": len({row["seed"] for row in rows}),
    }


def infer_mechanism(payload: dict) -> dict:
    action_groups = payload["action_unit"]["summary"]["groups"]
    immune_groups = payload["immune"]["summary"]["groups"]
    action_thresholds = {
        group: data["empirical_robust_safe_budget"]
        for group, data in action_groups.items()
    }
    immune_thresholds = {
        group: data["empirical_robust_safe_budget"]
        for group, data in immune_groups.items()
    }
    no_immune_threshold = immune_thresholds.get("none")
    default_threshold = immune_thresholds.get("default")
    action_threshold_values = [
        value for value in action_thresholds.values() if value is not None
    ]
    threshold_range = (
        max(action_threshold_values) - min(action_threshold_values)
        if action_threshold_values
        else None
    )
    return {
        "action_unit_thresholds": action_thresholds,
        "immune_thresholds": immune_thresholds,
        "action_unit_threshold_range": threshold_range,
        "immune_bootstrap_supported": (
            no_immune_threshold is not None
            and default_threshold is not None
            and default_threshold < no_immune_threshold
        ),
        "action_unit_artifact_supported": (
            threshold_range is not None and threshold_range > 0.25
        ),
    }


def run_mechanism_sensitivity(
    budgets: list[float] | None = None,
    action_units: list[float] | None = None,
    seeds: int = 3,
    episodes: int = 5,
    agents: int = 64,
    rounds: int = 6,
    steps_per_round: int = 8,
    seed_base: int = 20260610,
) -> dict:
    payload = {
        "action_unit": run_action_unit_sensitivity(
            action_units=action_units,
            budgets=budgets,
            seeds=seeds,
            episodes=episodes,
            agents=agents,
            rounds=rounds,
            steps_per_round=steps_per_round,
            seed_base=seed_base,
        ),
        "immune": run_immune_sensitivity(
            budgets=budgets,
            seeds=seeds,
            episodes=episodes,
            agents=agents,
            rounds=rounds,
            steps_per_round=steps_per_round,
            seed_base=seed_base,
        ),
    }
    payload["mechanism_inference"] = infer_mechanism(payload)
    return payload


def _render_axis_table(summary: dict) -> list[str]:
    lines = [
        "| Group | Robust Safe | Conservative CI Safe | Violations | Budget | Mean Peak | Max Peak | Crossing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, data in summary["groups"].items():
        first = True
        for budget in data["budgets_tested"]:
            row = data["budget_summary"][f"{budget:.2f}"]
            prefix = (
                f"| {group} | {data['empirical_robust_safe_budget']} | "
                f"{data['conservative_mean_ci_safe_budget']} | "
                f"{data['monotonicity_violation_count']} |"
                if first
                else "|  |  |  |  |"
            )
            lines.append(
                f"{prefix} {budget:.2f} | {row['mean_q_peak_risk']:.3f} | "
                f"{row['max_q_peak_risk']:.3f} | {row['threshold_crossing_rate']:.1%} |"
            )
            first = False
    return lines


def render_report(payload: dict) -> str:
    inference = payload["mechanism_inference"]
    lines = [
        "# Mechanism Sensitivity Report",
        "",
        "## Purpose",
        "",
        (
            "Tests whether the dense frontier's `0.25` defender-budget threshold is driven by "
            "action-unit granularity, dynamic immune-response bootstrapping, or a genuine "
            "topology-repair minimum under the current Q-adversary."
        ),
        "",
        "## Inference",
        "",
        f"- Action-unit thresholds: `{inference['action_unit_thresholds']}`",
        f"- Immune-regime thresholds: `{inference['immune_thresholds']}`",
        f"- Action-unit threshold range: `{inference['action_unit_threshold_range']}`",
        f"- Immune bootstrap supported: `{inference['immune_bootstrap_supported']}`",
        f"- Action-unit artifact supported: `{inference['action_unit_artifact_supported']}`",
        "",
        "## Action-Unit Sensitivity",
        "",
    ]
    lines.extend(_render_axis_table(payload["action_unit"]["summary"]))
    lines.extend([
        "",
        "## Immune-Response Sensitivity",
        "",
    ])
    lines.extend(_render_axis_table(payload["immune"]["summary"]))
    lines.extend([
        "",
        "## Reading",
        "",
        (
            "A lower threshold under the default immune regime than under `none` means the first "
            "defensive action is acting as a bootstrap into later immune replenishment. A threshold "
            "that moves substantially with `action_unit` means the frontier is also a control-resolution "
            "effect. If thresholds stay fixed across action units and immune regimes, the result is "
            "stronger evidence for an intrinsic topology-repair minimum."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("influence_mechanism_sensitivity_output.json"),
    output_md: Path = Path("INFLUENCE_MECHANISM_SENSITIVITY_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mechanism sensitivity analysis.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--steps-per-round", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=20260610)
    parser.add_argument("--output-json", type=Path, default=Path("influence_mechanism_sensitivity_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("INFLUENCE_MECHANISM_SENSITIVITY_REPORT.md"))
    args = parser.parse_args()

    payload = run_mechanism_sensitivity(
        budgets=[0.0, 0.25, 0.5] if args.quick else DEFAULT_BUDGETS,
        action_units=[0.5, 1.0] if args.quick else DEFAULT_ACTION_UNITS,
        seeds=args.seeds if args.seeds is not None else (1 if args.quick else 5),
        episodes=args.episodes if args.episodes is not None else (1 if args.quick else 8),
        agents=args.agents if args.agents is not None else (32 if args.quick else 96),
        rounds=args.rounds if args.rounds is not None else (3 if args.quick else 8),
        steps_per_round=args.steps_per_round if args.steps_per_round is not None else (4 if args.quick else 12),
        seed_base=args.seed_base,
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps(payload["mechanism_inference"], indent=2))


if __name__ == "__main__":
    main()

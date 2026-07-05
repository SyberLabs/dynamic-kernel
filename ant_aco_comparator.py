"""
Classical ACO comparator for the DTE ant foraging experiments.

This is a controlled null model, not a full ant-colony optimization package.
It uses the same terrain, depletion, shock, and hazard semantics as
ant_foraging_dte.py, but replaces DTE's telemetry/node-feature transition
kernel with a classical pheromone-times-heuristic transition rule:

    P_ij proportional to pheromone_ij^alpha * heuristic_ij^beta

The purpose is institutional validity: determine whether DTE is merely
renaming ACO, or whether telemetry-coupled routing creates a richer control
surface.

Usage:
    .venv\\Scripts\\python.exe ant_aco_comparator.py --quick
    .venv\\Scripts\\python.exe ant_aco_comparator.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from ant_foraging_dte import (
    BRANCH_NODES,
    FOOD_NODES,
    ForagingPolicy,
    ForagingScenario,
    SimulationConfig,
    _entropy,
    simulate as simulate_dte,
    topology,
)
from ant_foraging_phase_diagram import _policy_for, phase_scenarios


@dataclass(frozen=True)
class ACOConfig:
    agents: int = 96
    steps: int = 110
    seed: int = 17
    pheromone_alpha: float = 1.15
    heuristic_beta: float = 2.20
    initial_pheromone: float = 1.0
    max_path_memory: int = 18


@dataclass(frozen=True)
class ComparisonPoint:
    name: str
    pheromone_deposit: float
    evaporation: float
    scout_share: float
    temperature: float = 0.95


def comparison_points(quick: bool = False) -> list[ComparisonPoint]:
    points = [
        ComparisonPoint("no_memory", 0.0, 0.005, 0.18, 1.05),
        ComparisonPoint("adaptive_candidate", 0.65, 0.120, 0.44, 1.00),
        ComparisonPoint("danger_candidate", 0.65, 0.005, 0.18, 0.85),
        ComparisonPoint("strong_memory_low_scout", 1.20, 0.035, 0.02, 0.80),
    ]
    if quick:
        return points
    return points + [
        ComparisonPoint("moderate_memory", 0.16, 0.035, 0.18, 0.95),
        ComparisonPoint("strong_memory_high_evap", 1.20, 0.120, 0.44, 1.00),
    ]


def _to_dte_policy(point: ComparisonPoint) -> ForagingPolicy:
    template = ForagingPolicy(
        name=point.name,
        pheromone_deposit=point.pheromone_deposit,
        evaporation=point.evaporation,
        scout_share=point.scout_share,
        risk_averse_share=0.16,
        temperature=point.temperature,
        pheromone_cap=max(2.6, point.pheromone_deposit * 6.5),
    )
    return _policy_for(
        point.pheromone_deposit,
        point.evaporation,
        point.scout_share,
        template,
    )


def _friction_matrix(topo, scenario: ForagingScenario, shock_active: bool) -> np.ndarray:
    idx = {label: i for i, label in enumerate(topo.labels)}
    matrix = np.zeros((topo.N, topo.N), dtype=float)
    for (src, dst), value in scenario.initial_friction.items():
        matrix[idx[src], idx[dst]] += value
    if shock_active:
        for (src, dst), value in scenario.shock_friction.items():
            matrix[idx[src], idx[dst]] += value
    return matrix


def _edge_neighbors(topo) -> list[np.ndarray]:
    neighbors = []
    for i in range(topo.N):
        row = np.where(np.isfinite(topo.distance_matrix[i]) & (topo.distance_matrix[i] > 0.0))[0]
        neighbors.append(row.astype(int))
    return neighbors


def _aco_probs(
    topo,
    neighbors: np.ndarray,
    src: int,
    carrying: bool,
    pheromone: np.ndarray,
    friction: np.ndarray,
    config: ACOConfig,
    policy: ComparisonPoint,
) -> np.ndarray:
    labels = topo.labels
    distances = topo.distance_matrix[src, neighbors]
    effective_cost = np.maximum(0.05, distances - friction[src, neighbors])
    heuristic = 1.0 / effective_cost

    for k, dst in enumerate(neighbors):
        label = labels[int(dst)]
        if carrying:
            if label == "Nest":
                heuristic[k] *= 6.0
            elif label == "Return Corridor":
                heuristic[k] *= 4.0
            elif label in FOOD_NODES:
                heuristic[k] *= 0.25
        else:
            if label in FOOD_NODES:
                heuristic[k] *= 4.0
            elif label == "Hazard Zone":
                heuristic[k] *= 0.12
            elif label == "Return Corridor":
                heuristic[k] *= 0.35

    trail = np.maximum(1e-6, pheromone[src, neighbors])
    weights = (trail ** config.pheromone_alpha) * (heuristic ** config.heuristic_beta)
    if float(np.sum(weights)) <= 0.0:
        probs = np.full(len(neighbors), 1.0 / len(neighbors))
    else:
        probs = weights / np.sum(weights)

    exploration = max(0.0, min(policy.scout_share, 0.95))
    if exploration > 0.0:
        uniform = np.full(len(neighbors), 1.0 / len(neighbors))
        probs = (1.0 - exploration) * probs + exploration * uniform
    return probs


def simulate_aco(
    config: ACOConfig,
    scenario: ForagingScenario,
    policy: ComparisonPoint,
    seed_offset: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed + seed_offset)
    topo = topology()
    idx = {label: i for i, label in enumerate(topo.labels)}
    neighbors = _edge_neighbors(topo)
    edge_mask = np.isfinite(topo.distance_matrix) & (topo.distance_matrix > 0.0)

    nest = idx["Nest"]
    hazard = idx["Hazard Zone"]
    fork = idx["Trail Fork"]
    food_idx = {idx[name]: name for name in FOOD_NODES}
    branch_idx = {idx[name]: name for name in BRANCH_NODES}

    positions = np.full(config.agents, nest, dtype=int)
    carrying = np.zeros(config.agents, dtype=bool)
    carried_source: list[str | None] = [None for _ in range(config.agents)]
    paths: list[list[tuple[int, int]]] = [[] for _ in range(config.agents)]
    pheromone = np.zeros((topo.N, topo.N), dtype=float)
    pheromone[edge_mask] = config.initial_pheromone

    food_remaining = dict(scenario.food_inventory)
    initial_food_total = sum(food_remaining.values())
    edge_counts = np.zeros((topo.N, topo.N), dtype=int)
    fork_counts = {name: 0 for name in BRANCH_NODES}
    returns_by_source = {name: 0 for name in FOOD_NODES}
    visits_by_source = {name: 0 for name in FOOD_NODES}
    empty_food_visits = 0
    hazard_hits = 0
    food_returns = 0
    path_lengths_to_return: list[int] = []
    returns_by_step: list[int] = []

    for step in range(config.steps):
        shock_active = scenario.shock_step is not None and step >= scenario.shock_step
        friction = _friction_matrix(topo, scenario, shock_active)
        pheromone[edge_mask] = (
            config.initial_pheromone
            + (pheromone[edge_mask] - config.initial_pheromone) * max(0.0, 1.0 - policy.evaporation)
        )

        step_returns = 0
        for ant in range(config.agents):
            src = int(positions[ant])
            node_neighbors = neighbors[src]
            if len(node_neighbors) == 0:
                continue
            probs = _aco_probs(
                topo,
                node_neighbors,
                src,
                bool(carrying[ant]),
                pheromone,
                friction,
                config,
                policy,
            )
            dst = int(rng.choice(node_neighbors, p=probs))

            edge_counts[src, dst] += 1
            paths[ant].append((src, dst))
            if len(paths[ant]) > config.max_path_memory:
                paths[ant] = paths[ant][-config.max_path_memory :]
            if src == fork and dst in branch_idx:
                fork_counts[branch_idx[dst]] += 1

            if dst == hazard:
                hazard_hits += 1
                carrying[ant] = False
                carried_source[ant] = None
                paths[ant] = []
                positions[ant] = nest
                continue

            if carrying[ant] and dst == nest:
                food_returns += 1
                step_returns += 1
                source = carried_source[ant]
                if source is not None:
                    returns_by_source[source] += 1
                path_lengths_to_return.append(len(paths[ant]))
                if policy.pheromone_deposit > 0.0 and paths[ant]:
                    deposit = policy.pheromone_deposit / math.sqrt(max(1, len(paths[ant])))
                    for i, j in paths[ant]:
                        pheromone[i, j] += deposit
                    cap = max(2.6, policy.pheromone_deposit * 6.5)
                    np.clip(pheromone, 0.0, cap, out=pheromone)
                carrying[ant] = False
                carried_source[ant] = None
                paths[ant] = []
                positions[ant] = dst
                continue

            if not carrying[ant] and dst in food_idx:
                food_name = food_idx[dst]
                visits_by_source[food_name] += 1
                if food_remaining.get(food_name, 0) > 0:
                    food_remaining[food_name] -= 1
                    carrying[ant] = True
                    carried_source[ant] = food_name
                else:
                    empty_food_visits += 1

            if not carrying[ant] and dst == nest:
                paths[ant] = []
            positions[ant] = dst
        returns_by_step.append(step_returns)

    fork_total = sum(fork_counts.values())
    food_visit_total = max(1, sum(visits_by_source.values()))
    rich_returns = returns_by_source["Rich Food Patch"]
    sparse_returns = returns_by_source["Sparse Food Patch"]
    return_total = max(1, rich_returns + sparse_returns)

    shock_recovery_steps: int | None = None
    if scenario.shock_step is not None and scenario.shock_step >= 12 and scenario.shock_step < len(returns_by_step):
        pre = np.mean(returns_by_step[max(0, scenario.shock_step - 12) : scenario.shock_step])
        target = 0.8 * pre
        if target > 0:
            for t in range(scenario.shock_step + 6, config.steps):
                window = returns_by_step[max(0, t - 8) : t + 1]
                if np.mean(window) >= target:
                    shock_recovery_steps = int(t - scenario.shock_step)
                    break

    return {
        "framework": "ACO",
        "scenario": scenario.name,
        "policy": policy.name,
        "agents": config.agents,
        "steps": config.steps,
        "food_returned": int(food_returns),
        "returns_per_100_ant_steps": round(100.0 * food_returns / (config.agents * config.steps), 4),
        "food_completion_rate": round(food_returns / max(1, initial_food_total), 4),
        "food_remaining": {key: int(value) for key, value in food_remaining.items()},
        "returns_by_source": {key: int(value) for key, value in returns_by_source.items()},
        "visits_by_source": {key: int(value) for key, value in visits_by_source.items()},
        "empty_food_visits": int(empty_food_visits),
        "empty_food_visit_rate": round(empty_food_visits / food_visit_total, 4),
        "rich_return_share": round(rich_returns / return_total, 4),
        "rich_visit_share": round(visits_by_source["Rich Food Patch"] / food_visit_total, 4),
        "sparse_visit_share": round(visits_by_source["Sparse Food Patch"] / food_visit_total, 4),
        "hazard_hits": int(hazard_hits),
        "hazard_rate": round(hazard_hits / (config.agents * config.steps), 4),
        "fork_counts": {key: int(value) for key, value in fork_counts.items()},
        "fork_entropy": round(_entropy(fork_counts), 4),
        "lock_in_index": round(max(fork_counts.values()) / fork_total if fork_total else 0.0, 4),
        "dominant_branch": max(fork_counts, key=fork_counts.get) if fork_total else "none",
        "mean_return_path_length": round(float(np.mean(path_lengths_to_return)), 4)
        if path_lengths_to_return
        else 0.0,
        "pheromone_mass": round(float(np.sum(pheromone[edge_mask] - config.initial_pheromone)), 4),
        "pheromone_max": round(float(np.max(pheromone[edge_mask])), 4),
        "shock_recovery_steps": shock_recovery_steps,
    }


def _score(row: dict[str, Any]) -> float:
    return float(
        row["food_completion_rate"]
        - 0.32 * row["empty_food_visit_rate"]
        - 0.70 * row["hazard_rate"]
        - 0.06 * row["lock_in_index"]
        + 0.03 * row["fork_entropy"]
    )


def _mean_row(items: list[dict[str, Any]]) -> dict[str, Any]:
    first = items[0]
    numeric = [
        "food_returned",
        "returns_per_100_ant_steps",
        "food_completion_rate",
        "empty_food_visit_rate",
        "rich_visit_share",
        "hazard_rate",
        "fork_entropy",
        "lock_in_index",
        "mean_return_path_length",
        "pheromone_mass",
        "pheromone_max",
        "score",
    ]
    row = {
        "framework": first["framework"],
        "scenario": first["scenario"],
        "point": first["point"],
        "runs": len(items),
    }
    for key in numeric:
        row[key] = float(mean(item[key] for item in items))
    recoveries = [item["shock_recovery_steps"] for item in items if item["shock_recovery_steps"] is not None]
    row["mean_shock_recovery_steps"] = float(mean(recoveries)) if recoveries else None
    return row


def summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["framework"], row["scenario"], row["point"])].append(row)

    aggregate_rows = [_mean_row(items) for items in by_group.values()]
    by_pair = {
        (row["scenario"], row["point"], row["framework"]): row
        for row in aggregate_rows
    }

    comparisons = []
    win_counts = {"DTE": 0, "ACO": 0, "tie": 0}
    for scenario in sorted({row["scenario"] for row in aggregate_rows}):
        for point in sorted({row["point"] for row in aggregate_rows if row["scenario"] == scenario}):
            dte = by_pair.get((scenario, point, "DTE"))
            aco = by_pair.get((scenario, point, "ACO"))
            if dte is None or aco is None:
                continue
            score_delta = dte["score"] - aco["score"]
            if score_delta > 0.02:
                verdict = "DTE"
            elif score_delta < -0.02:
                verdict = "ACO"
            else:
                verdict = "tie"
            win_counts[verdict] += 1
            comparisons.append({
                "scenario": scenario,
                "point": point,
                "dte_completion_minus_aco": dte["food_completion_rate"] - aco["food_completion_rate"],
                "dte_empty_minus_aco": dte["empty_food_visit_rate"] - aco["empty_food_visit_rate"],
                "dte_hazard_minus_aco": dte["hazard_rate"] - aco["hazard_rate"],
                "dte_entropy_minus_aco": dte["fork_entropy"] - aco["fork_entropy"],
                "dte_score_minus_aco": score_delta,
                "verdict": verdict,
            })

    return {
        "aggregate_rows": aggregate_rows,
        "comparisons": comparisons,
        "win_counts": win_counts,
        "best_dte_advantage": max(comparisons, key=lambda row: row["dte_score_minus_aco"], default=None),
        "best_aco_advantage": min(comparisons, key=lambda row: row["dte_score_minus_aco"], default=None),
    }


def run_comparison(
    dte_config: SimulationConfig | None = None,
    aco_config: ACOConfig | None = None,
    quick: bool = False,
    seeds: int = 3,
) -> dict[str, Any]:
    dte_config = dte_config or (SimulationConfig(agents=48, steps=90) if quick else SimulationConfig())
    aco_config = aco_config or ACOConfig(
        agents=dte_config.agents,
        steps=dte_config.steps,
        seed=dte_config.seed,
    )
    selected_scenarios = phase_scenarios(quick)
    selected_points = comparison_points(quick)
    rows: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(selected_scenarios):
        for point_index, point in enumerate(selected_points):
            dte_policy = _to_dte_policy(point)
            for seed_index in range(seeds):
                seed_offset = scenario_index * 100_000 + point_index * 10_000 + seed_index * 997
                dte_row = simulate_dte(dte_config, scenario, dte_policy, seed_offset=seed_offset)
                dte_row["framework"] = "DTE"
                dte_row["point"] = point.name
                dte_row["score"] = _score(dte_row)
                rows.append(dte_row)

                aco_row = simulate_aco(aco_config, scenario, point, seed_offset=seed_offset)
                aco_row["point"] = point.name
                aco_row["score"] = _score(aco_row)
                rows.append(aco_row)

    return {
        "config": {
            "dte": asdict(dte_config),
            "aco": asdict(aco_config),
            "quick": quick,
            "seeds": seeds,
        },
        "points": [asdict(point) for point in selected_points],
        "scenarios": [scenario.name for scenario in selected_scenarios],
        "summary": summarize_comparison(rows),
        "rows": rows,
    }


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Ant DTE vs Classical ACO Comparator",
        "",
        "## Scope",
        "",
        (
            "Controlled comparison on the same ant terrain. DTE uses telemetry, node "
            "features, and pheromone-as-friction; ACO uses a classical pheromone-times-"
            "heuristic transition rule with the same depletion and shock semantics."
        ),
        "",
        f"- Raw runs: `{len(payload['rows'])}`",
        f"- Seeds: `{payload['config']['seeds']}`",
        f"- Scenarios: `{len(payload['scenarios'])}`",
        f"- Policy points: `{len(payload['points'])}`",
        "",
        "## Framework Wins",
        "",
        "| Verdict | Count |",
        "|---|---:|",
    ]
    for key, value in summary["win_counts"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## Pairwise Comparisons",
            "",
            "| Scenario | Point | Completion DTE-ACO | Empty DTE-ACO | Hazard DTE-ACO | Entropy DTE-ACO | Score DTE-ACO | Verdict |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['scenario']} | {row['point']} | "
            f"{row['dte_completion_minus_aco']:.3f} | {row['dte_empty_minus_aco']:.3f} | "
            f"{row['dte_hazard_minus_aco']:.3f} | {row['dte_entropy_minus_aco']:.3f} | "
            f"{row['dte_score_minus_aco']:.3f} | {row['verdict']} |"
        )

    if summary["best_dte_advantage"] is not None:
        best = summary["best_dte_advantage"]
        worst = summary["best_aco_advantage"]
        lines.extend(
            [
                "",
                "## Extremes",
                "",
                f"- Best DTE advantage: `{best['scenario']} / {best['point']}` "
                f"with score delta `{best['dte_score_minus_aco']:.3f}`",
                f"- Best ACO advantage: `{worst['scenario']} / {worst['point']}` "
                f"with score delta `{worst['dte_score_minus_aco']:.3f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "If ACO dominates, the ant result is mostly a pheromone-memory result. "
                "If DTE wins or ties while exposing different empty-visit, hazard, and "
                "entropy tradeoffs, the kernel is doing more: it separates individual "
                "intent, terrain semantics, and collective path memory."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], json_path: Path, report_path: Path) -> None:
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ant DTE vs ACO comparator.")
    parser.add_argument("--quick", action="store_true", help="Run a reduced comparator.")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--json", default="ant_aco_comparator_output.json")
    parser.add_argument("--report", default="ANT_ACO_COMPARATOR_REPORT.md")
    args = parser.parse_args()

    config = SimulationConfig(agents=48, steps=90) if args.quick else SimulationConfig()
    seeds = args.seeds if args.seeds is not None else (2 if args.quick else 5)
    payload = run_comparison(dte_config=config, quick=args.quick, seeds=seeds)
    write_outputs(payload, Path(args.json), Path(args.report))
    print(render_report(payload))


if __name__ == "__main__":
    main()

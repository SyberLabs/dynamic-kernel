"""
Seed-robust ant foraging memory sweep.

The phase diagram proves that adaptive memory and deadly familiarity both
exist in the ant topology. This harness asks whether the boundary survives
random colony histories by classifying each seed against its own matched
no-pheromone baseline.

Usage:
    .venv\\Scripts\\python.exe ant_foraging_seed_sweep.py --quick
    .venv\\Scripts\\python.exe ant_foraging_seed_sweep.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from ant_foraging_dte import SimulationConfig, policies, simulate
from ant_foraging_phase_diagram import (
    PHASE_CODES,
    _classify_phase,
    _policy_for,
    _score,
    phase_grid,
    phase_scenarios,
)


def _mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    array = np.array(values, dtype=np.float64)
    avg = float(array.mean())
    half_width = float(z * array.std(ddof=1) / np.sqrt(len(array)))
    return avg - half_width, avg + half_width


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _param_offset(
    scenario_index: int,
    scout_share: float,
    deposit: float,
    evaporation: float,
) -> int:
    return (
        scenario_index * 100_000
        + int(round(scout_share * 1000)) * 100
        + int(round(deposit * 1000)) * 10
        + int(round(evaporation * 1000))
    )


def _enrich_row(
    row: dict[str, Any],
    baseline: dict[str, Any],
    seed_index: int,
    seed: int,
    deposit: float,
    evaporation: float,
    scout_share: float,
) -> dict[str, Any]:
    row["seed_index"] = seed_index
    row["seed"] = seed
    row["pheromone_deposit"] = deposit
    row["evaporation"] = evaporation
    row["scout_share"] = scout_share
    row["memory_ratio"] = float(deposit / max(evaporation, 1e-9))
    row["food_completion_gain"] = row["food_completion_rate"] - baseline["food_completion_rate"]
    row["empty_visit_delta"] = row["empty_food_visit_rate"] - baseline["empty_food_visit_rate"]
    row["hazard_delta"] = row["hazard_rate"] - baseline["hazard_rate"]
    row["phase"] = _classify_phase(row, baseline)
    row["phase_code"] = PHASE_CODES[row["phase"]]
    row["score"] = _score(row)
    return row


def run_seed_robust_phase_sweep(
    config: SimulationConfig | None = None,
    quick: bool = False,
    seeds: int = 5,
) -> dict[str, Any]:
    config = config or (SimulationConfig(agents=48, steps=90) if quick else SimulationConfig())
    deposits, evaporations, scout_shares = phase_grid(quick)
    selected_scenarios = phase_scenarios(quick)
    template = policies()[1]
    baseline_evaporation = evaporations[0]

    rows: list[dict[str, Any]] = []
    baselines: dict[tuple[int, str, float], dict[str, Any]] = {}

    for seed_index in range(seeds):
        seed = config.seed + seed_index * 100_003
        seed_config = replace(config, seed=seed)
        for scenario_index, scenario in enumerate(selected_scenarios):
            for scout_share in scout_shares:
                baseline_policy = _policy_for(0.0, baseline_evaporation, scout_share, template)
                baseline = simulate(
                    seed_config,
                    scenario,
                    baseline_policy,
                    seed_offset=_param_offset(
                        scenario_index,
                        scout_share,
                        0.0,
                        baseline_evaporation,
                    ),
                )
                baseline = _enrich_row(
                    baseline,
                    baseline,
                    seed_index,
                    seed,
                    0.0,
                    baseline_evaporation,
                    scout_share,
                )
                baselines[(seed_index, scenario.name, scout_share)] = baseline
                rows.append(baseline)

                for deposit in deposits:
                    if deposit == 0.0:
                        continue
                    for evaporation in evaporations:
                        policy = _policy_for(deposit, evaporation, scout_share, template)
                        row = simulate(
                            seed_config,
                            scenario,
                            policy,
                            seed_offset=_param_offset(
                                scenario_index,
                                scout_share,
                                deposit,
                                evaporation,
                            ),
                        )
                        rows.append(
                            _enrich_row(
                                row,
                                baseline,
                                seed_index,
                                seed,
                                deposit,
                                evaporation,
                                scout_share,
                            )
                        )

    summary = summarize_seed_robust_phase(rows)
    return {
        "config": asdict(config) | {"quick": quick, "seeds": seeds},
        "deposits": deposits,
        "evaporations": evaporations,
        "scout_shares": scout_shares,
        "phase_codes": PHASE_CODES,
        "summary": summary,
        "rows": rows,
    }


def summarize_seed_robust_phase(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phase_counts = Counter(row["phase"] for row in rows)
    by_cell: dict[tuple[str, float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario"],
            row["pheromone_deposit"],
            row["evaporation"],
            row["scout_share"],
        )
        by_cell[key].append(row)

    cell_summary = []
    for (scenario, deposit, evaporation, scout_share), items in sorted(by_cell.items()):
        deadly = sum(row["phase"] == "deadly_familiarity" for row in items)
        adaptive = sum(row["phase"] == "adaptive_memory" for row in items)
        neutral = sum(row["phase"] == "neutral_memory" for row in items)
        deadly_low, deadly_high = _wilson_interval(deadly, len(items))
        adaptive_low, adaptive_high = _wilson_interval(adaptive, len(items))
        completion_low, completion_high = _mean_ci([row["food_completion_rate"] for row in items])
        empty_low, empty_high = _mean_ci([row["empty_food_visit_rate"] for row in items])
        score_low, score_high = _mean_ci([row["score"] for row in items])
        cell_summary.append({
            "scenario": scenario,
            "pheromone_deposit": deposit,
            "evaporation": evaporation,
            "scout_share": scout_share,
            "memory_ratio": float(deposit / max(evaporation, 1e-9)),
            "runs": len(items),
            "deadly_rate": float(deadly / len(items)),
            "deadly_rate_ci95": [deadly_low, deadly_high],
            "adaptive_rate": float(adaptive / len(items)),
            "adaptive_rate_ci95": [adaptive_low, adaptive_high],
            "neutral_rate": float(neutral / len(items)),
            "mean_completion": float(mean(row["food_completion_rate"] for row in items)),
            "mean_completion_ci95": [completion_low, completion_high],
            "mean_empty_visit_rate": float(mean(row["empty_food_visit_rate"] for row in items)),
            "mean_empty_visit_rate_ci95": [empty_low, empty_high],
            "mean_score": float(mean(row["score"] for row in items)),
            "mean_score_ci95": [score_low, score_high],
        })

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cell_summary:
        by_scenario[cell["scenario"]].append(cell)

    scenario_summary = {}
    for scenario, cells in sorted(by_scenario.items()):
        non_baseline = [cell for cell in cells if cell["pheromone_deposit"] > 0.0]
        deadly_any = [cell for cell in non_baseline if cell["deadly_rate"] > 0.0]
        deadly_majority = [cell for cell in non_baseline if cell["deadly_rate"] >= 0.50]
        deadly_all_seed = [cell for cell in non_baseline if cell["deadly_rate"] >= 1.0]
        adaptive_majority = [cell for cell in non_baseline if cell["adaptive_rate"] >= 0.50]
        scenario_summary[scenario] = {
            "cells": len(cells),
            "non_baseline_cells": len(non_baseline),
            "any_deadly_min_memory_ratio": min(
                (cell["memory_ratio"] for cell in deadly_any),
                default=None,
            ),
            "majority_deadly_min_memory_ratio": min(
                (cell["memory_ratio"] for cell in deadly_majority),
                default=None,
            ),
            "all_seed_deadly_min_memory_ratio": min(
                (cell["memory_ratio"] for cell in deadly_all_seed),
                default=None,
            ),
            "majority_adaptive_max_memory_ratio": max(
                (cell["memory_ratio"] for cell in adaptive_majority),
                default=None,
            ),
            "best_mean_score": max(non_baseline or cells, key=lambda cell: cell["mean_score"]),
            "highest_deadly_rate": max(non_baseline or cells, key=lambda cell: cell["deadly_rate"]),
            "highest_adaptive_rate": max(non_baseline or cells, key=lambda cell: cell["adaptive_rate"]),
        }

    monotonicity = _deadly_monotonicity(cell_summary)
    return {
        "runs": len(rows),
        "seeds": len({row["seed"] for row in rows}),
        "phase_counts": dict(phase_counts),
        "cell_summary": cell_summary,
        "scenario_summary": scenario_summary,
        "deadly_monotonicity": monotonicity,
    }


def _deadly_monotonicity(cell_summary: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for cell in cell_summary:
        if cell["pheromone_deposit"] == 0.0:
            continue
        groups[(cell["scenario"], cell["evaporation"], cell["scout_share"])].append(cell)

    violations = []
    comparisons = 0
    for (scenario, evaporation, scout_share), cells in groups.items():
        ordered = sorted(cells, key=lambda cell: cell["memory_ratio"])
        for prev, curr in zip(ordered, ordered[1:]):
            comparisons += 1
            if curr["deadly_rate"] + 1e-9 < prev["deadly_rate"]:
                violations.append({
                    "scenario": scenario,
                    "evaporation": evaporation,
                    "scout_share": scout_share,
                    "previous_ratio": prev["memory_ratio"],
                    "current_ratio": curr["memory_ratio"],
                    "previous_deadly_rate": prev["deadly_rate"],
                    "current_deadly_rate": curr["deadly_rate"],
                })
    return {
        "comparisons": comparisons,
        "violation_count": len(violations),
        "violations": violations,
    }


def _fmt_cell(cell: dict[str, Any]) -> str:
    return (
        f"deposit={cell['pheromone_deposit']:.3f}, evap={cell['evaporation']:.3f}, "
        f"scout={cell['scout_share']:.2f}, ratio={cell['memory_ratio']:.2f}, "
        f"deadly={cell['deadly_rate']:.1%}, adaptive={cell['adaptive_rate']:.1%}, "
        f"completion={cell['mean_completion']:.3f}, empty={cell['mean_empty_visit_rate']:.3f}"
    )


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Seed-Robust Ant Memory Sweep",
        "",
        "## Scope",
        "",
        (
            "Seed-level validation of the ant foraging phase diagram. Each pheromone "
            "cell is compared against a no-pheromone baseline with the same seed, "
            "scenario, and scout share."
        ),
        "",
        f"- Runs: `{summary['runs']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Agents per run: `{payload['config']['agents']}`",
        f"- Steps per run: `{payload['config']['steps']}`",
        "",
        "## Phase Counts",
        "",
        "| Phase | Count |",
        "|---|---:|",
    ]
    for phase, count in sorted(summary["phase_counts"].items()):
        lines.append(f"| `{phase}` | {count} |")

    lines.extend(
        [
            "",
            "## Robust Thresholds",
            "",
            "| Scenario | Any deadly min ratio | Majority deadly min ratio | All-seed deadly min ratio | Majority adaptive max ratio | Best mean-score cell |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for scenario, item in summary["scenario_summary"].items():
        any_deadly = (
            "n/a"
            if item["any_deadly_min_memory_ratio"] is None
            else f"{item['any_deadly_min_memory_ratio']:.2f}"
        )
        majority_deadly = (
            "n/a"
            if item["majority_deadly_min_memory_ratio"] is None
            else f"{item['majority_deadly_min_memory_ratio']:.2f}"
        )
        all_deadly = (
            "n/a"
            if item["all_seed_deadly_min_memory_ratio"] is None
            else f"{item['all_seed_deadly_min_memory_ratio']:.2f}"
        )
        adaptive = (
            "n/a"
            if item["majority_adaptive_max_memory_ratio"] is None
            else f"{item['majority_adaptive_max_memory_ratio']:.2f}"
        )
        lines.append(
            f"| {scenario} | {any_deadly} | {majority_deadly} | {all_deadly} | "
            f"{adaptive} | {_fmt_cell(item['best_mean_score'])} |"
        )

    mono = summary["deadly_monotonicity"]
    lines.extend(
        [
            "",
            "## Monotonicity Check",
            "",
            f"- Comparisons: `{mono['comparisons']}`",
            f"- Violations: `{mono['violation_count']}`",
            "",
            "This check asks whether deadly-familiarity rate increases monotonically "
            "as memory ratio rises when scenario, evaporation, and scout share are fixed.",
            "",
            "## Highest-Risk Cells",
            "",
            "| Scenario | Cell | Deadly CI95 | Adaptive CI95 | Score CI95 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    risky_cells = sorted(
        [cell for cell in summary["cell_summary"] if cell["pheromone_deposit"] > 0.0],
        key=lambda cell: (cell["deadly_rate"], cell["mean_empty_visit_rate"]),
        reverse=True,
    )[:8]
    for cell in risky_cells:
        lines.append(
            f"| {cell['scenario']} | {_fmt_cell(cell)} | "
            f"[{cell['deadly_rate_ci95'][0]:.2f}, {cell['deadly_rate_ci95'][1]:.2f}] | "
            f"[{cell['adaptive_rate_ci95'][0]:.2f}, {cell['adaptive_rate_ci95'][1]:.2f}] | "
            f"[{cell['mean_score_ci95'][0]:.3f}, {cell['mean_score_ci95'][1]:.3f}] |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The seed-robust object is a probabilistic boundary, not a single magic "
                "ratio. A memory ratio enters the danger region when multiple colony "
                "histories independently classify as deadly familiarity. This is the "
                "right standard for later comparison with classical ACO and other "
                "memory-based routing algorithms."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], json_path: Path, report_path: Path) -> None:
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seed-robust ant memory sweep.")
    parser.add_argument("--quick", action="store_true", help="Run a reduced seed sweep.")
    parser.add_argument("--seeds", type=int, default=None, help="Number of random seeds.")
    parser.add_argument("--json", default="ant_foraging_seed_sweep_output.json")
    parser.add_argument("--report", default="ANT_FORAGING_SEED_SWEEP_REPORT.md")
    args = parser.parse_args()

    config = SimulationConfig(agents=48, steps=90) if args.quick else SimulationConfig()
    seeds = args.seeds if args.seeds is not None else (3 if args.quick else 5)
    payload = run_seed_robust_phase_sweep(config=config, quick=args.quick, seeds=seeds)
    write_outputs(payload, Path(args.json), Path(args.report))
    print(render_report(payload))


if __name__ == "__main__":
    main()

"""
Phase diagram for ant foraging memory in DTE.

Sweeps pheromone deposit, evaporation, and scout share to identify the
boundary between adaptive collective memory and deadly familiarity. The core
question is whether prior success continues to route ants through productive
terrain, or traps the colony on depleted/obstructed paths.

Usage:
    .venv\\Scripts\\python.exe ant_foraging_phase_diagram.py --quick
    .venv\\Scripts\\python.exe ant_foraging_phase_diagram.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

from ant_foraging_dte import (
    ForagingPolicy,
    ForagingScenario,
    SimulationConfig,
    policies,
    scenarios,
    simulate,
)


PHASE_CODES = {
    "baseline": ".",
    "adaptive_memory": "A",
    "deadly_familiarity": "D",
    "costly_adaptation": "C",
    "unsafe_shortcut": "U",
    "over_evaporation": "E",
    "neutral_memory": "N",
}


def phase_grid(quick: bool = False) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if quick:
        return (0.0, 0.16, 0.65, 1.20), (0.005, 0.035, 0.120), (0.02, 0.18, 0.44)
    return (
        (0.0, 0.08, 0.16, 0.30, 0.65, 1.20),
        (0.003, 0.005, 0.015, 0.035, 0.060, 0.120),
        (0.02, 0.08, 0.18, 0.30, 0.44, 0.56),
    )


def phase_scenarios(quick: bool = False) -> list[ForagingScenario]:
    by_name = {scenario.name: scenario for scenario in scenarios()}
    delayed_depletion = ForagingScenario(
        name="rich_patch_delayed_depletion",
        food_inventory={"Rich Food Patch": 120, "Sparse Food Patch": 1800},
        notes=(
            "The rich patch persists long enough for pheromone familiarity to form, "
            "then becomes empty while the sparse patch remains productive."
        ),
    )
    selected = [delayed_depletion, by_name["short_path_obstruction"]]
    if not quick:
        selected.append(by_name["baseline_rich_short_path"])
    return selected


def _policy_for(
    deposit: float,
    evaporation: float,
    scout_share: float,
    template: ForagingPolicy,
) -> ForagingPolicy:
    return replace(
        template,
        name=f"deposit_{deposit:.3f}_evap_{evaporation:.3f}_scout_{scout_share:.2f}",
        pheromone_deposit=deposit,
        evaporation=evaporation,
        scout_share=scout_share,
        pheromone_cap=max(template.pheromone_cap, deposit * 6.5),
    )


def _mean_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    numeric_keys = [
        "food_returned",
        "returns_per_100_ant_steps",
        "food_completion_rate",
        "empty_food_visits",
        "empty_food_visit_rate",
        "rich_return_share",
        "rich_visit_share",
        "hazard_hits",
        "hazard_rate",
        "fork_entropy",
        "lock_in_index",
        "mean_return_path_length",
        "pheromone_mass",
        "pheromone_max",
        "flow_entropy_production",
        "irreversible_flux",
    ]
    aggregated = {
        "scenario": first["scenario"],
        "policy": first["policy"],
        "runs": len(rows),
    }
    for key in numeric_keys:
        aggregated[key] = float(mean(row[key] for row in rows))
    recovery_values = [row["shock_recovery_steps"] for row in rows if row["shock_recovery_steps"] is not None]
    aggregated["mean_shock_recovery_steps"] = (
        float(mean(recovery_values)) if recovery_values else None
    )
    aggregated["dominant_branch_modes"] = {
        branch: sum(row["dominant_branch"] == branch for row in rows)
        for branch in sorted({row["dominant_branch"] for row in rows})
    }
    return aggregated


def _classify_phase(row: dict[str, Any], baseline: dict[str, Any]) -> str:
    if row["pheromone_deposit"] == 0.0:
        return "baseline"

    gain = row["food_completion_gain"]
    empty_delta = row["empty_visit_delta"]
    hazard_delta = row["hazard_delta"]

    if row["hazard_rate"] >= 0.06 or hazard_delta >= 0.025:
        return "unsafe_shortcut"
    if row["empty_food_visit_rate"] >= 0.43 and gain < 0.045:
        return "deadly_familiarity"
    if row["empty_food_visit_rate"] >= 0.38 and empty_delta >= 0.04 and gain < 0.08:
        return "deadly_familiarity"
    if gain >= 0.035 and empty_delta <= 0.035 and hazard_delta <= 0.015:
        return "adaptive_memory"
    if gain >= 0.025 and empty_delta > 0.035:
        return "costly_adaptation"
    if row["evaporation"] >= 0.10 and gain <= 0.005 and row["pheromone_mass"] < 2.0:
        return "over_evaporation"
    return "neutral_memory"


def _score(row: dict[str, Any]) -> float:
    return float(
        row["food_completion_rate"]
        + 0.40 * row["food_completion_gain"]
        - 0.32 * row["empty_food_visit_rate"]
        - 0.70 * row["hazard_rate"]
        - 0.06 * row["lock_in_index"]
        + 0.03 * row["fork_entropy"]
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        label: sum(row["phase"] == label for row in rows)
        for label in sorted({row["phase"] for row in rows})
    }
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)

    scenario_summary = {}
    for scenario, scenario_rows in sorted(by_scenario.items()):
        non_baseline = [row for row in scenario_rows if row["phase"] != "baseline"]
        deadly = [row for row in non_baseline if row["phase"] == "deadly_familiarity"]
        adaptive = [row for row in non_baseline if row["phase"] == "adaptive_memory"]
        scenario_summary[scenario] = {
            "runs": len(scenario_rows),
            "best_score": max(non_baseline or scenario_rows, key=lambda row: row["score"]),
            "best_completion": max(non_baseline or scenario_rows, key=lambda row: row["food_completion_rate"]),
            "lowest_empty_visit": min(non_baseline or scenario_rows, key=lambda row: row["empty_food_visit_rate"]),
            "deadly_min_memory_ratio": min(
                (row["memory_ratio"] for row in deadly),
                default=None,
            ),
            "adaptive_max_memory_ratio": max(
                (row["memory_ratio"] for row in adaptive),
                default=None,
            ),
            "adaptive_count": len(adaptive),
            "deadly_count": len(deadly),
        }

    non_baseline = [row for row in rows if row["phase"] != "baseline"]
    return {
        "classification_counts": counts,
        "best_global_score": max(non_baseline or rows, key=lambda row: row["score"]),
        "most_deadly_familiarity": max(
            non_baseline or rows,
            key=lambda row: row["empty_food_visit_rate"] - row["food_completion_gain"],
        ),
        "scenario_summary": scenario_summary,
    }


def run_phase_diagram(
    config: SimulationConfig | None = None,
    quick: bool = False,
    seeds: int | None = None,
) -> dict[str, Any]:
    config = config or SimulationConfig()
    deposits, evaporations, scout_shares = phase_grid(quick)
    selected_scenarios = phase_scenarios(quick)
    seeds = seeds if seeds is not None else (1 if quick else 3)
    template = policies()[1]

    raw_rows: list[dict[str, Any]] = []
    aggregated_rows: list[dict[str, Any]] = []
    baselines: dict[tuple[str, float], dict[str, Any]] = {}

    for scenario_index, scenario in enumerate(selected_scenarios):
        for scout_share in scout_shares:
            for deposit in deposits:
                for evaporation in evaporations:
                    if deposit == 0.0 and evaporation != evaporations[0]:
                        continue
                    policy = _policy_for(deposit, evaporation, scout_share, template)
                    seed_rows = []
                    for seed_index in range(seeds):
                        seed_offset = (
                            scenario_index * 100_000
                            + int(round(scout_share * 1000)) * 100
                            + int(round(deposit * 1000)) * 10
                            + int(round(evaporation * 1000))
                            + seed_index * 997
                        )
                        row = simulate(config, scenario, policy, seed_offset=seed_offset)
                        row["seed_index"] = seed_index
                        row["pheromone_deposit"] = deposit
                        row["evaporation"] = evaporation
                        row["scout_share"] = scout_share
                        raw_rows.append(row)
                        seed_rows.append(row)

                    aggregated = _mean_row(seed_rows)
                    aggregated["pheromone_deposit"] = deposit
                    aggregated["evaporation"] = evaporation
                    aggregated["scout_share"] = scout_share
                    aggregated["memory_ratio"] = float(deposit / max(evaporation, 1e-9))
                    if deposit == 0.0:
                        baselines[(scenario.name, scout_share)] = aggregated
                    aggregated_rows.append(aggregated)

    for row in aggregated_rows:
        baseline = baselines[(row["scenario"], row["scout_share"])]
        row["food_completion_gain"] = row["food_completion_rate"] - baseline["food_completion_rate"]
        row["empty_visit_delta"] = row["empty_food_visit_rate"] - baseline["empty_food_visit_rate"]
        row["hazard_delta"] = row["hazard_rate"] - baseline["hazard_rate"]
        row["phase"] = _classify_phase(row, baseline)
        row["phase_code"] = PHASE_CODES[row["phase"]]
        row["score"] = _score(row)

    summary = _summarize(aggregated_rows)
    return {
        "config": asdict(config) | {"quick": quick, "seeds": seeds},
        "deposits": deposits,
        "evaporations": evaporations,
        "scout_shares": scout_shares,
        "phase_codes": PHASE_CODES,
        "summary": summary,
        "rows": aggregated_rows,
        "raw_rows": raw_rows,
    }


def _fmt(row: dict[str, Any]) -> str:
    return (
        f"scenario={row['scenario']}, deposit={row['pheromone_deposit']:.3f}, "
        f"evap={row['evaporation']:.3f}, scout={row['scout_share']:.2f}, "
        f"ratio={row['memory_ratio']:.2f}, completion={row['food_completion_rate']:.3f}, "
        f"empty={row['empty_food_visit_rate']:.3f}, hazard={row['hazard_rate']:.3f}, "
        f"phase={row['phase']}"
    )


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = payload["rows"]
    lines = [
        "# Ant Foraging Phase Diagram",
        "",
        "## Scope",
        "",
        (
            "Sweep over pheromone deposit, evaporation, and scout share. The central "
            "quantity is the memory ratio `deposit / evaporation`: high values create "
            "strong collective familiarity, while low values erase path memory quickly."
        ),
        "",
        f"- Aggregated grid points: `{len(rows)}`",
        f"- Seeds per point: `{payload['config']['seeds']}`",
        f"- Agents per run: `{payload['config']['agents']}`",
        f"- Steps per run: `{payload['config']['steps']}`",
        "",
        "## Phase Legend",
        "",
        "| Code | Phase |",
        "|---|---|",
    ]
    for phase, code in PHASE_CODES.items():
        lines.append(f"| `{code}` | `{phase}` |")

    lines.extend(["", "## Classification Counts", "", "| Phase | Count |", "|---|---:|"])
    for phase, count in sorted(summary["classification_counts"].items()):
        lines.append(f"| `{phase}` | {count} |")

    best = summary["best_global_score"]
    deadly = summary["most_deadly_familiarity"]
    lines.extend(
        [
            "",
            "## Best Global Score",
            "",
            f"- {_fmt(best)}",
            f"- Score: `{best['score']:.4f}`",
            "",
            "## Strongest Familiarity Trap",
            "",
            f"- {_fmt(deadly)}",
            f"- Empty visit delta vs matched baseline: `{deadly['empty_visit_delta']:.3f}`",
            f"- Completion gain vs matched baseline: `{deadly['food_completion_gain']:.3f}`",
            "",
            "## Scenario Boundaries",
            "",
            "| Scenario | Adaptive points | Deadly points | Max adaptive ratio | Min deadly ratio | Best phase |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for scenario, item in summary["scenario_summary"].items():
        adaptive_ratio = (
            "n/a"
            if item["adaptive_max_memory_ratio"] is None
            else f"{item['adaptive_max_memory_ratio']:.2f}"
        )
        deadly_ratio = (
            "n/a"
            if item["deadly_min_memory_ratio"] is None
            else f"{item['deadly_min_memory_ratio']:.2f}"
        )
        lines.append(
            f"| {scenario} | {item['adaptive_count']} | {item['deadly_count']} | "
            f"{adaptive_ratio} | {deadly_ratio} | {item['best_score']['phase']} |"
        )

    lines.extend(["", "## Phase Matrices", ""])
    for scenario in sorted({row["scenario"] for row in rows}):
        lines.extend([f"### `{scenario}`", ""])
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        for scout_share in payload["scout_shares"]:
            subset = [row for row in scenario_rows if row["scout_share"] == scout_share]
            if not subset:
                continue
            lines.extend(
                [
                    f"Scout share `{scout_share:.2f}`",
                    "",
                    "| Deposit \\ Evap | " + " | ".join(f"{evap:.3f}" for evap in payload["evaporations"]) + " |",
                    "|---|" + "|".join("---:" for _ in payload["evaporations"]) + "|",
                ]
            )
            for deposit in payload["deposits"]:
                cells = []
                for evaporation in payload["evaporations"]:
                    match = [
                        row
                        for row in subset
                        if row["pheromone_deposit"] == deposit and row["evaporation"] == evaporation
                    ]
                    cells.append(match[0]["phase_code"] if match else "")
                lines.append(f"| {deposit:.3f} | " + " | ".join(cells) + " |")
            lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "The actionable biological boundary is not pheromone alone; it is pheromone "
                "relative to evaporation and exploration labor. DTE exposes this as a "
                "transition surface where path memory changes role: below the boundary it "
                "is an efficiency pump, above the boundary it becomes familiarity with a "
                "dead environment."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], json_path: Path, report_path: Path) -> None:
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ant foraging memory phase diagram.")
    parser.add_argument("--quick", action="store_true", help="Run a reduced phase diagram.")
    parser.add_argument("--json", default="ant_foraging_phase_output.json")
    parser.add_argument("--report", default="ANT_FORAGING_PHASE_REPORT.md")
    args = parser.parse_args()

    config = SimulationConfig(agents=48, steps=90) if args.quick else SimulationConfig()
    payload = run_phase_diagram(config=config, quick=args.quick)
    write_outputs(payload, Path(args.json), Path(args.report))
    print(render_report(payload))


if __name__ == "__main__":
    main()

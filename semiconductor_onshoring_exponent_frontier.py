"""
High-load exponent frontier for all-resource onshoring scaling.

The scaling-law ablation shows that viability is non-monotone in both load and
resource exponent. This script maps the phase islands directly by applying one
uniform exponent to renewal, domestic wafer inflow, node capacity, and gate
service capacity.

Usage:
    .venv\\Scripts\\python.exe semiconductor_onshoring_exponent_frontier.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semiconductor_onshoring import OnshoringConfig, simulate
from semiconductor_onshoring_frontier import FrontierConfig, control_for, import_dominant_baseline_point, scenario_for
from semiconductor_onshoring_scaling_law import (
    RANDOMIZATION_KEY,
    ScalingPolicy,
    _evaluate,
    resources_for,
)


def run_exponent_frontier(
    agent_levels: tuple[int, ...] = (160, 240, 320),
    exponents: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25),
    seeds: tuple[int, ...] = (20260606, 20260607, 20260608, 20260609, 20260610),
    steps: int = 40,
) -> dict:
    rows = []
    for agents in agent_levels:
        for seed in seeds:
            config = FrontierConfig(agents=agents, steps=steps, seed=seed)
            sim_config = OnshoringConfig(
                agents=agents,
                steps=steps,
                seed=seed,
                gate_initial_inventory=config.gate_initial_inventory,
                randomization_key=RANDOMIZATION_KEY,
            )
            baseline_point = import_dominant_baseline_point()
            baseline = simulate(
                sim_config,
                scenario_for(baseline_point),
                control_for(baseline_point),
                enforce_gates=True,
            )
            for exponent in exponents:
                policy = ScalingPolicy(
                    name=f"all_{exponent:.2f}",
                    renewal_exponent=exponent,
                    wafer_exponent=exponent,
                    node_capacity_exponent=exponent,
                    gate_capacity_exponent=exponent,
                )
                resources = resources_for(policy, agents)
                row = _evaluate(config, baseline, policy, resources)
                row["exponent"] = exponent
                rows.append(row)

    grouped = []
    for agents in agent_levels:
        for exponent in exponents:
            subset = [
                row for row in rows
                if row["agents"] == agents and row["exponent"] == exponent
            ]
            pressures = [
                max(row["gate_backlog_pressure"], row["gate_starvation_index"])
                for row in subset
            ]
            grouped.append({
                "agents": agents,
                "exponent": exponent,
                "runs": len(subset),
                "viable_rate": sum(row["classification"] == "viable_onshoring_transition" for row in subset) / len(subset),
                "fake_rate": sum(row["classification"] == "fake_onshoring" for row in subset) / len(subset),
                "mean_onshore_share": sum(row["onshore_share"] for row in subset) / len(subset),
                "mean_dependency_pressure": sum(pressures) / len(pressures),
                "mean_overflow": sum(row["capacity_overflow_rate"] for row in subset) / len(subset),
                "mean_completion": sum(row["lot_completion_rate"] for row in subset) / len(subset),
                "resources": {
                    "renewal": subset[0]["inventory_renewal"],
                    "wafer_inflow": subset[0]["domestic_wafer_inflow"],
                    "fab_capacity_multiplier": subset[0]["capacity_multiplier"],
                    "packaging_capacity_multiplier": subset[0]["packaging_capacity_multiplier"],
                    "gate_capacity_caps": subset[0]["gate_capacity_caps"],
                },
            })

    robust_cells = [row for row in grouped if row["viable_rate"] >= 2 / 3]
    return {
        "config": {
            "agent_levels": agent_levels,
            "exponents": exponents,
            "seeds": seeds,
            "steps": steps,
            "randomization_key": RANDOMIZATION_KEY,
        },
        "rows": rows,
        "grouped": grouped,
        "robust_cells": robust_cells,
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Semiconductor Onshoring All-Resource Exponent Frontier",
        "",
        "## Scope",
        "",
        (
            "High-load sweep where one exponent scales renewal, domestic wafer inflow, "
            "node capacity, and gate service capacity together. The purpose is to map "
            "non-monotone viability islands rather than assume a single threshold."
        ),
        "",
        f"- Agent levels: `{', '.join(str(x) for x in payload['config']['agent_levels'])}`",
        f"- Exponents: `{', '.join(f'{x:.2f}' for x in payload['config']['exponents'])}`",
        f"- Seeds: `{', '.join(str(x) for x in payload['config']['seeds'])}`",
        f"- Steps per run: `{payload['config']['steps']}`",
        f"- Paired randomization key: `{payload['config']['randomization_key']}`",
        f"- Runs: `{len(payload['rows'])}`",
        "",
        "## Robust Cells",
        "",
        "| Agents | Exponent | Viable Rate | Mean Share | Mean Dependency | Mean Overflow | Mean Completion | Renewal | Wafer Inflow | FabCap | PkgCap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(payload["robust_cells"], key=lambda item: (item["agents"], item["exponent"])):
        resources = row["resources"]
        lines.append(
            f"| {row['agents']} | {row['exponent']:.2f} | {row['viable_rate']:.1%} | "
            f"{row['mean_onshore_share']:.3f} | {row['mean_dependency_pressure']:.1%} | "
            f"{row['mean_overflow']:.1%} | {row['mean_completion']:.1%} | "
            f"{resources['renewal']} | {resources['wafer_inflow']} | "
            f"{resources['fab_capacity_multiplier']:.2f} | {resources['packaging_capacity_multiplier']:.2f} |"
        )
    if not payload["robust_cells"]:
        lines.append("| none | none | none | none | none | none | none | none | none | none | none |")

    lines.extend([
        "",
        "## Full Frontier",
        "",
        "| Agents | Exponent | Viable Rate | Fake Rate | Mean Share | Mean Dependency | Mean Overflow | Mean Completion |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(payload["grouped"], key=lambda item: (item["agents"], item["exponent"])):
        lines.append(
            f"| {row['agents']} | {row['exponent']:.2f} | {row['viable_rate']:.1%} | "
            f"{row['fake_rate']:.1%} | {row['mean_onshore_share']:.3f} | "
            f"{row['mean_dependency_pressure']:.1%} | {row['mean_overflow']:.1%} | "
            f"{row['mean_completion']:.1%} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "A robust cell has viable-rate at or above two-thirds. Because routing responds "
            "to the changing feasible set, a higher exponent can lower dependency pressure "
            "without guaranteeing majority onshore share. The frontier should therefore be "
            "read as a set of phase islands, not as a monotone resource threshold. No sampled "
            "exponent is a proven lower bound; the shared randomization key only makes the "
            "finite-grid comparisons paired and reproducible."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_exponent_frontier_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_EXPONENT_FRONTIER_REPORT.md"))
    args = parser.parse_args()
    payload = run_exponent_frontier(steps=args.steps)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": len(payload["rows"]),
        "robust_cells": payload["robust_cells"],
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

"""
Model benchmark for the semiconductor feasibility-preference surface.

The benchmark compares:

1. full DTE feedback routing,
2. a frozen-telemetry heterogeneous agent model with identical hard constraints,
3. a static expected-flow Markov model using mean telemetry and no BOM dynamics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from semiconductor_onshoring import (
    US_DEMAND_NODES,
    OnshoringConfig,
    _apply_scenario,
    _initial_telemetries,
    build_kernel,
    simulate,
)
from semiconductor_onshoring_feasibility_preference import (
    RESOURCE_POLICY,
    run_feasibility_preference_surface,
)
from semiconductor_onshoring_frontier import (
    FrontierConfig,
    _classify,
    control_for,
    import_dominant_baseline_point,
    scenario_for,
)
from semiconductor_onshoring_scale_stability import NEAR_VIABLE_DOCTRINE
from semiconductor_onshoring_scaling_law import RANDOMIZATION_KEY, _scaling_control, resources_for


def static_expected_flow_share(
    config: OnshoringConfig,
    scenario,
    control,
) -> float:
    kernel = build_kernel(config)
    labels = kernel.topo.labels
    _apply_scenario(kernel, labels, scenario)
    _apply_scenario(kernel, labels, control)
    telemetry = _initial_telemetries(config).mean(axis=0)
    telemetry /= max(float(np.linalg.norm(telemetry)), 1e-12)
    transition = kernel.transition_matrix(telemetry, step=0)

    occupancy = np.zeros(kernel.topo.N, dtype=np.float64)
    occupancy[labels.index("Market Allocation Desk")] = 1.0
    edge_flow = np.zeros_like(transition)
    for _ in range(config.steps):
        flow = occupancy[:, np.newaxis] * transition
        edge_flow += flow
        occupancy = occupancy @ transition

    demand_indices = [labels.index(name) for name in US_DEMAND_NODES]
    onshore = edge_flow[labels.index("US Finished Packaged Chips"), demand_indices].sum()
    imported = edge_flow[labels.index("US West Coast Port"), demand_indices].sum()
    reserve = edge_flow[labels.index("Strategic Chip Reserve"), demand_indices].sum()
    return float(onshore / max(onshore + imported + reserve, 1e-12))


def run_model_benchmark(surface: dict | None = None) -> dict:
    surface = surface or run_feasibility_preference_surface()
    frozen_rows = []
    static_cells = []

    for agents in surface["config"]["agent_levels"]:
        for seed in surface["config"]["seeds"]:
            frontier_config = FrontierConfig(
                agents=agents,
                steps=surface["config"]["steps"],
                seed=seed,
            )
            frozen_config = OnshoringConfig(
                agents=agents,
                steps=frontier_config.steps,
                seed=seed,
                gate_initial_inventory=frontier_config.gate_initial_inventory,
                feedback_rate=0.0,
                randomization_key=RANDOMIZATION_KEY,
            )
            baseline_point = import_dominant_baseline_point()
            frozen_baseline = simulate(
                frozen_config,
                scenario_for(baseline_point),
                control_for(baseline_point),
                enforce_gates=True,
            )
            resources = resources_for(RESOURCE_POLICY, agents)
            for domestic_pull in surface["config"]["domestic_pulls"]:
                for tariff in surface["config"]["tariffs"]:
                    point = replace(
                        NEAR_VIABLE_DOCTRINE,
                        tariff=tariff,
                        domestic_pull=domestic_pull,
                        capacity_multiplier=resources["fab_capacity_multiplier"],
                        packaging_capacity_multiplier=resources["packaging_capacity_multiplier"],
                    )
                    scenario = scenario_for(point)
                    control = _scaling_control(point, resources, RESOURCE_POLICY)
                    row = simulate(frozen_config, scenario, control, enforce_gates=True)
                    row.update({
                        "seed": seed,
                        "agents": agents,
                        "domestic_pull": domestic_pull,
                        "tariff": tariff,
                        "model": "frozen_telemetry_agents",
                        "finished_flow_ratio": row["lot_total_us_finished"] / max(frozen_baseline["lot_total_us_finished"], 1.0),
                        "classification": _classify(row, frozen_baseline, frontier_config),
                    })
                    frozen_rows.append(row)

        static_config = OnshoringConfig(
            agents=agents,
            steps=surface["config"]["steps"],
            feedback_rate=0.0,
        )
        resources = resources_for(RESOURCE_POLICY, agents)
        for domestic_pull in surface["config"]["domestic_pulls"]:
            for tariff in surface["config"]["tariffs"]:
                point = replace(
                    NEAR_VIABLE_DOCTRINE,
                    tariff=tariff,
                    domestic_pull=domestic_pull,
                    capacity_multiplier=resources["fab_capacity_multiplier"],
                    packaging_capacity_multiplier=resources["packaging_capacity_multiplier"],
                )
                share = static_expected_flow_share(
                    static_config,
                    scenario_for(point),
                    _scaling_control(point, resources, RESOURCE_POLICY),
                )
                static_cells.append({
                    "agents": agents,
                    "domestic_pull": domestic_pull,
                    "tariff": tariff,
                    "model": "static_expected_flow",
                    "predicted_onshore_share": share,
                    "predicts_majority_onshore": share >= 0.50,
                })

    dte_grouped = surface["grouped"]
    frozen_grouped = []
    for cell in dte_grouped:
        subset = [
            row for row in frozen_rows
            if row["agents"] == cell["agents"]
            and row["domestic_pull"] == cell["domestic_pull"]
            and row["tariff"] == cell["tariff"]
        ]
        frozen_grouped.append({
            "agents": cell["agents"],
            "domestic_pull": cell["domestic_pull"],
            "tariff": cell["tariff"],
            "viable_rate": sum(row["classification"] == "viable_onshoring_transition" for row in subset) / len(subset),
            "mean_onshore_share": sum(row["onshore_share"] for row in subset) / len(subset),
            "mean_completion": sum(row["lot_completion_rate"] for row in subset) / len(subset),
        })

    comparisons = []
    for dte in dte_grouped:
        frozen = next(
            row for row in frozen_grouped
            if row["agents"] == dte["agents"]
            and row["domestic_pull"] == dte["domestic_pull"]
            and row["tariff"] == dte["tariff"]
        )
        static = next(
            row for row in static_cells
            if row["agents"] == dte["agents"]
            and row["domestic_pull"] == dte["domestic_pull"]
            and row["tariff"] == dte["tariff"]
        )
        comparisons.append({
            "agents": dte["agents"],
            "domestic_pull": dte["domestic_pull"],
            "tariff": dte["tariff"],
            "dte_viable_rate": dte["viable_rate"],
            "frozen_viable_rate": frozen["viable_rate"],
            "dte_mean_share": dte["mean_onshore_share"],
            "frozen_mean_share": frozen["mean_onshore_share"],
            "static_predicted_share": static["predicted_onshore_share"],
            "static_predicts_majority": static["predicts_majority_onshore"],
            "dte_robust": dte["viable_rate"] >= 2 / 3,
            "frozen_robust": frozen["viable_rate"] >= 2 / 3,
        })

    return {
        "config": surface["config"],
        "comparisons": comparisons,
        "frozen_rows": frozen_rows,
        "static_cells": static_cells,
        "summary": {
            "cells": len(comparisons),
            "dte_robust_cells": sum(row["dte_robust"] for row in comparisons),
            "frozen_robust_cells": sum(row["frozen_robust"] for row in comparisons),
            "feedback_changes_robust_class": sum(row["dte_robust"] != row["frozen_robust"] for row in comparisons),
            "static_majority_cells": sum(row["static_predicts_majority"] for row in comparisons),
            "static_majority_but_dte_not_robust": sum(
                row["static_predicts_majority"] and not row["dte_robust"]
                for row in comparisons
            ),
        },
    }


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Semiconductor Onshoring Model Benchmark",
        "",
        "## Scope",
        "",
        (
            "Comparison of full DTE feedback routing, frozen-telemetry heterogeneous agents "
            "with identical hard constraints, and a static mean-telemetry expected-flow model "
            "without BOM or capacity dynamics."
        ),
        "",
        f"- Policy cells: `{summary['cells']}`",
        f"- DTE robust cells: `{summary['dte_robust_cells']}`",
        f"- Frozen-agent robust cells: `{summary['frozen_robust_cells']}`",
        f"- Cells where feedback changes robust classification: `{summary['feedback_changes_robust_class']}`",
        f"- Static majority-onshore predictions: `{summary['static_majority_cells']}`",
        f"- Static majority predictions not robust under DTE: `{summary['static_majority_but_dte_not_robust']}`",
        "",
        "## Cell Comparison",
        "",
        "| Agents | Pull | Tariff | DTE Viable | Frozen Viable | DTE Share | Frozen Share | Static Share | Static Majority |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["comparisons"]:
        lines.append(
            f"| {row['agents']} | {row['domestic_pull']:.1f} | {row['tariff']:.1f} | "
            f"{row['dte_viable_rate']:.1%} | {row['frozen_viable_rate']:.1%} | "
            f"{row['dte_mean_share']:.3f} | {row['frozen_mean_share']:.3f} | "
            f"{row['static_predicted_share']:.3f} | {'yes' if row['static_predicts_majority'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "The static model is a route-attractiveness benchmark, not a production model: "
            "it cannot represent BOM starvation, gate service limits, or adaptive telemetry. "
            "The frozen-agent model isolates the value of DTE feedback while preserving the "
            "same heterogeneous population and hard constraints."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_model_benchmark_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_MODEL_BENCHMARK_REPORT.md"))
    args = parser.parse_args()
    payload = run_model_benchmark()
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

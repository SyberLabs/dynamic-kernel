"""
Classification-threshold robustness analysis for the feasibility-preference surface.

This module reuses one set of simulated runs and varies only the definition of
a viable onshoring transition. It tests whether the qualitative conclusion is
an artifact of a particular threshold choice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

from semiconductor_onshoring_feasibility_preference import run_feasibility_preference_surface
from semiconductor_onshoring_frontier import FrontierConfig, _classify


def _threshold_grid() -> list[FrontierConfig]:
    return [
        FrontierConfig(
            transition_threshold=transition,
            max_dependency_pressure=dependency,
            max_overflow=overflow,
            min_finished_flow_ratio=finished,
        )
        for transition, dependency, overflow, finished in product(
            (0.45, 0.50, 0.55),
            (0.55, 0.65, 0.75),
            (0.05, 0.10),
            (0.80, 0.85, 0.95),
        )
    ]


def run_classification_robustness(surface: dict | None = None) -> dict:
    surface = surface or run_feasibility_preference_surface()
    configs = _threshold_grid()
    grouped = surface["grouped"]
    rows = surface["rows"]
    results = []

    for config_id, config in enumerate(configs):
        robust_cells = []
        for cell in grouped:
            subset = [
                row for row in rows
                if row["agents"] == cell["agents"]
                and row["domestic_pull"] == cell["domestic_pull"]
                and row["tariff"] == cell["tariff"]
            ]
            viable_rate = sum(
                _classify(
                    row,
                    {
                        "onshore_share": row["baseline_onshore_share"],
                        "lot_total_us_finished": row["baseline_lot_total_us_finished"],
                    },
                    config,
                )
                == "viable_onshoring_transition"
                for row in subset
            ) / len(subset)
            if viable_rate >= 2 / 3:
                robust_cells.append({
                    "agents": cell["agents"],
                    "domestic_pull": cell["domestic_pull"],
                    "tariff": cell["tariff"],
                    "viable_rate": viable_rate,
                })

        no_preference = [cell for cell in robust_cells if cell["domestic_pull"] == 0.0 and cell["tariff"] == 0.0]
        allocation_cells = [cell for cell in robust_cells if cell["domestic_pull"] > 0.0 or cell["tariff"] > 0.0]
        max_agents = max(surface["config"]["agent_levels"])
        high_load_cells = [cell for cell in robust_cells if cell["agents"] == max_agents]
        high_load_no_preference = [
            cell for cell in high_load_cells
            if cell["domestic_pull"] == 0.0 and cell["tariff"] == 0.0
        ]
        high_load_allocation = [
            cell for cell in high_load_cells
            if cell["domestic_pull"] > 0.0 or cell["tariff"] > 0.0
        ]
        results.append({
            "config_id": config_id,
            "thresholds": asdict(config),
            "robust_cell_count": len(robust_cells),
            "robust_no_preference_count": len(no_preference),
            "robust_allocation_cell_count": len(allocation_cells),
            "max_robust_agents": max((cell["agents"] for cell in robust_cells), default=0),
            "allocation_required": bool(allocation_cells) and not bool(no_preference),
            "high_load_allocation_required": bool(high_load_allocation) and not bool(high_load_no_preference),
            "high_load_no_preference_robust": bool(high_load_no_preference),
            "robust_cells": robust_cells,
        })

    configs_with_robust = [row for row in results if row["robust_cell_count"] > 0]
    return {
        "surface_config": surface["config"],
        "threshold_configs": len(configs),
        "results": results,
        "summary": {
            "configs_with_robust_cells": len(configs_with_robust),
            "configs_where_allocation_required": sum(row["allocation_required"] for row in results),
            "configs_with_robust_no_preference": sum(row["robust_no_preference_count"] > 0 for row in results),
            "configs_with_robust_320": sum(row["max_robust_agents"] >= 320 for row in results),
            "configs_where_high_load_allocation_required": sum(row["high_load_allocation_required"] for row in results),
            "configs_with_high_load_no_preference": sum(row["high_load_no_preference_robust"] for row in results),
        },
    }


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Semiconductor Onshoring Classification-Robustness Report",
        "",
        "## Scope",
        "",
        (
            "Sensitivity analysis over the definition of a viable onshoring transition. "
            "All classifications reuse the same feasibility-preference simulation rows."
        ),
        "",
        f"- Threshold configurations: `{payload['threshold_configs']}`",
        f"- Configurations with at least one robust cell: `{summary['configs_with_robust_cells']}`",
        f"- Configurations where allocation pressure is required: `{summary['configs_where_allocation_required']}`",
        f"- Configurations with a robust no-preference cell: `{summary['configs_with_robust_no_preference']}`",
        f"- Configurations with a robust 320-agent cell: `{summary['configs_with_robust_320']}`",
        f"- Configurations where high-load allocation pressure is required: `{summary['configs_where_high_load_allocation_required']}`",
        f"- Configurations with high-load no-preference robustness: `{summary['configs_with_high_load_no_preference']}`",
        "",
        "## Threshold Results",
        "",
        "| ID | Share Threshold | Max Dependency | Max Overflow | Min Finished Ratio | Robust Cells | High-Load Allocation Required | Max Robust Agents |",
        "|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in payload["results"]:
        thresholds = row["thresholds"]
        lines.append(
            f"| {row['config_id']} | {thresholds['transition_threshold']:.2f} | "
            f"{thresholds['max_dependency_pressure']:.2f} | {thresholds['max_overflow']:.2f} | "
            f"{thresholds['min_finished_flow_ratio']:.2f} | {row['robust_cell_count']} | "
            f"{'yes' if row['high_load_allocation_required'] else 'no'} | {row['max_robust_agents']} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "The conclusion is threshold-robust when feasible classifications repeatedly "
            "require nonzero allocation pressure and when capacity-only cells remain absent "
            "across reasonable definitions. Threshold configurations that admit no robust "
            "cell are evidence of a strict standard, not evidence against the mechanism."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_classification_robustness_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_CLASSIFICATION_ROBUSTNESS_REPORT.md"))
    args = parser.parse_args()
    payload = run_classification_robustness()
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

"""
Coupled feasibility-preference phase surface for semiconductor onshoring.

Physical resources are scaled together with the best robust sampled exponent
from the resource-scaling experiment. The remaining axes vary allocation
pressure: domestic procurement pull and offshore import friction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from semiconductor_onshoring import OnshoringConfig, simulate
from semiconductor_onshoring_frontier import (
    FrontierConfig,
    _classify,
    _score,
    control_for,
    import_dominant_baseline_point,
    scenario_for,
)
from semiconductor_onshoring_scale_stability import NEAR_VIABLE_DOCTRINE
from semiconductor_onshoring_scaling_law import (
    RANDOMIZATION_KEY,
    ScalingPolicy,
    _scaling_control,
    resources_for,
)


RESOURCE_EXPONENT = 1.25
RESOURCE_POLICY = ScalingPolicy(
    "feasible_all_1_25",
    RESOURCE_EXPONENT,
    RESOURCE_EXPONENT,
    RESOURCE_EXPONENT,
    RESOURCE_EXPONENT,
)


def _evaluate(
    config: FrontierConfig,
    baseline: dict,
    domestic_pull: float,
    tariff: float,
) -> dict:
    resources = resources_for(RESOURCE_POLICY, config.agents)
    point = replace(
        NEAR_VIABLE_DOCTRINE,
        tariff=tariff,
        domestic_pull=domestic_pull,
        capacity_multiplier=resources["fab_capacity_multiplier"],
        packaging_capacity_multiplier=resources["packaging_capacity_multiplier"],
    )
    sim_config = OnshoringConfig(
        agents=config.agents,
        steps=config.steps,
        seed=config.seed,
        gate_initial_inventory=config.gate_initial_inventory,
        randomization_key=RANDOMIZATION_KEY,
    )
    row = simulate(
        sim_config,
        scenario_for(point),
        _scaling_control(point, resources, RESOURCE_POLICY),
        enforce_gates=True,
    )
    finished_ratio = row["lot_total_us_finished"] / max(baseline["lot_total_us_finished"], 1.0)
    row.update({
        "seed": config.seed,
        "tariff": tariff,
        "domestic_pull": domestic_pull,
        "resource_exponent": RESOURCE_EXPONENT,
        "inventory_renewal": resources["renewal"],
        "domestic_wafer_inflow": resources["wafer_inflow"],
        "capacity_multiplier": point.capacity_multiplier,
        "packaging_capacity_multiplier": point.packaging_capacity_multiplier,
        "finished_flow_ratio": finished_ratio,
        "baseline_lot_total_us_finished": baseline["lot_total_us_finished"],
        "baseline_onshore_share": baseline["onshore_share"],
        "score": _score(row, baseline),
        "classification": _classify(row, baseline, config),
        "onshore_delta_vs_baseline": row["onshore_share"] - baseline["onshore_share"],
    })
    return row


def run_feasibility_preference_surface(
    agent_levels: tuple[int, ...] = (160, 240, 320),
    domestic_pulls: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0),
    tariffs: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0),
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
            for domestic_pull in domestic_pulls:
                for tariff in tariffs:
                    rows.append(_evaluate(config, baseline, domestic_pull, tariff))

    grouped = []
    for agents in agent_levels:
        for domestic_pull in domestic_pulls:
            for tariff in tariffs:
                subset = [
                    row for row in rows
                    if row["agents"] == agents
                    and row["domestic_pull"] == domestic_pull
                    and row["tariff"] == tariff
                ]
                pressures = [
                    max(row["gate_backlog_pressure"], row["gate_starvation_index"])
                    for row in subset
                ]
                grouped.append({
                    "agents": agents,
                    "domestic_pull": domestic_pull,
                    "tariff": tariff,
                    "runs": len(subset),
                    "viable_rate": sum(row["classification"] == "viable_onshoring_transition" for row in subset) / len(subset),
                    "fake_rate": sum(row["classification"] == "fake_onshoring" for row in subset) / len(subset),
                    "mean_onshore_share": sum(row["onshore_share"] for row in subset) / len(subset),
                    "mean_dependency_pressure": sum(pressures) / len(pressures),
                    "mean_overflow": sum(row["capacity_overflow_rate"] for row in subset) / len(subset),
                    "mean_completion": sum(row["lot_completion_rate"] for row in subset) / len(subset),
                    "mean_finished_flow_ratio": sum(row["finished_flow_ratio"] for row in subset) / len(subset),
                })

    robust_cells = [row for row in grouped if row["viable_rate"] >= 2 / 3]
    tariff_spreads = []
    for agents in agent_levels:
        for domestic_pull in domestic_pulls:
            subset = [
                row for row in grouped
                if row["agents"] == agents and row["domestic_pull"] == domestic_pull
            ]
            tariff_spreads.append(max(row["mean_onshore_share"] for row in subset) - min(row["mean_onshore_share"] for row in subset))
    high_load = max(agent_levels)
    high_load_no_pull = next(
        row for row in grouped
        if row["agents"] == high_load and row["domestic_pull"] == 0.0 and row["tariff"] == 0.0
    )
    high_load_with_pull = [
        row for row in grouped
        if row["agents"] == high_load and row["domestic_pull"] > 0.0
    ]
    return {
        "config": {
            "agent_levels": agent_levels,
            "domestic_pulls": domestic_pulls,
            "tariffs": tariffs,
            "seeds": seeds,
            "steps": steps,
            "resource_exponent": RESOURCE_EXPONENT,
            "resource_policy": asdict(RESOURCE_POLICY),
            "randomization_key": RANDOMIZATION_KEY,
        },
        "rows": rows,
        "grouped": grouped,
        "robust_cells": robust_cells,
        "summary": {
            "max_tariff_share_spread": max(tariff_spreads, default=0.0),
            "high_load_agents": high_load,
            "high_load_no_pull_viable_rate": high_load_no_pull["viable_rate"],
            "high_load_best_pull_viable_rate": max(row["viable_rate"] for row in high_load_with_pull),
            "high_load_no_pull_mean_share": high_load_no_pull["mean_onshore_share"],
            "high_load_best_pull_mean_share": max(row["mean_onshore_share"] for row in high_load_with_pull),
        },
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Semiconductor Onshoring Feasibility-Preference Surface",
        "",
        "## Scope",
        "",
        (
            "The production system is scaled with the all-resource exponent `1.25`. "
            "The sweep then varies domestic procurement pull and offshore import friction "
            "to test whether feasible domestic production captures allocation."
        ),
        "",
        f"- Agent levels: `{', '.join(str(x) for x in payload['config']['agent_levels'])}`",
        f"- Domestic pulls: `{', '.join(f'{x:.1f}' for x in payload['config']['domestic_pulls'])}`",
        f"- Tariffs: `{', '.join(f'{x:.1f}' for x in payload['config']['tariffs'])}`",
        f"- Seeds: `{', '.join(str(x) for x in payload['config']['seeds'])}`",
        f"- Paired randomization key: `{payload['config']['randomization_key']}`",
        f"- Runs: `{len(payload['rows'])}`",
        "",
        "## Mechanism Summary",
        "",
        f"- Maximum mean-share spread across tariff levels: `{payload['summary']['max_tariff_share_spread']:.6f}`",
        f"- High-load no-pull viable rate: `{payload['summary']['high_load_no_pull_viable_rate']:.1%}`",
        f"- High-load best positive-pull viable rate: `{payload['summary']['high_load_best_pull_viable_rate']:.1%}`",
        f"- High-load no-pull mean share: `{payload['summary']['high_load_no_pull_mean_share']:.3f}`",
        f"- High-load best positive-pull mean share: `{payload['summary']['high_load_best_pull_mean_share']:.3f}`",
        "",
        "## Robust Cells",
        "",
        "| Agents | Domestic Pull | Tariff | Viable Rate | Mean Share | Mean Dependency | Mean Overflow | Mean Completion |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(payload["robust_cells"], key=lambda item: (item["agents"], item["domestic_pull"], item["tariff"])):
        lines.append(
            f"| {row['agents']} | {row['domestic_pull']:.1f} | {row['tariff']:.1f} | "
            f"{row['viable_rate']:.1%} | {row['mean_onshore_share']:.3f} | "
            f"{row['mean_dependency_pressure']:.1%} | {row['mean_overflow']:.1%} | "
            f"{row['mean_completion']:.1%} |"
        )
    if not payload["robust_cells"]:
        lines.append("| none | none | none | none | none | none | none | none |")

    lines.extend([
        "",
        "## Full Surface",
        "",
        "| Agents | Domestic Pull | Tariff | Viable Rate | Fake Rate | Mean Share | Mean Dependency | Mean Completion |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(payload["grouped"], key=lambda item: (item["agents"], item["domestic_pull"], item["tariff"])):
        lines.append(
            f"| {row['agents']} | {row['domestic_pull']:.1f} | {row['tariff']:.1f} | "
            f"{row['viable_rate']:.1%} | {row['fake_rate']:.1%} | "
            f"{row['mean_onshore_share']:.3f} | {row['mean_dependency_pressure']:.1%} | "
            f"{row['mean_completion']:.1%} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "If robust cells appear only after allocation pressure is introduced, the "
            "capacity-only result is confirmed: physical feasibility is necessary but does "
            "not select the domestic route. If no robust cells appear, the current topology "
            "requires structural change rather than stronger preference on existing edges. "
            "A flat tariff axis means the tariff is applied after route commitment on a "
            "serial corridor, where no competing outgoing edge exists."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_feasibility_preference_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_FEASIBILITY_PREFERENCE_REPORT.md"))
    args = parser.parse_args()
    payload = run_feasibility_preference_surface(steps=args.steps)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": len(payload["rows"]),
        "robust_cells": payload["robust_cells"],
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

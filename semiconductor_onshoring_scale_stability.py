"""
Scale-stability sweep for the near-viable onshoring pocket.

The inventory-renewal phase diagram found viable cells at 80 agents but no
viable cells at 160 agents. This script holds the near-viable doctrine fixed
and sweeps population scale, renewal rate, and seeds to identify whether the
pocket is robust or collapses under load.

Usage:
    .venv\\Scripts\\python.exe semiconductor_onshoring_scale_stability.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from semiconductor_onshoring import OnshoringConfig, simulate
from semiconductor_onshoring_frontier import (
    FrontierConfig,
    StrategyPoint,
    control_for,
    import_dominant_baseline_point,
    scenario_for,
)
from semiconductor_onshoring_inventory_phase_diagram import _evaluate_inventory_point


NEAR_VIABLE_DOCTRINE = StrategyPoint(
    tariff=0.0,
    capacity_multiplier=3.0,
    packaging_capacity_multiplier=0.0,
    subsidy=1.2,
    offset=0.0,
    domestic_ramp=1.5,
    packaging_ramp=1.5,
    materials=1.6,
    domestic_pull=0.0,
)


def _fmt_point(row: dict) -> str:
    return (
        f"agents={row['agents']}, renewal={row['inventory_renewal']}, seed={row['seed']}, "
        f"fabcap={row['capacity_multiplier']:.1f}, pkgcap={row['packaging_capacity_multiplier']:.1f}"
    )


def run_scale_stability(
    agent_levels: tuple[int, ...] = (40, 80, 120, 160, 240, 320),
    renewals: tuple[int, ...] = (2, 3, 4, 5),
    seeds: tuple[int, ...] = (20260606, 20260607, 20260608),
    steps: int = 40,
    packaging_caps: tuple[float, ...] = (0.0, 3.0, 6.0),
) -> dict:
    rows = []
    for agents in agent_levels:
        for seed in seeds:
            base_config = FrontierConfig(agents=agents, steps=steps, seed=seed)
            sim_config = OnshoringConfig(
                agents=base_config.agents,
                steps=base_config.steps,
                seed=base_config.seed,
                gate_initial_inventory=base_config.gate_initial_inventory,
            )
            baseline_point = import_dominant_baseline_point()
            baseline = simulate(
                sim_config,
                scenario_for(baseline_point),
                control_for(baseline_point),
                enforce_gates=True,
            )
            for renewal in renewals:
                for packaging_cap in packaging_caps:
                    point = replace(
                        NEAR_VIABLE_DOCTRINE,
                        packaging_capacity_multiplier=packaging_cap,
                    )
                    row = _evaluate_inventory_point(base_config, point, baseline, renewal)
                    row["seed"] = seed
                    row["phase_code"] = {
                        "viable_onshoring_transition": "V",
                        "fake_onshoring": "F",
                        "capacity_blocked": "C",
                        "dependency_blocked": "D",
                        "partial_onshoring": "P",
                        "no_transition": ".",
                    }[row["classification"]]
                    rows.append(row)

    grouped = []
    for agents in agent_levels:
        for renewal in renewals:
            for packaging_cap in packaging_caps:
                subset = [
                    row for row in rows
                    if row["agents"] == agents
                    and row["inventory_renewal"] == renewal
                    and row["packaging_capacity_multiplier"] == packaging_cap
                ]
                if not subset:
                    continue
                pressures = [
                    max(row["gate_backlog_pressure"], row["gate_starvation_index"])
                    for row in subset
                ]
                grouped.append({
                    "agents": agents,
                    "renewal": renewal,
                    "packaging_capacity_multiplier": packaging_cap,
                    "runs": len(subset),
                    "viable_rate": sum(row["classification"] == "viable_onshoring_transition" for row in subset) / len(subset),
                    "fake_rate": sum(row["classification"] == "fake_onshoring" for row in subset) / len(subset),
                    "mean_onshore_share": sum(row["onshore_share"] for row in subset) / len(subset),
                    "max_onshore_share": max(row["onshore_share"] for row in subset),
                    "mean_dependency_pressure": sum(pressures) / len(pressures),
                    "min_dependency_pressure": min(pressures),
                    "mean_overflow": sum(row["capacity_overflow_rate"] for row in subset) / len(subset),
                    "mean_lot_completion": sum(row["lot_completion_rate"] for row in subset) / len(subset),
                })

    viable_groups = [item for item in grouped if item["viable_rate"] > 0]
    robust_groups = [item for item in grouped if item["viable_rate"] >= 2 / 3]
    return {
        "config": {
            "agent_levels": agent_levels,
            "renewals": renewals,
            "seeds": seeds,
            "steps": steps,
            "packaging_caps": packaging_caps,
            "doctrine": asdict(NEAR_VIABLE_DOCTRINE),
        },
        "rows": rows,
        "grouped": grouped,
        "best_run": max(rows, key=lambda row: row["score"]),
        "best_viable_run": max(
            [row for row in rows if row["classification"] == "viable_onshoring_transition"],
            key=lambda row: row["score"],
            default=None,
        ),
        "viable_groups": viable_groups,
        "robust_groups": robust_groups,
    }


def render_report(payload: dict) -> str:
    grouped = payload["grouped"]
    best = payload["best_run"]
    lines = [
        "# Semiconductor Onshoring Scale-Stability Report",
        "",
        "## Scope",
        "",
        (
            "Seed-aware sweep around the near-viable inventory-renewal pocket. "
            "The doctrine is held fixed while agent population, renewal rate, "
            "and U.S. advanced-packaging capacity vary."
        ),
        "",
        f"- Steps per run: `{payload['config']['steps']}`",
        f"- Agent levels: `{', '.join(str(x) for x in payload['config']['agent_levels'])}`",
        f"- Renewal levels: `{', '.join(str(x) for x in payload['config']['renewals'])}`",
        f"- Packaging capacity multipliers: `{', '.join(str(x) for x in payload['config']['packaging_caps'])}`",
        f"- Seeds: `{', '.join(str(x) for x in payload['config']['seeds'])}`",
        f"- Runs: `{len(payload['rows'])}`",
        "",
        "## Best Run",
        "",
        f"- Point: `{_fmt_point(best)}`",
        f"- Classification: `{best['classification']}`",
        f"- Onshore share: `{best['onshore_share']:.3f}`",
        f"- Finished-lot ratio: `{best['finished_flow_ratio']:.3f}`",
        f"- Overflow: `{best['capacity_overflow_rate']:.1%}`",
        f"- Dependency pressure: `{max(best['gate_backlog_pressure'], best['gate_starvation_index']):.1%}`",
        f"- Lot completion rate: `{best['lot_completion_rate']:.1%}`",
        "",
        "## Best Viable Run",
        "",
    ]
    if payload["best_viable_run"] is None:
        lines.append("No viable run was found.")
    else:
        viable = payload["best_viable_run"]
        lines.extend([
            f"- Point: `{_fmt_point(viable)}`",
            f"- Onshore share: `{viable['onshore_share']:.3f}`",
            f"- Finished-lot ratio: `{viable['finished_flow_ratio']:.3f}`",
            f"- Overflow: `{viable['capacity_overflow_rate']:.1%}`",
            f"- Dependency pressure: `{max(viable['gate_backlog_pressure'], viable['gate_starvation_index']):.1%}`",
        ])

    lines.extend([
        "",
        "## Grouped Frontier",
        "",
        "| Agents | Renewal | PkgCap | Viable Rate | Fake Rate | Mean Share | Mean Dependency | Mean Overflow | Mean Completion |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(grouped, key=lambda item: (item["agents"], item["renewal"], item["packaging_capacity_multiplier"])):
        lines.append(
            f"| {row['agents']} | {row['renewal']} | {row['packaging_capacity_multiplier']:.1f} | "
            f"{row['viable_rate']:.1%} | {row['fake_rate']:.1%} | "
            f"{row['mean_onshore_share']:.3f} | {row['mean_dependency_pressure']:.1%} | "
            f"{row['mean_overflow']:.1%} | {row['mean_lot_completion']:.1%} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "A robust scale-stable pocket should preserve viable-rate as agents increase. "
            "If viable runs exist only at low population or only in isolated seeds, the "
            "onshoring phase is fragile rather than institutionally stable."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_scale_stability_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_SCALE_STABILITY_REPORT.md"))
    args = parser.parse_args()
    payload = run_scale_stability(steps=args.steps)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": len(payload["rows"]),
        "viable_groups": len(payload["viable_groups"]),
        "robust_groups": len(payload["robust_groups"]),
        "best_viable_run": payload["best_viable_run"],
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

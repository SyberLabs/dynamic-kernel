"""
Resource-scaling law experiment for semiconductor onshoring.

The scale-stability sweep found a fragile viable phase, while fixed per-tick
resources failed at larger populations. This experiment tests which resource
scaling laws preserve viability as demand load grows.

The controlled resources are:

- consumable stock renewal
- domestic wafer inflow
- domestic fab and packaging node capacity
- gate service capacity

Usage:
    .venv\\Scripts\\python.exe semiconductor_onshoring_scaling_law.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from semiconductor_onshoring import OnshoringConfig, Scenario, simulate
from semiconductor_onshoring_frontier import (
    FrontierConfig,
    StrategyPoint,
    _classify,
    _score,
    control_for,
    import_dominant_baseline_point,
    scenario_for,
)
from semiconductor_onshoring_inventory_phase_diagram import REPLENISHED_PARTS
from semiconductor_onshoring_scale_stability import NEAR_VIABLE_DOCTRINE


REFERENCE_AGENTS = 80
BASE_RENEWAL = 4
BASE_WAFER_INFLOW = 1
BASE_FAB_CAPACITY_MULTIPLIER = 3.0
BASE_PACKAGING_CAPACITY_MULTIPLIER = 6.0
BASE_GATE_CAPS = {
    "us_fab_gate": 8,
    "us_advanced_packaging_gate": 8,
    "taiwan_export_gate": 8,
}
RANDOMIZATION_KEY = "semiconductor_onshoring_scaling_law"


@dataclass(frozen=True)
class ScalingPolicy:
    name: str
    renewal_exponent: float
    wafer_exponent: float
    node_capacity_exponent: float
    gate_capacity_exponent: float


POLICIES = (
    ScalingPolicy("fixed", 0.0, 0.0, 0.0, 0.0),
    ScalingPolicy("linear_renewal", 1.0, 0.0, 0.0, 0.0),
    ScalingPolicy("linear_stock", 1.0, 1.0, 0.0, 0.0),
    ScalingPolicy("linear_capacity", 0.0, 0.0, 1.0, 1.0),
    ScalingPolicy("linear_stock_service", 1.0, 1.0, 0.0, 1.0),
    ScalingPolicy("linear_all", 1.0, 1.0, 1.0, 1.0),
    ScalingPolicy("superlinear_all", 1.25, 1.25, 1.25, 1.25),
    ScalingPolicy("all_1_50", 1.50, 1.50, 1.50, 1.50),
    ScalingPolicy("all_1_75", 1.75, 1.75, 1.75, 1.75),
    ScalingPolicy("all_2_00", 2.00, 2.00, 2.00, 2.00),
)


def _scaled_int(base: int, agents: int, exponent: float) -> int:
    return max(1, int(math.ceil(base * (agents / REFERENCE_AGENTS) ** exponent)))


def _scaled_float(base: float, agents: int, exponent: float) -> float:
    return float(base * (agents / REFERENCE_AGENTS) ** exponent)


def resources_for(policy: ScalingPolicy, agents: int) -> dict:
    return {
        "renewal": _scaled_int(BASE_RENEWAL, agents, policy.renewal_exponent),
        "wafer_inflow": _scaled_int(BASE_WAFER_INFLOW, agents, policy.wafer_exponent),
        "fab_capacity_multiplier": _scaled_float(
            BASE_FAB_CAPACITY_MULTIPLIER,
            agents,
            policy.node_capacity_exponent,
        ),
        "packaging_capacity_multiplier": _scaled_float(
            BASE_PACKAGING_CAPACITY_MULTIPLIER,
            agents,
            policy.node_capacity_exponent,
        ),
        "gate_capacity_caps": {
            gate: _scaled_int(capacity, agents, policy.gate_capacity_exponent)
            for gate, capacity in BASE_GATE_CAPS.items()
        },
    }


def _scaling_control(point: StrategyPoint, resources: dict, policy: ScalingPolicy) -> Scenario:
    base = control_for(point)
    replenishment = {
        key: int(resources["renewal"])
        for key in REPLENISHED_PARTS
    }
    replenishment[("us_fab_gate", "domestic_wafers")] = int(resources["wafer_inflow"])
    return Scenario(
        name=f"{base.name}_{policy.name}",
        family="scaling_law_control",
        cost=base.cost + float(sum(replenishment.values()) + sum(resources["gate_capacity_caps"].values())),
        friction_edges=base.friction_edges,
        friction_delta=base.friction_delta,
        friction_edge_deltas=base.friction_edge_deltas,
        beta_edges=base.beta_edges,
        beta_boost=base.beta_boost,
        beta_edge_boosts=base.beta_edge_boosts,
        node_capacity_caps=base.node_capacity_caps,
        edge_capacity_caps=base.edge_capacity_caps,
        gate_replenishment=replenishment,
        gate_capacity_caps=resources["gate_capacity_caps"],
    )


def _evaluate(
    config: FrontierConfig,
    baseline: dict,
    policy: ScalingPolicy,
    resources: dict,
) -> dict:
    point = replace(
        NEAR_VIABLE_DOCTRINE,
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
        _scaling_control(point, resources, policy),
        enforce_gates=True,
    )
    finished_ratio = row["lot_total_us_finished"] / max(baseline["lot_total_us_finished"], 1.0)
    row.update({
        "seed": config.seed,
        "policy": policy.name,
        "renewal_exponent": policy.renewal_exponent,
        "wafer_exponent": policy.wafer_exponent,
        "node_capacity_exponent": policy.node_capacity_exponent,
        "gate_capacity_exponent": policy.gate_capacity_exponent,
        "inventory_renewal": resources["renewal"],
        "domestic_wafer_inflow": resources["wafer_inflow"],
        "capacity_multiplier": point.capacity_multiplier,
        "packaging_capacity_multiplier": point.packaging_capacity_multiplier,
        "finished_flow_ratio": finished_ratio,
        "score": _score(row, baseline),
        "classification": _classify(row, baseline, config),
        "onshore_delta_vs_baseline": row["onshore_share"] - baseline["onshore_share"],
        "finished_delta_vs_baseline": row["lot_total_us_finished"] - baseline["lot_total_us_finished"],
    })
    return row


def run_scaling_law(
    agent_levels: tuple[int, ...] = (40, 80, 120, 160, 240, 320),
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
            for policy in POLICIES:
                resources = resources_for(policy, agents)
                rows.append(_evaluate(config, baseline, policy, resources))

    grouped = []
    for policy in POLICIES:
        for agents in agent_levels:
            subset = [
                row for row in rows
                if row["policy"] == policy.name and row["agents"] == agents
            ]
            pressures = [
                max(row["gate_backlog_pressure"], row["gate_starvation_index"])
                for row in subset
            ]
            grouped.append({
                "policy": policy.name,
                "agents": agents,
                "runs": len(subset),
                "viable_rate": sum(row["classification"] == "viable_onshoring_transition" for row in subset) / len(subset),
                "fake_rate": sum(row["classification"] == "fake_onshoring" for row in subset) / len(subset),
                "mean_onshore_share": sum(row["onshore_share"] for row in subset) / len(subset),
                "mean_dependency_pressure": sum(pressures) / len(pressures),
                "mean_overflow": sum(row["capacity_overflow_rate"] for row in subset) / len(subset),
                "mean_completion": sum(row["lot_completion_rate"] for row in subset) / len(subset),
                "mean_gate_capacity_blocked": sum(
                    sum(row["gate_capacity_blocked"].values()) / max(sum(row["gate_attempts"].values()), 1)
                    for row in subset
                ) / len(subset),
                "resources": {
                    "renewal": subset[0]["inventory_renewal"],
                    "wafer_inflow": subset[0]["domestic_wafer_inflow"],
                    "fab_capacity_multiplier": subset[0]["capacity_multiplier"],
                    "packaging_capacity_multiplier": subset[0]["packaging_capacity_multiplier"],
                    "gate_capacity_caps": subset[0]["gate_capacity_caps"],
                },
            })

    policy_summary = []
    for policy in POLICIES:
        subset = [row for row in grouped if row["policy"] == policy.name]
        robust = [row for row in subset if row["viable_rate"] >= 2 / 3]
        any_viable = [row for row in subset if row["viable_rate"] > 0]
        policy_summary.append({
            "policy": policy.name,
            "max_robust_agents": max((row["agents"] for row in robust), default=0),
            "max_any_viable_agents": max((row["agents"] for row in any_viable), default=0),
            "robust_agent_levels": [row["agents"] for row in robust],
            "any_viable_agent_levels": [row["agents"] for row in any_viable],
            "robust_levels": len(robust),
            "best_viable_rate": max(row["viable_rate"] for row in subset),
        })

    viable_rows = [row for row in rows if row["classification"] == "viable_onshoring_transition"]
    return {
        "config": {
            "agent_levels": agent_levels,
            "seeds": seeds,
            "steps": steps,
            "reference_agents": REFERENCE_AGENTS,
            "randomization_key": RANDOMIZATION_KEY,
            "policies": [asdict(policy) for policy in POLICIES],
        },
        "rows": rows,
        "grouped": grouped,
        "policy_summary": policy_summary,
        "best_viable_run": max(viable_rows, key=lambda row: row["score"], default=None),
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Semiconductor Onshoring Resource-Scaling Law Report",
        "",
        "## Scope",
        "",
        (
            "Seed-aware comparison of resource scaling laws for the near-viable onshoring "
            "doctrine. The experiment tests whether renewal, domestic wafer inflow, node "
            "capacity, and gate service capacity must scale with demand load."
        ),
        "",
        f"- Reference agents: `{payload['config']['reference_agents']}`",
        f"- Agent levels: `{', '.join(str(x) for x in payload['config']['agent_levels'])}`",
        f"- Seeds: `{', '.join(str(x) for x in payload['config']['seeds'])}`",
        f"- Steps per run: `{payload['config']['steps']}`",
        f"- Paired randomization key: `{payload['config']['randomization_key']}`",
        f"- Runs: `{len(payload['rows'])}`",
        "",
        "## Robust Phase Summary",
        "",
        "| Policy | Max Robust Agents | Max Any-Viable Agents | Robust Agent Levels | Any-Viable Agent Levels | Best Viable Rate |",
        "|---|---:|---:|---|---|---:|",
    ]
    for row in payload["policy_summary"]:
        lines.append(
            f"| `{row['policy']}` | {row['max_robust_agents']} | "
            f"{row['max_any_viable_agents']} | "
            f"`{', '.join(str(x) for x in row['robust_agent_levels']) or 'none'}` | "
            f"`{', '.join(str(x) for x in row['any_viable_agent_levels']) or 'none'}` | "
            f"{row['best_viable_rate']:.1%} |"
        )

    lines.extend([
        "",
        "## Grouped Results",
        "",
        "| Policy | Agents | Viable Rate | Mean Share | Mean Dependency | Mean Overflow | Mean Completion | Gate-Cap Block | Renewal | Wafer Inflow | FabCap | PkgCap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(payload["grouped"], key=lambda item: (item["policy"], item["agents"])):
        resources = row["resources"]
        lines.append(
            f"| `{row['policy']}` | {row['agents']} | {row['viable_rate']:.1%} | "
            f"{row['mean_onshore_share']:.3f} | {row['mean_dependency_pressure']:.1%} | "
            f"{row['mean_overflow']:.1%} | {row['mean_completion']:.1%} | "
            f"{row['mean_gate_capacity_blocked']:.1%} | {resources['renewal']} | "
            f"{resources['wafer_inflow']} | {resources['fab_capacity_multiplier']:.2f} | "
            f"{resources['packaging_capacity_multiplier']:.2f} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "Robust viability means at least two-thirds of seeds remain viable. The summary "
            "reports robust load levels explicitly because viability can be non-monotone: "
            "resource scaling changes routing, not only capacity, and can create phase islands "
            "rather than a single critical-load threshold. The shared randomization key makes "
            "baseline and policy comparisons paired within each seed."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_scaling_law_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_SCALING_LAW_REPORT.md"))
    args = parser.parse_args()
    payload = run_scaling_law(steps=args.steps)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": len(payload["rows"]),
        "policy_summary": payload["policy_summary"],
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

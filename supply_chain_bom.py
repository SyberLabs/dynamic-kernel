"""
BOM-gated supply-chain feasibility experiments.

The stress harness measures circulation: where shipment-intent agents move.
This module compares circulation against production feasibility by enforcing a
bill-of-material gate at Final Assembly. Finished goods can leave Final
Assembly only when battery, electronics, chassis, and packaging inputs are all
available.

Usage:
    .venv\\Scripts\\python.exe supply_chain_bom.py --quick
    .venv\\Scripts\\python.exe supply_chain_bom.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from supply_chain_resilience import SimulationConfig
from supply_chain_stress import StressShock, hard_capacity_shocks, severity_shocks, simulate_stress


BOM_CONTROLS = [
    "no_control",
    "dual_source_chips",
    "expedite_air",
    "buffer_release",
    "port_reroute",
    "generic_resilience",
]


@dataclass(frozen=True)
class BOMConfig:
    agents: int = 192
    steps: int = 96
    seed: int = 20260604
    bom_initial_inventory: int = 4


def bom_shocks(quick: bool = False) -> list[StressShock]:
    nominal = [StressShock(name="nominal", family="nominal")]
    if quick:
        selected = {
            "chip_fab_a_hard_failure",
            "chip_fab_b_capacity_04",
            "regional_dc_capacity_18",
            "port_and_air_capacity",
        }
        return nominal + [shock for shock in hard_capacity_shocks() if shock.name in selected]
    return (
        nominal
        + severity_shocks([2.0, 4.0])
        + hard_capacity_shocks()
    )


def _limiting_component(row: dict) -> str:
    inventory = row["bom_inventory_end"]
    if not inventory:
        return "none"
    return min(inventory, key=lambda part: (inventory[part], row["bom_arrivals"].get(part, 0)))


def _bom_score(row: dict) -> float:
    return (
        row["bom_fulfillment_share"]
        + 0.75 * row["bom_critical_service_share"]
        + 0.50 * row["bom_completion_rate"]
        - 0.10 * row["bom_block_rate"]
        - 0.25 * row["bom_capacity_overflow_rate"]
    )


def _classify(row: dict) -> str:
    if row["bom_completion_per_agent"] >= 0.40 and row["feasibility_gap"] >= 0.03:
        return "produced_but_delayed"
    if row["bom_block_rate"] >= 0.75:
        return "component_limited"
    if row["feasibility_gap"] >= 0.03:
        return "circulation_overstates_feasibility"
    if row["bom_capacity_overflow_rate"] >= 0.10:
        return "capacity_limited"
    if row["bom_completion_flow_share"] > 0.0:
        return "bom_feasible"
    return "low_throughput"


def simulate_bom_pair(config: BOMConfig, shock: StressShock, control_name: str) -> dict:
    sim_config = SimulationConfig(
        agents=config.agents,
        steps=config.steps,
        seed=config.seed,
    )
    circulation = simulate_stress(sim_config, shock, control_name, enforce_bom=False)
    bom = simulate_stress(
        sim_config,
        shock,
        control_name,
        enforce_bom=True,
        bom_initial_inventory=config.bom_initial_inventory,
    )
    row = {
        "shock": shock.name,
        "family": shock.family,
        "severity": shock.severity,
        "control": control_name,
        "control_cost": bom["control_cost"],
        "agents": config.agents,
        "steps": config.steps,
        "bom_initial_inventory": config.bom_initial_inventory,
        "circulation_fulfillment_share": circulation["fulfillment_share"],
        "circulation_critical_service_share": circulation["critical_service_share"],
        "circulation_capacity_overflow_rate": circulation["capacity_overflow_rate"],
        "bom_fulfillment_share": bom["fulfillment_share"],
        "bom_critical_service_share": bom["critical_service_share"],
        "bom_capacity_overflow_rate": bom["capacity_overflow_rate"],
        "bom_edge_capacity_overflow_rate": bom["edge_capacity_overflow_rate"],
        "bom_node_capacity_overflow_rate": bom["node_capacity_overflow_rate"],
        "bom_block_rate": bom["bom_block_rate"],
        "bom_attempts": bom["bom_attempts"],
        "bom_completion_events": bom["bom_completion_events"],
        "bom_completion_per_agent": bom["bom_completion_per_agent"],
        "bom_completion_rate": bom["bom_completion_rate"],
        "bom_completion_flow_share": bom["bom_completion_flow_share"],
        "bom_blocked_flow_share": bom["bom_blocked_flow_share"],
        "bom_inventory_end": bom["bom_inventory_end"],
        "bom_arrivals": bom["bom_arrivals"],
        "limiting_component": _limiting_component(bom),
    }
    row["feasibility_gap"] = row["circulation_fulfillment_share"] - row["bom_fulfillment_share"]
    row["critical_gap"] = row["circulation_critical_service_share"] - row["bom_critical_service_share"]
    row["bom_score"] = _bom_score(row)
    row["classification"] = _classify(row)
    return row


def _with_baseline_deltas(rows: list[dict]) -> list[dict]:
    baselines = {
        row["shock"]: row
        for row in rows
        if row["control"] == "no_control"
    }
    enriched = []
    for row in rows:
        baseline = baselines[row["shock"]]
        item = dict(row)
        item["bom_fulfillment_delta_vs_baseline"] = (
            row["bom_fulfillment_share"] - baseline["bom_fulfillment_share"]
        )
        item["bom_critical_delta_vs_baseline"] = (
            row["bom_critical_service_share"] - baseline["bom_critical_service_share"]
        )
        item["bom_score_delta_vs_baseline"] = row["bom_score"] - baseline["bom_score"]
        item["bom_block_rate_delta_vs_baseline"] = row["bom_block_rate"] - baseline["bom_block_rate"]
        enriched.append(item)
    return enriched


def run_bom_suite(config: BOMConfig | None = None, quick: bool = False) -> dict:
    config = config or BOMConfig()
    rows = [
        simulate_bom_pair(config, shock, control)
        for shock in bom_shocks(quick=quick)
        for control in BOM_CONTROLS
    ]
    rows = _with_baseline_deltas(rows)
    classifications = {
        label: sum(1 for row in rows if row["classification"] == label)
        for label in sorted({row["classification"] for row in rows})
    }
    best_by_shock = {}
    for shock in sorted({row["shock"] for row in rows}):
        candidates = [row for row in rows if row["shock"] == shock and row["control"] != "no_control"]
        best = max(candidates, key=lambda row: (row["bom_score_delta_vs_baseline"], row["bom_fulfillment_delta_vs_baseline"]))
        best_by_shock[shock] = {
            "control": best["control"],
            "score_delta": best["bom_score_delta_vs_baseline"],
            "fulfillment_delta": best["bom_fulfillment_delta_vs_baseline"],
            "critical_delta": best["bom_critical_delta_vs_baseline"],
            "block_rate_delta": best["bom_block_rate_delta_vs_baseline"],
            "classification": best["classification"],
            "limiting_component": best["limiting_component"],
        }
    return {
        "config": asdict(config) | {"quick": quick},
        "classification_counts": classifications,
        "best_by_shock": best_by_shock,
        "rows": rows,
    }


def render_report(payload: dict) -> str:
    rows = payload["rows"]
    worst_gaps = sorted(rows, key=lambda row: row["feasibility_gap"], reverse=True)[:10]
    best_by_shock = payload["best_by_shock"]
    component_counts = {
        part: sum(1 for row in rows if row["limiting_component"] == part)
        for part in sorted({row["limiting_component"] for row in rows})
    }
    lines = [
        "# Supply Chain BOM Feasibility Report",
        "",
        "## Scope",
        "",
        (
            "Paired circulation-vs-BOM simulations. Circulation allows agents to move through "
            "Final Assembly normally; BOM enforcement requires battery, electronics, chassis, "
            "and packaging inventory before finished goods can leave Final Assembly."
        ),
        "",
        f"- Agents per run: `{payload['config']['agents']}`",
        f"- Steps per run: `{payload['config']['steps']}`",
        f"- Initial WIP inventory per component: `{payload['config']['bom_initial_inventory']}`",
        f"- Scenario-control rows: `{len(rows)}`",
        "",
        "## Classification Counts",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(payload["classification_counts"].items()):
        lines.append(f"| `{label}` | {count} |")

    lines.extend([
        "",
        "## Limiting Components",
        "",
        "| Component | Rows Limited |",
        "|---|---:|",
    ])
    for part, count in component_counts.items():
        lines.append(f"| `{part}` | {count} |")

    lines.extend([
        "",
        "## Largest Circulation-Feasibility Gaps",
        "",
        "| Shock | Control | Gap | Circulation Fulfillment | BOM Fulfillment | Completions / Agent | BOM Block Rate | Limiting Component |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in worst_gaps:
        lines.append(
            f"| `{row['shock']}` | `{row['control']}` | {row['feasibility_gap']:+.3f} | "
            f"{row['circulation_fulfillment_share']:.3f} | {row['bom_fulfillment_share']:.3f} | "
            f"{row['bom_completion_per_agent']:.2f} | {row['bom_block_rate']:.1%} | `{row['limiting_component']}` |"
        )

    lines.extend([
        "",
        "## Best BOM-Aware Control By Shock",
        "",
        "| Shock | Best Control | Score Delta | Fulfillment Delta | Critical Delta | Block Rate Delta | Classification | Limiting Component |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ])
    for shock, row in best_by_shock.items():
        lines.append(
            f"| `{shock}` | `{row['control']}` | {row['score_delta']:+.4f} | "
            f"{row['fulfillment_delta']:+.3f} | {row['critical_delta']:+.3f} | "
            f"{row['block_rate_delta']:+.3f} | `{row['classification']}` | `{row['limiting_component']}` |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "The BOM gate converts DTE from a pure circulation model into a production-feasibility "
            "model. The gap column is the warning signal: it estimates how much apparent logistics "
            "success disappears once component complementarity is enforced. This is the required "
            "mechanic before a Taiwan semiconductor adapter, where wafers, tools, chemicals, energy, "
            "packaging, export controls, shipping, and demand channels must align."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("supply_chain_bom_output.json"),
    output_md: Path = Path("SUPPLY_CHAIN_BOM_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BOM-gated supply-chain feasibility suite.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--agents", type=int, default=BOMConfig.agents)
    parser.add_argument("--steps", type=int, default=BOMConfig.steps)
    parser.add_argument("--seed", type=int, default=BOMConfig.seed)
    parser.add_argument("--bom-initial-inventory", type=int, default=BOMConfig.bom_initial_inventory)
    parser.add_argument("--output-json", type=Path, default=Path("supply_chain_bom_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SUPPLY_CHAIN_BOM_REPORT.md"))
    args = parser.parse_args()

    config = BOMConfig(
        agents=64 if args.quick and args.agents == BOMConfig.agents else args.agents,
        steps=32 if args.quick and args.steps == BOMConfig.steps else args.steps,
        seed=args.seed,
        bom_initial_inventory=args.bom_initial_inventory,
    )
    payload = run_bom_suite(config, quick=args.quick)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "classification_counts": payload["classification_counts"],
        "rows": len(payload["rows"]),
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

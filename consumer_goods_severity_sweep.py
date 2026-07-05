"""
Seed-robust severity sweeps for the consumer-goods cold-chain pilot.

The sweep varies two stress axes:

- cold-chain capacity severity
- promotion-pressure severity

and reports which controls remain robust across seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consumer_goods_circulation import (
    SimulationConfig,
    Scenario,
    _classify,
    _edge_cost,
    controls,
    simulate,
    static_expected_service_share,
)


COLD_EDGES = (
    ("Finished Goods Staging", "Cold Chain Carrier"),
    ("Cold Chain Carrier", "Regional DC"),
)
PROMOTION_EDGES = (
    ("Planning Desk", "Promotion Demand"),
    ("Promotion Demand", "Standard Retail Accounts"),
)


def severity_scenario(family: str, severity: float) -> Scenario:
    if family == "cold_chain":
        cold_gate_cap = max(2, int(round(8 - 3 * severity)))
        carrier_cap = max(3, int(round(10 - 3 * severity)))
        return Scenario(
            name=f"cold_chain_severity_{severity:.2f}",
            family=family,
            cost=_edge_cost(COLD_EDGES, severity),
            friction_edges=COLD_EDGES,
            friction_delta=-0.8 * severity,
            gate_capacity_caps={"cold_chain_gate": cold_gate_cap},
            edge_capacity_caps={("Cold Chain Carrier", "Regional DC"): carrier_cap},
        )
    if family == "promotion_pressure":
        standard_cap = max(6, int(round(14 - 3 * severity)))
        return Scenario(
            name=f"promotion_pressure_{severity:.2f}",
            family=family,
            cost=_edge_cost(PROMOTION_EDGES, severity),
            beta_edges=PROMOTION_EDGES,
            beta_boost=1.2 * severity,
            node_capacity_caps={"Standard Retail Accounts": standard_cap},
        )
    if family == "combined":
        cold = severity_scenario("cold_chain", severity)
        promo = severity_scenario("promotion_pressure", severity)
        return Scenario(
            name=f"combined_severity_{severity:.2f}",
            family=family,
            cost=cold.cost + promo.cost,
            friction_edges=cold.friction_edges,
            friction_delta=cold.friction_delta,
            beta_edges=promo.beta_edges,
            beta_boost=promo.beta_boost,
            gate_capacity_caps=cold.gate_capacity_caps,
            edge_capacity_caps=cold.edge_capacity_caps,
            node_capacity_caps=promo.node_capacity_caps,
        )
    raise ValueError(f"unknown severity family: {family}")


def _evaluate(config: SimulationConfig, scenario: Scenario, control: Scenario, baseline: dict, frozen_baseline: dict) -> dict:
    row = simulate(config, scenario, control)
    frozen_config = SimulationConfig(
        agents=config.agents,
        steps=config.steps,
        feedback_rate=0.0,
        temperature=config.temperature,
        seed=config.seed,
        gate_initial_inventory=config.gate_initial_inventory,
        randomization_key=config.randomization_key,
    )
    frozen = simulate(frozen_config, scenario, control)
    row["classification"] = _classify(row, baseline)
    row["service_delta_vs_baseline"] = row["service_completion_rate"] - baseline["service_completion_rate"]
    row["priority_delta_vs_baseline"] = row["priority_service_rate"] - baseline["priority_service_rate"]
    row["lost_delta_vs_baseline"] = row["lost_demand_rate"] - baseline["lost_demand_rate"]
    row["static_expected_service_share"] = static_expected_service_share(config, scenario, control)
    row["frozen_service_completion_rate"] = frozen["service_completion_rate"]
    row["frozen_classification"] = _classify(frozen, frozen_baseline)
    row["resilience_score"] = (
        row["service_delta_vs_baseline"]
        + 0.70 * row["priority_delta_vs_baseline"]
        - 0.50 * max(row["lost_delta_vs_baseline"], 0.0)
        - 0.40 * row["capacity_overflow_rate"]
        - 0.05 * row["gate_pressure_rate"]
        - 0.002 * row["control_cost"]
    )
    return row


def run_severity_sweep(
    families: tuple[str, ...] = ("cold_chain", "promotion_pressure", "combined"),
    severities: tuple[float, ...] = (0.0, 0.75, 1.5, 2.25),
    seeds: tuple[int, ...] = (20260611, 20260612, 20260613),
    agents: int = 128,
    steps: int = 40,
) -> dict:
    rows = []
    control_list = controls()
    for family in families:
        for severity in severities:
            scenario = severity_scenario(family, severity)
            for seed in seeds:
                config = SimulationConfig(agents=agents, steps=steps, seed=seed)
                frozen_config = SimulationConfig(
                    agents=agents,
                    steps=steps,
                    feedback_rate=0.0,
                    seed=seed,
                    gate_initial_inventory=config.gate_initial_inventory,
                    randomization_key=config.randomization_key,
                )
                baseline = simulate(config, scenario, control_list[0])
                frozen_baseline = simulate(frozen_config, scenario, control_list[0])
                for control in control_list:
                    row = _evaluate(config, scenario, control, baseline, frozen_baseline)
                    row.update({
                        "family": family,
                        "severity": severity,
                        "seed": seed,
                        "baseline_service_completion_rate": baseline["service_completion_rate"],
                    })
                    rows.append(row)

    grouped = []
    for family in families:
        for severity in severities:
            for control in [item.name for item in control_list]:
                subset = [
                    row for row in rows
                    if row["family"] == family
                    and row["severity"] == severity
                    and row["control"] == control
                ]
                class_counts = {
                    label: sum(row["classification"] == label for row in subset)
                    for label in sorted({row["classification"] for row in subset})
                }
                grouped.append({
                    "family": family,
                    "severity": severity,
                    "control": control,
                    "runs": len(subset),
                    "viable_rate": class_counts.get("viable_service_recovery", 0) / len(subset),
                    "partial_rate": class_counts.get("partial_recovery", 0) / len(subset),
                    "backfire_rate": class_counts.get("capacity_backfire", 0) / len(subset),
                    "no_recovery_rate": class_counts.get("no_recovery", 0) / len(subset),
                    "mean_service": sum(row["service_completion_rate"] for row in subset) / len(subset),
                    "mean_priority": sum(row["priority_service_rate"] for row in subset) / len(subset),
                    "mean_lost": sum(row["lost_demand_rate"] for row in subset) / len(subset),
                    "mean_overflow": sum(row["capacity_overflow_rate"] for row in subset) / len(subset),
                    "mean_gate_pressure": sum(row["gate_pressure_rate"] for row in subset) / len(subset),
                    "mean_gate_starvation": sum(row["gate_starvation_rate"] for row in subset) / len(subset),
                    "mean_gate_capacity_block": sum(row["gate_service_capacity_block_rate"] for row in subset) / len(subset),
                    "mean_gate_contention": sum(row["gate_contention_rate"] for row in subset) / len(subset),
                    "mean_score": sum(row["resilience_score"] for row in subset) / len(subset),
                    "class_counts": class_counts,
                })

    best_by_family_severity = {}
    for family in families:
        for severity in severities:
            subset = [
                row for row in grouped
                if row["family"] == family and row["severity"] == severity
            ]
            best = max(
                subset,
                key=lambda row: (
                    row["viable_rate"],
                    row["partial_rate"],
                    -row["backfire_rate"],
                    row["mean_score"],
                ),
            )
            best_by_family_severity[f"{family}:{severity:.2f}"] = best

    robust_controls = [
        row for row in grouped
        if row["viable_rate"] >= 2 / 3
    ]
    useful_controls = [
        row for row in grouped
        if row["viable_rate"] + row["partial_rate"] >= 2 / 3
        and row["backfire_rate"] < 2 / 3
    ]
    boundary_summary = {}
    for family in families:
        robust_family_rows = [row for row in robust_controls if row["family"] == family]
        useful_family_rows = [row for row in useful_controls if row["family"] == family]
        all_backfire = []
        for severity in severities:
            family_cell = [
                row for row in grouped
                if row["family"] == family and row["severity"] == severity
            ]
            if family_cell and all(row["backfire_rate"] >= 2 / 3 for row in family_cell):
                all_backfire.append(severity)
        useful_severity = max((row["severity"] for row in useful_family_rows), default=None)
        useful_at_boundary = [
            row for row in useful_family_rows
            if useful_severity is not None and row["severity"] == useful_severity
        ]
        best_useful = max(
            useful_at_boundary,
            key=lambda row: (
                row["viable_rate"] + row["partial_rate"],
                -row["backfire_rate"],
                row["mean_score"],
            ),
            default=None,
        )
        boundary_summary[family] = {
            "max_robust_viable_severity": max(
                (row["severity"] for row in robust_family_rows),
                default=None,
            ),
            "max_useful_non_backfire_severity": useful_severity,
            "best_useful_control_at_boundary": best_useful["control"] if best_useful else None,
            "first_all_controls_backfire_severity": min(all_backfire) if all_backfire else None,
        }
    return {
        "config": {
            "families": families,
            "severities": severities,
            "seeds": seeds,
            "agents": agents,
            "steps": steps,
        },
        "rows": rows,
        "grouped": grouped,
        "best_by_family_severity": best_by_family_severity,
        "robust_controls": robust_controls,
        "useful_controls": useful_controls,
        "boundary_summary": boundary_summary,
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Consumer Goods Severity Robustness Report",
        "",
        "## Scope",
        "",
        (
            "Seed-robust severity sweep over cold-chain capacity stress, promotion "
            "pressure, and combined stress for the refrigerated packaged-goods pilot."
        ),
        "",
        f"- Families: `{', '.join(payload['config']['families'])}`",
        f"- Severities: `{', '.join(f'{x:.2f}' for x in payload['config']['severities'])}`",
        f"- Seeds: `{', '.join(str(x) for x in payload['config']['seeds'])}`",
        f"- Agents: `{payload['config']['agents']}`",
        f"- Steps: `{payload['config']['steps']}`",
        f"- Runs: `{len(payload['rows'])}`",
        "",
        "## Boundary Summary",
        "",
        "| Family | Max Robust Viable Severity | Max Useful Non-Backfire Severity | Boundary Control | First All-Backfire Severity |",
        "|---|---:|---:|---|---:|",
    ]
    for family, row in payload["boundary_summary"].items():
        robust = "none" if row["max_robust_viable_severity"] is None else f"{row['max_robust_viable_severity']:.2f}"
        useful = "none" if row["max_useful_non_backfire_severity"] is None else f"{row['max_useful_non_backfire_severity']:.2f}"
        all_backfire = (
            "none"
            if row["first_all_controls_backfire_severity"] is None
            else f"{row['first_all_controls_backfire_severity']:.2f}"
        )
        lines.append(
            f"| `{family}` | {robust} | {useful} | "
            f"`{row['best_useful_control_at_boundary'] or 'none'}` | {all_backfire} |"
        )

    lines.extend([
        "",
        "## Robust Controls",
        "",
        "| Family | Severity | Control | Viable Rate | Mean Service | Mean Overflow | Starvation | Gate Capacity | Gate Load |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["robust_controls"]:
        lines.append(
            f"| `{row['family']}` | {row['severity']:.2f} | `{row['control']}` | "
            f"{row['viable_rate']:.1%} | {row['mean_service']:.1%} | "
            f"{row['mean_overflow']:.1%} | {row['mean_gate_starvation']:.1%} | "
            f"{row['mean_gate_capacity_block']:.1%} | {row['mean_gate_contention']:.1%} |"
        )
    if not payload["robust_controls"]:
        lines.append("| none | none | none | none | none | none | none | none | none |")

    lines.extend([
        "",
        "## Useful Non-Backfire Controls",
        "",
        "| Family | Severity | Control | Viable+Partial Rate | Backfire Rate | Mean Service | Mean Priority | Mean Score |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["useful_controls"]:
        lines.append(
            f"| `{row['family']}` | {row['severity']:.2f} | `{row['control']}` | "
            f"{(row['viable_rate'] + row['partial_rate']):.1%} | {row['backfire_rate']:.1%} | "
            f"{row['mean_service']:.1%} | {row['mean_priority']:.1%} | {row['mean_score']:.4f} |"
        )

    lines.extend([
        "",
        "## Best Control By Family And Severity",
        "",
        "| Family | Severity | Best Control | Viable | Partial | Backfire | Service | Lost | Score |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key in sorted(payload["best_by_family_severity"]):
        row = payload["best_by_family_severity"][key]
        lines.append(
            f"| `{row['family']}` | {row['severity']:.2f} | `{row['control']}` | "
            f"{row['viable_rate']:.1%} | {row['partial_rate']:.1%} | {row['backfire_rate']:.1%} | "
            f"{row['mean_service']:.1%} | {row['mean_lost']:.1%} | {row['mean_score']:.4f} |"
        )

    lines.extend([
        "",
        "## Full Grouped Surface",
        "",
        "| Family | Severity | Control | Viable | Partial | Backfire | No Recovery | Service | Priority | Overflow | Starvation | Gate Capacity | Gate Load |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["grouped"]:
        lines.append(
            f"| `{row['family']}` | {row['severity']:.2f} | `{row['control']}` | "
            f"{row['viable_rate']:.1%} | {row['partial_rate']:.1%} | {row['backfire_rate']:.1%} | "
            f"{row['no_recovery_rate']:.1%} | {row['mean_service']:.1%} | "
            f"{row['mean_priority']:.1%} | {row['mean_overflow']:.1%} | "
            f"{row['mean_gate_starvation']:.1%} | {row['mean_gate_capacity_block']:.1%} | "
            f"{row['mean_gate_contention']:.1%} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "A robust control is viable in at least two-thirds of seeds. Useful "
            "non-backfire controls may be only partial recoveries, but they avoid becoming "
            "capacity backfires in most seeds. This is the appropriate standard for a "
            "first industrial module slice: separate reliable recovery, partial mitigation, "
            "and attractive-but-overloading interventions."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("consumer_goods_severity_sweep_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("CONSUMER_GOODS_SEVERITY_SWEEP_REPORT.md"))
    args = parser.parse_args()
    payload = run_severity_sweep(
        severities=(0.0, 1.5) if args.quick else (0.0, 0.75, 1.5, 2.25),
        seeds=(20260611,) if args.quick else (20260611, 20260612, 20260613),
        agents=48 if args.quick else 128,
        steps=20 if args.quick else 40,
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": len(payload["rows"]),
        "robust_controls": len(payload["robust_controls"]),
        "useful_controls": len(payload["useful_controls"]),
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

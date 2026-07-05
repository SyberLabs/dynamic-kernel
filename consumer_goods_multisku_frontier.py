"""
Calibrated frontier for the multi-SKU consumer-goods topology.

The frontier stresses four industrial levers:

- promotion intensity
- regional receiving caps
- reserved cold-chain slot availability
- spot-market cold-chain availability

and includes a combined ladder where all four worsen together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consumer_goods_multisku import (
    MultiSKUConfig,
    Scenario,
    _classify,
    _edge_cost,
    controls,
    simulate,
)


PROMO_EDGES = (
    ("Demand Control Tower", "Promo Calendar"),
    ("Promo Calendar", "Yogurt Finished Staging"),
    ("Promo Calendar", "North Standard Accounts"),
    ("Promo Calendar", "South Standard Accounts"),
)
RESERVED_EDGES = (
    ("Core Finished Staging", "Reserved Cold Carrier"),
    ("Yogurt Finished Staging", "Reserved Cold Carrier"),
    ("Frozen Finished Staging", "Reserved Cold Carrier"),
    ("Reserved Cold Carrier", "North DC"),
    ("Reserved Cold Carrier", "South DC"),
)
SPOT_EDGES = (
    ("Core Finished Staging", "Spot Cold Carrier"),
    ("Yogurt Finished Staging", "Spot Cold Carrier"),
    ("Frozen Finished Staging", "Spot Cold Carrier"),
    ("Spot Cold Carrier", "North DC"),
    ("Spot Cold Carrier", "South DC"),
)


def frontier_scenario(family: str, severity: float) -> Scenario:
    promo_boost = 1.15 * severity
    standard_cap = max(5, int(round(18 - 5 * severity)))
    south_receiving_cap = max(5, int(round(18 - 5 * severity)))
    reserved_core_cap = max(2, int(round(8 - 2.5 * severity)))
    reserved_frozen_cap = max(1, int(round(6 - 2 * severity)))
    reserved_edge_cap = max(4, int(round(12 - 3 * severity)))
    spot_gate_cap = max(1, int(round(5 - 1.5 * severity)))
    spot_edge_cap = max(2, int(round(7 - 2 * severity)))

    if family == "promotion_intensity":
        return Scenario(
            name=f"promotion_intensity_{severity:.2f}",
            family=family,
            cost=_edge_cost(PROMO_EDGES, promo_boost),
            beta_edges=PROMO_EDGES,
            beta_boost=promo_boost,
            node_capacity_caps={
                "North Standard Accounts": standard_cap,
                "South Standard Accounts": standard_cap,
            },
        )
    if family == "receiving_cap":
        return Scenario(
            name=f"receiving_cap_{severity:.2f}",
            family=family,
            cost=float(severity),
            node_capacity_caps={
                "North DC": south_receiving_cap,
                "South DC": south_receiving_cap,
                "North Standard Accounts": standard_cap,
                "South Standard Accounts": standard_cap,
            },
        )
    if family == "reserved_slots":
        return Scenario(
            name=f"reserved_slots_{severity:.2f}",
            family=family,
            cost=_edge_cost(RESERVED_EDGES, severity),
            friction_edges=RESERVED_EDGES,
            friction_delta=-0.85 * severity,
            gate_capacity_caps={
                "core_reserved_gate": reserved_core_cap,
                "yogurt_reserved_gate": reserved_core_cap,
                "frozen_reserved_gate": reserved_frozen_cap,
            },
            edge_capacity_caps={
                ("Reserved Cold Carrier", "North DC"): reserved_edge_cap,
                ("Reserved Cold Carrier", "South DC"): reserved_edge_cap,
            },
        )
    if family == "spot_market":
        return Scenario(
            name=f"spot_market_{severity:.2f}",
            family=family,
            cost=_edge_cost(SPOT_EDGES, severity),
            friction_edges=SPOT_EDGES,
            friction_delta=-1.05 * severity,
            gate_capacity_caps={
                "core_spot_gate": spot_gate_cap,
                "yogurt_spot_gate": spot_gate_cap,
                "frozen_spot_gate": spot_gate_cap,
            },
            edge_capacity_caps={
                ("Spot Cold Carrier", "North DC"): spot_edge_cap,
                ("Spot Cold Carrier", "South DC"): spot_edge_cap,
            },
        )
    if family == "combined":
        return Scenario(
            name=f"combined_frontier_{severity:.2f}",
            family=family,
            cost=_edge_cost(PROMO_EDGES + RESERVED_EDGES + SPOT_EDGES, severity),
            beta_edges=PROMO_EDGES,
            beta_boost=promo_boost,
            friction_edges=RESERVED_EDGES + SPOT_EDGES,
            friction_delta=-0.80 * severity,
            gate_capacity_caps={
                "core_reserved_gate": reserved_core_cap,
                "yogurt_reserved_gate": reserved_core_cap,
                "frozen_reserved_gate": reserved_frozen_cap,
                "core_spot_gate": spot_gate_cap,
                "yogurt_spot_gate": spot_gate_cap,
                "frozen_spot_gate": spot_gate_cap,
            },
            edge_capacity_caps={
                ("Reserved Cold Carrier", "North DC"): reserved_edge_cap,
                ("Reserved Cold Carrier", "South DC"): reserved_edge_cap,
                ("Spot Cold Carrier", "North DC"): spot_edge_cap,
                ("Spot Cold Carrier", "South DC"): spot_edge_cap,
            },
            node_capacity_caps={
                "North DC": south_receiving_cap,
                "South DC": south_receiving_cap,
                "North Standard Accounts": standard_cap,
                "South Standard Accounts": standard_cap,
            },
        )
    raise ValueError(f"unknown frontier family: {family}")


def _score(row: dict, baseline: dict) -> float:
    return (
        row["service_completion_rate"]
        - baseline["service_completion_rate"]
        + 0.70 * (row["priority_service_rate"] - baseline["priority_service_rate"])
        + 0.20 * (row["substitution_rate"] - baseline["substitution_rate"])
        - 0.50 * max(row["lost_demand_rate"] - baseline["lost_demand_rate"], 0.0)
        - 0.40 * row["capacity_overflow_rate"]
        - 0.05 * row["gate_pressure_rate"]
        - 0.002 * row["control_cost"]
    )


def _evaluate(config: MultiSKUConfig, scenario: Scenario, control: Scenario, baseline: dict) -> dict:
    row = simulate(config, scenario, control)
    row["classification"] = _classify(row, baseline)
    row["service_delta_vs_baseline"] = row["service_completion_rate"] - baseline["service_completion_rate"]
    row["priority_delta_vs_baseline"] = row["priority_service_rate"] - baseline["priority_service_rate"]
    row["lost_delta_vs_baseline"] = row["lost_demand_rate"] - baseline["lost_demand_rate"]
    row["substitution_delta_vs_baseline"] = row["substitution_rate"] - baseline["substitution_rate"]
    row["frontier_score"] = _score(row, baseline)
    return row


def run_frontier(
    families: tuple[str, ...] = (
        "promotion_intensity",
        "receiving_cap",
        "reserved_slots",
        "spot_market",
        "combined",
    ),
    severities: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0),
    seeds: tuple[int, ...] = (20260617, 20260618, 20260619),
    agents: int = 160,
    steps: int = 44,
) -> dict:
    rows = []
    control_list = controls()
    for family in families:
        for severity in severities:
            scenario = frontier_scenario(family, severity)
            for seed in seeds:
                config = MultiSKUConfig(agents=agents, steps=steps, seed=seed)
                baseline = simulate(config, scenario, control_list[0])
                for control in control_list:
                    row = _evaluate(config, scenario, control, baseline)
                    row.update({
                        "family": family,
                        "severity": severity,
                        "seed": seed,
                        "baseline_service_completion_rate": baseline["service_completion_rate"],
                        "baseline_overflow_rate": baseline["capacity_overflow_rate"],
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
                    "mean_substitution": sum(row["substitution_rate"] for row in subset) / len(subset),
                    "mean_lost": sum(row["lost_demand_rate"] for row in subset) / len(subset),
                    "mean_overflow": sum(row["capacity_overflow_rate"] for row in subset) / len(subset),
                    "mean_starvation": sum(row["gate_starvation_rate"] for row in subset) / len(subset),
                    "mean_gate_capacity": sum(row["gate_service_capacity_block_rate"] for row in subset) / len(subset),
                    "mean_score": sum(row["frontier_score"] for row in subset) / len(subset),
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

    robust_controls = [row for row in grouped if row["viable_rate"] >= 2 / 3]
    useful_controls = [
        row for row in grouped
        if row["viable_rate"] + row["partial_rate"] >= 2 / 3
        and row["backfire_rate"] < 2 / 3
    ]
    recommendable_controls = [
        row for row in useful_controls
        if row["mean_score"] >= 0.0
    ]
    boundary_summary = {}
    for family in families:
        family_useful = [row for row in useful_controls if row["family"] == family]
        family_robust = [row for row in robust_controls if row["family"] == family]
        all_backfire = []
        for severity in severities:
            cell = [
                row for row in grouped
                if row["family"] == family and row["severity"] == severity
            ]
            if cell and all(row["backfire_rate"] >= 2 / 3 for row in cell):
                all_backfire.append(severity)
        useful_severity = max((row["severity"] for row in family_useful), default=None)
        useful_at_boundary = [
            row for row in family_useful
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
                (row["severity"] for row in family_robust),
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
        "recommendable_controls": recommendable_controls,
        "boundary_summary": boundary_summary,
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Consumer Goods Multi-SKU Calibrated Frontier",
        "",
        "## Scope",
        "",
        (
            "Seed-robust frontier over promotion intensity, regional receiving caps, "
            "reserved cold-chain availability, spot-market availability, and a combined ladder."
        ),
        "",
        f"- Families: `{', '.join(payload['config']['families'])}`",
        f"- Severities: `{', '.join(f'{item:.2f}' for item in payload['config']['severities'])}`",
        f"- Seeds: `{', '.join(str(item) for item in payload['config']['seeds'])}`",
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
        all_backfire = "none" if row["first_all_controls_backfire_severity"] is None else f"{row['first_all_controls_backfire_severity']:.2f}"
        lines.append(
            f"| `{family}` | {robust} | {useful} | "
            f"`{row['best_useful_control_at_boundary'] or 'none'}` | {all_backfire} |"
        )

    lines.extend([
        "",
        "## Best Control By Family And Severity",
        "",
        "| Family | Severity | Best Control | Viable | Partial | Backfire | Service | Priority | Overflow | Starvation | Gate Capacity | Score |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key in sorted(payload["best_by_family_severity"]):
        row = payload["best_by_family_severity"][key]
        lines.append(
            f"| `{row['family']}` | {row['severity']:.2f} | `{row['control']}` | "
            f"{row['viable_rate']:.1%} | {row['partial_rate']:.1%} | {row['backfire_rate']:.1%} | "
            f"{row['mean_service']:.1%} | {row['mean_priority']:.1%} | {row['mean_overflow']:.1%} | "
            f"{row['mean_starvation']:.1%} | {row['mean_gate_capacity']:.1%} | {row['mean_score']:.4f} |"
        )

    lines.extend([
        "",
        "## Recommendable Controls",
        "",
        "| Family | Severity | Control | Viable+Partial | Backfire | Service | Priority | Overflow | Score |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["recommendable_controls"]:
        lines.append(
            f"| `{row['family']}` | {row['severity']:.2f} | `{row['control']}` | "
            f"{(row['viable_rate'] + row['partial_rate']):.1%} | {row['backfire_rate']:.1%} | "
            f"{row['mean_service']:.1%} | {row['mean_priority']:.1%} | "
            f"{row['mean_overflow']:.1%} | {row['mean_score']:.4f} |"
        )
    if not payload["recommendable_controls"]:
        lines.append("| none | none | none | none | none | none | none | none | none |")

    lines.extend([
        "",
        "## Useful Non-Backfire Controls",
        "",
        "| Family | Severity | Control | Viable+Partial | Backfire | Service | Priority | Overflow | Score |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["useful_controls"]:
        lines.append(
            f"| `{row['family']}` | {row['severity']:.2f} | `{row['control']}` | "
            f"{(row['viable_rate'] + row['partial_rate']):.1%} | {row['backfire_rate']:.1%} | "
            f"{row['mean_service']:.1%} | {row['mean_priority']:.1%} | "
            f"{row['mean_overflow']:.1%} | {row['mean_score']:.4f} |"
        )
    if not payload["useful_controls"]:
        lines.append("| none | none | none | none | none | none | none | none | none |")

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "A robust control is viable in at least two-thirds of seeds. A useful "
            "non-backfire control may be only partial, but it must avoid becoming a "
            "capacity backfire in most seeds. A recommendable control is useful and "
            "has non-negative score after service, priority, substitution, lost-demand, "
            "overflow, gate, and cost penalties. The combined ladder is the strongest "
            "test: it asks whether the topology can absorb promotion, receiving, "
            "reserved-carrier, and spot-market stress simultaneously."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("consumer_goods_multisku_frontier_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("CONSUMER_GOODS_MULTISKU_FRONTIER_REPORT.md"))
    args = parser.parse_args()
    payload = run_frontier(
        families=("promotion_intensity", "reserved_slots") if args.quick else (
            "promotion_intensity",
            "receiving_cap",
            "reserved_slots",
            "spot_market",
            "combined",
        ),
        severities=(0.0, 2.0) if args.quick else (0.0, 1.0, 2.0, 3.0),
        seeds=(20260617,) if args.quick else (20260617, 20260618, 20260619),
        agents=64 if args.quick else 160,
        steps=20 if args.quick else 44,
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": len(payload["rows"]),
        "robust_controls": len(payload["robust_controls"]),
        "useful_controls": len(payload["useful_controls"]),
        "recommendable_controls": len(payload["recommendable_controls"]),
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

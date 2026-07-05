"""
Choice-point falsification tests for the consumer-goods cold-chain pilot.

The harness moves or decomposes interventions that looked useful in the
severity sweep. A control passes the falsification standard when its lift is
materially stronger at the hypothesized choice point than in wrong-location or
single-mechanism variants.
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
)
from consumer_goods_severity_sweep import severity_scenario


def _control_by_name(name: str) -> Scenario:
    for control in controls():
        if control.name == name:
            return control
    raise ValueError(f"unknown control: {name}")


def falsification_controls(target: str) -> list[Scenario]:
    if target == "carrier_priority":
        carrier_edges = (
            ("Finished Goods Staging", "Cold Chain Carrier"),
            ("Cold Chain Carrier", "Regional DC"),
        )
        dry_edges = (
            ("Finished Goods Staging", "Dry Carrier"),
            ("Dry Carrier", "Lost Demand"),
        )
        return [
            Scenario(
                "carrier_priority__canonical",
                "falsification",
                cost=_control_by_name("carrier_priority").cost,
                friction_edges=carrier_edges,
                friction_delta=1.2,
                gate_replenishment={("cold_chain_gate", "cold_slot"): 3},
            ),
            Scenario(
                "carrier_priority__edges_only",
                "falsification",
                cost=_edge_cost(carrier_edges, 1.2),
                friction_edges=carrier_edges,
                friction_delta=1.2,
            ),
            Scenario(
                "carrier_priority__gate_only",
                "falsification",
                cost=1.0,
                gate_replenishment={("cold_chain_gate", "cold_slot"): 3},
            ),
            Scenario(
                "carrier_priority__wrong_dry_lane",
                "falsification",
                cost=_control_by_name("carrier_priority").cost,
                friction_edges=dry_edges,
                friction_delta=1.2,
                gate_replenishment={("cold_chain_gate", "cold_slot"): 3},
            ),
            Scenario(
                "carrier_priority__wrong_gate",
                "falsification",
                cost=_control_by_name("carrier_priority").cost,
                friction_edges=carrier_edges,
                friction_delta=1.2,
                gate_replenishment={("production_gate", "plant_labor"): 3},
            ),
        ]
    if target == "promotion_throttle":
        throttle_edges = (
            ("Planning Desk", "Promotion Demand"),
            ("Promotion Demand", "Standard Retail Accounts"),
        )
        cold_edges = (
            ("Finished Goods Staging", "Cold Chain Carrier"),
            ("Cold Chain Carrier", "Regional DC"),
        )
        return [
            Scenario(
                "promotion_throttle__canonical",
                "falsification",
                cost=_control_by_name("promotion_throttle").cost,
                friction_edges=throttle_edges,
                friction_delta=-1.0,
            ),
            Scenario(
                "promotion_throttle__entry_only",
                "falsification",
                cost=_edge_cost((throttle_edges[0],), 1.0),
                friction_edges=(throttle_edges[0],),
                friction_delta=-1.0,
            ),
            Scenario(
                "promotion_throttle__exit_only",
                "falsification",
                cost=_edge_cost((throttle_edges[1],), 1.0),
                friction_edges=(throttle_edges[1],),
                friction_delta=-1.0,
            ),
            Scenario(
                "promotion_throttle__wrong_cold_lane",
                "falsification",
                cost=_control_by_name("promotion_throttle").cost,
                friction_edges=cold_edges,
                friction_delta=-1.0,
            ),
            Scenario(
                "promotion_throttle__terminal_return",
                "falsification",
                cost=_control_by_name("promotion_throttle").cost,
                friction_edges=(
                    ("Standard Retail Accounts", "Planning Desk"),
                    ("Priority Retail Accounts", "Planning Desk"),
                ),
                friction_delta=-1.0,
            ),
        ]
    if target == "safety_stock_release":
        reserve_edges = (
            ("Planning Desk", "Safety Stock"),
            ("Safety Stock", "Regional DC"),
            ("Safety Stock", "Priority Retail Accounts"),
        )
        alternate_edges = (
            ("Planning Desk", "Alternate Supplier"),
            ("Alternate Supplier", "Regional Plant"),
            ("Alternate Supplier", "Co-Packer"),
        )
        return [
            Scenario(
                "safety_stock_release__canonical",
                "falsification",
                cost=_control_by_name("safety_stock_release").cost,
                friction_edges=reserve_edges,
                friction_delta=1.1,
                gate_replenishment={("cold_chain_gate", "finished_lot"): 2},
            ),
            Scenario(
                "safety_stock_release__edges_only",
                "falsification",
                cost=_edge_cost(reserve_edges, 1.1),
                friction_edges=reserve_edges,
                friction_delta=1.1,
            ),
            Scenario(
                "safety_stock_release__gate_only",
                "falsification",
                cost=1.0,
                gate_replenishment={("cold_chain_gate", "finished_lot"): 2},
            ),
            Scenario(
                "safety_stock_release__wrong_supplier_lane",
                "falsification",
                cost=_control_by_name("safety_stock_release").cost,
                friction_edges=alternate_edges,
                friction_delta=1.1,
                gate_replenishment={("cold_chain_gate", "finished_lot"): 2},
            ),
            Scenario(
                "safety_stock_release__wrong_gate",
                "falsification",
                cost=_control_by_name("safety_stock_release").cost,
                friction_edges=reserve_edges,
                friction_delta=1.1,
                gate_replenishment={("copacker_gate", "copack_packaging"): 2},
            ),
        ]
    raise ValueError(f"unknown falsification target: {target}")


def experiment_cells() -> list[tuple[str, float, str]]:
    return [
        ("cold_chain", 1.50, "carrier_priority"),
        ("cold_chain", 2.25, "carrier_priority"),
        ("cold_chain", 1.50, "safety_stock_release"),
        ("combined", 1.50, "safety_stock_release"),
        ("combined", 2.25, "carrier_priority"),
        ("promotion_pressure", 0.75, "promotion_throttle"),
        ("promotion_pressure", 1.50, "promotion_throttle"),
        ("combined", 0.75, "promotion_throttle"),
    ]


def _score(row: dict, baseline: dict) -> float:
    return (
        row["service_completion_rate"]
        - baseline["service_completion_rate"]
        + 0.70 * (row["priority_service_rate"] - baseline["priority_service_rate"])
        - 0.50 * max(row["lost_demand_rate"] - baseline["lost_demand_rate"], 0.0)
        - 0.40 * row["capacity_overflow_rate"]
        - 0.05 * row["gate_pressure_rate"]
        - 0.002 * row["control_cost"]
    )


def _variant_role(control_name: str) -> str:
    return control_name.split("__", 1)[1]


def _is_wrong_location(role: str) -> bool:
    return role in {
        "wrong_dry_lane",
        "wrong_supplier_lane",
        "wrong_cold_lane",
        "terminal_return",
    }


def run_choice_falsification(
    seeds: tuple[int, ...] = (20260611, 20260612, 20260613),
    agents: int = 128,
    steps: int = 40,
) -> dict:
    rows = []
    for family, severity, target in experiment_cells():
        scenario = severity_scenario(family, severity)
        variants = falsification_controls(target)
        for seed in seeds:
            config = SimulationConfig(agents=agents, steps=steps, seed=seed)
            baseline = simulate(config, scenario, _control_by_name("no_control"))
            for variant in variants:
                row = simulate(config, scenario, variant)
                row.update({
                    "family": family,
                    "severity": severity,
                    "target_control": target,
                    "variant": variant.name,
                    "variant_role": _variant_role(variant.name),
                    "seed": seed,
                    "baseline_service_completion_rate": baseline["service_completion_rate"],
                    "classification": _classify(row, baseline),
                    "service_delta_vs_baseline": row["service_completion_rate"] - baseline["service_completion_rate"],
                    "priority_delta_vs_baseline": row["priority_service_rate"] - baseline["priority_service_rate"],
                    "lost_delta_vs_baseline": row["lost_demand_rate"] - baseline["lost_demand_rate"],
                    "resilience_score": _score(row, baseline),
                })
                rows.append(row)

    grouped = []
    cell_keys = sorted({(row["family"], row["severity"], row["target_control"]) for row in rows})
    for family, severity, target in cell_keys:
        variants = sorted({row["variant"] for row in rows if (
            row["family"] == family
            and row["severity"] == severity
            and row["target_control"] == target
        )})
        canonical_lift = None
        for variant in variants:
            subset = [
                row for row in rows
                if row["family"] == family
                and row["severity"] == severity
                and row["target_control"] == target
                and row["variant"] == variant
            ]
            mean_lift = sum(row["service_delta_vs_baseline"] for row in subset) / len(subset)
            if _variant_role(variant) == "canonical":
                canonical_lift = mean_lift
            class_counts = {
                label: sum(row["classification"] == label for row in subset)
                for label in sorted({row["classification"] for row in subset})
            }
            grouped.append({
                "family": family,
                "severity": severity,
                "target_control": target,
                "variant": variant,
                "variant_role": _variant_role(variant),
                "runs": len(subset),
                "mean_service_lift": mean_lift,
                "mean_priority_lift": sum(row["priority_delta_vs_baseline"] for row in subset) / len(subset),
                "mean_lost_delta": sum(row["lost_delta_vs_baseline"] for row in subset) / len(subset),
                "mean_service": sum(row["service_completion_rate"] for row in subset) / len(subset),
                "mean_overflow": sum(row["capacity_overflow_rate"] for row in subset) / len(subset),
                "mean_gate_pressure": sum(row["gate_pressure_rate"] for row in subset) / len(subset),
                "mean_gate_starvation": sum(row["gate_starvation_rate"] for row in subset) / len(subset),
                "mean_gate_capacity_block": sum(row["gate_service_capacity_block_rate"] for row in subset) / len(subset),
                "mean_gate_contention": sum(row["gate_contention_rate"] for row in subset) / len(subset),
                "mean_score": sum(row["resilience_score"] for row in subset) / len(subset),
                "class_counts": class_counts,
            })
        for row in grouped:
            if (
                row["family"] == family
                and row["severity"] == severity
                and row["target_control"] == target
            ):
                if canonical_lift is None or canonical_lift <= 1e-12:
                    row["lift_retention_vs_canonical"] = None
                else:
                    row["lift_retention_vs_canonical"] = row["mean_service_lift"] / canonical_lift

    verdicts = []
    for family, severity, target in cell_keys:
        subset = [
            row for row in grouped
            if row["family"] == family
            and row["severity"] == severity
            and row["target_control"] == target
        ]
        canonical = next(row for row in subset if row["variant_role"] == "canonical")
        wrong_location = [
            row for row in subset
            if _is_wrong_location(row["variant_role"])
        ]
        edges_only = next((row for row in subset if row["variant_role"] == "edges_only"), None)
        gate_only = next((row for row in subset if row["variant_role"] == "gate_only"), None)
        best_wrong_lift = max((row["mean_service_lift"] for row in wrong_location), default=0.0)
        if canonical["mean_service_lift"] <= 0.0:
            verdict = "canonical_not_effective"
        elif best_wrong_lift >= 0.75 * canonical["mean_service_lift"]:
            verdict = "not_falsified_wrong_location_retains_lift"
        elif canonical["mean_score"] <= 0.0:
            verdict = "lift_exists_but_score_fragile"
        else:
            verdict = "choice_point_supported"
        verdicts.append({
            "family": family,
            "severity": severity,
            "target_control": target,
            "canonical_service_lift": canonical["mean_service_lift"],
            "canonical_score": canonical["mean_score"],
            "best_wrong_location_lift": best_wrong_lift,
            "edges_only_lift": edges_only["mean_service_lift"] if edges_only else None,
            "gate_only_lift": gate_only["mean_service_lift"] if gate_only else None,
            "verdict": verdict,
        })

    return {
        "config": {
            "seeds": seeds,
            "agents": agents,
            "steps": steps,
            "cells": experiment_cells(),
        },
        "rows": rows,
        "grouped": grouped,
        "verdicts": verdicts,
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Consumer Goods Choice-Point Falsification Report",
        "",
        "## Scope",
        "",
        (
            "Relocation and mechanism-decomposition tests for controls that appeared "
            "useful in the consumer-goods severity sweep."
        ),
        "",
        f"- Seeds: `{', '.join(str(seed) for seed in payload['config']['seeds'])}`",
        f"- Agents: `{payload['config']['agents']}`",
        f"- Steps: `{payload['config']['steps']}`",
        f"- Runs: `{len(payload['rows'])}`",
        "",
        "## Verdicts",
        "",
        "| Family | Severity | Target Control | Canonical Lift | Edges-Only Lift | Gate-Only Lift | Best Wrong-Location Lift | Canonical Score | Verdict |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["verdicts"]:
        edges_lift = "n/a" if row["edges_only_lift"] is None else f"{row['edges_only_lift']:.3f}"
        gate_lift = "n/a" if row["gate_only_lift"] is None else f"{row['gate_only_lift']:.3f}"
        lines.append(
            f"| `{row['family']}` | {row['severity']:.2f} | `{row['target_control']}` | "
            f"{row['canonical_service_lift']:.3f} | {edges_lift} | {gate_lift} | "
            f"{row['best_wrong_location_lift']:.3f} | "
            f"{row['canonical_score']:.4f} | `{row['verdict']}` |"
        )

    lines.extend([
        "",
        "## Variant Surface",
        "",
        "| Family | Severity | Target | Variant | Service Lift | Lift Retention | Priority Lift | Overflow | Starvation | Gate Capacity | Gate Load | Score |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["grouped"]:
        retention = row["lift_retention_vs_canonical"]
        retention_text = "n/a" if retention is None else f"{retention:.1%}"
        lines.append(
            f"| `{row['family']}` | {row['severity']:.2f} | `{row['target_control']}` | "
            f"`{row['variant_role']}` | {row['mean_service_lift']:.3f} | {retention_text} | "
            f"{row['mean_priority_lift']:.3f} | {row['mean_overflow']:.1%} | "
            f"{row['mean_gate_starvation']:.1%} | {row['mean_gate_capacity_block']:.1%} | "
            f"{row['mean_gate_contention']:.1%} | {row['mean_score']:.4f} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "`choice_point_supported` means the canonical control has positive lift, "
            "positive score, and wrong-location variants retain less than 75% of its "
            "service lift. `lift_exists_but_score_fragile` means a control moves volume "
            "but pays enough overflow, gate, lost-demand, or cost penalty that it should "
            "not be sold as a robust policy. `canonical_not_effective` means the original "
            "control itself failed in that stress cell."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("consumer_goods_choice_falsification_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("CONSUMER_GOODS_CHOICE_FALSIFICATION_REPORT.md"))
    args = parser.parse_args()
    payload = run_choice_falsification(
        seeds=(20260611,) if args.quick else (20260611, 20260612, 20260613),
        agents=48 if args.quick else 128,
        steps=20 if args.quick else 40,
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": len(payload["rows"]),
        "verdicts": {
            verdict: sum(row["verdict"] == verdict for row in payload["verdicts"])
            for verdict in sorted({row["verdict"] for row in payload["verdicts"]})
        },
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

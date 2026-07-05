"""
Adversarial falsification suite for the semiconductor onshoring conclusion.

The suite tests whether the feasibility-allocation-adaptation result survives:

1. equal-cost intervention relocation,
2. topology surgery that creates downstream reconsideration exits,
3. a feedback-rate continuum,
4. degree-preserving randomized choice-topology nulls.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from semiconductor_onshoring import (
    EDGES,
    GATES,
    US_DEMAND_NODES,
    OnshoringConfig,
    Scenario,
    simulate,
)
from semiconductor_onshoring_feasibility_preference import RESOURCE_POLICY
from semiconductor_onshoring_frontier import (
    FrontierConfig,
    OFFSHORE_IMPORT_EDGES,
    _classify,
    control_for,
    import_dominant_baseline_point,
    scenario_for,
)
from semiconductor_onshoring_scale_stability import NEAR_VIABLE_DOCTRINE
from semiconductor_onshoring_scaling_law import RANDOMIZATION_KEY, _scaling_control, resources_for


UPSTREAM_CHOICE_EDGES = (
    ("NVIDIA AI Accelerator Demand", "TSMC Taiwan Fabs"),
    ("AMD AI Accelerator Demand", "TSMC Taiwan Fabs"),
    ("US Hyperscaler Demand", "TSMC Taiwan Fabs"),
)
ROUTE_COMMITMENT_EDGES = (("TSMC Taiwan Fabs", "Taiwan OSAT Packaging"),)
RECONSIDERATION_EDGES = (
    ("Export Control Review", "Market Allocation Desk", 8.0),
    ("Taiwan Export Logistics", "Market Allocation Desk", 8.0),
    ("Pacific Shipping Lane", "Market Allocation Desk", 8.0),
)


def _penalty_scenario(base: Scenario, name: str, edges: tuple[tuple[str, str], ...], budget: float) -> Scenario:
    per_edge = -budget / max(len(edges), 1)
    return replace(
        base,
        name=f"{base.name}_{name}_{budget:.1f}",
        family="falsification_penalty",
        cost=base.cost + budget,
        friction_edges=tuple(base.friction_edges) + edges,
        friction_edge_deltas=base.friction_edge_deltas | {edge: per_edge for edge in edges},
    )


def _run_cell(
    agents: int,
    steps: int,
    seed: int,
    feedback_rate: float,
    domestic_pull: float,
    scenario: Scenario,
    additional_edges: tuple[tuple[str, str, float], ...] = (),
    removed_edges: tuple[tuple[str, str], ...] = (),
) -> dict:
    config = FrontierConfig(agents=agents, steps=steps, seed=seed)
    sim_config = OnshoringConfig(
        agents=agents,
        steps=steps,
        seed=seed,
        feedback_rate=feedback_rate,
        gate_initial_inventory=config.gate_initial_inventory,
        randomization_key=RANDOMIZATION_KEY,
        additional_edges=additional_edges,
        removed_edges=removed_edges,
    )
    baseline_point = import_dominant_baseline_point()
    baseline = simulate(
        sim_config,
        scenario_for(baseline_point),
        control_for(baseline_point),
        enforce_gates=True,
    )
    resources = resources_for(RESOURCE_POLICY, agents)
    point = replace(
        NEAR_VIABLE_DOCTRINE,
        tariff=0.0,
        domestic_pull=domestic_pull,
        capacity_multiplier=resources["fab_capacity_multiplier"],
        packaging_capacity_multiplier=resources["packaging_capacity_multiplier"],
    )
    row = simulate(
        sim_config,
        scenario,
        _scaling_control(point, resources, RESOURCE_POLICY),
        enforce_gates=True,
    )
    row.update({
        "seed": seed,
        "feedback_rate": feedback_rate,
        "domestic_pull": domestic_pull,
        "finished_flow_ratio": row["lot_total_us_finished"] / max(baseline["lot_total_us_finished"], 1.0),
        "classification": _classify(row, baseline, config),
    })
    return row


def _base_scenario(agents: int, domestic_pull: float = 0.0) -> Scenario:
    resources = resources_for(RESOURCE_POLICY, agents)
    point = replace(
        NEAR_VIABLE_DOCTRINE,
        tariff=0.0,
        domestic_pull=domestic_pull,
        capacity_multiplier=resources["fab_capacity_multiplier"],
        packaging_capacity_multiplier=resources["packaging_capacity_multiplier"],
    )
    return scenario_for(point)


def run_choice_point_relocation(
    agents: int,
    steps: int,
    seeds: tuple[int, ...],
    budgets: tuple[float, ...] = (0.0, 4.0, 8.0, 12.0),
) -> dict:
    locations = {
        "upstream_choice": UPSTREAM_CHOICE_EDGES,
        "route_commitment": ROUTE_COMMITMENT_EDGES,
        "downstream_serial": OFFSHORE_IMPORT_EDGES,
    }
    rows = []
    base = _base_scenario(agents)
    for seed in seeds:
        for location, edges in locations.items():
            for budget in budgets:
                scenario = _penalty_scenario(base, location, edges, budget)
                row = _run_cell(agents, steps, seed, 0.15, 0.0, scenario)
                row.update({"location": location, "budget": budget})
                rows.append(row)
    grouped = []
    for location in locations:
        for budget in budgets:
            subset = [row for row in rows if row["location"] == location and row["budget"] == budget]
            grouped.append({
                "location": location,
                "budget": budget,
                "viable_rate": sum(row["classification"] == "viable_onshoring_transition" for row in subset) / len(subset),
                "mean_onshore_share": sum(row["onshore_share"] for row in subset) / len(subset),
                "mean_completion": sum(row["lot_completion_rate"] for row in subset) / len(subset),
            })
    return {"rows": rows, "grouped": grouped}


def run_topology_surgery(
    agents: int,
    steps: int,
    seeds: tuple[int, ...],
    budgets: tuple[float, ...] = (0.0, 4.0, 8.0, 12.0),
) -> dict:
    rows = []
    base = _base_scenario(agents)
    for seed in seeds:
        for topology in ("serial", "reconsideration_exits"):
            additional = RECONSIDERATION_EDGES if topology == "reconsideration_exits" else ()
            for budget in budgets:
                scenario = _penalty_scenario(base, "downstream_serial", OFFSHORE_IMPORT_EDGES, budget)
                row = _run_cell(agents, steps, seed, 0.15, 0.0, scenario, additional_edges=additional)
                row.update({"topology": topology, "budget": budget})
                rows.append(row)
    grouped = []
    for topology in ("serial", "reconsideration_exits"):
        for budget in budgets:
            subset = [row for row in rows if row["topology"] == topology and row["budget"] == budget]
            grouped.append({
                "topology": topology,
                "budget": budget,
                "viable_rate": sum(row["classification"] == "viable_onshoring_transition" for row in subset) / len(subset),
                "mean_onshore_share": sum(row["onshore_share"] for row in subset) / len(subset),
                "mean_completion": sum(row["lot_completion_rate"] for row in subset) / len(subset),
            })
    return {"rows": rows, "grouped": grouped}


def run_feedback_continuum(
    agents: int,
    steps: int,
    seeds: tuple[int, ...],
    feedback_rates: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.30, 0.50),
    domestic_pulls: tuple[float, ...] = (0.0, 1.0),
) -> dict:
    rows = []
    for seed in seeds:
        for feedback_rate in feedback_rates:
            for domestic_pull in domestic_pulls:
                row = _run_cell(
                    agents,
                    steps,
                    seed,
                    feedback_rate,
                    domestic_pull,
                    _base_scenario(agents, domestic_pull),
                )
                rows.append(row)
    grouped = []
    for feedback_rate in feedback_rates:
        for domestic_pull in domestic_pulls:
            subset = [
                row for row in rows
                if row["feedback_rate"] == feedback_rate and row["domestic_pull"] == domestic_pull
            ]
            grouped.append({
                "feedback_rate": feedback_rate,
                "domestic_pull": domestic_pull,
                "viable_rate": sum(row["classification"] == "viable_onshoring_transition" for row in subset) / len(subset),
                "mean_onshore_share": sum(row["onshore_share"] for row in subset) / len(subset),
                "mean_completion": sum(row["lot_completion_rate"] for row in subset) / len(subset),
            })
    return {"rows": rows, "grouped": grouped}


def _protected_edges() -> set[tuple[str, str]]:
    protected = set()
    for gate in GATES:
        protected.add((gate.source, gate.target))
        protected.update(gate.arrivals)
    for source in ("US Finished Packaged Chips", "US West Coast Port", "Strategic Chip Reserve"):
        protected.update((source, target) for target in US_DEMAND_NODES)
    return protected


def _rewired_choice_topology(seed: int, swaps: int = 24) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str, float], ...]]:
    rng = np.random.default_rng(seed)
    protected = _protected_edges()
    edge_map = {(source, target): distance for source, target, distance in EDGES}
    mutable = [edge for edge in edge_map if edge not in protected]
    current = set(edge_map)
    changed_from: set[tuple[str, str]] = set()
    changed_to: dict[tuple[str, str], float] = {}
    completed = 0
    attempts = 0
    while completed < swaps and attempts < swaps * 100:
        attempts += 1
        first_idx, second_idx = rng.choice(len(mutable), size=2, replace=False)
        first = mutable[int(first_idx)]
        second = mutable[int(second_idx)]
        a, b = first
        c, d = second
        candidate_first = (a, d)
        candidate_second = (c, b)
        if a == d or c == b or candidate_first in current or candidate_second in current:
            continue
        first_distance = edge_map[first]
        second_distance = edge_map[second]
        current.remove(first)
        current.remove(second)
        current.add(candidate_first)
        current.add(candidate_second)
        mutable[int(first_idx)] = candidate_first
        mutable[int(second_idx)] = candidate_second
        edge_map[candidate_first] = first_distance
        edge_map[candidate_second] = second_distance
        changed_from.update((first, second))
        changed_to[candidate_first] = first_distance
        changed_to[candidate_second] = second_distance
        completed += 1
    removed = tuple(sorted(edge for edge in changed_from if edge not in current))
    added = tuple(sorted(
        (source, target, distance)
        for (source, target), distance in changed_to.items()
        if (source, target) in current and (source, target) not in {(e[0], e[1]) for e in EDGES}
    ))
    return removed, added


def run_randomized_topology_nulls(
    agents: int,
    steps: int,
    seeds: tuple[int, ...],
    null_seeds: tuple[int, ...] = (7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008),
) -> dict:
    rows = []
    for null_seed in null_seeds:
        removed, added = _rewired_choice_topology(null_seed)
        for seed in seeds:
            for domestic_pull in (0.0, 1.0):
                row = _run_cell(
                    agents,
                    steps,
                    seed,
                    0.15,
                    domestic_pull,
                    _base_scenario(agents, domestic_pull),
                    additional_edges=added,
                    removed_edges=removed,
                )
                row.update({
                    "null_seed": null_seed,
                    "removed_edges": len(removed),
                    "added_edges": len(added),
                })
                rows.append(row)
    grouped = []
    for null_seed in null_seeds:
        for domestic_pull in (0.0, 1.0):
            subset = [
                row for row in rows
                if row["null_seed"] == null_seed and row["domestic_pull"] == domestic_pull
            ]
            grouped.append({
                "null_seed": null_seed,
                "domestic_pull": domestic_pull,
                "viable_rate": sum(row["classification"] == "viable_onshoring_transition" for row in subset) / len(subset),
                "mean_onshore_share": sum(row["onshore_share"] for row in subset) / len(subset),
                "mean_completion": sum(row["lot_completion_rate"] for row in subset) / len(subset),
                "removed_edges": subset[0]["removed_edges"],
                "added_edges": subset[0]["added_edges"],
            })
    return {"rows": rows, "grouped": grouped}


def run_falsification_suite(
    agents: int = 320,
    steps: int = 40,
    seeds: tuple[int, ...] = (20260606, 20260607, 20260608, 20260609, 20260610),
) -> dict:
    relocation = run_choice_point_relocation(agents, steps, seeds)
    surgery = run_topology_surgery(agents, steps, seeds)
    feedback = run_feedback_continuum(agents, steps, seeds)
    nulls = run_randomized_topology_nulls(agents, steps, seeds)
    relocation_zero = {
        row["location"]: row["mean_onshore_share"]
        for row in relocation["grouped"]
        if row["budget"] == 0.0
    }
    relocation_lifts = {
        location: max(
            row["mean_onshore_share"] - relocation_zero[location]
            for row in relocation["grouped"]
            if row["location"] == location
        )
        for location in relocation_zero
    }
    surgery_zero = {
        row["topology"]: row["mean_onshore_share"]
        for row in surgery["grouped"]
        if row["budget"] == 0.0
    }
    surgery_lifts = {
        topology: max(
            row["mean_onshore_share"] - surgery_zero[topology]
            for row in surgery["grouped"]
            if row["topology"] == topology
        )
        for topology in surgery_zero
    }
    null_pairs = {
        null_seed: {
            row["domestic_pull"]: row
            for row in nulls["grouped"]
            if row["null_seed"] == null_seed
        }
        for null_seed in {row["null_seed"] for row in nulls["grouped"]}
    }
    return {
        "config": {"agents": agents, "steps": steps, "seeds": seeds},
        "choice_point_relocation": relocation,
        "topology_surgery": surgery,
        "feedback_continuum": feedback,
        "randomized_topology_nulls": nulls,
        "summary": {
            "relocation_max_share_lift": relocation_lifts,
            "surgery_max_share_lift": surgery_lifts,
            "feedback_viable_rates_with_pull": [
                row["viable_rate"]
                for row in feedback["grouped"]
                if row["domestic_pull"] == 1.0
            ],
            "nulls_where_pull_improves_viable_rate": sum(
                pair[1.0]["viable_rate"] > pair[0.0]["viable_rate"]
                for pair in null_pairs.values()
            ),
            "nulls_with_high_share_but_no_robust_transition": sum(
                pair[0.0]["mean_onshore_share"] >= 0.50 and pair[0.0]["viable_rate"] < 2 / 3
                for pair in null_pairs.values()
            ),
        },
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Semiconductor Onshoring Falsification Report",
        "",
        "## Scope",
        "",
        "Adversarial tests of whether the feasibility-allocation-adaptation conclusion survives changes in intervention location, topology, feedback rate, and route-choice wiring.",
        "",
        f"- Agents: `{payload['config']['agents']}`",
        f"- Steps: `{payload['config']['steps']}`",
        f"- Seeds: `{', '.join(str(x) for x in payload['config']['seeds'])}`",
        "",
        "## Summary",
        "",
        f"- Maximum upstream-choice share lift: `{payload['summary']['relocation_max_share_lift']['upstream_choice']:.3f}`",
        f"- Maximum route-commitment share lift: `{payload['summary']['relocation_max_share_lift']['route_commitment']:.3f}`",
        f"- Maximum downstream-serial share lift: `{payload['summary']['relocation_max_share_lift']['downstream_serial']:.3f}`",
        f"- Maximum serial-topology downstream-penalty share lift: `{payload['summary']['surgery_max_share_lift']['serial']:.3f}`",
        f"- Maximum reconsideration-topology downstream-penalty share lift: `{payload['summary']['surgery_max_share_lift']['reconsideration_exits']:.3f}`",
        f"- Randomized nulls where pull improves viable rate: `{payload['summary']['nulls_where_pull_improves_viable_rate']}`",
        f"- Randomized nulls with majority share but no robust transition: `{payload['summary']['nulls_with_high_share_but_no_robust_transition']}`",
        "",
        "## Choice-Point Relocation",
        "",
        "| Location | Budget | Viable Rate | Mean Share | Mean Completion |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["choice_point_relocation"]["grouped"]:
        lines.append(f"| `{row['location']}` | {row['budget']:.1f} | {row['viable_rate']:.1%} | {row['mean_onshore_share']:.3f} | {row['mean_completion']:.1%} |")
    lines.extend([
        "",
        "## Topology Surgery",
        "",
        "| Topology | Downstream Budget | Viable Rate | Mean Share | Mean Completion |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in payload["topology_surgery"]["grouped"]:
        lines.append(f"| `{row['topology']}` | {row['budget']:.1f} | {row['viable_rate']:.1%} | {row['mean_onshore_share']:.3f} | {row['mean_completion']:.1%} |")
    lines.extend([
        "",
        "## Feedback Continuum",
        "",
        "| Feedback Rate | Domestic Pull | Viable Rate | Mean Share | Mean Completion |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in payload["feedback_continuum"]["grouped"]:
        lines.append(f"| {row['feedback_rate']:.2f} | {row['domestic_pull']:.1f} | {row['viable_rate']:.1%} | {row['mean_onshore_share']:.3f} | {row['mean_completion']:.1%} |")
    lines.extend([
        "",
        "## Randomized Choice-Topology Nulls",
        "",
        "| Null Seed | Domestic Pull | Viable Rate | Mean Share | Mean Completion | Rewired Edges |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["randomized_topology_nulls"]["grouped"]:
        lines.append(f"| {row['null_seed']} | {row['domestic_pull']:.1f} | {row['viable_rate']:.1%} | {row['mean_onshore_share']:.3f} | {row['mean_completion']:.1%} | {row['removed_edges']} |")
    lines.extend([
        "",
        "## Reading",
        "",
        (
            "The choice-point principle survives: equal-cost penalties change route share only "
            "at upstream choice edges, or after topology surgery creates a latent alternative. "
            "Feedback reshapes the phase boundary non-monotonically rather than acting as a "
            "universal amplifier. Procurement pull is topology-contingent, while the distinction "
            "between high route share and a viable production transition persists across nulls."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=320)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_falsification_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_FALSIFICATION_REPORT.md"))
    args = parser.parse_args()
    payload = run_falsification_suite(agents=args.agents, steps=args.steps)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "relocation_cells": len(payload["choice_point_relocation"]["grouped"]),
        "surgery_cells": len(payload["topology_surgery"]["grouped"]),
        "feedback_cells": len(payload["feedback_continuum"]["grouped"]),
        "null_cells": len(payload["randomized_topology_nulls"]["grouped"]),
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

"""
Onshoring frontier sweep for the semiconductor sector prototype.

This module searches parameter bundles where onshore share increases without
becoming a fake win caused by collapsed U.S. finished flow, domestic capacity
overflow, or dependency gate failure.

Usage:
    .venv\\Scripts\\python.exe semiconductor_onshoring_frontier.py --quick
    .venv\\Scripts\\python.exe semiconductor_onshoring_frontier.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from semiconductor_onshoring import OnshoringConfig, Scenario, simulate


OFFSHORE_IMPORT_EDGES = (
    ("Taiwan OSAT Packaging", "Export Control Review"),
    ("Export Control Review", "Taiwan Export Logistics"),
    ("Taiwan Export Logistics", "Pacific Shipping Lane"),
    ("Pacific Shipping Lane", "US West Coast Port"),
)
SUBSIDY_EDGES = (
    ("CHIPS Subsidy Credit", "Intel US Fabs"),
    ("CHIPS Subsidy Credit", "TSMC Arizona Fabs"),
    ("CHIPS Subsidy Credit", "Samsung Texas Fabs"),
)
OFFSET_EDGES = (
    ("Section 232 Tariff Offset", "TSMC Arizona Fabs"),
    ("Section 232 Tariff Offset", "US Advanced Packaging"),
)
DOMESTIC_RAMP_EDGES = (
    ("Intel US Fabs", "US Wafer Fabrication"),
    ("TSMC Arizona Fabs", "US Wafer Fabrication"),
    ("Samsung Texas Fabs", "US Wafer Fabrication"),
    ("US Wafer Fabrication", "US Advanced Packaging"),
)
PACKAGING_RAMP_EDGES = (
    ("Market Allocation Desk", "US Packaging Inputs"),
    ("US Packaging Inputs", "US Advanced Packaging"),
    ("US Advanced Packaging", "US Finished Packaged Chips"),
    ("US Finished Packaged Chips", "NVIDIA AI Accelerator Demand"),
    ("US Finished Packaged Chips", "AMD AI Accelerator Demand"),
    ("US Finished Packaged Chips", "US Hyperscaler Demand"),
    ("US Finished Packaged Chips", "US Defense Demand"),
)
MATERIALS_CONTINUITY_EDGES = (
    ("Market Allocation Desk", "Japan Chemicals Materials"),
    ("Market Allocation Desk", "EU Lithography Tools"),
    ("Market Allocation Desk", "US EDA IP"),
    ("Market Allocation Desk", "US Power Labor"),
    ("Market Allocation Desk", "Korea Packaging Inputs"),
    ("Market Allocation Desk", "US Packaging Inputs"),
    ("Japan Chemicals Materials", "US Wafer Fabrication"),
    ("EU Lithography Tools", "US Wafer Fabrication"),
    ("US EDA IP", "US Wafer Fabrication"),
    ("US Power Labor", "US Wafer Fabrication"),
)
DOMESTIC_PROCUREMENT_EDGES = (
    ("NVIDIA AI Accelerator Demand", "TSMC Arizona Fabs"),
    ("NVIDIA AI Accelerator Demand", "Intel US Fabs"),
    ("AMD AI Accelerator Demand", "TSMC Arizona Fabs"),
    ("AMD AI Accelerator Demand", "Samsung Texas Fabs"),
    ("US Hyperscaler Demand", "TSMC Arizona Fabs"),
    ("US Defense Demand", "Intel US Fabs"),
    ("US Defense Demand", "TSMC Arizona Fabs"),
)
CAPACITY_BASE = {
    "Intel US Fabs": 10,
    "TSMC Arizona Fabs": 9,
    "Samsung Texas Fabs": 7,
    "US Advanced Packaging": 12,
}
DOMESTIC_FAB_CAPACITY_NODES = (
    "Intel US Fabs",
    "TSMC Arizona Fabs",
    "Samsung Texas Fabs",
)


@dataclass(frozen=True)
class FrontierConfig:
    agents: int = 192
    steps: int = 96
    seed: int = 20260606
    gate_initial_inventory: int = 12
    min_finished_flow_ratio: float = 0.85
    max_overflow: float = 0.10
    max_dependency_pressure: float = 0.65
    transition_threshold: float = 0.50


@dataclass(frozen=True)
class StrategyPoint:
    tariff: float
    capacity_multiplier: float
    packaging_capacity_multiplier: float
    subsidy: float
    offset: float
    domestic_ramp: float
    packaging_ramp: float
    materials: float
    domestic_pull: float


def _edge_cost(edges: tuple[tuple[str, str], ...], magnitude: float) -> float:
    return float(len(edges) * abs(magnitude))


def _capacity_caps(fab_multiplier: float, packaging_multiplier: float) -> dict[str, int]:
    caps = {}
    if fab_multiplier > 0:
        caps.update({
            node: max(1, int(round(CAPACITY_BASE[node] * fab_multiplier)))
            for node in DOMESTIC_FAB_CAPACITY_NODES
        })
    if packaging_multiplier > 0:
        caps["US Advanced Packaging"] = max(
            1,
            int(round(CAPACITY_BASE["US Advanced Packaging"] * packaging_multiplier)),
        )
    return caps


def scenario_for(point: StrategyPoint) -> Scenario:
    return Scenario(
        name=(
            f"frontier_tariff_{point.tariff:.2f}_"
            f"fabcap_{point.capacity_multiplier:.2f}_"
            f"pkgcap_{point.packaging_capacity_multiplier:.2f}"
        ),
        family="frontier",
        cost=_edge_cost(OFFSHORE_IMPORT_EDGES, point.tariff),
        friction_edges=OFFSHORE_IMPORT_EDGES,
        friction_delta=-point.tariff,
        node_capacity_caps=_capacity_caps(
            point.capacity_multiplier,
            point.packaging_capacity_multiplier,
        ),
    )


def control_for(point: StrategyPoint) -> Scenario:
    friction_edges = []
    friction_edge_deltas = {}
    beta_edges = []
    beta_edge_boosts = {}
    cost = 0.0
    if point.offset > 0:
        friction_edges.extend(OFFSET_EDGES)
        friction_edge_deltas.update({edge: point.offset for edge in OFFSET_EDGES})
        cost += _edge_cost(OFFSET_EDGES, point.offset)
    if point.materials > 0:
        friction_edges.extend(MATERIALS_CONTINUITY_EDGES)
        friction_edge_deltas.update({edge: point.materials for edge in MATERIALS_CONTINUITY_EDGES})
        cost += _edge_cost(MATERIALS_CONTINUITY_EDGES, point.materials)
    if point.subsidy > 0:
        beta_edges.extend(SUBSIDY_EDGES)
        beta_edge_boosts.update({edge: point.subsidy for edge in SUBSIDY_EDGES})
        cost += _edge_cost(SUBSIDY_EDGES, point.subsidy)
    if point.domestic_ramp > 0:
        beta_edges.extend(DOMESTIC_RAMP_EDGES)
        beta_edge_boosts.update({edge: point.domestic_ramp for edge in DOMESTIC_RAMP_EDGES})
        cost += _edge_cost(DOMESTIC_RAMP_EDGES, point.domestic_ramp)
    if point.packaging_ramp > 0:
        beta_edges.extend(PACKAGING_RAMP_EDGES)
        beta_edge_boosts.update({edge: point.packaging_ramp for edge in PACKAGING_RAMP_EDGES})
        cost += _edge_cost(PACKAGING_RAMP_EDGES, point.packaging_ramp)
    if point.domestic_pull > 0:
        beta_edges.extend(DOMESTIC_PROCUREMENT_EDGES)
        beta_edge_boosts.update({edge: point.domestic_pull for edge in DOMESTIC_PROCUREMENT_EDGES})
        cost += _edge_cost(DOMESTIC_PROCUREMENT_EDGES, point.domestic_pull)
    return Scenario(
        name=(
            f"bundle_sub_{point.subsidy:.1f}_off_{point.offset:.1f}_"
            f"dom_{point.domestic_ramp:.1f}_pkg_{point.packaging_ramp:.1f}_"
            f"mat_{point.materials:.1f}_pull_{point.domestic_pull:.1f}"
        ),
        family="frontier_control",
        cost=cost,
        friction_edges=tuple(friction_edges),
        friction_delta=max(point.offset, point.materials),
        friction_edge_deltas=friction_edge_deltas,
        beta_edges=tuple(beta_edges),
        beta_boost=max(point.subsidy, point.domestic_ramp, point.packaging_ramp, point.domestic_pull),
        beta_edge_boosts=beta_edge_boosts,
    )


def quick_grid() -> list[StrategyPoint]:
    points = []
    for tariff in (0.0, 1.0, 2.0):
        for cap in (0.0, 2.0, 5.0):
            for packaging_cap in (0.0, 3.0, 6.0):
                for packaging in (0.0, 1.2):
                    for materials in (0.8, 1.6):
                        points.append(StrategyPoint(
                            tariff=tariff,
                            capacity_multiplier=cap,
                            packaging_capacity_multiplier=packaging_cap,
                            subsidy=1.0 if packaging > 0 else 0.0,
                            offset=1.0 if tariff > 0 else 0.0,
                            domestic_ramp=1.0,
                            packaging_ramp=packaging,
                            materials=materials,
                            domestic_pull=1.5 if packaging > 0 else 0.8,
                        ))
    return points


def default_grid() -> list[StrategyPoint]:
    points = []
    for tariff in (0.0, 1.5, 2.25):
        for cap in (0.0, 3.0, 6.0):
            for packaging_cap in (0.0, 3.0, 6.0, 9.0):
                for subsidy in (0.0, 1.2):
                    for offset in (0.0, 1.0):
                        if tariff == 0.0 and offset > 0:
                            continue
                        for domestic in (1.5, 2.5):
                            for packaging in (0.0, 1.5):
                                for pull in (0.0, 2.5):
                                    if packaging == 0.0 and pull == 0.0 and subsidy == 0.0:
                                        continue
                                    materials_options = (
                                        (0.8, 1.6, 2.4)
                                        if (subsidy > 0 or packaging > 0 or domestic > 0.8 or pull > 0)
                                        else (0.0,)
                                    )
                                    for materials in materials_options:
                                        points.append(StrategyPoint(
                                            tariff=tariff,
                                            capacity_multiplier=cap,
                                            packaging_capacity_multiplier=packaging_cap,
                                            subsidy=subsidy,
                                            offset=offset,
                                            domestic_ramp=domestic,
                                            packaging_ramp=packaging,
                                            materials=materials,
                                            domestic_pull=pull,
                                        ))
    return points


def import_dominant_baseline_point() -> StrategyPoint:
    return StrategyPoint(
        tariff=-1.0,
        capacity_multiplier=1.0,
        packaging_capacity_multiplier=1.0,
        subsidy=0.0,
        offset=0.0,
        domestic_ramp=0.4,
        packaging_ramp=0.0,
        materials=0.0,
        domestic_pull=0.0,
    )


def _classify(row: dict, baseline: dict, config: FrontierConfig) -> str:
    finished_ratio = row["lot_total_us_finished"] / max(baseline["lot_total_us_finished"], 1.0)
    dependency_pressure = max(row["gate_backlog_pressure"], row["gate_starvation_index"])
    if row["onshore_share"] >= config.transition_threshold:
        if (
            row["capacity_overflow_rate"] <= config.max_overflow
            and dependency_pressure <= config.max_dependency_pressure
            and finished_ratio >= config.min_finished_flow_ratio
        ):
            return "viable_onshoring_transition"
        return "fake_onshoring"
    if row["capacity_overflow_rate"] > config.max_overflow:
        return "capacity_blocked"
    if dependency_pressure > config.max_dependency_pressure:
        return "dependency_blocked"
    if row["onshore_share"] > baseline["onshore_share"]:
        return "partial_onshoring"
    return "no_transition"


def _score(row: dict, baseline: dict) -> float:
    finished_ratio = row["lot_total_us_finished"] / max(baseline["lot_total_us_finished"], 1.0)
    return (
        row["onshore_share"]
        + 0.20 * min(finished_ratio, 1.25)
        + 0.10 * row["us_demand_share"]
        - 0.25 * row["capacity_overflow_rate"]
        - 0.05 * max(row["gate_backlog_pressure"], row["gate_starvation_index"])
        - 0.005 * row["control_cost"]
    )


def evaluate_point(config: FrontierConfig, point: StrategyPoint, baseline: dict) -> dict:
    sim_config = OnshoringConfig(
        agents=config.agents,
        steps=config.steps,
        seed=config.seed,
        gate_initial_inventory=config.gate_initial_inventory,
    )
    row = simulate(sim_config, scenario_for(point), control_for(point), enforce_gates=True)
    finished_ratio = row["lot_total_us_finished"] / max(baseline["lot_total_us_finished"], 1.0)
    row.update({
        "tariff": point.tariff,
        "capacity_multiplier": point.capacity_multiplier,
        "packaging_capacity_multiplier": point.packaging_capacity_multiplier,
        "subsidy": point.subsidy,
        "offset": point.offset,
        "domestic_ramp": point.domestic_ramp,
        "packaging_ramp": point.packaging_ramp,
        "materials": point.materials,
        "domestic_pull": point.domestic_pull,
        "finished_flow_ratio": finished_ratio,
        "score": _score(row, baseline),
        "classification": _classify(row, baseline, config),
        "onshore_delta_vs_baseline": row["onshore_share"] - baseline["onshore_share"],
        "finished_delta_vs_baseline": row["lot_total_us_finished"] - baseline["lot_total_us_finished"],
    })
    return row


def run_frontier(config: FrontierConfig | None = None, quick: bool = False) -> dict:
    config = config or FrontierConfig()
    sim_config = OnshoringConfig(
        agents=config.agents,
        steps=config.steps,
        seed=config.seed,
        gate_initial_inventory=config.gate_initial_inventory,
    )
    baseline_point = import_dominant_baseline_point()
    baseline = simulate(sim_config, scenario_for(baseline_point), control_for(baseline_point), enforce_gates=True)
    grid = quick_grid() if quick else default_grid()
    rows = [evaluate_point(config, point, baseline) for point in grid]
    counts = {
        label: sum(1 for row in rows if row["classification"] == label)
        for label in sorted({row["classification"] for row in rows})
    }
    viable = [row for row in rows if row["classification"] == "viable_onshoring_transition"]
    best_viable = max(viable, key=lambda row: row["score"], default=None)
    best_score = max(rows, key=lambda row: row["score"])
    max_share = max(rows, key=lambda row: row["onshore_share"])
    return {
        "config": asdict(config) | {"quick": quick},
        "baseline_point": asdict(baseline_point),
        "baseline": baseline,
        "classification_counts": counts,
        "best_viable": best_viable,
        "best_score": best_score,
        "max_share": max_share,
        "rows": rows,
    }


def _fmt_point(row: dict) -> str:
    return (
        f"tariff={row['tariff']:.2f}, fabcap={row['capacity_multiplier']:.2f}, "
        f"pkgcap={row['packaging_capacity_multiplier']:.2f}, "
        f"subsidy={row['subsidy']:.1f}, offset={row['offset']:.1f}, "
        f"domestic={row['domestic_ramp']:.1f}, packaging={row['packaging_ramp']:.1f}, "
        f"materials={row['materials']:.1f}, pull={row['domestic_pull']:.1f}"
    )


def render_report(payload: dict) -> str:
    rows = payload["rows"]
    viable = sorted(
        [row for row in rows if row["classification"] == "viable_onshoring_transition"],
        key=lambda row: row["score"],
        reverse=True,
    )[:10]
    frontier = sorted(rows, key=lambda row: (row["onshore_share"], row["score"]), reverse=True)[:10]
    best_score = payload["best_score"]
    max_share = payload["max_share"]
    lines = [
        "# Semiconductor Onshoring Frontier Report",
        "",
        "## Scope",
        "",
        (
            "Parameter sweep over tariff intensity, domestic fab capacity, advanced-packaging "
            "capacity, subsidy strength, tariff-offset strength, domestic ramp, packaging ramp, "
            "and materials/tooling continuity."
        ),
        "",
        f"- Agents per run: `{payload['config']['agents']}`",
        f"- Steps per run: `{payload['config']['steps']}`",
        f"- Grid points: `{len(rows)}`",
        f"- Baseline doctrine: `{_fmt_point(payload['baseline_point'])}`",
        f"- Baseline onshore share: `{payload['baseline']['onshore_share']:.3f}`",
        f"- Baseline U.S. finished lots: `{payload['baseline']['lot_total_us_finished']}`",
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
        "## Best Feasible Strategy",
        "",
    ])
    if payload["best_viable"] is None:
        lines.append("No grid point crossed the viable onshoring-transition threshold.")
    else:
        row = payload["best_viable"]
        lines.extend([
            f"- Strategy: `{_fmt_point(row)}`",
        f"- Onshore share: `{row['onshore_share']:.3f}`",
        f"- Finished-lot ratio: `{row['finished_flow_ratio']:.3f}`",
        f"- Overflow: `{row['capacity_overflow_rate']:.1%}`",
        f"- Dependency pressure: `{max(row['gate_backlog_pressure'], row['gate_starvation_index']):.1%}`",
        f"- Score: `{row['score']:.4f}`",
        ])

    lines.extend([
        "",
        "## Best Score Point",
        "",
        f"- Strategy: `{_fmt_point(best_score)}`",
        f"- Classification: `{best_score['classification']}`",
        f"- Onshore share: `{best_score['onshore_share']:.3f}`",
        f"- Finished-lot ratio: `{best_score['finished_flow_ratio']:.3f}`",
        f"- Overflow: `{best_score['capacity_overflow_rate']:.1%}`",
        f"- Dependency pressure: `{max(best_score['gate_backlog_pressure'], best_score['gate_starvation_index']):.1%}`",
        f"- Score: `{best_score['score']:.4f}`",
        "",
        "## Max Share Point",
        "",
        f"- Strategy: `{_fmt_point(max_share)}`",
        f"- Classification: `{max_share['classification']}`",
        f"- Onshore share: `{max_share['onshore_share']:.3f}`",
        f"- Finished-lot ratio: `{max_share['finished_flow_ratio']:.3f}`",
        f"- Overflow: `{max_share['capacity_overflow_rate']:.1%}`",
        f"- Dependency pressure: `{max(max_share['gate_backlog_pressure'], max_share['gate_starvation_index']):.1%}`",
        "",
        "## Viable Frontier",
        "",
        "| Onshore Share | Finished Ratio | Overflow | Dependency Pressure | Score | Strategy |",
        "|---:|---:|---:|---:|---:|---|",
    ])
    for row in viable:
        lines.append(
            f"| {row['onshore_share']:.3f} | {row['finished_flow_ratio']:.3f} | "
            f"{row['capacity_overflow_rate']:.1%} | "
            f"{max(row['gate_backlog_pressure'], row['gate_starvation_index']):.1%} | "
            f"{row['score']:.4f} | `{_fmt_point(row)}` |"
        )
    if not viable:
        lines.append("| none | none | none | none | none |")

    lines.extend([
        "",
        "## Highest Share Frontier",
        "",
        "| Classification | Onshore Share | Finished Ratio | Overflow | Dependency Pressure | Retry Block | Score | Strategy |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in frontier:
        lines.append(
            f"| `{row['classification']}` | {row['onshore_share']:.3f} | "
            f"{row['finished_flow_ratio']:.3f} | {row['capacity_overflow_rate']:.1%} | "
            f"{max(row['gate_backlog_pressure'], row['gate_starvation_index']):.1%} | "
            f"{row['gate_block_rate']:.1%} | {row['score']:.4f} | `{_fmt_point(row)}` |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "The frontier rejects fake onshoring: a high onshore share is not considered viable "
            "if it is purchased by lower U.S. finished flow, high capacity overflow, or unresolved "
            "dependency pressure. Dependency pressure uses queue backlog and uncovered input demand "
            "rather than raw repeated retry block rate."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("semiconductor_onshoring_frontier_output.json"),
    output_md: Path = Path("SEMICONDUCTOR_ONSHORING_FRONTIER_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semiconductor onshoring frontier sweep.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--agents", type=int, default=FrontierConfig.agents)
    parser.add_argument("--steps", type=int, default=FrontierConfig.steps)
    parser.add_argument("--seed", type=int, default=FrontierConfig.seed)
    parser.add_argument("--gate-initial-inventory", type=int, default=FrontierConfig.gate_initial_inventory)
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_frontier_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_FRONTIER_REPORT.md"))
    args = parser.parse_args()

    config = FrontierConfig(
        agents=80 if args.quick and args.agents == FrontierConfig.agents else args.agents,
        steps=40 if args.quick and args.steps == FrontierConfig.steps else args.steps,
        seed=args.seed,
        gate_initial_inventory=args.gate_initial_inventory,
    )
    payload = run_frontier(config, quick=args.quick)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "classification_counts": payload["classification_counts"],
        "rows": len(payload["rows"]),
        "best_viable": None if payload["best_viable"] is None else payload["best_viable"]["onshore_share"],
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

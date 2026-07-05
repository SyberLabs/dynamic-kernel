"""
Stock-renewal onshoring phase diagram.

The input-continuity phase diagram treated materials/tooling continuity as
routing/friction support and still found no viable cells. This script tests the
next mechanism: exogenous per-step inventory renewal at the gate parts that
have behaved like binding stock constraints.

Usage:
    .venv\\Scripts\\python.exe semiconductor_onshoring_inventory_phase_diagram.py --quick
    .venv\\Scripts\\python.exe semiconductor_onshoring_inventory_phase_diagram.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
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
from semiconductor_onshoring_phase_diagram import DOCTRINES, PHASE_CODES


REPLENISHED_PARTS = (
    ("us_fab_gate", "materials"),
    ("us_fab_gate", "power_labor"),
    ("us_advanced_packaging_gate", "us_packaging_inputs"),
    ("taiwan_export_gate", "taiwan_packaging_inputs"),
)


def _fmt_point(row: dict) -> str:
    return (
        f"tariff={row['tariff']:.2f}, fabcap={row['capacity_multiplier']:.2f}, "
        f"pkgcap={row['packaging_capacity_multiplier']:.2f}, renewal={row['inventory_renewal']}, "
        f"subsidy={row['subsidy']:.1f}, offset={row['offset']:.1f}, "
        f"domestic={row['domestic_ramp']:.1f}, packaging={row['packaging_ramp']:.1f}, "
        f"materials={row['materials']:.1f}, pull={row['domestic_pull']:.1f}"
    )


def phase_axes(quick: bool = False) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    if quick:
        return (0.0, 3.0, 6.0), (0.0, 6.0, 12.0), (0, 1, 3)
    return (0.0, 3.0, 6.0, 9.0), (0.0, 3.0, 6.0, 9.0, 12.0), (0, 1, 2, 3, 4)


def _renewal_control(point: StrategyPoint, renewal: int) -> Scenario:
    base = control_for(point)
    replenishment = {key: int(renewal) for key in REPLENISHED_PARTS if renewal > 0}
    return Scenario(
        name=f"{base.name}_renew_{renewal}",
        family=base.family,
        cost=base.cost + float(len(replenishment) * renewal),
        friction_edges=base.friction_edges,
        friction_delta=base.friction_delta,
        friction_edge_deltas=base.friction_edge_deltas,
        beta_edges=base.beta_edges,
        beta_boost=base.beta_boost,
        beta_edge_boosts=base.beta_edge_boosts,
        node_capacity_caps=base.node_capacity_caps,
        edge_capacity_caps=base.edge_capacity_caps,
        gate_replenishment=replenishment,
        gate_capacity_caps=base.gate_capacity_caps,
    )


def _evaluate_inventory_point(
    config: FrontierConfig,
    point: StrategyPoint,
    baseline: dict,
    renewal: int,
) -> dict:
    sim_config = OnshoringConfig(
        agents=config.agents,
        steps=config.steps,
        seed=config.seed,
        gate_initial_inventory=config.gate_initial_inventory,
    )
    row = simulate(
        sim_config,
        scenario_for(point),
        _renewal_control(point, renewal),
        enforce_gates=True,
    )
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
        "inventory_renewal": int(renewal),
        "finished_flow_ratio": finished_ratio,
        "score": _score(row, baseline),
        "classification": _classify(row, baseline, config),
        "onshore_delta_vs_baseline": row["onshore_share"] - baseline["onshore_share"],
        "finished_delta_vs_baseline": row["lot_total_us_finished"] - baseline["lot_total_us_finished"],
    })
    return row


def run_inventory_phase_diagram(config: FrontierConfig | None = None, quick: bool = False) -> dict:
    config = config or FrontierConfig(agents=160, steps=80)
    fab_caps, packaging_caps, renewals = phase_axes(quick)
    sim_config = OnshoringConfig(
        agents=config.agents,
        steps=config.steps,
        seed=config.seed,
        gate_initial_inventory=config.gate_initial_inventory,
    )
    baseline_point = import_dominant_baseline_point()
    baseline = simulate(
        sim_config,
        scenario_for(baseline_point),
        control_for(baseline_point),
        enforce_gates=True,
    )
    rows = []
    for doctrine, template in DOCTRINES.items():
        for renewal in renewals:
            for fab_cap in fab_caps:
                for packaging_cap in packaging_caps:
                    point = replace(
                        template,
                        capacity_multiplier=fab_cap,
                        packaging_capacity_multiplier=packaging_cap,
                    )
                    row = _evaluate_inventory_point(config, point, baseline, renewal)
                    row["doctrine"] = doctrine
                    row["phase_code"] = PHASE_CODES[row["classification"]]
                    rows.append(row)
    counts = {
        label: sum(1 for row in rows if row["classification"] == label)
        for label in sorted({row["classification"] for row in rows})
    }
    viable = [row for row in rows if row["classification"] == "viable_onshoring_transition"]
    return {
        "config": asdict(config) | {"quick": quick},
        "baseline_point": asdict(baseline_point),
        "baseline": baseline,
        "fab_caps": fab_caps,
        "packaging_caps": packaging_caps,
        "inventory_renewals": renewals,
        "classification_counts": counts,
        "best_viable": max(viable, key=lambda row: row["score"], default=None),
        "best_score": max(rows, key=lambda row: row["score"]),
        "max_share": max(rows, key=lambda row: row["onshore_share"]),
        "rows": rows,
    }


def _matrix_for(rows: list[dict], fab_caps: tuple[float, ...], packaging_caps: tuple[float, ...]) -> list[str]:
    lines = [
        "| FabCap \\ PkgCap | " + " | ".join(f"{cap:.1f}" for cap in packaging_caps) + " |",
        "|---|" + "|".join("---:" for _ in packaging_caps) + "|",
    ]
    for fab_cap in fab_caps:
        cells = []
        for packaging_cap in packaging_caps:
            row = next(
                item for item in rows
                if item["capacity_multiplier"] == fab_cap
                and item["packaging_capacity_multiplier"] == packaging_cap
            )
            cells.append(row["phase_code"])
        lines.append(f"| {fab_cap:.1f} | " + " | ".join(f"`{cell}`" for cell in cells) + " |")
    return lines


def render_report(payload: dict) -> str:
    rows = payload["rows"]
    best = payload["best_score"]
    max_share = payload["max_share"]
    lines = [
        "# Semiconductor Onshoring Inventory-Renewal Phase Diagram",
        "",
        "## Scope",
        "",
        (
            "Focused sweep over domestic fab capacity, U.S. advanced-packaging capacity, "
            "and per-step gate inventory renewal for materials, power/labor, and packaging "
            "inputs. This tests whether stock timing, not routing preference, is the "
            "missing transition mechanism."
        ),
        "",
        f"- Agents per run: `{payload['config']['agents']}`",
        f"- Steps per run: `{payload['config']['steps']}`",
        f"- Grid points: `{len(rows)}`",
        f"- Baseline onshore share: `{payload['baseline']['onshore_share']:.3f}`",
        f"- Baseline U.S. finished lots: `{payload['baseline']['lot_total_us_finished']}`",
        "",
        "## Phase Legend",
        "",
        "| Code | Classification |",
        "|---|---|",
    ]
    for classification, code in PHASE_CODES.items():
        lines.append(f"| `{code}` | `{classification}` |")

    lines.extend(["", "## Classification Counts", "", "| Classification | Count |", "|---|---:|"])
    for label, count in sorted(payload["classification_counts"].items()):
        lines.append(f"| `{label}` | {count} |")

    lines.extend([
        "",
        "## Renewal Summary",
        "",
        "| Renewal | Mean Dependency | Min Dependency | Max Onshore Share | Mean Overflow | Fake Cells | Viable Cells |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for renewal in payload["inventory_renewals"]:
        renewal_rows = [row for row in rows if row["inventory_renewal"] == renewal]
        pressures = [
            max(row["gate_backlog_pressure"], row["gate_starvation_index"])
            for row in renewal_rows
        ]
        fake_count = sum(1 for row in renewal_rows if row["classification"] == "fake_onshoring")
        viable_count = sum(1 for row in renewal_rows if row["classification"] == "viable_onshoring_transition")
        lines.append(
            f"| {renewal} | {sum(pressures) / len(pressures):.1%} | "
            f"{min(pressures):.1%} | {max(row['onshore_share'] for row in renewal_rows):.3f} | "
            f"{sum(row['capacity_overflow_rate'] for row in renewal_rows) / len(renewal_rows):.1%} | "
            f"{fake_count} | {viable_count} |"
        )

    lines.extend([
        "",
        "## Best Score",
        "",
        f"- Doctrine: `{best['doctrine']}`",
        f"- Strategy: `{_fmt_point(best)}`",
        f"- Classification: `{best['classification']}`",
        f"- Onshore share: `{best['onshore_share']:.3f}`",
        f"- Finished-lot ratio: `{best['finished_flow_ratio']:.3f}`",
        f"- Overflow: `{best['capacity_overflow_rate']:.1%}`",
        f"- Dependency pressure: `{max(best['gate_backlog_pressure'], best['gate_starvation_index']):.1%}`",
        f"- Score: `{best['score']:.4f}`",
        "",
        "## Max Share",
        "",
        f"- Doctrine: `{max_share['doctrine']}`",
        f"- Strategy: `{_fmt_point(max_share)}`",
        f"- Classification: `{max_share['classification']}`",
        f"- Onshore share: `{max_share['onshore_share']:.3f}`",
        f"- Finished-lot ratio: `{max_share['finished_flow_ratio']:.3f}`",
        f"- Overflow: `{max_share['capacity_overflow_rate']:.1%}`",
        f"- Dependency pressure: `{max(max_share['gate_backlog_pressure'], max_share['gate_starvation_index']):.1%}`",
    ])

    if payload["best_viable"] is not None:
        viable = payload["best_viable"]
        lines.extend([
            "",
            "## Best Viable Point",
            "",
            f"- Doctrine: `{viable['doctrine']}`",
            f"- Strategy: `{_fmt_point(viable)}`",
            f"- Onshore share: `{viable['onshore_share']:.3f}`",
            f"- Finished-lot ratio: `{viable['finished_flow_ratio']:.3f}`",
            f"- Overflow: `{viable['capacity_overflow_rate']:.1%}`",
            f"- Dependency pressure: `{max(viable['gate_backlog_pressure'], viable['gate_starvation_index']):.1%}`",
        ])
    else:
        lines.extend(["", "## Best Viable Point", "", "No viable cell was found."])

    lines.extend(["", "## Renewal Slices", ""])
    for doctrine in DOCTRINES:
        for renewal in payload["inventory_renewals"]:
            slice_rows = [
                row for row in rows
                if row["doctrine"] == doctrine and row["inventory_renewal"] == renewal
            ]
            lines.extend([
                f"### `{doctrine}` renewal `{renewal}`",
                "",
                *_matrix_for(slice_rows, payload["fab_caps"], payload["packaging_caps"]),
                "",
            ])

    lines.extend([
        "## Reading",
        "",
        (
            "Inventory renewal is the first experiment that tests stock timing directly. "
            "A viable cell means the kernel has found a regime where policy pressure, "
            "capacity, and replenishment synchronize rather than merely redirecting flow."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--agents", type=int, default=160)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_inventory_phase_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_INVENTORY_PHASE_REPORT.md"))
    args = parser.parse_args()
    config = FrontierConfig(
        agents=80 if args.quick and args.agents == 160 else args.agents,
        steps=40 if args.quick and args.steps == 80 else args.steps,
    )
    payload = run_inventory_phase_diagram(config, quick=args.quick)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "classification_counts": payload["classification_counts"],
        "rows": len(payload["rows"]),
        "best_viable": payload["best_viable"],
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

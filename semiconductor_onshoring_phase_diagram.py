"""
Focused onshoring phase diagram for domestic fab and advanced-packaging capacity.

The broad frontier asks which policy bundles look good. This script asks the
cleaner threshold question: holding a doctrine fixed, how much fab capacity and
packaging capacity are required before the model stops reporting fake
onshoring?

Usage:
    .venv\\Scripts\\python.exe semiconductor_onshoring_phase_diagram.py --quick
    .venv\\Scripts\\python.exe semiconductor_onshoring_phase_diagram.py
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
    evaluate_point,
    import_dominant_baseline_point,
    scenario_for,
    control_for,
)


DOCTRINES = {
    "tariff_offset": StrategyPoint(
        tariff=2.25,
        capacity_multiplier=0.0,
        packaging_capacity_multiplier=0.0,
        subsidy=0.0,
        offset=1.0,
        domestic_ramp=1.5,
        packaging_ramp=0.0,
        materials=1.6,
        domestic_pull=0.8,
    ),
    "subsidy_packaging": StrategyPoint(
        tariff=0.0,
        capacity_multiplier=0.0,
        packaging_capacity_multiplier=0.0,
        subsidy=1.2,
        offset=0.0,
        domestic_ramp=1.5,
        packaging_ramp=1.5,
        materials=1.6,
        domestic_pull=0.0,
    ),
    "full_stack": StrategyPoint(
        tariff=2.25,
        capacity_multiplier=0.0,
        packaging_capacity_multiplier=0.0,
        subsidy=1.2,
        offset=1.0,
        domestic_ramp=2.5,
        packaging_ramp=1.5,
        materials=2.4,
        domestic_pull=2.5,
    ),
}


PHASE_CODES = {
    "viable_onshoring_transition": "V",
    "fake_onshoring": "F",
    "capacity_blocked": "C",
    "dependency_blocked": "D",
    "partial_onshoring": "P",
    "no_transition": ".",
}


def _fmt_point(row: dict) -> str:
    return (
        f"tariff={row['tariff']:.2f}, fabcap={row['capacity_multiplier']:.2f}, "
        f"pkgcap={row['packaging_capacity_multiplier']:.2f}, subsidy={row['subsidy']:.1f}, "
        f"offset={row['offset']:.1f}, domestic={row['domestic_ramp']:.1f}, "
        f"packaging={row['packaging_ramp']:.1f}, materials={row['materials']:.1f}, "
        f"pull={row['domestic_pull']:.1f}"
    )


def phase_grid(quick: bool = False) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if quick:
        return (0.0, 3.0, 6.0), (0.0, 3.0, 6.0, 9.0)
    return (0.0, 1.5, 3.0, 6.0, 9.0), (0.0, 1.5, 3.0, 6.0, 9.0, 12.0)


def run_phase_diagram(config: FrontierConfig | None = None, quick: bool = False) -> dict:
    config = config or FrontierConfig(agents=160, steps=80)
    fab_caps, packaging_caps = phase_grid(quick)
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
        for fab_cap in fab_caps:
            for packaging_cap in packaging_caps:
                point = replace(
                    template,
                    capacity_multiplier=fab_cap,
                    packaging_capacity_multiplier=packaging_cap,
                )
                row = evaluate_point(config, point, baseline)
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
        "classification_counts": counts,
        "best_viable": max(viable, key=lambda row: row["score"], default=None),
        "best_score": max(rows, key=lambda row: row["score"]),
        "max_share": max(rows, key=lambda row: row["onshore_share"]),
        "rows": rows,
    }


def render_report(payload: dict) -> str:
    rows = payload["rows"]
    lines = [
        "# Semiconductor Onshoring Phase Diagram",
        "",
        "## Scope",
        "",
        (
            "Focused sweep over domestic fab capacity and advanced-packaging capacity. "
            "Each doctrine holds tariff, subsidy, procurement pull, routing preference, "
            "and materials continuity fixed while the two capacity axes move."
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

    best = payload["best_score"]
    max_share = payload["max_share"]
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

    lines.extend(["", "## Doctrine Phase Matrices", ""])
    for doctrine in DOCTRINES:
        doctrine_rows = [row for row in rows if row["doctrine"] == doctrine]
        lines.extend([
            f"### `{doctrine}`",
            "",
            "| FabCap \\ PkgCap | " + " | ".join(f"{cap:.1f}" for cap in payload["packaging_caps"]) + " |",
            "|---|" + "|".join("---:" for _ in payload["packaging_caps"]) + "|",
        ])
        for fab_cap in payload["fab_caps"]:
            cells = []
            for packaging_cap in payload["packaging_caps"]:
                row = next(
                    item for item in doctrine_rows
                    if item["capacity_multiplier"] == fab_cap
                    and item["packaging_capacity_multiplier"] == packaging_cap
                )
                cells.append(row["phase_code"])
            lines.append(f"| {fab_cap:.1f} | " + " | ".join(f"`{cell}`" for cell in cells) + " |")
        lines.append("")

    lines.extend([
        "## Reading",
        "",
        (
            "This diagram isolates whether advanced-packaging capacity is the missing "
            "threshold variable. A viable transition requires majority onshore share, "
            "preserved finished flow, low overflow, and low dependency pressure."
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
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_phase_diagram_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_PHASE_DIAGRAM_REPORT.md"))
    args = parser.parse_args()
    config = FrontierConfig(
        agents=80 if args.quick and args.agents == 160 else args.agents,
        steps=40 if args.quick and args.steps == 80 else args.steps,
    )
    payload = run_phase_diagram(config, quick=args.quick)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "classification_counts": payload["classification_counts"],
        "rows": len(payload["rows"]),
        "best_viable": payload["best_viable"],
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

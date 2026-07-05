"""
Third-axis onshoring phase diagram for input replenishment.

The 2D phase diagram shows that domestic fab capacity plus advanced-packaging
capacity is not sufficient. This script adds materials/tooling continuity as a
third axis and asks whether upstream input replenishment opens viable
onshoring cells.

Usage:
    .venv\\Scripts\\python.exe semiconductor_onshoring_input_phase_diagram.py --quick
    .venv\\Scripts\\python.exe semiconductor_onshoring_input_phase_diagram.py
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
from semiconductor_onshoring_phase_diagram import DOCTRINES, PHASE_CODES


def _fmt_point(row: dict) -> str:
    return (
        f"tariff={row['tariff']:.2f}, fabcap={row['capacity_multiplier']:.2f}, "
        f"pkgcap={row['packaging_capacity_multiplier']:.2f}, subsidy={row['subsidy']:.1f}, "
        f"offset={row['offset']:.1f}, domestic={row['domestic_ramp']:.1f}, "
        f"packaging={row['packaging_ramp']:.1f}, materials={row['materials']:.1f}, "
        f"pull={row['domestic_pull']:.1f}"
    )


def phase_axes(quick: bool = False) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if quick:
        return (0.0, 3.0, 6.0), (0.0, 6.0, 12.0), (0.8, 2.4, 4.8)
    return (0.0, 3.0, 6.0, 9.0), (0.0, 3.0, 6.0, 9.0, 12.0), (0.8, 1.6, 2.4, 3.2, 4.8)


def run_input_phase_diagram(config: FrontierConfig | None = None, quick: bool = False) -> dict:
    config = config or FrontierConfig(agents=160, steps=80)
    fab_caps, packaging_caps, materials_levels = phase_axes(quick)
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
        for materials in materials_levels:
            for fab_cap in fab_caps:
                for packaging_cap in packaging_caps:
                    point = replace(
                        template,
                        capacity_multiplier=fab_cap,
                        packaging_capacity_multiplier=packaging_cap,
                        materials=materials,
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
        "materials_levels": materials_levels,
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
        "# Semiconductor Onshoring Input-Replenishment Phase Diagram",
        "",
        "## Scope",
        "",
        (
            "Focused 3D sweep over domestic fab capacity, U.S. advanced-packaging capacity, "
            "and materials/tooling continuity. The experiment tests whether upstream "
            "input replenishment opens viable onshoring cells after the 2D capacity "
            "phase diagram found no viable transition."
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

    lines.extend(["", "## Materials Slices", ""])
    for doctrine in DOCTRINES:
        for materials in payload["materials_levels"]:
            slice_rows = [
                row for row in rows
                if row["doctrine"] == doctrine and row["materials"] == materials
            ]
            lines.extend([
                f"### `{doctrine}` materials `{materials:.1f}`",
                "",
                *_matrix_for(slice_rows, payload["fab_caps"], payload["packaging_caps"]),
                "",
            ])

    lines.extend([
        "## Reading",
        "",
        (
            "The third axis tests the minimal extension implied by the 2D result. "
            "If high materials continuity still fails, the remaining missing variables "
            "are likely inventory renewal timing, packaging-input independence, or a "
            "more explicit split between U.S. and Taiwan packaging dependencies."
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
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_input_phase_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_INPUT_PHASE_REPORT.md"))
    args = parser.parse_args()
    config = FrontierConfig(
        agents=80 if args.quick and args.agents == 160 else args.agents,
        steps=40 if args.quick and args.steps == 80 else args.steps,
    )
    payload = run_input_phase_diagram(config, quick=args.quick)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "classification_counts": payload["classification_counts"],
        "rows": len(payload["rows"]),
        "best_viable": payload["best_viable"],
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

"""
Timing sweep for the abstract microbiome DTE ecology prototype.

This sweep holds the probiotic-like intervention content mostly fixed and
varies the start time relative to antibiotic clearance. It estimates a
recoverability window: too early can wash out, on-phase can recover, and too
late can leave residual pathobiont lock-in.

This is not a medical model and does not make treatment recommendations.

Usage:
    .venv\\Scripts\\python.exe microbiome_timing_sweep.py --quick
    .venv\\Scripts\\python.exe microbiome_timing_sweep.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from microbiome_ecology_dte import MicrobiomeConfig, MicrobiomeScenario, render_report as render_pilot_report, simulate


def _mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    arr = np.array(values, dtype=float)
    avg = float(arr.mean())
    half = float(z * arr.std(ddof=1) / np.sqrt(len(arr)))
    return avg - half, avg + half


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def timing_offsets(quick: bool = False) -> tuple[int, ...]:
    if quick:
        return (-28, -14, 0, 8, 20, 34, 48, 60)
    return (-34, -28, -21, -14, -7, 0, 4, 8, 12, 20, 28, 34, 42, 48, 56, 64)


def scenario_for_offset(offset: int) -> MicrobiomeScenario:
    antibiotic_end = 42
    probiotic_start = antibiotic_end + offset
    return MicrobiomeScenario(
        name=f"timing_offset_{offset:+d}",
        antibiotic_start=8,
        antibiotic_end=antibiotic_end,
        probiotic_start=probiotic_start,
        probiotic_duration=22,
        probiotic_dose=7,
        prebiotic_start=42,
        prebiotic_duration=42,
        prebiotic_strength=1.05,
        antibiotic_strength=0.78,
        initial_pathobiont_share=0.10,
        notes="Timing sweep scenario with fixed intervention content.",
    )


def _phase_label(offset: int) -> str:
    if offset < 0:
        return "antibiotic_overlap"
    if offset <= 12:
        return "recovery_window"
    return "late_window"


def run_timing_sweep(
    config: MicrobiomeConfig | None = None,
    quick: bool = False,
    seeds: int | None = None,
) -> dict[str, Any]:
    config = config or (MicrobiomeConfig(agents=96, steps=126) if quick else MicrobiomeConfig(steps=132))
    seeds = seeds if seeds is not None else (2 if quick else 5)
    rows: list[dict[str, Any]] = []

    for offset in timing_offsets(quick):
        scenario = scenario_for_offset(offset)
        for seed_index in range(seeds):
            seed_config = replace(config, seed=config.seed + seed_index * 100_003)
            row = simulate(seed_config, scenario, seed_offset=offset * 97 + seed_index * 997)
            row["timing_offset"] = offset
            row["timing_phase"] = _phase_label(offset)
            row["seed_index"] = seed_index
            rows.append(row)

    summary = summarize_timing_sweep(rows)
    return {
        "config": asdict(config) | {"quick": quick, "seeds": seeds},
        "offsets": timing_offsets(quick),
        "summary": summary,
        "rows": rows,
    }


def summarize_timing_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_offset: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_offset[row["timing_offset"]].append(row)

    offset_summary = []
    for offset, items in sorted(by_offset.items()):
        class_counts = Counter(row["classification"] for row in items)
        recovery_count = sum(
            row["final_beneficial_occupancy"] >= 0.56 and row["final_pathobiont_occupancy"] <= 0.28
            for row in items
        )
        washout_count = class_counts.get("early_washout", 0)
        lockin_count = sum(
            row["classification"] in {"late_lockin", "pathobiont_lockin"}
            or row["final_pathobiont_occupancy"] >= 0.30
            for row in items
        )
        recovery_low, recovery_high = _wilson_interval(recovery_count, len(items))
        beneficial_low, beneficial_high = _mean_ci([row["final_beneficial_occupancy"] for row in items])
        path_low, path_high = _mean_ci([row["final_pathobiont_occupancy"] for row in items])
        phase_error_low, phase_error_high = _mean_ci([row["intervention_phase_error"] for row in items])
        offset_summary.append({
            "timing_offset": offset,
            "timing_phase": _phase_label(offset),
            "runs": len(items),
            "class_counts": dict(class_counts),
            "recovery_rate": float(recovery_count / len(items)),
            "recovery_rate_ci95": [recovery_low, recovery_high],
            "washout_rate": float(washout_count / len(items)),
            "lockin_rate": float(lockin_count / len(items)),
            "mean_beneficial_occupancy": float(mean(row["final_beneficial_occupancy"] for row in items)),
            "mean_beneficial_occupancy_ci95": [beneficial_low, beneficial_high],
            "mean_pathobiont_occupancy": float(mean(row["final_pathobiont_occupancy"] for row in items)),
            "mean_pathobiont_occupancy_ci95": [path_low, path_high],
            "mean_phase_error": float(mean(row["intervention_phase_error"] for row in items)),
            "mean_phase_error_ci95": [phase_error_low, phase_error_high],
            "dominant_stale_layers": dict(Counter(row["dominant_stale_memory_layer"] for row in items)),
        })

    viable = [
        row for row in offset_summary
        if row["recovery_rate"] >= 0.50 and row["washout_rate"] < 0.50 and row["lockin_rate"] < 0.50
    ]
    best = max(offset_summary, key=lambda row: row["mean_beneficial_occupancy"])
    worst = max(offset_summary, key=lambda row: row["mean_pathobiont_occupancy"])
    return {
        "runs": len(rows),
        "seeds": len({row["seed_index"] for row in rows}),
        "offset_summary": offset_summary,
        "viable_window": [min(row["timing_offset"] for row in viable), max(row["timing_offset"] for row in viable)]
        if viable
        else None,
        "best_offset": best,
        "worst_offset": worst,
        "classification_counts": dict(Counter(row["classification"] for row in rows)),
    }


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Microbiome Timing Sweep",
        "",
        "## Scope",
        "",
        (
            "Abstract, non-medical timing sweep. Intervention content is held mostly "
            "fixed while probiotic start time moves relative to antibiotic clearance. "
            "The output estimates a recoverability window rather than prescribing any "
            "real-world treatment."
        ),
        "",
        f"- Runs: `{summary['runs']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Viable recovery window: `{summary['viable_window']}`",
        f"- Best offset: `{summary['best_offset']['timing_offset']}` "
        f"beneficial={summary['best_offset']['mean_beneficial_occupancy']:.3f}",
        f"- Worst offset: `{summary['worst_offset']['timing_offset']}` "
        f"pathobiont={summary['worst_offset']['mean_pathobiont_occupancy']:.3f}",
        "",
        "## Classification Counts",
        "",
    ]
    for label, count in sorted(summary["classification_counts"].items()):
        lines.append(f"- `{label}`: `{count}`")

    lines.extend(
        [
            "",
            "## Offset Frontier",
            "",
            "| Offset vs clearance | Phase | Recovery rate | Washout | Lock-in | Beneficial | Pathobiont | Phase error | Dominant stale layers |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["offset_summary"]:
        layers = ", ".join(f"{key}:{value}" for key, value in sorted(row["dominant_stale_layers"].items()))
        lines.append(
            f"| {row['timing_offset']} | {row['timing_phase']} | "
            f"{row['recovery_rate']:.2f} | {row['washout_rate']:.2f} | {row['lockin_rate']:.2f} | "
            f"{row['mean_beneficial_occupancy']:.3f} | {row['mean_pathobiont_occupancy']:.3f} | "
            f"{row['mean_phase_error']:.3f} | {layers} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The timing sweep operationalizes phase-aware control. The same "
                "intervention can be washed out before clearance, productive in a "
                "recovery window, or too late after pathobiont memory has formed. "
                "The important variable is not intervention content alone; it is "
                "intervention content relative to ecological phase and memory state."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], json_path: Path, report_path: Path) -> None:
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run microbiome timing sweep.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--json", default="microbiome_timing_sweep_output.json")
    parser.add_argument("--report", default="MICROBIOME_TIMING_SWEEP_REPORT.md")
    args = parser.parse_args()

    payload = run_timing_sweep(quick=args.quick, seeds=args.seeds)
    write_outputs(payload, Path(args.json), Path(args.report))
    print(render_report(payload))


if __name__ == "__main__":
    main()

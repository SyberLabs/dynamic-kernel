"""
Seed-robust intervention test for the Social Media DTE prototype.

This is the first local thesis test after the phase diagram. It asks whether
topology-aware off-ramp interventions beat generic diversity boosting under
common random numbers across stressed feed regimes.

Usage:
    .venv\\Scripts\\python.exe social_media_robustness.py
    .venv\\Scripts\\python.exe social_media_robustness.py --quick
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from social_media_phase import SimulationConfig, run_interventions


@dataclass(frozen=True)
class RobustnessCell:
    intent: str
    feedback_rate: float
    temperature: float
    noise_sigma: float


DEFAULT_STRESSED_CELLS = [
    RobustnessCell("High-Arousal Scroll", 0.20, 0.50, 0.00),
    RobustnessCell("High-Arousal Scroll", 0.20, 0.80, 0.00),
    RobustnessCell("High-Arousal Scroll", 0.35, 0.50, 0.00),
    RobustnessCell("High-Arousal Scroll", 0.35, 0.80, 0.00),
    RobustnessCell("High-Arousal Scroll", 0.35, 0.50, 0.03),
    RobustnessCell("High-Arousal Scroll", 0.35, 0.80, 0.03),
    RobustnessCell("High-Arousal Scroll", 0.50, 0.50, 0.00),
    RobustnessCell("High-Arousal Scroll", 0.50, 0.80, 0.03),
]

PRIMARY_INTERVENTION = "off_ramp_friction"
BASELINE_INTERVENTION = "baseline"
COMPARATOR_INTERVENTION = "generic_diversity_beta"
SAFETY_INTERVENTIONS = {
    "off_ramp_beta",
    "science_beta",
    "off_ramp_friction",
    "generic_diversity_beta",
}


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _bootstrap_mean_ci(values: list[float], seed: int = 20260602, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    array = np.array(values, dtype=np.float64)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _strip_series(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "risk_series"}


def summarize_replicates(
    replicates: list[dict],
    cells: int,
    seeds: int,
    agents: int,
    steps: int,
) -> dict:
    paired_roi_deltas = [float(row["primary_minus_comparator_roi"]) for row in replicates]
    paired_risk_deltas = [float(row["primary_minus_comparator_risk"]) for row in replicates]
    primary_wins = sum(row["best_intervention"] == PRIMARY_INTERVENTION for row in replicates)
    primary_beats_comparator = sum(row["primary_minus_comparator_roi"] > 0 for row in replicates)
    total = len(replicates)

    win_low, win_high = _wilson_interval(primary_wins, total)
    beat_low, beat_high = _wilson_interval(primary_beats_comparator, total)
    roi_low, roi_high = _bootstrap_mean_ci(paired_roi_deltas)
    risk_low, risk_high = _bootstrap_mean_ci(paired_risk_deltas, seed=20260603)

    baseline_risks = [row["baseline_risk"] for row in replicates]
    baseline_currents = [row["baseline_edge_current_norm"] for row in replicates]
    current_risk_corr = 0.0
    if len(replicates) > 1 and np.std(baseline_currents) > 1e-12 and np.std(baseline_risks) > 1e-12:
        current_risk_corr = float(np.corrcoef(baseline_currents, baseline_risks)[0, 1])

    return {
        "cells": cells,
        "seeds": seeds,
        "agents": agents,
        "steps": steps,
        "replicates": total,
        "primary_intervention": PRIMARY_INTERVENTION,
        "comparator_intervention": COMPARATOR_INTERVENTION,
        "primary_best_win_rate": float(primary_wins / max(total, 1)),
        "primary_best_win_rate_ci95": [win_low, win_high],
        "primary_beats_comparator_rate": float(primary_beats_comparator / max(total, 1)),
        "primary_beats_comparator_rate_ci95": [beat_low, beat_high],
        "mean_primary_minus_comparator_roi": float(np.mean(paired_roi_deltas)) if paired_roi_deltas else 0.0,
        "mean_primary_minus_comparator_roi_ci95": [roi_low, roi_high],
        "mean_primary_minus_comparator_risk": float(np.mean(paired_risk_deltas)) if paired_risk_deltas else 0.0,
        "mean_primary_minus_comparator_risk_ci95": [risk_low, risk_high],
        "mean_baseline_risk": float(np.mean(baseline_risks)) if baseline_risks else 0.0,
        "edge_current_risk_correlation": current_risk_corr,
        "institutional_result": "pass"
        if primary_beats_comparator / max(total, 1) >= 0.70 and beat_low > 0.50
        else "inconclusive",
    }


def run_robustness(
    cells: list[RobustnessCell] | None = None,
    seeds: int = 10,
    agents: int = 256,
    steps: int = 80,
    seed_base: int = 20260602,
) -> dict:
    cells = cells or DEFAULT_STRESSED_CELLS
    replicates = []

    for cell_index, cell in enumerate(cells):
        for seed_index in range(seeds):
            config = SimulationConfig(
                intent=cell.intent,
                feedback_rate=cell.feedback_rate,
                temperature=cell.temperature,
                noise_sigma=cell.noise_sigma,
                agents=agents,
                steps=steps,
                seed=seed_base + 100_000 * cell_index + seed_index,
            )
            rows = run_interventions(config)
            by_name = {row["intervention"]: row for row in rows}
            safety_rows = [row for row in rows if row["intervention"] in SAFETY_INTERVENTIONS]
            best = max(safety_rows, key=lambda row: row["risk_reduction_per_cost"])
            primary = by_name[PRIMARY_INTERVENTION]
            comparator = by_name[COMPARATOR_INTERVENTION]
            baseline = by_name[BASELINE_INTERVENTION]

            paired_roi_delta = float(
                primary["risk_reduction_per_cost"] - comparator["risk_reduction_per_cost"]
            )
            paired_risk_delta = float(primary["risk_share"] - comparator["risk_share"])

            replicates.append({
                "cell": {
                    "intent": cell.intent,
                    "lambda": cell.feedback_rate,
                    "tau": cell.temperature,
                    "sigma": cell.noise_sigma,
                },
                "seed": config.seed,
                "baseline_risk": float(baseline["risk_share"]),
                "baseline_escape_probability": float(baseline["escape_probability"]),
                "baseline_entropy_production": float(baseline["entropy_production"]),
                "baseline_edge_current_norm": float(baseline["edge_current_norm"]),
                "best_intervention": best["intervention"],
                "primary_roi": float(primary["risk_reduction_per_cost"]),
                "comparator_roi": float(comparator["risk_reduction_per_cost"]),
                "primary_minus_comparator_roi": paired_roi_delta,
                "primary_minus_comparator_risk": paired_risk_delta,
                "interventions": [_strip_series(row) for row in rows],
            })

    summary = summarize_replicates(replicates, len(cells), seeds, agents, steps)
    return {"summary": summary, "replicates": replicates}


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Social Media Robustness Report",
        "",
        "## Thesis Test",
        "",
        (
            f"Primary claim: `{summary['primary_intervention']}` should beat "
            f"`{summary['comparator_intervention']}` on risk-reduction-per-cost "
            "in stressed high-arousal regimes."
        ),
        "",
        "## Summary",
        "",
        f"- Replicates: `{summary['replicates']}` (`{summary['cells']}` cells x `{summary['seeds']}` seeds)",
        f"- Agents per replicate: `{summary['agents']}`",
        f"- Steps per replicate: `{summary['steps']}`",
        f"- Mean baseline risk: `{summary['mean_baseline_risk']:.3f}`",
        (
            f"- Primary beats comparator: `{summary['primary_beats_comparator_rate']:.1%}` "
            f"CI95 `[{summary['primary_beats_comparator_rate_ci95'][0]:.1%}, "
            f"{summary['primary_beats_comparator_rate_ci95'][1]:.1%}]`"
        ),
        (
            f"- Primary is best safety intervention: `{summary['primary_best_win_rate']:.1%}` "
            f"CI95 `[{summary['primary_best_win_rate_ci95'][0]:.1%}, "
            f"{summary['primary_best_win_rate_ci95'][1]:.1%}]`"
        ),
        (
            f"- Mean ROI edge over comparator: `{summary['mean_primary_minus_comparator_roi']:.5f}` "
            f"CI95 `[{summary['mean_primary_minus_comparator_roi_ci95'][0]:.5f}, "
            f"{summary['mean_primary_minus_comparator_roi_ci95'][1]:.5f}]`"
        ),
        (
            f"- Mean risk difference vs comparator: `{summary['mean_primary_minus_comparator_risk']:.5f}` "
            f"CI95 `[{summary['mean_primary_minus_comparator_risk_ci95'][0]:.5f}, "
            f"{summary['mean_primary_minus_comparator_risk_ci95'][1]:.5f}]`"
        ),
        f"- Edge-current/risk correlation: `{summary['edge_current_risk_correlation']:.3f}`",
        f"- Institutional result: `{summary['institutional_result']}`",
        "",
        "## Decision Rule",
        "",
        (
            "Promote this claim to a larger Tier 2 or SLURM study only if the primary "
            "beats comparator rate is at least 70% and the 95% lower bound exceeds 50%."
        ),
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("social_media_robustness_output.json"),
    output_md: Path = Path("SOCIAL_MEDIA_ROBUSTNESS_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run social-media intervention robustness tests.")
    parser.add_argument("--quick", action="store_true", help="Use a small smoke-size robustness run.")
    parser.add_argument("--cell-index", type=int, default=None, help="Run one default stressed cell by index.")
    parser.add_argument("--seed-base", type=int, default=20260602)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("social_media_robustness_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SOCIAL_MEDIA_ROBUSTNESS_REPORT.md"))
    args = parser.parse_args()

    cells = DEFAULT_STRESSED_CELLS
    seed_base = args.seed_base
    if args.cell_index is not None:
        if args.cell_index < 0 or args.cell_index >= len(DEFAULT_STRESSED_CELLS):
            raise SystemExit(f"--cell-index must be between 0 and {len(DEFAULT_STRESSED_CELLS) - 1}")
        cells = [DEFAULT_STRESSED_CELLS[args.cell_index]]
        seed_base = args.seed_base + 100_000 * args.cell_index

    payload = run_robustness(
        cells=cells,
        seeds=args.seeds if args.seeds is not None else (2 if args.quick else 10),
        agents=args.agents if args.agents is not None else (64 if args.quick else 256),
        steps=args.steps if args.steps is not None else (24 if args.quick else 80),
        seed_base=seed_base,
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

"""
Local 2D phase surface for Phase 2 influence-game defenses.

This sweep fixes an initial defender budget and maps threshold-crossing
probability over `(immune_gain, action_unit)`. It is meant as the local
pre-SLURM contour experiment after the dense frontier and mechanism analysis.

Usage:
    .venv\\Scripts\\python.exe influence_phase_surface.py
    .venv\\Scripts\\python.exe influence_phase_surface.py --quick
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from influence_game import GameConfig
from influence_qlearn import train_q_adversary


DEFAULT_GAINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
DEFAULT_ACTION_UNITS = [0.25, 0.5, 1.0, 2.0]


def _mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    array = np.array(values, dtype=np.float64)
    mean = float(array.mean())
    half_width = float(z * array.std(ddof=1) / np.sqrt(len(array)))
    return mean - half_width, mean + half_width


def _classify_cell(crossing_rate: float, max_peak: float, risk_threshold: float) -> str:
    if crossing_rate == 0.0 and max_peak < risk_threshold:
        return "safe"
    if crossing_rate == 0.0:
        return "near_boundary"
    if crossing_rate <= 0.25:
        return "marginal"
    if crossing_rate < 0.75:
        return "vulnerable"
    return "unstable"


def _game_config(
    seed: int,
    defender_budget: float,
    action_unit: float,
    immune_gain: float,
    immune_cap: float,
    max_extra: float,
    agents: int,
    rounds: int,
    steps_per_round: int,
    adversary_budget: float,
) -> GameConfig:
    dynamic = immune_gain > 0.0
    return GameConfig(
        rounds=rounds,
        agents=agents,
        steps_per_round=steps_per_round,
        adversary_budget=adversary_budget,
        defender_budget=defender_budget,
        action_unit=action_unit,
        seed=seed,
        dynamic_defense=dynamic,
        immune_response_gain=immune_gain if dynamic else 0.0,
        immune_response_cap=immune_cap if dynamic else 0.0,
        max_defender_budget=max(defender_budget, defender_budget + (max_extra if dynamic else 0.0)),
        trust_hysteresis=True,
    )


def run_phase_surface(
    gains: list[float] | None = None,
    action_units: list[float] | None = None,
    defender_budget: float = 0.25,
    immune_cap: float = 3.0,
    max_extra: float = 5.0,
    seeds: int = 5,
    episodes: int = 8,
    agents: int = 96,
    rounds: int = 8,
    steps_per_round: int = 12,
    adversary_budget: float = 10.0,
    seed_base: int = 20260610,
) -> dict:
    gains = gains or DEFAULT_GAINS
    action_units = action_units or DEFAULT_ACTION_UNITS
    rows = []
    for gain in gains:
        for action_unit in action_units:
            for seed_index in range(seeds):
                seed = seed_base + seed_index
                config = _game_config(
                    seed=seed,
                    defender_budget=defender_budget,
                    action_unit=action_unit,
                    immune_gain=gain,
                    immune_cap=immune_cap,
                    max_extra=max_extra,
                    agents=agents,
                    rounds=rounds,
                    steps_per_round=steps_per_round,
                    adversary_budget=adversary_budget,
                )
                summary = train_q_adversary(config, episodes=episodes)["summary"]
                rows.append({
                    "immune_gain": gain,
                    "action_unit": action_unit,
                    "seed": seed,
                    "defender_budget": defender_budget,
                    "immune_cap": immune_cap,
                    "episodes": episodes,
                    "q_eval_peak_risk": summary["q_eval_peak_risk"],
                    "q_eval_final_risk": summary["q_eval_final_risk"],
                    "baseline_peak_risk": summary["baseline_peak_risk"],
                    "baseline_final_risk": summary["baseline_final_risk"],
                    "q_minus_baseline_final_risk": summary["q_minus_baseline_final_risk"],
                    "q_threshold_crossed": summary["q_threshold_crossed"],
                    "q_pause_rate": summary["q_pause_rate"],
                })
    return {
        "summary": summarize_phase_surface(rows),
        "rows": rows,
        "config": {
            "gains": gains,
            "action_units": action_units,
            "defender_budget": defender_budget,
            "immune_cap": immune_cap,
            "max_extra": max_extra,
            "seeds": seeds,
            "episodes": episodes,
            "agents": agents,
            "rounds": rounds,
            "steps_per_round": steps_per_round,
            "adversary_budget": adversary_budget,
            "seed_base": seed_base,
        },
    }


def summarize_phase_surface(rows: list[dict], risk_threshold: float = 0.18) -> dict:
    by_cell: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["immune_gain"], row["action_unit"])].append(row)

    gains = sorted({row["immune_gain"] for row in rows})
    action_units = sorted({row["action_unit"] for row in rows})
    cells = {}
    for gain in gains:
        for action_unit in action_units:
            items = by_cell[(gain, action_unit)]
            peaks = [row["q_eval_peak_risk"] for row in items]
            finals = [row["q_eval_final_risk"] for row in items]
            crossed = sum(row["q_threshold_crossed"] for row in items)
            peak_low, peak_high = _mean_ci(peaks)
            crossing_rate = float(crossed / max(len(items), 1))
            max_peak = float(np.max(peaks)) if peaks else 0.0
            cells[f"{gain:.2f}|{action_unit:.2f}"] = {
                "immune_gain": gain,
                "action_unit": action_unit,
                "runs": len(items),
                "mean_q_peak_risk": float(np.mean(peaks)) if peaks else 0.0,
                "mean_q_peak_risk_ci95": [peak_low, peak_high],
                "max_q_peak_risk": max_peak,
                "mean_q_final_risk": float(np.mean(finals)) if finals else 0.0,
                "threshold_crossing_rate": crossing_rate,
                "mean_q_pause_rate": float(np.mean([row["q_pause_rate"] for row in items])) if items else 0.0,
                "classification": _classify_cell(crossing_rate, max_peak, risk_threshold),
            }

    robust_safe_gains = {}
    for action_unit in action_units:
        robust_safe_gains[f"{action_unit:.2f}"] = None
        for gain in gains:
            cell = cells[f"{gain:.2f}|{action_unit:.2f}"]
            future_gains = gains[gains.index(gain):]
            if all(
                cells[f"{future_gain:.2f}|{action_unit:.2f}"]["threshold_crossing_rate"] == 0.0
                and cells[f"{future_gain:.2f}|{action_unit:.2f}"]["max_q_peak_risk"] < risk_threshold
                for future_gain in future_gains
            ):
                robust_safe_gains[f"{action_unit:.2f}"] = gain
                break

    return {
        "gains": gains,
        "action_units": action_units,
        "risk_threshold": risk_threshold,
        "runs": len(rows),
        "seeds": len({row["seed"] for row in rows}),
        "robust_safe_gain_by_action_unit": robust_safe_gains,
        "cells": cells,
    }


def _render_matrix(summary: dict, field: str, formatter) -> list[str]:
    action_units = summary["action_units"]
    gains = summary["gains"]
    lines = [
        "| Immune Gain | " + " | ".join(f"unit {unit:.2f}" for unit in action_units) + " |",
        "|---:" + "|---:" * len(action_units) + "|",
    ]
    for gain in gains:
        values = []
        for action_unit in action_units:
            cell = summary["cells"][f"{gain:.2f}|{action_unit:.2f}"]
            values.append(formatter(cell[field]))
        lines.append(f"| {gain:.2f} | " + " | ".join(values) + " |")
    return lines


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    config = payload["config"]
    lines = [
        "# Influence Phase Surface Report",
        "",
        "## Scope",
        "",
        (
            "Local 2D crossing-probability surface over immune-response gain and defender "
            "action granularity under a Q-learning adversary."
        ),
        "",
        "## Configuration",
        "",
        f"- Defender budget: `{config['defender_budget']}`",
        f"- Immune cap: `{config['immune_cap']}`",
        f"- Seeds: `{summary['seeds']}`",
        f"- Runs: `{summary['runs']}`",
        f"- Episodes per run: `{config['episodes']}`",
        f"- Risk threshold: `{summary['risk_threshold']:.3f}`",
        f"- Robust safe gain by action unit: `{summary['robust_safe_gain_by_action_unit']}`",
        "",
        "## Crossing Rate Surface",
        "",
    ]
    lines.extend(_render_matrix(
        summary,
        "threshold_crossing_rate",
        lambda value: f"{value:.1%}",
    ))
    lines.extend([
        "",
        "## Max Peak Risk Surface",
        "",
    ])
    lines.extend(_render_matrix(
        summary,
        "max_q_peak_risk",
        lambda value: f"{value:.3f}",
    ))
    lines.extend([
        "",
        "## Classification Surface",
        "",
    ])
    lines.extend(_render_matrix(
        summary,
        "classification",
        str,
    ))
    lines.extend([
        "",
        "## Reading",
        "",
        (
            "`safe` means zero crossings and max peak risk below threshold across the local seed grid. "
            "`marginal` means a low but nonzero crossing rate. `vulnerable` and `unstable` indicate "
            "larger crossing rates. This surface is a local pre-SLURM contour map, not a final theorem."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("influence_phase_surface_output.json"),
    output_md: Path = Path("INFLUENCE_PHASE_SURFACE_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local immune/action-unit phase surface.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--defender-budget", type=float, default=0.25)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--steps-per-round", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("influence_phase_surface_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("INFLUENCE_PHASE_SURFACE_REPORT.md"))
    args = parser.parse_args()

    payload = run_phase_surface(
        gains=[0.0, 0.75, 1.25] if args.quick else DEFAULT_GAINS,
        action_units=[0.5, 1.0] if args.quick else DEFAULT_ACTION_UNITS,
        defender_budget=args.defender_budget,
        seeds=args.seeds if args.seeds is not None else (1 if args.quick else 5),
        episodes=args.episodes if args.episodes is not None else (1 if args.quick else 8),
        agents=args.agents if args.agents is not None else (32 if args.quick else 96),
        rounds=args.rounds if args.rounds is not None else (3 if args.quick else 8),
        steps_per_round=args.steps_per_round if args.steps_per_round is not None else (4 if args.quick else 12),
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": payload["summary"]["runs"],
        "robust_safe_gain_by_action_unit": payload["summary"]["robust_safe_gain_by_action_unit"],
    }, indent=2))


if __name__ == "__main__":
    main()

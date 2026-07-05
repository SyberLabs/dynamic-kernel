"""
Local scaling benchmark for the Social Media DTE research program.

The goal is not to produce final science. It estimates how far local desktop
experiments can go before SLURM/HPC becomes justified.

Usage:
    .venv\\Scripts\\python.exe social_media_scaling.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from social_media_phase import (
    DEFAULT_INTENTS,
    SimulationConfig,
    run_interventions,
    run_phase_grid,
)


@dataclass(frozen=True)
class ScalingCase:
    name: str
    agents: int
    steps: int
    cells: int
    seeds: int
    interventions: int


DEFAULT_CASES = [
    ScalingCase("tier0_smoke", agents=32, steps=16, cells=4, seeds=1, interventions=1),
    ScalingCase("tier1_grid", agents=192, steps=64, cells=96, seeds=1, interventions=1),
    ScalingCase("tier1_interventions", agents=256, steps=80, cells=1, seeds=1, interventions=6),
    ScalingCase("tier2_probe", agents=384, steps=96, cells=96, seeds=5, interventions=6),
]


def _small_grid_for_cells(cells: int):
    lambdas = [0.0, 0.10, 0.20, 0.35, 0.50]
    taus = [0.5, 0.8, 1.0, 1.5, 2.5]
    sigmas = [0.0, 0.03, 0.05]
    intents = DEFAULT_INTENTS

    selected = []
    for intent in intents:
        for lam in lambdas:
            for tau in taus:
                for sigma in sigmas:
                    selected.append((intent, lam, tau, sigma))
                    if len(selected) >= cells:
                        return selected
    return selected


def _run_cell(intent: str, lam: float, tau: float, sigma: float, agents: int, steps: int, seed: int):
    return run_phase_grid(
        lambdas=[lam],
        taus=[tau],
        sigmas=[sigma],
        intents=[intent],
        agents=agents,
        steps=steps,
        seed=seed,
    )[0]


def benchmark_case(case: ScalingCase) -> dict:
    grid = _small_grid_for_cells(case.cells)
    start = time.perf_counter()
    lock_count = 0
    risk_values = []

    for seed_offset in range(case.seeds):
        for intent, lam, tau, sigma in grid:
            row = _run_cell(
                intent=intent,
                lam=lam,
                tau=tau,
                sigma=sigma,
                agents=case.agents,
                steps=case.steps,
                seed=20260602 + seed_offset,
            )
            lock_count += int(row["lock_in"])
            risk_values.append(float(row["risk_share"]))

        if case.interventions > 1:
            rows = run_interventions(
                SimulationConfig(
                    intent="High-Arousal Scroll",
                    feedback_rate=0.35,
                    temperature=0.5,
                    noise_sigma=0.0,
                    agents=case.agents,
                    steps=case.steps,
                    seed=20260602 + seed_offset,
                )
            )
            risk_values.extend(float(row["risk_share"]) for row in rows[:case.interventions])

    elapsed = time.perf_counter() - start
    simulated_runs = case.seeds * (len(grid) + (case.interventions if case.interventions > 1 else 0))
    agent_steps = case.agents * case.steps * simulated_runs
    seconds_per_agent_step = elapsed / max(agent_steps, 1)

    return {
        "name": case.name,
        "agents": case.agents,
        "steps": case.steps,
        "cells": len(grid),
        "seeds": case.seeds,
        "interventions": case.interventions,
        "simulated_runs": simulated_runs,
        "agent_steps": agent_steps,
        "elapsed_seconds": elapsed,
        "seconds_per_agent_step": seconds_per_agent_step,
        "agent_steps_per_second": 1.0 / seconds_per_agent_step if seconds_per_agent_step > 0 else 0.0,
        "lock_count": lock_count,
        "mean_risk_share": float(np.mean(risk_values)) if risk_values else 0.0,
    }


def estimate_runtime_seconds(
    seconds_per_agent_step: float,
    agents: int,
    steps: int,
    cells: int,
    seeds: int,
    interventions: int,
) -> float:
    runs = seeds * cells * max(interventions, 1)
    return float(seconds_per_agent_step * agents * steps * runs)


def summarize_scaling(results: list[dict]) -> dict:
    best_rate = min(row["seconds_per_agent_step"] for row in results)
    sustained = max(results, key=lambda row: row["agent_steps"])
    sustained_rate = sustained["seconds_per_agent_step"]
    tier2_runtime = estimate_runtime_seconds(
        sustained_rate,
        agents=512,
        steps=128,
        cells=300,
        seeds=20,
        interventions=6,
    )
    hpc_runtime = estimate_runtime_seconds(
        sustained_rate,
        agents=1024,
        steps=192,
        cells=1000,
        seeds=200,
        interventions=6,
    )
    return {
        "best_seconds_per_agent_step": best_rate,
        "sustained_seconds_per_agent_step": sustained_rate,
        "sustained_case": sustained["name"],
        "tier2_estimated_hours": tier2_runtime / 3600.0,
        "hpc_target_estimated_hours_single_core": hpc_runtime / 3600.0,
        "recommendation": (
            "prepare_slurm"
            if tier2_runtime > 6 * 3600
            else "stay_local_for_tier2"
        ),
    }


def render_report(results: list[dict], summary: dict) -> str:
    lines = [
        "# Social Media Scaling Report",
        "",
        "## Local Benchmarks",
        "",
        "| Case | Agents | Steps | Cells | Seeds | Interventions | Runs | Seconds | Agent Steps/s | Mean Risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['name']} | {row['agents']} | {row['steps']} | {row['cells']} | "
            f"{row['seeds']} | {row['interventions']} | {row['simulated_runs']} | "
            f"{row['elapsed_seconds']:.3f} | {row['agent_steps_per_second']:.0f} | "
            f"{row['mean_risk_share']:.3f} |"
        )
    lines.extend([
        "",
        "## Runtime Estimates",
        "",
        f"- Best seconds per agent-step: `{summary['best_seconds_per_agent_step']:.3e}`",
        (
            f"- Sustained planning rate: `{summary['sustained_seconds_per_agent_step']:.3e}` "
            f"from `{summary['sustained_case']}`"
        ),
        f"- Tier 2 estimate: `{summary['tier2_estimated_hours']:.2f}` desktop hours",
        f"- HPC target estimate on one core: `{summary['hpc_target_estimated_hours_single_core']:.2f}` hours",
        f"- Recommendation: `{summary['recommendation']}`",
        "",
        "## Interpretation",
        "",
        "Use HPC only after Tier 2 local robustness exceeds a practical desktop budget or after the reducer/report schema is frozen.",
    ])
    return "\n".join(lines) + "\n"


def run_scaling(cases: list[ScalingCase] | None = None) -> dict:
    results = [benchmark_case(case) for case in (cases or DEFAULT_CASES)]
    summary = summarize_scaling(results)
    return {"summary": summary, "results": results}


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload["results"], payload["summary"]), encoding="utf-8")


def main() -> None:
    payload = run_scaling()
    write_outputs(
        payload,
        Path("social_media_scaling_output.json"),
        Path("SOCIAL_MEDIA_SCALING_REPORT.md"),
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

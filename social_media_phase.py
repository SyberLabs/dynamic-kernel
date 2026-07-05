"""
Social Media Phase Diagram and Intervention ROI Harness.

Runs the SOCIAL_MEDIA adapter as an augmented feedback process over
(position, telemetry), then reports filter-bubble thresholds and intervention
cost effectiveness.

Usage:
    .venv\\Scripts\\python.exe social_media_phase.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from adapters import SOCIAL_MEDIA


RISK_NODES = ("Conflict Commentary", "Conspiracy/Rumor")
PROTECTIVE_NODES = ("Science Explainers", "Longform Off-Ramp", "Local News")
OFF_RAMP_NODE = "Longform Off-Ramp"
CONFLICT_FEATURE = "Conflict"
CREDIBILITY_FEATURE = "Credibility"


DEFAULT_LAMBDAS = [0.0, 0.10, 0.20, 0.35]
DEFAULT_TAUS = [0.5, 1.0, 1.5, 2.5]
DEFAULT_SIGMAS = [0.0, 0.03]
DEFAULT_INTENTS = ["High-Arousal Scroll", "Civic Learner", "Deep Research"]


@dataclass(frozen=True)
class SimulationConfig:
    intent: str
    feedback_rate: float
    temperature: float
    noise_sigma: float
    agents: int = 192
    steps: int = 64
    seed: int = 20260602


@dataclass(frozen=True)
class Intervention:
    name: str
    kind: str
    cost: float
    apply: Callable


def _idx(labels: list[str], names: tuple[str, ...] | list[str]) -> list[int]:
    return [labels.index(name) for name in names]


def _normalized_intent(intent: str) -> np.ndarray:
    telemetry = np.array(SOCIAL_MEDIA.intent_presets[intent], dtype=np.float64)
    norm = np.linalg.norm(telemetry)
    return telemetry / norm if norm > 0 else telemetry


def _seed_for(config: SimulationConfig, salt: int = 0) -> int:
    raw = (
        config.seed
        + 1009 * DEFAULT_INTENTS.index(config.intent)
        + 9173 * int(round(config.feedback_rate * 1000))
        + 6113 * int(round(config.temperature * 1000))
        + 7919 * int(round(config.noise_sigma * 1000))
        + salt
    )
    return int(raw % (2**32 - 1))


def build_kernel(config: SimulationConfig):
    return SOCIAL_MEDIA.build_kernel(
        feedback_rate=config.feedback_rate,
        feedback_noise=0.0,
        temperature=config.temperature,
    )


def social_interventions(labels: list[str]) -> list[Intervention]:
    def edge_cost(edge_names: list[tuple[str, str]], magnitude: float) -> float:
        return float(len(edge_names) * abs(magnitude))

    def beta_boost(edge_names: list[tuple[str, str]], boost: float):
        def apply(kernel):
            for source, target in edge_names:
                i, j = labels.index(source), labels.index(target)
                if kernel.topo.adjacency_mask[i, j]:
                    kernel.sponsor_edge(i, j, boost)
        return apply

    def friction_boost(edge_names: list[tuple[str, str]], reduction: float):
        def apply(kernel):
            for source, target in edge_names:
                i, j = labels.index(source), labels.index(target)
                if kernel.topo.adjacency_mask[i, j]:
                    kernel.sponsor_edge_friction(i, j, reduction)
        return apply

    off_ramp_edges = [
        ("Local News", "Longform Off-Ramp"),
        ("Civic Debate", "Longform Off-Ramp"),
        ("Conflict Commentary", "Longform Off-Ramp"),
        ("Conspiracy/Rumor", "Longform Off-Ramp"),
        ("Music & Culture", "Longform Off-Ramp"),
    ]
    science_edges = [
        ("Local News", "Science Explainers"),
        ("Civic Debate", "Science Explainers"),
        ("Conspiracy/Rumor", "Science Explainers"),
        ("Longform Off-Ramp", "Science Explainers"),
        ("Wellness", "Science Explainers"),
    ]
    adversarial_edges = [
        ("Civic Debate", "Conflict Commentary"),
        ("Conflict Commentary", "Conspiracy/Rumor"),
        ("Conspiracy/Rumor", "Conflict Commentary"),
    ]
    generic_protective = off_ramp_edges + science_edges

    return [
        Intervention("baseline", "baseline", 0.0, lambda kernel: None),
        Intervention(
            "off_ramp_beta",
            "safety",
            edge_cost(off_ramp_edges, 1.0),
            beta_boost(off_ramp_edges, 1.0),
        ),
        Intervention(
            "science_beta",
            "safety",
            edge_cost(science_edges, 1.0),
            beta_boost(science_edges, 1.0),
        ),
        Intervention(
            "off_ramp_friction",
            "safety",
            edge_cost(off_ramp_edges, 0.75),
            friction_boost(off_ramp_edges, 0.75),
        ),
        Intervention(
            "generic_diversity_beta",
            "safety",
            edge_cost(generic_protective, 0.35),
            beta_boost(generic_protective, 0.35),
        ),
        Intervention(
            "adversarial_rumor_beta",
            "adversarial",
            edge_cost(adversarial_edges, 2.0),
            beta_boost(adversarial_edges, 2.0),
        ),
    ]


def simulate(config: SimulationConfig, intervention: Intervention | None = None) -> dict:
    kernel = build_kernel(config)
    labels = kernel.topo.labels
    if intervention is not None:
        intervention.apply(kernel)

    rng = np.random.default_rng(_seed_for(config))
    n = kernel.topo.N
    risk_idx = np.array(_idx(labels, RISK_NODES), dtype=int)
    protective_idx = np.array(_idx(labels, PROTECTIVE_NODES), dtype=int)
    start = labels.index("Onboarding")

    positions = np.full(config.agents, start, dtype=int)
    telemetries = np.tile(_normalized_intent(config.intent), (config.agents, 1))

    conflict_i = SOCIAL_MEDIA.feature_labels.index(CONFLICT_FEATURE)
    credibility_i = SOCIAL_MEDIA.feature_labels.index(CREDIBILITY_FEATURE)

    risk_series = []
    protective_series = []
    escape_series = []
    drift_series = []
    edge_counts = np.zeros((n, n), dtype=np.float64)

    for step in range(config.steps):
        P_all = kernel.transition_matrix_batch(telemetries, step=step)
        rows = P_all[np.arange(config.agents), positions, :]

        in_risk = np.isin(positions, risk_idx)
        if np.any(in_risk):
            escape_series.append(float(np.mean(1.0 - rows[in_risk][:, risk_idx].sum(axis=1))))
        else:
            escape_series.append(1.0)

        cdf = np.cumsum(rows, axis=1)
        draws = rng.random((config.agents, 1))
        next_positions = np.argmax(cdf >= draws, axis=1)
        row_sums = rows.sum(axis=1)
        next_positions[row_sums < 1e-12] = positions[row_sums < 1e-12]

        np.add.at(edge_counts, (positions, next_positions), 1.0)

        visited = kernel.topo.node_features[next_positions]
        lam = config.feedback_rate
        telemetries = (1.0 - lam) * telemetries + lam * visited
        if config.noise_sigma > 0:
            telemetries += rng.normal(scale=config.noise_sigma, size=telemetries.shape)
        norms = np.linalg.norm(telemetries, axis=1, keepdims=True)
        telemetries = np.where(norms > 0, telemetries / norms, telemetries)

        positions = next_positions
        risk_series.append(float(np.mean(np.isin(positions, risk_idx))))
        protective_series.append(float(np.mean(np.isin(positions, protective_idx))))
        drift_series.append(float(np.mean(telemetries[:, conflict_i] - telemetries[:, credibility_i])))

    edge_flow = edge_counts / max(float(edge_counts.sum()), 1.0)
    edge_current = edge_flow - edge_flow.T
    eps = 1e-12
    rev = edge_flow.T
    mask = (edge_flow > eps) & (rev > eps)
    entropy_production = 0.0
    if np.any(mask):
        entropy_production = 0.5 * float(np.sum(edge_flow[mask] * np.log(edge_flow[mask] / rev[mask])))

    final_window = max(8, config.steps // 4)
    final_risk = float(np.mean(risk_series[-final_window:]))
    final_protective = float(np.mean(protective_series[-final_window:]))
    escape_probability = float(np.mean(escape_series[-final_window:]))
    conflict_credibility_drift = float(np.mean(drift_series[-final_window:]))

    lock_in = (
        final_risk >= 0.18
        and escape_probability <= 0.55
        and conflict_credibility_drift > -0.05
    )

    return {
        "intent": config.intent,
        "lambda": config.feedback_rate,
        "tau": config.temperature,
        "sigma": config.noise_sigma,
        "agents": config.agents,
        "steps": config.steps,
        "intervention": intervention.name if intervention else "baseline",
        "risk_share": final_risk,
        "protective_share": final_protective,
        "escape_probability": escape_probability,
        "conflict_credibility_drift": conflict_credibility_drift,
        "edge_current_norm": float(np.linalg.norm(edge_current, ord="fro")),
        "entropy_production": entropy_production,
        "lock_in": bool(lock_in),
        "risk_series": risk_series,
    }


def run_phase_grid(
    lambdas: list[float] | None = None,
    taus: list[float] | None = None,
    sigmas: list[float] | None = None,
    intents: list[str] | None = None,
    agents: int = 192,
    steps: int = 64,
    seed: int = 20260602,
) -> list[dict]:
    results = []
    for intent in intents or DEFAULT_INTENTS:
        for lam in lambdas or DEFAULT_LAMBDAS:
            for tau in taus or DEFAULT_TAUS:
                for sigma in sigmas or DEFAULT_SIGMAS:
                    config = SimulationConfig(
                        intent=intent,
                        feedback_rate=lam,
                        temperature=tau,
                        noise_sigma=sigma,
                        agents=agents,
                        steps=steps,
                        seed=seed,
                    )
                    results.append(simulate(config))
    return results


def run_interventions(
    config: SimulationConfig | None = None,
) -> list[dict]:
    if config is None:
        config = SimulationConfig(
            intent="High-Arousal Scroll",
            feedback_rate=0.35,
            temperature=0.5,
            noise_sigma=0.0,
            agents=256,
            steps=80,
        )
    labels = SOCIAL_MEDIA.build_topology().labels
    interventions = social_interventions(labels)
    rows = [simulate(config, intervention) for intervention in interventions]
    baseline = next(row for row in rows if row["intervention"] == "baseline")
    for row in rows:
        cost = next(i.cost for i in interventions if i.name == row["intervention"])
        row["cost"] = cost
        row["risk_delta_vs_baseline"] = row["risk_share"] - baseline["risk_share"]
        if row["intervention"] == "baseline" or cost <= 0:
            row["risk_reduction_per_cost"] = 0.0
        else:
            row["risk_reduction_per_cost"] = (baseline["risk_share"] - row["risk_share"]) / cost
    return rows


def summarize(grid: list[dict], interventions: list[dict]) -> dict:
    def compact(row: dict | None) -> dict | None:
        if row is None:
            return None
        return {key: value for key, value in row.items() if key != "risk_series"}

    lock_cells = [row for row in grid if row["lock_in"]]
    by_intent = {}
    for intent in sorted({row["intent"] for row in grid}):
        rows = [row for row in grid if row["intent"] == intent]
        by_intent[intent] = {
            "cells": len(rows),
            "lock_in_cells": int(sum(row["lock_in"] for row in rows)),
            "max_risk_share": float(max(row["risk_share"] for row in rows)),
            "min_escape_probability": float(min(row["escape_probability"] for row in rows)),
        }

    safety = [row for row in interventions if row["intervention"] != "baseline" and row["risk_reduction_per_cost"] > 0]
    safety_rank = sorted(safety, key=lambda row: row["risk_reduction_per_cost"], reverse=True)
    adversarial = [row for row in interventions if row["intervention"] == "adversarial_rumor_beta"]

    return {
        "grid_cells": len(grid),
        "lock_in_cells": len(lock_cells),
        "lock_in_rate": float(len(lock_cells) / max(len(grid), 1)),
        "by_intent": by_intent,
        "best_safety_intervention": compact(safety_rank[0]) if safety_rank else None,
        "adversarial_result": compact(adversarial[0]) if adversarial else None,
    }


def write_outputs(
    grid: list[dict],
    interventions: list[dict],
    output_json: Path = Path("social_media_phase_output.json"),
    output_md: Path = Path("SOCIAL_MEDIA_PHASE_REPORT.md"),
) -> dict:
    summary = summarize(grid, interventions)
    payload = {
        "summary": summary,
        "grid": grid,
        "interventions": interventions,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(summary, interventions), encoding="utf-8")
    return payload


def render_report(summary: dict, interventions: list[dict]) -> str:
    lines = [
        "# Social Media Phase Diagram Report",
        "",
        "## Summary",
        "",
        f"- Grid cells: {summary['grid_cells']}",
        f"- Lock-in cells: {summary['lock_in_cells']} ({summary['lock_in_rate']:.1%})",
        "",
        "## Intent Thresholds",
        "",
        "| Intent | Cells | Lock-In Cells | Max Risk Share | Min Escape Probability |",
        "|---|---:|---:|---:|---:|",
    ]
    for intent, row in summary["by_intent"].items():
        lines.append(
            f"| {intent} | {row['cells']} | {row['lock_in_cells']} | "
            f"{row['max_risk_share']:.3f} | {row['min_escape_probability']:.3f} |"
        )

    lines.extend([
        "",
        "## Intervention ROI",
        "",
        "| Intervention | Type | Cost | Risk Share | Risk Delta | Risk Reduction / Cost | Escape Probability |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    ranked = sorted(
        interventions,
        key=lambda row: row.get("risk_reduction_per_cost", 0.0),
        reverse=True,
    )
    for row in ranked:
        kind = "adversarial" if row["intervention"] == "adversarial_rumor_beta" else "safety"
        if row["intervention"] == "baseline":
            kind = "baseline"
        lines.append(
            f"| {row['intervention']} | {kind} | {row['cost']:.2f} | "
            f"{row['risk_share']:.3f} | {row['risk_delta_vs_baseline']:+.3f} | "
            f"{row['risk_reduction_per_cost']:.4f} | {row['escape_probability']:.3f} |"
        )

    best = summary.get("best_safety_intervention")
    if best:
        lines.extend([
            "",
            "## Proposition From This Run",
            "",
            (
                f"The highest risk-reduction-per-cost intervention is "
                f"`{best['intervention']}` at {best['risk_reduction_per_cost']:.4f}. "
                "This is a concrete DTE claim: topology-aware intervention beats generic "
                "diversity only if its risk reduction per unit cost is higher on the same graph."
            ),
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    grid = run_phase_grid()
    interventions = run_interventions()
    payload = write_outputs(grid, interventions)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

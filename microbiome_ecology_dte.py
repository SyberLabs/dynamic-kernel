"""
Abstract microbiome ecology prototype for DTE.

This module is not a medical model and does not make treatment claims. It is a
cybernetic ecology simulator for studying timing-sensitive interventions in a
gut-flora-like topology: antibiotic pressure, probiotic introduction,
prebiotic support, pathobiont lock-in, and recovery diagnostics.

Usage:
    .venv\\Scripts\\python.exe microbiome_ecology_dte.py --quick
    .venv\\Scripts\\python.exe microbiome_ecology_dte.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kernel import DynamicTopologyKernel, topology_from_edges


FEATURE_LABELS = [
    "Fiber Use",
    "Mucosal Fit",
    "Antibiotic Resilience",
    "Inflammation Tolerance",
    "Cross Feeding",
    "Acid Balance",
    "Probiotic Compatibility",
    "Pathobiont Virulence",
    "Recovery Support",
]


NODES = {
    "Luminal Nutrient Pool": [0.78, 0.20, 0.36, 0.30, 0.55, 0.42, 0.48, 0.12, 0.48],
    "Fiber Substrate": [1.00, 0.18, 0.28, 0.22, 0.84, 0.44, 0.56, 0.04, 0.68],
    "Fermenter Guild": [0.94, 0.42, 0.32, 0.28, 1.00, 0.62, 0.46, 0.05, 0.72],
    "Butyrate Producer Guild": [0.82, 0.62, 0.26, 0.20, 0.90, 0.88, 0.36, 0.02, 1.00],
    "Probiotic Guild": [0.70, 0.52, 0.22, 0.24, 0.58, 0.80, 1.00, 0.02, 0.78],
    "Mucosal Niche": [0.44, 1.00, 0.36, 0.34, 0.46, 0.62, 0.52, 0.10, 0.92],
    "Recovery Reservoir": [0.60, 0.74, 0.42, 0.28, 0.60, 0.66, 0.58, 0.04, 0.96],
    "Antibiotic Clearance": [0.08, 0.04, 0.06, 0.12, 0.05, 0.08, 0.02, 0.04, 0.04],
    "Inflammatory Pocket": [0.18, 0.34, 0.72, 1.00, 0.16, 0.12, 0.06, 0.74, 0.08],
    "Pathobiont Bloom": [0.16, 0.50, 0.86, 0.92, 0.12, 0.10, 0.02, 1.00, 0.02],
}


EDGES = [
    ("Luminal Nutrient Pool", "Fiber Substrate", 1.0),
    ("Fiber Substrate", "Fermenter Guild", 0.9),
    ("Fermenter Guild", "Butyrate Producer Guild", 1.0),
    ("Butyrate Producer Guild", "Mucosal Niche", 1.0),
    ("Mucosal Niche", "Recovery Reservoir", 1.2),
    ("Recovery Reservoir", "Luminal Nutrient Pool", 1.3),
    ("Luminal Nutrient Pool", "Probiotic Guild", 1.1),
    ("Probiotic Guild", "Fermenter Guild", 1.2),
    ("Probiotic Guild", "Mucosal Niche", 1.3),
    ("Antibiotic Clearance", "Luminal Nutrient Pool", 1.6),
    ("Antibiotic Clearance", "Recovery Reservoir", 2.0),
    ("Luminal Nutrient Pool", "Inflammatory Pocket", 1.3),
    ("Inflammatory Pocket", "Pathobiont Bloom", 0.8),
    ("Pathobiont Bloom", "Inflammatory Pocket", 0.8),
    ("Pathobiont Bloom", "Mucosal Niche", 1.6),
    ("Pathobiont Bloom", "Recovery Reservoir", 2.4),
    ("Mucosal Niche", "Inflammatory Pocket", 1.4),
    ("Inflammatory Pocket", "Recovery Reservoir", 2.0),
    ("Inflammatory Pocket", "Butyrate Producer Guild", 2.6),
    ("Fermenter Guild", "Inflammatory Pocket", 2.2),
    ("Butyrate Producer Guild", "Pathobiont Bloom", 2.6),
]


INTENTS = {
    "Commensal": [0.86, 0.62, 0.24, 0.18, 0.92, 0.76, 0.34, 0.02, 0.88],
    "Butyrate": [0.78, 0.70, 0.20, 0.10, 0.88, 1.00, 0.26, 0.00, 1.00],
    "Probiotic": [0.70, 0.58, 0.16, 0.16, 0.62, 0.86, 1.00, 0.00, 0.82],
    "Pathobiont": [0.12, 0.48, 0.96, 0.96, 0.10, 0.08, 0.00, 1.00, 0.00],
    "Stressed": [0.22, 0.18, 0.72, 0.66, 0.12, 0.12, 0.04, 0.44, 0.10],
}


BENEFICIAL_NODES = (
    "Fermenter Guild",
    "Butyrate Producer Guild",
    "Probiotic Guild",
    "Mucosal Niche",
    "Recovery Reservoir",
)
PATHOBIONT_NODES = ("Inflammatory Pocket", "Pathobiont Bloom")
CLEARANCE_NODE = "Antibiotic Clearance"


@dataclass(frozen=True)
class MicrobiomeConfig:
    agents: int = 144
    steps: int = 96
    seed: int = 202606
    alpha: float = 1.0
    beta_strength: float = 0.95
    feedback_rate: float = 0.14
    pathobiont_memory_rate: float = 0.10
    beneficial_memory_rate: float = 0.06
    memory_decay: float = 0.035


@dataclass(frozen=True)
class MicrobiomeScenario:
    name: str
    antibiotic_start: int
    antibiotic_end: int
    probiotic_start: int
    probiotic_duration: int
    probiotic_dose: int
    prebiotic_start: int
    prebiotic_duration: int
    prebiotic_strength: float
    antibiotic_strength: float
    initial_pathobiont_share: float
    notes: str = ""


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec if norm == 0.0 else vec / norm


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return arr / norms


def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    values = np.array([value / total for value in counts.values() if value > 0])
    if len(values) <= 1:
        return 0.0
    return float(-np.sum(values * np.log(values)) / math.log(len(counts)))


def topology():
    return topology_from_edges(
        nodes={label: np.array(vec, dtype=float) for label, vec in NODES.items()},
        edges=EDGES,
        undirected=False,
    )


def scenarios() -> list[MicrobiomeScenario]:
    return [
        MicrobiomeScenario(
            name="early_probiotic_washout",
            antibiotic_start=8,
            antibiotic_end=42,
            probiotic_start=14,
            probiotic_duration=16,
            probiotic_dose=5,
            prebiotic_start=44,
            prebiotic_duration=28,
            prebiotic_strength=0.55,
            antibiotic_strength=0.78,
            initial_pathobiont_share=0.10,
            notes="Probiotic arrives while antibiotic pressure remains high.",
        ),
        MicrobiomeScenario(
            name="on_phase_recovery",
            antibiotic_start=8,
            antibiotic_end=42,
            probiotic_start=44,
            probiotic_duration=24,
            probiotic_dose=8,
            prebiotic_start=42,
            prebiotic_duration=44,
            prebiotic_strength=1.15,
            antibiotic_strength=0.78,
            initial_pathobiont_share=0.07,
            notes="Probiotic arrives after clearance with prebiotic support.",
        ),
        MicrobiomeScenario(
            name="late_pathobiont_lockin",
            antibiotic_start=8,
            antibiotic_end=42,
            probiotic_start=72,
            probiotic_duration=18,
            probiotic_dose=5,
            prebiotic_start=42,
            prebiotic_duration=22,
            prebiotic_strength=0.45,
            antibiotic_strength=0.78,
            initial_pathobiont_share=0.16,
            notes="Probiotic arrives after pathobiont memory has had time to form.",
        ),
        MicrobiomeScenario(
            name="prebiotic_only_recovery",
            antibiotic_start=8,
            antibiotic_end=42,
            probiotic_start=200,
            probiotic_duration=0,
            probiotic_dose=0,
            prebiotic_start=42,
            prebiotic_duration=40,
            prebiotic_strength=0.85,
            antibiotic_strength=0.72,
            initial_pathobiont_share=0.10,
            notes="No probiotic packets; recovery depends on substrate support.",
        ),
    ]


def _build_kernel(config: MicrobiomeConfig) -> DynamicTopologyKernel:
    topo = topology()
    beta = np.full((topo.N, topo.N), config.beta_strength, dtype=float)
    node_bias = np.zeros(topo.N, dtype=float)
    node_bias[topo.labels.index("Luminal Nutrient Pool")] = 0.08
    node_bias[topo.labels.index("Recovery Reservoir")] = 0.05
    return DynamicTopologyKernel(
        topology=topo,
        alpha=config.alpha,
        beta=beta,
        feedback_rate=config.feedback_rate,
        temperature=1.0,
        feedback_noise=0.0,
        node_bias=node_bias,
    )


def _sample_next(rng: np.random.Generator, rows: np.ndarray) -> np.ndarray:
    u = rng.random(rows.shape[0])
    cdf = np.cumsum(rows, axis=1)
    return np.array(
        [min(int(np.searchsorted(cdf[i], u[i], side="right")), rows.shape[1] - 1) for i in range(rows.shape[0])],
        dtype=int,
    )


def _initial_population(
    rng: np.random.Generator,
    config: MicrobiomeConfig,
    scenario: MicrobiomeScenario,
    idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pathobiont_count = int(round(config.agents * scenario.initial_pathobiont_share))
    butyrate_count = int(round(config.agents * 0.24))
    commensal_count = config.agents - pathobiont_count - butyrate_count
    classes = (
        ["Pathobiont"] * pathobiont_count
        + ["Butyrate"] * butyrate_count
        + ["Commensal"] * commensal_count
    )
    rng.shuffle(classes)
    positions = np.empty(config.agents, dtype=int)
    for i, name in enumerate(classes):
        if name == "Pathobiont":
            positions[i] = rng.choice([idx["Inflammatory Pocket"], idx["Pathobiont Bloom"]])
        elif name == "Butyrate":
            positions[i] = rng.choice([idx["Fermenter Guild"], idx["Butyrate Producer Guild"]])
        else:
            positions[i] = rng.choice([idx["Luminal Nutrient Pool"], idx["Fiber Substrate"], idx["Mucosal Niche"]])
    telemetries = np.array([_normalize(np.array(INTENTS[name], dtype=float)) for name in classes])
    return positions, telemetries, classes


def _active(start: int, duration: int, step: int) -> bool:
    return duration > 0 and start <= step < start + duration


def _antibiotic_pressure(scenario: MicrobiomeScenario, step: int) -> float:
    if scenario.antibiotic_start <= step < scenario.antibiotic_end:
        return scenario.antibiotic_strength
    if step >= scenario.antibiotic_end:
        return scenario.antibiotic_strength * math.exp(-(step - scenario.antibiotic_end) / 4.0)
    return 0.0


def _node_bias_for_ecology(
    kernel: DynamicTopologyKernel,
    scenario: MicrobiomeScenario,
    step: int,
    pathobiont_share: float,
) -> np.ndarray:
    labels = kernel.topo.labels
    bias = np.zeros(kernel.topo.N, dtype=float)
    pressure = _antibiotic_pressure(scenario, step)
    prebiotic = scenario.prebiotic_strength if _active(scenario.prebiotic_start, scenario.prebiotic_duration, step) else 0.0

    for label in ("Fermenter Guild", "Butyrate Producer Guild", "Fiber Substrate"):
        bias[labels.index(label)] += 0.55 * prebiotic
    for label in ("Probiotic Guild", "Mucosal Niche", "Recovery Reservoir"):
        bias[labels.index(label)] += 0.25 * prebiotic

    for label in ("Fermenter Guild", "Butyrate Producer Guild", "Probiotic Guild"):
        bias[labels.index(label)] -= 0.45 * pressure
    bias[labels.index("Antibiotic Clearance")] += 0.95 * pressure
    bias[labels.index("Pathobiont Bloom")] += 0.25 * pressure + 0.45 * max(0.0, pathobiont_share - 0.32)
    bias[labels.index("Inflammatory Pocket")] += 0.20 * pressure + 0.35 * max(0.0, pathobiont_share - 0.32)
    return bias


def _memory_diagnostics(
    topo,
    telemetries: np.ndarray,
    positions: np.ndarray,
    preference_memory: np.ndarray,
    edge_counts: np.ndarray,
) -> dict[str, Any]:
    idx = {label: i for i, label in enumerate(topo.labels)}
    path_idx = [idx[name] for name in PATHOBIONT_NODES]
    bene_idx = [idx[name] for name in BENEFICIAL_NODES]
    edge_mask = np.isfinite(topo.distance_matrix) & (topo.distance_matrix > 0.0)

    structural_stale_flow = float(np.mean(np.isin(positions, path_idx)))
    path_incident = np.zeros_like(edge_mask, dtype=bool)
    for node_idx in path_idx:
        path_incident[:, node_idx] = edge_mask[:, node_idx]
        path_incident[node_idx, :] = edge_mask[node_idx, :]
    total_pref = float(np.sum(np.maximum(0.0, preference_memory[edge_mask])))
    path_pref = float(np.sum(np.maximum(0.0, preference_memory[path_incident])))
    preference_stale_concentration = path_pref / total_pref if total_pref > 0.0 else 0.0

    path_alignment = telemetries @ topo.node_features[path_idx].T
    bene_alignment = telemetries @ topo.node_features[bene_idx].T
    path_score = float(np.mean(np.max(path_alignment, axis=1)))
    bene_score = float(np.mean(np.max(bene_alignment, axis=1)))
    state_stale_alignment = path_score / max(path_score + bene_score, 1e-9)

    layer_scores = {
        "structural_memory": structural_stale_flow,
        "preference_memory": preference_stale_concentration,
        "state_memory": max(0.0, state_stale_alignment - 0.5) * 2.0,
    }
    dominant = max(layer_scores, key=layer_scores.get)
    if layer_scores[dominant] <= 0.0:
        dominant = "none"
    return {
        "structural_stale_flow": round(structural_stale_flow, 4),
        "preference_stale_concentration": round(preference_stale_concentration, 4),
        "state_stale_alignment": round(state_stale_alignment, 4),
        "dominant_stale_memory_layer": dominant,
    }


def _classify(row: dict[str, Any]) -> str:
    if row["antibiotic_overlap_fraction"] >= 0.45 and row["probiotic_survival_rate"] < 0.35:
        return "early_washout"
    if (
        row["pathobiont_at_probiotic_start"] >= 0.23
        and row["final_pathobiont_occupancy"] >= 0.23
        and row["final_beneficial_occupancy"] <= 0.55
    ):
        return "late_lockin"
    if row["final_beneficial_occupancy"] >= 0.58 and row["final_pathobiont_occupancy"] <= 0.22:
        return "on_phase_recovery"
    if row["final_pathobiont_occupancy"] >= 0.32:
        return "pathobiont_lockin"
    return "partial_recovery"


def simulate(
    config: MicrobiomeConfig,
    scenario: MicrobiomeScenario,
    seed_offset: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed + seed_offset)
    kernel = _build_kernel(config)
    topo = kernel.topo
    idx = {label: i for i, label in enumerate(topo.labels)}
    positions, telemetries, classes = _initial_population(rng, config, scenario, idx)

    beneficial_idx = np.array([idx[name] for name in BENEFICIAL_NODES])
    pathobiont_idx = np.array([idx[name] for name in PATHOBIONT_NODES])
    clearance = idx[CLEARANCE_NODE]
    probiotic_node = idx["Probiotic Guild"]

    preference_memory = np.zeros((topo.N, topo.N), dtype=float)
    edge_counts = np.zeros((topo.N, topo.N), dtype=int)
    introduced_probiotics = 0
    probiotic_survivors = 0
    antibiotic_overlap_steps = 0
    pathobiont_at_probiotic_start: float | None = None
    recovery_time: int | None = None
    series = []

    for step in range(config.steps):
        beneficial_share = float(np.mean(np.isin(positions, beneficial_idx)))
        pathobiont_share = float(np.mean(np.isin(positions, pathobiont_idx)))
        pressure = _antibiotic_pressure(scenario, step)
        probiotic_active = _active(scenario.probiotic_start, scenario.probiotic_duration, step)
        if probiotic_active and scenario.antibiotic_start <= step < scenario.antibiotic_end:
            antibiotic_overlap_steps += 1
        if step == scenario.probiotic_start:
            pathobiont_at_probiotic_start = pathobiont_share

        preference_memory *= max(0.0, 1.0 - config.memory_decay)
        if pathobiont_share > 0.30:
            for src, dst in (("Inflammatory Pocket", "Pathobiont Bloom"), ("Pathobiont Bloom", "Inflammatory Pocket")):
                preference_memory[idx[src], idx[dst]] += config.pathobiont_memory_rate * pathobiont_share
        if beneficial_share > 0.36:
            for src, dst in (("Fermenter Guild", "Butyrate Producer Guild"), ("Butyrate Producer Guild", "Mucosal Niche")):
                preference_memory[idx[src], idx[dst]] += config.beneficial_memory_rate * beneficial_share

        kernel._sponsor_friction = preference_memory.copy()
        kernel._node_bias = _node_bias_for_ecology(kernel, scenario, step, pathobiont_share)

        if probiotic_active and scenario.probiotic_dose > 0:
            candidates = np.where(
                np.isin(positions, [idx["Luminal Nutrient Pool"], idx["Antibiotic Clearance"], idx["Recovery Reservoir"]])
            )[0]
            if len(candidates) > 0:
                dose = min(scenario.probiotic_dose, len(candidates))
                selected = rng.choice(candidates, size=dose, replace=False)
                positions[selected] = probiotic_node
                for ant in selected:
                    classes[ant] = "Probiotic"
                    telemetries[ant] = _normalize(np.array(INTENTS["Probiotic"], dtype=float))
                introduced_probiotics += dose

        if pressure > 0.0:
            for ant, name in enumerate(classes):
                if name == "Pathobiont":
                    kill_prob = 0.025 * pressure
                elif name == "Probiotic":
                    kill_prob = 0.42 * pressure
                elif name == "Butyrate":
                    kill_prob = 0.34 * pressure
                else:
                    kill_prob = 0.28 * pressure
                if rng.random() < kill_prob:
                    positions[ant] = clearance
                    classes[ant] = "Stressed"
                    telemetries[ant] = _normalize(np.array(INTENTS["Stressed"], dtype=float))

        p_all = kernel.transition_matrix_batch(telemetries, step=step)
        previous = positions.copy()
        next_positions = _sample_next(rng, p_all[np.arange(config.agents), positions])
        for src, dst in zip(previous, next_positions):
            if src != dst and np.isfinite(topo.distance_matrix[int(src), int(dst)]):
                edge_counts[int(src), int(dst)] += 1
        positions = next_positions
        visited = topo.node_features[positions]
        telemetries = _normalize_rows((1.0 - config.feedback_rate) * telemetries + config.feedback_rate * visited)

        probiotic_survivors = sum(
            1 for ant, name in enumerate(classes) if name == "Probiotic" and positions[ant] != clearance
        )
        beneficial_share = float(np.mean(np.isin(positions, beneficial_idx)))
        pathobiont_share = float(np.mean(np.isin(positions, pathobiont_idx)))
        if recovery_time is None and step >= scenario.antibiotic_end:
            if beneficial_share >= 0.58 and pathobiont_share <= 0.18:
                recovery_time = step - scenario.antibiotic_end
        counts = {label: int(np.sum(positions == idx[label])) for label in topo.labels}
        series.append({
            "step": step,
            "beneficial_occupancy": beneficial_share,
            "pathobiont_occupancy": pathobiont_share,
            "diversity_entropy": _entropy(counts),
            "antibiotic_pressure": pressure,
            "probiotic_survivors": probiotic_survivors,
        })

    counts = {label: int(np.sum(positions == idx[label])) for label in topo.labels}
    final_beneficial = float(np.mean(np.isin(positions, beneficial_idx)))
    final_pathobiont = float(np.mean(np.isin(positions, pathobiont_idx)))
    overlap_fraction = antibiotic_overlap_steps / max(1, scenario.probiotic_duration)
    survival_rate = probiotic_survivors / max(1, introduced_probiotics)
    phase_error = max(overlap_fraction, pathobiont_at_probiotic_start or 0.0)
    memory = _memory_diagnostics(topo, telemetries, positions, preference_memory, edge_counts)

    row = {
        "scenario": scenario.name,
        "agents": config.agents,
        "steps": config.steps,
        "introduced_probiotics": int(introduced_probiotics),
        "probiotic_survivors": int(probiotic_survivors),
        "probiotic_survival_rate": round(float(survival_rate), 4),
        "antibiotic_overlap_fraction": round(float(overlap_fraction), 4),
        "pathobiont_at_probiotic_start": round(float(pathobiont_at_probiotic_start or 0.0), 4),
        "intervention_phase_error": round(float(phase_error), 4),
        "final_beneficial_occupancy": round(final_beneficial, 4),
        "final_pathobiont_occupancy": round(final_pathobiont, 4),
        "final_diversity_entropy": round(_entropy(counts), 4),
        "recovery_time": recovery_time,
        "memory_staleness": memory,
        "dominant_stale_memory_layer": memory["dominant_stale_memory_layer"],
        "classification": "",
        "series": series,
    }
    row["classification"] = _classify(row)
    return row


def run_pilot(config: MicrobiomeConfig | None = None, quick: bool = False) -> dict[str, Any]:
    config = config or (MicrobiomeConfig(agents=96, steps=96) if quick else MicrobiomeConfig())
    selected = scenarios()[:3] if quick else scenarios()
    rows = [simulate(config, scenario, seed_offset=i * 997) for i, scenario in enumerate(selected)]
    summary = {
        "runs": len(rows),
        "classification_counts": {
            label: sum(row["classification"] == label for row in rows)
            for label in sorted({row["classification"] for row in rows})
        },
        "best_recovery": max(rows, key=lambda row: row["final_beneficial_occupancy"]),
        "worst_lockin": max(rows, key=lambda row: row["final_pathobiont_occupancy"]),
        "lowest_phase_error": min(rows, key=lambda row: row["intervention_phase_error"]),
    }
    return {
        "config": config.__dict__ | {"quick": quick},
        "feature_labels": FEATURE_LABELS,
        "scenarios": [scenario.__dict__ for scenario in selected],
        "summary": summary,
        "rows": rows,
    }


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Microbiome Ecology DTE Prototype",
        "",
        "## Scope",
        "",
        (
            "Abstract, non-medical simulation of timing-sensitive microbiome ecology. "
            "Microbial packets circulate through functional niches under antibiotic "
            "pressure, probiotic introduction, prebiotic substrate support, and "
            "pathobiont memory."
        ),
        "",
        f"- Runs: `{summary['runs']}`",
        f"- Best recovery: `{summary['best_recovery']['scenario']}` "
        f"beneficial={summary['best_recovery']['final_beneficial_occupancy']:.3f}",
        f"- Worst lock-in: `{summary['worst_lockin']['scenario']}` "
        f"pathobiont={summary['worst_lockin']['final_pathobiont_occupancy']:.3f}",
        "",
        "## Classification Counts",
        "",
    ]
    for label, count in sorted(summary["classification_counts"].items()):
        lines.append(f"- `{label}`: `{count}`")

    lines.extend(
        [
            "",
            "## Result Table",
            "",
            "| Scenario | Class | Phase error | Probiotic survival | Beneficial | Pathobiont | Diversity | Recovery | Stale layer |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        recovery = "n/a" if row["recovery_time"] is None else str(row["recovery_time"])
        lines.append(
            f"| {row['scenario']} | {row['classification']} | "
            f"{row['intervention_phase_error']:.3f} | {row['probiotic_survival_rate']:.3f} | "
            f"{row['final_beneficial_occupancy']:.3f} | {row['final_pathobiont_occupancy']:.3f} | "
            f"{row['final_diversity_entropy']:.3f} | {recovery} | {row['dominant_stale_memory_layer']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The same probiotic-like intervention changes meaning with phase. During "
                "antibiotic pressure it can wash out; after clearance with substrate "
                "support it can aid recovery; after pathobiont memory forms it may be "
                "too late to dislodge the basin. This is the microbiome version of "
                "memory-ecology mismatch: intervention content is not enough; timing "
                "relative to ecological state and memory state matters."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], json_path: Path, report_path: Path) -> None:
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run abstract microbiome DTE ecology prototype.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", default="microbiome_ecology_output.json")
    parser.add_argument("--report", default="MICROBIOME_ECOLOGY_REPORT.md")
    args = parser.parse_args()

    payload = run_pilot(quick=args.quick)
    write_outputs(payload, Path(args.json), Path(args.report))
    print(render_report(payload))


if __name__ == "__main__":
    main()

"""
Rigorous reassessment of the current Neural DTE surface.

The current NeuralPort is a symmetry/beta optimizer over an abstract dense
topology. This script tests whether that surface has enough semantics to count
as a serious DTE application, or whether it should be reclassified as an
advanced optimizer instrument.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kernel import DynamicTopologyKernel, topology_from_edges
from optimizer import SymmetryMode, SymmetryOptimizer


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "neural_reassessment_output.json"
REPORT_PATH = ROOT / "NEURAL_DTE_REASSESSMENT.md"


def build_current_neural_kernel(
    n: int = 12,
    density: float = 1.0,
    directed: bool = False,
    beta_init: float = 3.0,
    seed: int = 42,
) -> DynamicTopologyKernel:
    """Reproduce the current API/adapter Neural construction."""
    feature_count = 4
    nodes = {
        f"Neuron {i + 1}": np.full(feature_count, 1.0 / feature_count)
        for i in range(n)
    }
    rng = np.random.default_rng(seed)
    edges: list[tuple[str, str, float]] = []
    if directed:
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() <= density:
                    edges.append((f"Neuron {i + 1}", f"Neuron {j + 1}", 1.0))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() <= density:
                    edges.append((f"Neuron {i + 1}", f"Neuron {j + 1}", 1.0))
    if not edges:
        edges = [
            (f"Neuron {i + 1}", f"Neuron {((i + 1) % n) + 1}", 1.0)
            for i in range(n)
        ]
    topo = topology_from_edges(nodes, edges, undirected=not directed)
    beta = np.zeros((topo.N, topo.N), dtype=np.float64)
    beta[topo.adjacency_mask] = float(beta_init)
    return DynamicTopologyKernel(
        topology=topo,
        alpha=1.0,
        beta=beta,
        feedback_rate=0.0,
        temperature=1.0,
        feedback_noise=0.0,
        node_bias=np.zeros(topo.N),
    )


def run_mode_probe(
    mode: SymmetryMode,
    n: int,
    density: float,
    directed: bool,
    steps: int = 120,
) -> dict[str, Any]:
    kernel = build_current_neural_kernel(n=n, density=density, directed=directed)
    optimizer = SymmetryOptimizer(kernel, mode=mode, noise_sigma=0.0)
    start_sigma = optimizer._eval_sigma(kernel._beta)
    for _ in range(steps):
        optimizer.step()
    end_sigma = optimizer._eval_sigma(kernel._beta)
    frame = optimizer.snapshot()
    active_beta = kernel._beta[kernel.topo.adjacency_mask]
    return {
        "mode": mode.value,
        "steps": steps,
        "start_sigma": float(start_sigma),
        "end_sigma": float(end_sigma),
        "delta_sigma": float(end_sigma - start_sigma),
        "beta_std": float(active_beta.std()) if active_beta.size else 0.0,
        "pi_std": float(np.std(frame["pi"])),
        "mixing_time": float(frame["mixing_time"]),
        "health": frame["health"],
    }


def run_composite_probe(steps: int = 120) -> dict[str, Any]:
    kernel = build_current_neural_kernel(n=12, density=1.0, directed=False)
    target = np.full(kernel.topo.N, 0.6 / (kernel.topo.N - 1))
    target[0] = 0.4
    optimizer = SymmetryOptimizer(
        kernel,
        mode=SymmetryMode.COMPOSITE,
        target_pi=target,
        composite_lambda=0.0,
        noise_sigma=0.0,
    )
    start_sigma = optimizer._eval_sigma(kernel._beta)
    for _ in range(steps):
        optimizer.step()
    end_sigma = optimizer._eval_sigma(kernel._beta)
    frame = optimizer.snapshot()
    return {
        "mode": "COMPOSITE_target_node_1",
        "steps": steps,
        "delta_sigma": float(end_sigma - start_sigma),
        "pi_node_1": float(frame["pi"][0]),
        "target_node_1": 0.4,
        "target_feasibility": frame["target_feasibility"],
    }


def run_reassessment() -> dict[str, Any]:
    cases = [
        ("complete_undirected", {"n": 12, "density": 1.0, "directed": False}),
        ("sparse_undirected", {"n": 12, "density": 0.35, "directed": False}),
        ("sparse_directed", {"n": 12, "density": 0.35, "directed": True}),
    ]
    modes = [
        SymmetryMode.ENTROPY_PI,
        SymmetryMode.ROW_ENTROPY,
        SymmetryMode.DETAILED_BALANCE,
        SymmetryMode.SPECTRAL_GAP,
        SymmetryMode.WEIGHT_SYMMETRY,
    ]
    mode_rows = []
    for label, kwargs in cases:
        for mode in modes:
            row = run_mode_probe(mode=mode, **kwargs)
            row["case"] = label
            mode_rows.append(row)
    payload = {
        "mode_rows": mode_rows,
        "composite_probe": run_composite_probe(),
        "verdict": "advanced_optimizer_instrument_not_validated_neural_application",
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(render_report(payload), encoding="utf-8")
    return payload


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Neural DTE Reassessment",
        "",
        "## Verdict",
        "",
        "The current Neural surface should be reclassified as an advanced optimizer",
        "instrument, not as a validated neural-computation application. It is useful",
        "for beta dynamics, phase chaining, feasibility reporting, gradient inspection,",
        "and topology stress tests. It does not yet model task performance, module",
        "specialization, loss reduction, activation routing, or adaptive computation.",
        "",
        "## Current Construction",
        "",
        "- Nodes are abstract neurons with identical feature vectors.",
        "- Edges are dense or random unit-distance links.",
        "- Telemetry is frozen in the optimizer backend.",
        "- Beta is optimized against stationary-distribution symmetry objectives.",
        "- There is no reward/ecology layer and no task-conditioned input stream.",
        "",
        "## Probe Results",
        "",
        "| Case | Mode | Delta Sigma | Beta Std | Pi Std | Mixing | Health |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["mode_rows"]:
        lines.append(
            f"| {row['case']} | {row['mode']} | {row['delta_sigma']:.6f} | "
            f"{row['beta_std']:.4f} | {row['pi_std']:.4f} | "
            f"{row['mixing_time']:.2f} | {row['health']} |"
        )
    comp = payload["composite_probe"]
    feas = comp["target_feasibility"]
    lines.extend(
        [
            "",
            "## Composite Feasibility Probe",
            "",
            (
                f"Targeting node 1 at probability `{comp['target_node_1']:.2f}` "
                f"on the complete current Neural graph reached pi_1 "
                f"`{comp['pi_node_1']:.4f}` after {comp['steps']} steps."
            ),
            (
                f"The feasibility probe classifies the target as `{feas['status']}` "
                f"with L1 error `{feas['l1_error']:.4f}`."
            ),
            "",
            "## Interpretation",
            "",
            "1. The complete undirected default is exactly symmetric, so all symmetry",
            "   objectives stall. This is mathematically expected, not a frontend bug.",
            "",
            "2. Sparse topology introduces structural asymmetry, but most objectives",
            "   still move weakly because all node features and distances remain equal.",
            "",
            "3. Directed sparse topology gives WEIGHT_SYMMETRY a real signal because",
            "   beta can repair directional asymmetry. This makes Neural a useful",
            "   optimizer stress test, but not yet a neural-domain model.",
            "",
            "4. COMPOSITE targeting can partially move stationary mass, but this is",
            "   routing-control over a graph, not learning or inference.",
            "",
            "## What Would Make It A Serious Neural Application",
            "",
            "Reframe nodes as modules or experts, not neurons. Then define:",
            "",
            "- node features: module capabilities, cost, latency, modality, specialization;",
            "- telemetry: task or input embedding;",
            "- reward: loss reduction, confidence gain, energy efficiency, or latency-adjusted utility;",
            "- beta memory: learned routing preference between modules;",
            "- local regret: traffic sent to a weaker module while a better reachable module exists;",
            "- ecology: task distribution drift, module degradation, or compute budget shocks.",
            "",
            "The resulting object would be closer to mixture-of-experts routing or",
            "adaptive-computation governance than to biological neural dynamics.",
            "",
            "## Recommendation",
            "",
            "Keep NeuralPort, but relabel it as `Optimizer Lab` or `Neural Routing Lab`.",
            "Do not use it as the paper's main proof demo yet. Use it as the advanced",
            "instrument surface after the ant/local-regret paper demo is clear.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_reassessment()
    print(json.dumps(result["composite_probe"], indent=2))

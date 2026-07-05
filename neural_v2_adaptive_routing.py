"""
Neural V2: adaptive computation routing.

This module intentionally starts clean. It does not model individual neurons.
It models routing among computational modules / experts under task-distribution
drift, performance feedback, and stale routing memory.

The witness:

1. Phase A: tasks are language-heavy, so routing memory learns Language Expert.
2. Phase B: tasks become symbolic-heavy. Language remains mildly useful, so
   stale routing is not catastrophic.
3. Local regret detects that Symbolic Expert is now the better reachable module
   and evaporates stale Language memory faster than surprise/base forgetting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from kernel import DynamicTopologyKernel, topology_from_edges


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "neural_v2_adaptive_routing_output.json"
REPORT_PATH = ROOT / "NEURAL_V2_ADAPTIVE_ROUTING_REPORT.md"

ROUTER = 0
LANGUAGE = 1
SYMBOLIC = 2
VISION = 3
GENERALIST = 4
OUTPUT = 5
MODULE_INDICES = [LANGUAGE, SYMBOLIC, VISION, GENERALIST]

FEATURE_LABELS = [
    "vision_demand",
    "language_demand",
    "symbolic_demand",
    "uncertainty",
    "latency_sensitivity",
    "accuracy_requirement",
]


@dataclass(frozen=True)
class NeuralV2Condition:
    label: str
    surprise_gain: float
    opportunity_gain: float


DEFAULT_CONDITIONS = [
    NeuralV2Condition("base_forgetting", surprise_gain=0.0, opportunity_gain=0.0),
    NeuralV2Condition("surprise_only", surprise_gain=0.2, opportunity_gain=0.0),
    NeuralV2Condition("local_regret", surprise_gain=0.2, opportunity_gain=4.0),
]


def _normalize(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def module_feature_table() -> dict[str, np.ndarray]:
    return {
        "Task Router": _normalize([0.20, 0.20, 0.20, 0.20, 0.20, 0.20]),
        "Language Expert": _normalize([0.05, 1.00, 0.25, 0.55, 0.35, 0.90]),
        "Symbolic Expert": _normalize([0.05, 0.25, 1.00, 0.85, 0.20, 1.00]),
        "Vision Expert": _normalize([1.00, 0.10, 0.10, 0.50, 0.35, 0.80]),
        "Fast Generalist": _normalize([0.45, 0.45, 0.38, 0.35, 0.95, 0.45]),
        "Output": _normalize([0.20, 0.20, 0.20, 0.20, 0.20, 0.20]),
    }


def task_templates() -> dict[str, np.ndarray]:
    return {
        "language": _normalize([0.00, 1.00, 0.10, 0.30, 0.25, 0.90]),
        "symbolic": _normalize([0.00, 0.20, 1.00, 0.80, 0.10, 1.00]),
        "vision": _normalize([1.00, 0.10, 0.10, 0.40, 0.30, 0.80]),
        "general": _normalize([0.35, 0.35, 0.35, 0.30, 0.85, 0.45]),
    }


def build_neural_v2_kernel(condition: NeuralV2Condition) -> DynamicTopologyKernel:
    nodes = module_feature_table()
    edges = [
        ("Task Router", "Language Expert", 1.0),
        ("Task Router", "Symbolic Expert", 1.0),
        ("Task Router", "Vision Expert", 1.0),
        ("Task Router", "Fast Generalist", 1.0),
        ("Language Expert", "Output", 1.0),
        ("Symbolic Expert", "Output", 1.0),
        ("Vision Expert", "Output", 1.0),
        ("Fast Generalist", "Output", 1.0),
    ]
    topo = topology_from_edges(nodes=nodes, edges=edges, undirected=False)
    kernel = DynamicTopologyKernel(
        topology=topo,
        alpha=1.0,
        beta=0.45,
        temperature=0.30,
        feedback_rate=0.0,
        feedback_noise=0.0,
    )
    kernel.configure_memory_law(
        mode="adaptive_eta",
        channel="friction",
        rho=0.55,
        eta=0.003,
        eta_max=0.55,
        surprise_gain=condition.surprise_gain,
        opportunity_gain=condition.opportunity_gain,
        reward_track_rate=0.08,
        initial_expectation=0.55,
    )
    return kernel


def sample_task_batch(
    rng: np.random.Generator,
    tick: int,
    batch_size: int,
    shift_tick: int,
) -> tuple[np.ndarray, np.ndarray]:
    templates = task_templates()
    names = list(templates)
    if tick < shift_tick:
        probs = [0.82, 0.04, 0.04, 0.10]
    else:
        probs = [0.08, 0.82, 0.04, 0.06]
    sampled = rng.choice(names, size=batch_size, p=probs)
    return np.array([templates[name] for name in sampled]), sampled


def reward_matrix(task_batch: np.ndarray, kernel: DynamicTopologyKernel) -> np.ndarray:
    features = kernel.topo.node_features
    demand = task_batch[:, :3]
    demand = demand / np.maximum(np.linalg.norm(demand, axis=1, keepdims=True), 1e-12)
    skill = features[:, :3]
    skill = skill / np.maximum(np.linalg.norm(skill, axis=1, keepdims=True), 1e-12)
    match = demand @ skill.T

    uncertainty = task_batch[:, 3:4]
    latency_sensitivity = task_batch[:, 4:5]
    accuracy_requirement = task_batch[:, 5:6]
    reliability = features[:, 3][np.newaxis, :]

    latency = np.array([0.0, 0.35, 0.55, 0.45, 0.08, 0.0])
    compute_cost = np.array([0.0, 0.25, 0.35, 0.30, 0.05, 0.0])
    rewards = (
        0.82 * accuracy_requirement * match
        + 0.24 * uncertainty * reliability
        - 0.25 * latency_sensitivity * latency[np.newaxis, :]
        - 0.08 * compute_cost[np.newaxis, :]
    )
    rewards[:, [ROUTER, OUTPUT]] = 0.0
    return np.clip(rewards, 0.0, 1.0)


def run_condition(
    condition: NeuralV2Condition,
    seed: int,
    ticks: int = 120,
    shift_tick: int = 50,
    batch_size: int = 240,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    kernel = build_neural_v2_kernel(condition)
    history: list[dict[str, Any]] = []

    for tick in range(ticks):
        tasks, task_names = sample_task_batch(rng, tick, batch_size, shift_tick)
        P_all = kernel.transition_matrix_batch(tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        cdf = np.cumsum(rows, axis=1)
        destinations = np.argmax(cdf >= rng.random((batch_size, 1)), axis=1)

        rewards = reward_matrix(tasks, kernel)
        chosen_reward = rewards[np.arange(batch_size), destinations]
        optimal_reward = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)

        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(batch_size, ROUTER, dtype=int), destinations),
            1.0 / batch_size,
        )
        update = kernel.memory_law_step(traffic, node_reward=node_reward)
        opportunity = update.get("opportunity_cost")

        history.append(
            {
                "tick": tick,
                "phase": "language" if tick < shift_tick else "symbolic",
                "mean_reward": float(chosen_reward.mean()),
                "mean_optimal_reward": float(optimal_reward.mean()),
                "mean_regret": float((optimal_reward - chosen_reward).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "vision_share": float(np.mean(destinations == VISION)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
                "language_memory": float(kernel._sponsor_friction[ROUTER, LANGUAGE]),
                "symbolic_memory": float(kernel._sponsor_friction[ROUTER, SYMBOLIC]),
                "language_eta": (
                    None
                    if kernel._last_eta_effective is None
                    else float(kernel._last_eta_effective[LANGUAGE])
                ),
                "mean_opportunity_cost": (
                    None if opportunity is None else opportunity["mean_opportunity_cost"]
                ),
                "task_mix": {
                    name: float(np.mean(task_names == name)) for name in task_templates()
                },
            }
        )

    post = history[shift_tick + 20 :]
    if not post:
        post = history[shift_tick:]
    recovery_tick = None
    for row in history[shift_tick:]:
        if row["symbolic_share"] > row["language_share"]:
            recovery_tick = row["tick"]
            break

    return {
        "condition": condition.label,
        "seed": seed,
        "ticks": ticks,
        "shift_tick": shift_tick,
        "post_shift_mean_reward": mean(row["mean_reward"] for row in post),
        "post_shift_mean_regret": mean(row["mean_regret"] for row in post),
        "post_shift_language_share": mean(row["language_share"] for row in post),
        "post_shift_symbolic_share": mean(row["symbolic_share"] for row in post),
        "recovery_tick": recovery_tick,
        "final_language_memory": history[-1]["language_memory"],
        "final_symbolic_memory": history[-1]["symbolic_memory"],
        "history": history,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)

    summary = {}
    for label, group in grouped.items():
        recovery_values = [
            row["recovery_tick"] for row in group if row["recovery_tick"] is not None
        ]
        summary[label] = {
            "runs": len(group),
            "mean_post_shift_reward": mean(row["post_shift_mean_reward"] for row in group),
            "mean_post_shift_regret": mean(row["post_shift_mean_regret"] for row in group),
            "mean_post_shift_language_share": mean(
                row["post_shift_language_share"] for row in group
            ),
            "mean_post_shift_symbolic_share": mean(
                row["post_shift_symbolic_share"] for row in group
            ),
            "mean_final_language_memory": mean(row["final_language_memory"] for row in group),
            "mean_final_symbolic_memory": mean(row["final_symbolic_memory"] for row in group),
            "mean_recovery_tick": (
                mean(recovery_values) if recovery_values else None
            ),
        }
    return summary


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Neural V2 Adaptive Routing Report",
        "",
        "## Scope",
        "",
        "Neural V2 models adaptive routing among computational modules / experts,",
        "not biological neurons. Tasks are routed from a task router to language,",
        "symbolic, vision, or fast-generalist modules. The task stream shifts from",
        "language-heavy to symbolic-heavy, testing stale routing memory.",
        "",
        "## Result",
        "",
        "| Condition | Runs | Reward | Regret | Language Share | Symbolic Share | Lang Memory | Sym Memory | Recovery Tick |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ["base_forgetting", "surprise_only", "local_regret"]:
        row = summary[label]
        recovery = (
            "n/a" if row["mean_recovery_tick"] is None else f"{row['mean_recovery_tick']:.1f}"
        )
        lines.append(
            f"| {label} | {row['runs']} | "
            f"{row['mean_post_shift_reward']:.3f} | "
            f"{row['mean_post_shift_regret']:.3f} | "
            f"{row['mean_post_shift_language_share']:.3f} | "
            f"{row['mean_post_shift_symbolic_share']:.3f} | "
            f"{row['mean_final_language_memory']:.3f} | "
            f"{row['mean_final_symbolic_memory']:.3f} | {recovery} |"
        )

    base = summary["surprise_only"]
    local = summary["local_regret"]
    regret_drop = base["mean_post_shift_regret"] - local["mean_post_shift_regret"]
    symbolic_gain = (
        local["mean_post_shift_symbolic_share"]
        - base["mean_post_shift_symbolic_share"]
    )
    language_drop = (
        base["mean_post_shift_language_share"]
        - local["mean_post_shift_language_share"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"Against surprise-only adaptation, local regret reduces post-shift "
                f"mean regret by `{regret_drop:.3f}`."
            ),
            (
                f"It increases symbolic-expert routing by `{symbolic_gain:.3f}` "
                f"and decreases stale language routing by `{language_drop:.3f}`."
            ),
            "",
            "This is the Neural V2 claim in miniature: a computational router can",
            "retain stale preference memory for a once-good module after the task",
            "ecology shifts. Local regret supplies the missing counterfactual signal:",
            "the chosen module still works, but a better reachable module exists.",
            "",
            "## V2 Status",
            "",
            "This is a rigorous minimal witness, not a neural-network benchmark.",
            "The next maturity step is to replace synthetic task rewards with real",
            "module performance traces: loss reduction, confidence gain, latency,",
            "or cost on a task suite.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(seeds: list[int] | None = None) -> dict[str, Any]:
    seeds = list(range(5)) if seeds is None else seeds
    rows = []
    for seed in seeds:
        for condition in DEFAULT_CONDITIONS:
            rows.append(run_condition(condition, seed=seed))
    payload = {
        "feature_labels": FEATURE_LABELS,
        "conditions": [condition.__dict__ for condition in DEFAULT_CONDITIONS],
        "seeds": seeds,
        "rows": rows,
        "summary": summarize(rows),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(render_report(payload), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result["summary"], indent=2))

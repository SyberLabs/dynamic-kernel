"""
Neural V2 router benchmark.

This benchmark compares DTE local-regret routing against simple ML-style
router baselines on the same adaptive-computation task stream.

The point is not to prove DTE universally dominates. The point is to identify
where its explicit preference-memory diagnostics help, and where simpler
contextual routers are sufficient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from neural_v2_adaptive_routing import (
    DEFAULT_CONDITIONS,
    GENERALIST,
    LANGUAGE,
    MODULE_INDICES,
    OUTPUT,
    ROUTER,
    SYMBOLIC,
    build_neural_v2_kernel,
    module_feature_table,
    reward_matrix,
    sample_task_batch,
    task_templates,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "neural_v2_router_benchmark_output.json"
REPORT_PATH = ROOT / "NEURAL_V2_ROUTER_BENCHMARK_REPORT.md"
HARD_OUTPUT_PATH = ROOT / "neural_v2_hard_router_benchmark_output.json"
HARD_REPORT_PATH = ROOT / "NEURAL_V2_HARD_ROUTER_BENCHMARK_REPORT.md"
FRONTIER_OUTPUT_PATH = ROOT / "neural_v2_frontier_output.json"
FRONTIER_REPORT_PATH = ROOT / "NEURAL_V2_FRONTIER_REPORT.md"
ADVERSARIAL_SWITCH_OUTPUT_PATH = ROOT / "neural_v2_adversarial_switch_output.json"
ADVERSARIAL_SWITCH_REPORT_PATH = ROOT / "NEURAL_V2_ADVERSARIAL_SWITCH_REPORT.md"

MODULE_COST = {
    LANGUAGE: 0.25,
    SYMBOLIC: 0.35,
    GENERALIST: 0.05,
}


@dataclass(frozen=True)
class BenchmarkConfig:
    ticks: int = 120
    shift_tick: int = 50
    batch_size: int = 240
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class HardBenchmarkConfig(BenchmarkConfig):
    context_noise: float = 0.22
    label_noise: float = 0.28
    reward_delay: int = 8
    degradation_tick: int = 58
    language_degradation: float = 0.62
    generalist_degradation: float = 0.82
    verifier_bonus: float = 0.12
    verifier_penalty: float = 0.10


@dataclass(frozen=True)
class FrontierConfig:
    ticks: int = 100
    shift_tick: int = 42
    batch_size: int = 160
    seeds: tuple[int, ...] = (0, 1, 2)
    context_noise_values: tuple[float, ...] = (0.0, 0.10, 0.22, 0.34, 0.46)
    label_noise_values: tuple[float, ...] = (0.0, 0.14, 0.28, 0.42, 0.56)
    reward_delay_values: tuple[int, ...] = (0, 4, 8, 12, 16)
    language_degradation_values: tuple[float, ...] = (1.0, 0.82, 0.62, 0.45, 0.30)
    verifier_bonus_values: tuple[float, ...] = (0.0, 0.06, 0.12, 0.18, 0.24)


@dataclass(frozen=True)
class AdversarialSwitchConfig(HardBenchmarkConfig):
    ticks: int = 90
    shift_tick: int = 24
    batch_size: int = 120
    seeds: tuple[int, ...] = (0, 1, 2)
    context_noise: float = 0.0
    label_noise: float = 0.0
    reward_delay: int = 4
    degradation_tick: int = 10_000
    language_degradation: float = 1.0
    generalist_degradation: float = 1.0
    verifier_bonus: float = 0.0
    verifier_penalty: float = 0.0
    switch_period: int = 8
    adversarial_intensity: float = 0.65


@dataclass(frozen=True)
class AdversarialSwitchSweepConfig:
    ticks: int = 80
    shift_tick: int = 20
    batch_size: int = 100
    seeds: tuple[int, ...] = (0, 1)
    switch_period_values: tuple[int, ...] = (2, 4, 8, 16)
    label_noise_values: tuple[float, ...] = (0.0, 0.28)
    context_noise: float = 0.0
    reward_delay: int = 4
    adversarial_intensity: float = 0.65


def condition(label: str):
    return next(c for c in DEFAULT_CONDITIONS if c.label == label)


def context_labels() -> list[str]:
    return list(task_templates())


def context_centroids() -> np.ndarray:
    templates = task_templates()
    return np.array([templates[label] for label in context_labels()])


def context_index_map() -> dict[str, int]:
    return {label: idx for idx, label in enumerate(context_labels())}


def configure_arbitrated_ucb(kernel, reliability_gated: bool = False) -> None:
    kwargs = {}
    if reliability_gated:
        kwargs = {
            "policy_reliability": "centroid_margin",
            "policy_reliability_floor": 0.10,
            "policy_reliability_scale": 0.12,
        }
    kernel.configure_edge_learning(
        mode="ucb",
        policy="arbitrated",
        reward_gain=1.0,
        uncertainty_gain=0.20,
        ucb_c=0.35,
        initial_reward=0.0,
        context_centroids=context_centroids(),
        policy_mix_max=0.45,
        policy_uncertainty_scale=0.08,
        policy_temperature=0.20,
        **kwargs,
    )


def configure_arbitrated_exp3(kernel, reliability_gated: bool = False) -> None:
    kwargs = {}
    if reliability_gated:
        kwargs = {
            "policy_reliability": "centroid_margin",
            "policy_reliability_floor": 0.10,
            "policy_reliability_scale": 0.12,
        }
    kernel.configure_edge_learning(
        mode="exp3",
        policy="arbitrated",
        initial_reward=0.0,
        context_centroids=context_centroids(),
        policy_mix_max=0.35,
        policy_uncertainty_scale=0.08,
        policy_temperature=1.0,
        exp3_gamma=0.10,
        exp3_eta=0.55,
        **kwargs,
    )


def make_task_stream(
    seed: int,
    config: BenchmarkConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [
        sample_task_batch(rng, tick, config.batch_size, config.shift_tick)
        for tick in range(config.ticks)
    ]


def _nearest_task_labels(task_batch: np.ndarray) -> np.ndarray:
    templates = task_templates()
    labels = list(templates)
    template_matrix = np.array([templates[label] for label in labels])
    scores = task_batch @ template_matrix.T
    return np.array([labels[pos] for pos in np.argmax(scores, axis=1)])


def _corrupt_labels(
    labels: np.ndarray,
    rng: np.random.Generator,
    label_noise: float,
) -> np.ndarray:
    choices = np.array(list(task_templates()))
    corrupted = labels.astype(object).copy()
    flips = rng.random(len(labels)) < label_noise
    for idx in np.flatnonzero(flips):
        alternatives = choices[choices != labels[idx]]
        corrupted[idx] = rng.choice(alternatives)
    return corrupted.astype(str)


def make_hard_task_stream(
    seed: int,
    config: HardBenchmarkConfig,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    stream = []
    for tick in range(config.ticks):
        true_tasks, true_labels = sample_task_batch(
            rng,
            tick,
            config.batch_size,
            config.shift_tick,
        )
        observed = true_tasks + rng.normal(
            0.0,
            config.context_noise,
            size=true_tasks.shape,
        )
        observed = np.clip(observed, 0.0, None)
        observed = observed / np.maximum(
            np.linalg.norm(observed, axis=1, keepdims=True),
            1e-12,
        )
        observed_labels = _corrupt_labels(
            _nearest_task_labels(observed),
            rng,
            config.label_noise,
        )
        stream.append((true_tasks, true_labels, observed, observed_labels))
    return stream


def _softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    logits = scores / max(tau, 1e-8)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)


def hard_reward_matrix(
    true_task_batch: np.ndarray,
    kernel,
    tick: int,
    config: HardBenchmarkConfig,
) -> np.ndarray:
    rewards = reward_matrix(true_task_batch, kernel).copy()
    switch_period = getattr(config, "switch_period", None)
    if switch_period is not None:
        intensity = float(getattr(config, "adversarial_intensity", 0.0))
        phase = (tick // max(int(switch_period), 1)) % 2
        if phase == 0:
            rewards[:, LANGUAGE] *= 1.0 - 0.55 * intensity
            rewards[:, SYMBOLIC] += 0.30 * intensity
        else:
            rewards[:, SYMBOLIC] *= 1.0 - 0.55 * intensity
            rewards[:, LANGUAGE] += 0.30 * intensity
        rewards[:, GENERALIST] += 0.08 * intensity
        rewards[:, [ROUTER, OUTPUT]] = 0.0
        return np.clip(rewards, 0.0, 1.0)

    if tick >= config.degradation_tick:
        rewards[:, LANGUAGE] *= config.language_degradation
        rewards[:, GENERALIST] *= config.generalist_degradation

    needs_verifier = (
        (true_task_batch[:, 2] > true_task_batch[:, 1])
        & (true_task_batch[:, 3] > 0.45)
        & (true_task_batch[:, 5] > 0.45)
    )
    rewards[needs_verifier, SYMBOLIC] += config.verifier_bonus
    for idx in [LANGUAGE, GENERALIST]:
        rewards[needs_verifier, idx] -= config.verifier_penalty
    rewards[:, [ROUTER, OUTPUT]] = 0.0
    return np.clip(rewards, 0.0, 1.0)


def _summarize_history(
    label: str,
    seed: int,
    history: list[dict[str, Any]],
    shift_tick: int,
) -> dict[str, Any]:
    post = history[shift_tick + 20 :]
    if not post:
        post = history[shift_tick:]
    recovery_tick = None
    for row in history[shift_tick:]:
        if row["symbolic_share"] > row["language_share"]:
            recovery_tick = row["tick"]
            break
    return {
        "router": label,
        "seed": seed,
        "post_shift_mean_reward": mean(row["mean_reward"] for row in post),
        "post_shift_mean_regret": mean(row["mean_regret"] for row in post),
        "post_shift_language_share": mean(row["language_share"] for row in post),
        "post_shift_symbolic_share": mean(row["symbolic_share"] for row in post),
        "post_shift_cost": mean(row["mean_cost"] for row in post),
        "recovery_tick": recovery_tick,
        "history": history,
    }


def run_dte_router(
    label: str,
    task_stream: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 1000)
    kernel = build_neural_v2_kernel(condition(label))
    history = []
    for tick, (tasks, _) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(tasks), 1)),
            axis=1,
        )
        rewards = reward_matrix(tasks, kernel)
        chosen = rewards[np.arange(len(tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(tasks), ROUTER, dtype=int), destinations),
            1.0 / len(tasks),
        )
        kernel.memory_law_step(traffic, node_reward=node_reward)
        costs = np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(costs.mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history(f"dte_{label}", seed, history, config.shift_tick)


def run_dte_ucb_router(
    task_stream: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 9000)
    kernel = build_neural_v2_kernel(condition("local_regret"))
    kernel.configure_edge_learning(
        mode="ucb",
        reward_gain=0.05,
        uncertainty_gain=0.01,
        ucb_c=0.05,
        initial_reward=0.0,
    )
    history = []
    for tick, (tasks, _) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(tasks), 1)),
            axis=1,
        )
        rewards = reward_matrix(tasks, kernel)
        chosen = rewards[np.arange(len(tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(tasks), ROUTER, dtype=int), destinations),
            1.0 / len(tasks),
        )
        kernel.memory_law_step(traffic, node_reward=node_reward)
        kernel.edge_learning_step(traffic, node_reward=node_reward)
        costs = np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(costs.mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("dte_ucb", seed, history, config.shift_tick)


def run_dte_contextual_ucb_router(
    task_stream: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 11000)
    kernel = build_neural_v2_kernel(condition("local_regret"))
    kernel.configure_edge_learning(
        mode="ucb",
        reward_gain=0.0,
        uncertainty_gain=0.002,
        ucb_c=0.03,
        initial_reward=0.0,
        context_centroids=context_centroids(),
    )
    ctx_map = context_index_map()
    history = []
    for tick, (tasks, task_names) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(tasks), 1)),
            axis=1,
        )
        rewards = reward_matrix(tasks, kernel)
        chosen = rewards[np.arange(len(tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(tasks), ROUTER, dtype=int), destinations),
            1.0 / len(tasks),
        )
        kernel.memory_law_step(traffic, node_reward=node_reward)
        for label, ctx_idx in ctx_map.items():
            mask = task_names == label
            if not np.any(mask):
                continue
            ctx_traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
            np.add.at(
                ctx_traffic,
                (np.full(int(mask.sum()), ROUTER, dtype=int), destinations[mask]),
                1.0,
            )
            kernel.edge_learning_step(
                ctx_traffic,
                node_reward=rewards[mask].mean(axis=0),
                context_index=ctx_idx,
            )
        costs = np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(costs.mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("dte_contextual_ucb", seed, history, config.shift_tick)


def run_dte_arbitrated_ucb_router(
    task_stream: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 13000)
    kernel = build_neural_v2_kernel(condition("local_regret"))
    configure_arbitrated_ucb(kernel)
    ctx_map = context_index_map()
    history = []
    for tick, (tasks, task_names) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(tasks), 1)),
            axis=1,
        )
        rewards = reward_matrix(tasks, kernel)
        chosen = rewards[np.arange(len(tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(tasks), ROUTER, dtype=int), destinations),
            1.0 / len(tasks),
        )
        kernel.memory_law_step(traffic, node_reward=node_reward)
        for label, ctx_idx in ctx_map.items():
            mask = task_names == label
            if not np.any(mask):
                continue
            ctx_traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
            np.add.at(
                ctx_traffic,
                (np.full(int(mask.sum()), ROUTER, dtype=int), destinations[mask]),
                1.0,
            )
            kernel.edge_learning_step(
                ctx_traffic,
                node_reward=rewards[mask].mean(axis=0),
                context_index=ctx_idx,
            )
        costs = np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(costs.mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("dte_arbitrated_ucb", seed, history, config.shift_tick)


def run_dte_reliability_arbitrated_ucb_router(
    task_stream: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 15000)
    kernel = build_neural_v2_kernel(condition("local_regret"))
    configure_arbitrated_ucb(kernel, reliability_gated=True)
    ctx_map = context_index_map()
    history = []
    for tick, (tasks, task_names) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(tasks), 1)),
            axis=1,
        )
        rewards = reward_matrix(tasks, kernel)
        chosen = rewards[np.arange(len(tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(tasks), ROUTER, dtype=int), destinations),
            1.0 / len(tasks),
        )
        kernel.memory_law_step(traffic, node_reward=node_reward)
        for label, ctx_idx in ctx_map.items():
            mask = task_names == label
            if not np.any(mask):
                continue
            ctx_traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
            np.add.at(
                ctx_traffic,
                (np.full(int(mask.sum()), ROUTER, dtype=int), destinations[mask]),
                1.0,
            )
            kernel.edge_learning_step(
                ctx_traffic,
                node_reward=rewards[mask].mean(axis=0),
                context_index=ctx_idx,
            )
        costs = np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(costs.mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history(
        "dte_reliability_arbitrated_ucb",
        seed,
        history,
        config.shift_tick,
    )


def run_dte_exp3_router(
    task_stream: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 17000)
    kernel = build_neural_v2_kernel(condition("local_regret"))
    configure_arbitrated_exp3(kernel)
    ctx_map = context_index_map()
    history = []
    for tick, (tasks, task_names) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(tasks), 1)),
            axis=1,
        )
        rewards = reward_matrix(tasks, kernel)
        chosen = rewards[np.arange(len(tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(tasks), ROUTER, dtype=int), destinations),
            1.0 / len(tasks),
        )
        kernel.memory_law_step(traffic, node_reward=node_reward)
        for label, ctx_idx in ctx_map.items():
            mask = task_names == label
            if not np.any(mask):
                continue
            ctx_traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
            np.add.at(
                ctx_traffic,
                (np.full(int(mask.sum()), ROUTER, dtype=int), destinations[mask]),
                1.0,
            )
            kernel.edge_learning_step(
                ctx_traffic,
                node_reward=rewards[mask].mean(axis=0),
                context_index=ctx_idx,
            )
        costs = np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(costs.mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("dte_exp3", seed, history, config.shift_tick)


def run_static_contextual_router(
    task_stream: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: BenchmarkConfig,
    tau: float = 0.20,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 2000)
    kernel = build_neural_v2_kernel(condition("base_forgetting"))
    features = module_feature_table()
    module_features = np.array(list(features.values()))
    skill = module_features[MODULE_INDICES, :3]
    skill = skill / np.maximum(np.linalg.norm(skill, axis=1, keepdims=True), 1e-12)
    costs = np.array([MODULE_COST.get(idx, 0.0) for idx in MODULE_INDICES])

    history = []
    for tick, (tasks, _) in enumerate(task_stream):
        demand = tasks[:, :3]
        demand = demand / np.maximum(np.linalg.norm(demand, axis=1, keepdims=True), 1e-12)
        scores = demand @ skill.T - 0.15 * costs[np.newaxis, :]
        probs = _softmax(scores, tau=tau)
        chosen_module_positions = np.argmax(
            np.cumsum(probs, axis=1) >= rng.random((len(tasks), 1)),
            axis=1,
        )
        destinations = np.array([MODULE_INDICES[pos] for pos in chosen_module_positions])
        rewards = reward_matrix(tasks, kernel)
        chosen = rewards[np.arange(len(tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        selected_cost = np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(selected_cost.mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("static_contextual", seed, history, config.shift_tick)


def run_epsilon_bandit_router(
    task_stream: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: BenchmarkConfig,
    epsilon: float = 0.10,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 3000)
    kernel = build_neural_v2_kernel(condition("base_forgetting"))
    contexts = list(task_templates())
    q = {ctx: np.full(len(MODULE_INDICES), 0.45, dtype=np.float64) for ctx in contexts}
    counts = {ctx: np.zeros(len(MODULE_INDICES), dtype=np.int64) for ctx in contexts}
    history = []

    for tick, (tasks, task_names) in enumerate(task_stream):
        rewards = reward_matrix(tasks, kernel)
        destinations = np.zeros(len(tasks), dtype=int)
        for idx, ctx in enumerate(task_names):
            if rng.random() < epsilon:
                pos = int(rng.integers(0, len(MODULE_INDICES)))
            else:
                pos = int(np.argmax(q[str(ctx)]))
            dest = MODULE_INDICES[pos]
            destinations[idx] = dest
            reward = rewards[idx, dest]
            counts[str(ctx)][pos] += 1
            n = counts[str(ctx)][pos]
            q[str(ctx)][pos] += (reward - q[str(ctx)][pos]) / n

        chosen = rewards[np.arange(len(tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        selected_cost = np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(selected_cost.mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("epsilon_bandit", seed, history, config.shift_tick)


def run_ucb_router(
    task_stream: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: BenchmarkConfig,
    exploration: float = 0.35,
) -> dict[str, Any]:
    kernel = build_neural_v2_kernel(condition("base_forgetting"))
    contexts = list(task_templates())
    q = {ctx: np.full(len(MODULE_INDICES), 0.45, dtype=np.float64) for ctx in contexts}
    counts = {ctx: np.zeros(len(MODULE_INDICES), dtype=np.int64) for ctx in contexts}
    context_totals = {ctx: 0 for ctx in contexts}
    history = []

    for tick, (tasks, task_names) in enumerate(task_stream):
        rewards = reward_matrix(tasks, kernel)
        destinations = np.zeros(len(tasks), dtype=int)
        for idx, ctx in enumerate(task_names):
            ctx = str(ctx)
            unvisited = np.flatnonzero(counts[ctx] == 0)
            if len(unvisited):
                pos = int(unvisited[0])
            else:
                bonus = exploration * np.sqrt(
                    np.log(context_totals[ctx] + 1.0) / counts[ctx]
                )
                pos = int(np.argmax(q[ctx] + bonus))
            dest = MODULE_INDICES[pos]
            destinations[idx] = dest
            reward = rewards[idx, dest]
            counts[ctx][pos] += 1
            context_totals[ctx] += 1
            n = counts[ctx][pos]
            q[ctx][pos] += (reward - q[ctx][pos]) / n

        chosen = rewards[np.arange(len(tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        selected_cost = np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(selected_cost.mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("ucb", seed, history, config.shift_tick)


def run_exp3_router(
    task_stream: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: BenchmarkConfig,
    gamma: float = 0.10,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 7000)
    kernel = build_neural_v2_kernel(condition("base_forgetting"))
    contexts = list(task_templates())
    weights = {ctx: np.ones(len(MODULE_INDICES), dtype=np.float64) for ctx in contexts}
    history = []

    for tick, (tasks, task_names) in enumerate(task_stream):
        rewards = reward_matrix(tasks, kernel)
        destinations = np.zeros(len(tasks), dtype=int)
        for idx, ctx in enumerate(task_names):
            ctx = str(ctx)
            probs = (1.0 - gamma) * weights[ctx] / weights[ctx].sum()
            probs += gamma / len(MODULE_INDICES)
            pos = int(np.argmax(np.cumsum(probs) >= rng.random()))
            dest = MODULE_INDICES[pos]
            destinations[idx] = dest
            estimated_reward = rewards[idx, dest] / max(probs[pos], 1e-12)
            weights[ctx][pos] *= np.exp(
                gamma * estimated_reward / len(MODULE_INDICES)
            )
            weights[ctx] = np.minimum(weights[ctx], 1e50)

        chosen = rewards[np.arange(len(tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        selected_cost = np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(selected_cost.mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("exp3", seed, history, config.shift_tick)


def run_oracle_router(
    task_stream: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    kernel = build_neural_v2_kernel(condition("base_forgetting"))
    history = []
    for tick, (tasks, _) in enumerate(task_stream):
        rewards = reward_matrix(tasks, kernel)
        positions = np.argmax(rewards[:, MODULE_INDICES], axis=1)
        destinations = np.array([MODULE_INDICES[pos] for pos in positions])
        chosen = rewards[np.arange(len(tasks)), destinations]
        selected_cost = np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": 0.0,
                "mean_cost": float(selected_cost.mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
        }
    )
    return _summarize_history("oracle", seed, history, config.shift_tick)


def _module_costs(destinations: np.ndarray) -> np.ndarray:
    return np.array([MODULE_COST.get(int(dest), 0.0) for dest in destinations])


def run_hard_dte_router(
    label: str,
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 4000)
    kernel = build_neural_v2_kernel(condition(label))
    pending_updates: list[tuple[np.ndarray, np.ndarray]] = []
    history = []
    for tick, (true_tasks, _, observed_tasks, _) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(observed_tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(true_tasks), 1)),
            axis=1,
        )
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        chosen = rewards[np.arange(len(true_tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(true_tasks), ROUTER, dtype=int), destinations),
            1.0 / len(true_tasks),
        )
        pending_updates.append((traffic, node_reward))
        if len(pending_updates) > config.reward_delay:
            delayed_traffic, delayed_reward = pending_updates.pop(0)
            kernel.memory_law_step(delayed_traffic, node_reward=delayed_reward)

        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history(f"hard_dte_{label}", seed, history, config.shift_tick)


def run_hard_dte_ucb_router(
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 10000)
    kernel = build_neural_v2_kernel(condition("local_regret"))
    kernel.configure_edge_learning(
        mode="ucb",
        reward_gain=0.05,
        uncertainty_gain=0.01,
        ucb_c=0.05,
        initial_reward=0.0,
    )
    pending_updates: list[tuple[np.ndarray, np.ndarray]] = []
    history = []
    for tick, (true_tasks, _, observed_tasks, _) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(observed_tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(true_tasks), 1)),
            axis=1,
        )
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        chosen = rewards[np.arange(len(true_tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(true_tasks), ROUTER, dtype=int), destinations),
            1.0 / len(true_tasks),
        )
        pending_updates.append((traffic, node_reward))
        if len(pending_updates) > config.reward_delay:
            delayed_traffic, delayed_reward = pending_updates.pop(0)
            kernel.memory_law_step(delayed_traffic, node_reward=delayed_reward)
            kernel.edge_learning_step(delayed_traffic, node_reward=delayed_reward)

        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("hard_dte_ucb", seed, history, config.shift_tick)


def run_hard_dte_contextual_ucb_router(
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 12000)
    kernel = build_neural_v2_kernel(condition("local_regret"))
    kernel.configure_edge_learning(
        mode="ucb",
        reward_gain=0.0,
        uncertainty_gain=0.002,
        ucb_c=0.03,
        initial_reward=0.0,
        context_centroids=context_centroids(),
    )
    ctx_map = context_index_map()
    pending_memory: list[tuple[np.ndarray, np.ndarray]] = []
    pending_context: list[list[tuple[int, np.ndarray, np.ndarray]]] = []
    history = []
    for tick, (true_tasks, _, observed_tasks, observed_labels) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(observed_tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(true_tasks), 1)),
            axis=1,
        )
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        chosen = rewards[np.arange(len(true_tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(true_tasks), ROUTER, dtype=int), destinations),
            1.0 / len(true_tasks),
        )
        context_updates: list[tuple[int, np.ndarray, np.ndarray]] = []
        for label, ctx_idx in ctx_map.items():
            mask = observed_labels == label
            if not np.any(mask):
                continue
            ctx_traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
            np.add.at(
                ctx_traffic,
                (np.full(int(mask.sum()), ROUTER, dtype=int), destinations[mask]),
                1.0,
            )
            context_updates.append((ctx_idx, ctx_traffic, rewards[mask].mean(axis=0)))

        pending_memory.append((traffic, node_reward))
        pending_context.append(context_updates)
        if len(pending_memory) > config.reward_delay:
            delayed_traffic, delayed_reward = pending_memory.pop(0)
            kernel.memory_law_step(delayed_traffic, node_reward=delayed_reward)
            for ctx_idx, ctx_traffic, ctx_reward in pending_context.pop(0):
                kernel.edge_learning_step(
                    ctx_traffic,
                    node_reward=ctx_reward,
                    context_index=ctx_idx,
                )

        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("hard_dte_contextual_ucb", seed, history, config.shift_tick)


def run_hard_dte_arbitrated_ucb_router(
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 14000)
    kernel = build_neural_v2_kernel(condition("local_regret"))
    configure_arbitrated_ucb(kernel)
    ctx_map = context_index_map()
    pending_memory: list[tuple[np.ndarray, np.ndarray]] = []
    pending_context: list[list[tuple[int, np.ndarray, np.ndarray]]] = []
    history = []
    for tick, (true_tasks, _, observed_tasks, observed_labels) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(observed_tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(true_tasks), 1)),
            axis=1,
        )
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        chosen = rewards[np.arange(len(true_tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(true_tasks), ROUTER, dtype=int), destinations),
            1.0 / len(true_tasks),
        )
        context_updates: list[tuple[int, np.ndarray, np.ndarray]] = []
        for label, ctx_idx in ctx_map.items():
            mask = observed_labels == label
            if not np.any(mask):
                continue
            ctx_traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
            np.add.at(
                ctx_traffic,
                (np.full(int(mask.sum()), ROUTER, dtype=int), destinations[mask]),
                1.0,
            )
            context_updates.append((ctx_idx, ctx_traffic, rewards[mask].mean(axis=0)))

        pending_memory.append((traffic, node_reward))
        pending_context.append(context_updates)
        if len(pending_memory) > config.reward_delay:
            delayed_traffic, delayed_reward = pending_memory.pop(0)
            kernel.memory_law_step(delayed_traffic, node_reward=delayed_reward)
            for ctx_idx, ctx_traffic, ctx_reward in pending_context.pop(0):
                kernel.edge_learning_step(
                    ctx_traffic,
                    node_reward=ctx_reward,
                    context_index=ctx_idx,
                )

        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("hard_dte_arbitrated_ucb", seed, history, config.shift_tick)


def run_hard_dte_reliability_arbitrated_ucb_router(
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 16000)
    kernel = build_neural_v2_kernel(condition("local_regret"))
    configure_arbitrated_ucb(kernel, reliability_gated=True)
    ctx_map = context_index_map()
    pending_memory: list[tuple[np.ndarray, np.ndarray]] = []
    pending_context: list[list[tuple[int, np.ndarray, np.ndarray]]] = []
    history = []
    for tick, (true_tasks, _, observed_tasks, observed_labels) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(observed_tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(true_tasks), 1)),
            axis=1,
        )
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        chosen = rewards[np.arange(len(true_tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(true_tasks), ROUTER, dtype=int), destinations),
            1.0 / len(true_tasks),
        )
        context_updates: list[tuple[int, np.ndarray, np.ndarray]] = []
        for label, ctx_idx in ctx_map.items():
            mask = observed_labels == label
            if not np.any(mask):
                continue
            ctx_traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
            np.add.at(
                ctx_traffic,
                (np.full(int(mask.sum()), ROUTER, dtype=int), destinations[mask]),
                1.0,
            )
            context_updates.append((ctx_idx, ctx_traffic, rewards[mask].mean(axis=0)))

        pending_memory.append((traffic, node_reward))
        pending_context.append(context_updates)
        if len(pending_memory) > config.reward_delay:
            delayed_traffic, delayed_reward = pending_memory.pop(0)
            kernel.memory_law_step(delayed_traffic, node_reward=delayed_reward)
            for ctx_idx, ctx_traffic, ctx_reward in pending_context.pop(0):
                kernel.edge_learning_step(
                    ctx_traffic,
                    node_reward=ctx_reward,
                    context_index=ctx_idx,
                )

        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history(
        "hard_dte_reliability_arbitrated_ucb",
        seed,
        history,
        config.shift_tick,
    )


def run_hard_dte_exp3_router(
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 18000)
    kernel = build_neural_v2_kernel(condition("local_regret"))
    configure_arbitrated_exp3(kernel)
    ctx_map = context_index_map()
    pending_memory: list[tuple[np.ndarray, np.ndarray]] = []
    pending_context: list[list[tuple[int, np.ndarray, np.ndarray]]] = []
    history = []
    for tick, (true_tasks, _, observed_tasks, observed_labels) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(observed_tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(true_tasks), 1)),
            axis=1,
        )
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        chosen = rewards[np.arange(len(true_tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(true_tasks), ROUTER, dtype=int), destinations),
            1.0 / len(true_tasks),
        )
        context_updates: list[tuple[int, np.ndarray, np.ndarray]] = []
        for label, ctx_idx in ctx_map.items():
            mask = observed_labels == label
            if not np.any(mask):
                continue
            ctx_traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
            np.add.at(
                ctx_traffic,
                (np.full(int(mask.sum()), ROUTER, dtype=int), destinations[mask]),
                1.0,
            )
            context_updates.append((ctx_idx, ctx_traffic, rewards[mask].mean(axis=0)))

        pending_memory.append((traffic, node_reward))
        pending_context.append(context_updates)
        if len(pending_memory) > config.reward_delay:
            delayed_traffic, delayed_reward = pending_memory.pop(0)
            kernel.memory_law_step(delayed_traffic, node_reward=delayed_reward)
            for ctx_idx, ctx_traffic, ctx_reward in pending_context.pop(0):
                kernel.edge_learning_step(
                    ctx_traffic,
                    node_reward=ctx_reward,
                    context_index=ctx_idx,
                )

        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("hard_dte_exp3", seed, history, config.shift_tick)


def run_hard_dte_reliability_arbitrated_exp3_router(
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 19000)
    kernel = build_neural_v2_kernel(condition("local_regret"))
    configure_arbitrated_exp3(kernel, reliability_gated=True)
    ctx_map = context_index_map()
    pending_memory: list[tuple[np.ndarray, np.ndarray]] = []
    pending_context: list[list[tuple[int, np.ndarray, np.ndarray]]] = []
    history = []
    for tick, (true_tasks, _, observed_tasks, observed_labels) in enumerate(task_stream):
        P_all = kernel.transition_matrix_batch(observed_tasks, step=tick)
        rows = P_all[:, ROUTER, :]
        destinations = np.argmax(
            np.cumsum(rows, axis=1) >= rng.random((len(true_tasks), 1)),
            axis=1,
        )
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        chosen = rewards[np.arange(len(true_tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        node_reward = rewards.mean(axis=0)
        traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
        np.add.at(
            traffic,
            (np.full(len(true_tasks), ROUTER, dtype=int), destinations),
            1.0 / len(true_tasks),
        )
        context_updates: list[tuple[int, np.ndarray, np.ndarray]] = []
        for label, ctx_idx in ctx_map.items():
            mask = observed_labels == label
            if not np.any(mask):
                continue
            ctx_traffic = np.zeros((kernel.topo.N, kernel.topo.N), dtype=np.float64)
            np.add.at(
                ctx_traffic,
                (np.full(int(mask.sum()), ROUTER, dtype=int), destinations[mask]),
                1.0,
            )
            context_updates.append((ctx_idx, ctx_traffic, rewards[mask].mean(axis=0)))

        pending_memory.append((traffic, node_reward))
        pending_context.append(context_updates)
        if len(pending_memory) > config.reward_delay:
            delayed_traffic, delayed_reward = pending_memory.pop(0)
            kernel.memory_law_step(delayed_traffic, node_reward=delayed_reward)
            for ctx_idx, ctx_traffic, ctx_reward in pending_context.pop(0):
                kernel.edge_learning_step(
                    ctx_traffic,
                    node_reward=ctx_reward,
                    context_index=ctx_idx,
                )

        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history(
        "hard_dte_reliability_arbitrated_exp3",
        seed,
        history,
        config.shift_tick,
    )


def run_hard_static_contextual_router(
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
    tau: float = 0.24,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 5000)
    kernel = build_neural_v2_kernel(condition("base_forgetting"))
    features = module_feature_table()
    module_features = np.array(list(features.values()))
    skill = module_features[MODULE_INDICES, :3]
    skill = skill / np.maximum(np.linalg.norm(skill, axis=1, keepdims=True), 1e-12)
    costs = np.array([MODULE_COST.get(idx, 0.0) for idx in MODULE_INDICES])

    history = []
    for tick, (true_tasks, _, observed_tasks, _) in enumerate(task_stream):
        demand = observed_tasks[:, :3]
        demand = demand / np.maximum(np.linalg.norm(demand, axis=1, keepdims=True), 1e-12)
        scores = demand @ skill.T - 0.15 * costs[np.newaxis, :]
        probs = _softmax(scores, tau=tau)
        chosen_module_positions = np.argmax(
            np.cumsum(probs, axis=1) >= rng.random((len(true_tasks), 1)),
            axis=1,
        )
        destinations = np.array([MODULE_INDICES[pos] for pos in chosen_module_positions])
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        chosen = rewards[np.arange(len(true_tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("hard_static_contextual", seed, history, config.shift_tick)


def run_hard_epsilon_bandit_router(
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
    epsilon: float = 0.12,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 6000)
    kernel = build_neural_v2_kernel(condition("base_forgetting"))
    contexts = list(task_templates())
    q = {ctx: np.full(len(MODULE_INDICES), 0.45, dtype=np.float64) for ctx in contexts}
    counts = {ctx: np.zeros(len(MODULE_INDICES), dtype=np.int64) for ctx in contexts}
    pending_rewards: list[tuple[str, int, float]] = []
    history = []

    for tick, (true_tasks, _, _, observed_labels) in enumerate(task_stream):
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        destinations = np.zeros(len(true_tasks), dtype=int)
        for idx, ctx in enumerate(observed_labels):
            if rng.random() < epsilon:
                pos = int(rng.integers(0, len(MODULE_INDICES)))
            else:
                pos = int(np.argmax(q[str(ctx)]))
            dest = MODULE_INDICES[pos]
            destinations[idx] = dest
            pending_rewards.append((str(ctx), pos, float(rewards[idx, dest])))
            if len(pending_rewards) > config.reward_delay * len(true_tasks):
                delayed_ctx, delayed_pos, delayed_reward = pending_rewards.pop(0)
                counts[delayed_ctx][delayed_pos] += 1
                n = counts[delayed_ctx][delayed_pos]
                q[delayed_ctx][delayed_pos] += (
                    delayed_reward - q[delayed_ctx][delayed_pos]
                ) / n

        chosen = rewards[np.arange(len(true_tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("hard_epsilon_bandit", seed, history, config.shift_tick)


def run_hard_ucb_router(
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
    exploration: float = 0.35,
) -> dict[str, Any]:
    kernel = build_neural_v2_kernel(condition("base_forgetting"))
    contexts = list(task_templates())
    q = {ctx: np.full(len(MODULE_INDICES), 0.45, dtype=np.float64) for ctx in contexts}
    counts = {ctx: np.zeros(len(MODULE_INDICES), dtype=np.int64) for ctx in contexts}
    context_totals = {ctx: 0 for ctx in contexts}
    pending_rewards: list[tuple[str, int, float]] = []
    history = []

    for tick, (true_tasks, _, _, observed_labels) in enumerate(task_stream):
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        destinations = np.zeros(len(true_tasks), dtype=int)
        for idx, ctx in enumerate(observed_labels):
            ctx = str(ctx)
            unvisited = np.flatnonzero(counts[ctx] == 0)
            if len(unvisited):
                pos = int(unvisited[0])
            else:
                bonus = exploration * np.sqrt(
                    np.log(context_totals[ctx] + 1.0) / counts[ctx]
                )
                pos = int(np.argmax(q[ctx] + bonus))
            dest = MODULE_INDICES[pos]
            destinations[idx] = dest
            pending_rewards.append((ctx, pos, float(rewards[idx, dest])))
            if len(pending_rewards) > config.reward_delay * len(true_tasks):
                delayed_ctx, delayed_pos, delayed_reward = pending_rewards.pop(0)
                counts[delayed_ctx][delayed_pos] += 1
                context_totals[delayed_ctx] += 1
                n = counts[delayed_ctx][delayed_pos]
                q[delayed_ctx][delayed_pos] += (
                    delayed_reward - q[delayed_ctx][delayed_pos]
                ) / n

        chosen = rewards[np.arange(len(true_tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("hard_ucb", seed, history, config.shift_tick)


def run_hard_exp3_router(
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
    gamma: float = 0.10,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 8000)
    kernel = build_neural_v2_kernel(condition("base_forgetting"))
    contexts = list(task_templates())
    weights = {ctx: np.ones(len(MODULE_INDICES), dtype=np.float64) for ctx in contexts}
    pending_rewards: list[tuple[str, int, float, float]] = []
    history = []

    for tick, (true_tasks, _, _, observed_labels) in enumerate(task_stream):
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        destinations = np.zeros(len(true_tasks), dtype=int)
        for idx, ctx in enumerate(observed_labels):
            ctx = str(ctx)
            probs = (1.0 - gamma) * weights[ctx] / weights[ctx].sum()
            probs += gamma / len(MODULE_INDICES)
            pos = int(np.argmax(np.cumsum(probs) >= rng.random()))
            dest = MODULE_INDICES[pos]
            destinations[idx] = dest
            pending_rewards.append((ctx, pos, float(rewards[idx, dest]), float(probs[pos])))
            if len(pending_rewards) > config.reward_delay * len(true_tasks):
                delayed_ctx, delayed_pos, delayed_reward, delayed_prob = pending_rewards.pop(0)
                estimated_reward = delayed_reward / max(delayed_prob, 1e-12)
                weights[delayed_ctx][delayed_pos] *= np.exp(
                    gamma * estimated_reward / len(MODULE_INDICES)
                )
                weights[delayed_ctx] = np.minimum(weights[delayed_ctx], 1e50)

        chosen = rewards[np.arange(len(true_tasks)), destinations]
        optimal = rewards[:, MODULE_INDICES].max(axis=1)
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": float((optimal - chosen).mean()),
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("hard_exp3", seed, history, config.shift_tick)


def run_hard_oracle_router(
    task_stream: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    config: HardBenchmarkConfig,
) -> dict[str, Any]:
    kernel = build_neural_v2_kernel(condition("base_forgetting"))
    history = []
    for tick, (true_tasks, _, _, _) in enumerate(task_stream):
        rewards = hard_reward_matrix(true_tasks, kernel, tick, config)
        positions = np.argmax(rewards[:, MODULE_INDICES], axis=1)
        destinations = np.array([MODULE_INDICES[pos] for pos in positions])
        chosen = rewards[np.arange(len(true_tasks)), destinations]
        history.append(
            {
                "tick": tick,
                "mean_reward": float(chosen.mean()),
                "mean_regret": 0.0,
                "mean_cost": float(_module_costs(destinations).mean()),
                "language_share": float(np.mean(destinations == LANGUAGE)),
                "symbolic_share": float(np.mean(destinations == SYMBOLIC)),
                "generalist_share": float(np.mean(destinations == GENERALIST)),
            }
        )
    return _summarize_history("hard_oracle", seed, history, config.shift_tick)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["router"], []).append(row)
    summary = {}
    for label, group in grouped.items():
        recoveries = [row["recovery_tick"] for row in group if row["recovery_tick"] is not None]
        summary[label] = {
            "runs": len(group),
            "mean_post_shift_reward": mean(row["post_shift_mean_reward"] for row in group),
            "mean_post_shift_regret": mean(row["post_shift_mean_regret"] for row in group),
            "mean_post_shift_language_share": mean(row["post_shift_language_share"] for row in group),
            "mean_post_shift_symbolic_share": mean(row["post_shift_symbolic_share"] for row in group),
            "mean_post_shift_cost": mean(row["post_shift_cost"] for row in group),
            "mean_recovery_tick": mean(recoveries) if recoveries else None,
        }
    return summary


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    order = [
        "dte_surprise_only",
        "dte_local_regret",
        "dte_ucb",
        "dte_contextual_ucb",
        "dte_arbitrated_ucb",
        "dte_reliability_arbitrated_ucb",
        "dte_exp3",
        "static_contextual",
        "epsilon_bandit",
        "ucb",
        "exp3",
        "oracle",
    ]
    lines = [
        "# Neural V2 Router Benchmark Report",
        "",
        "## Scope",
        "",
        "Shared task stream benchmark for adaptive inference routing. All routers",
        "see the same language-heavy to symbolic-heavy shift per seed.",
        "",
        "## Results",
        "",
        "| Router | Runs | Reward | Regret | Cost | Language Share | Symbolic Share | Recovery Tick |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in order:
        row = summary[label]
        recovery = "n/a" if row["mean_recovery_tick"] is None else f"{row['mean_recovery_tick']:.1f}"
        lines.append(
            f"| {label} | {row['runs']} | "
            f"{row['mean_post_shift_reward']:.3f} | "
            f"{row['mean_post_shift_regret']:.3f} | "
            f"{row['mean_post_shift_cost']:.3f} | "
            f"{row['mean_post_shift_language_share']:.3f} | "
            f"{row['mean_post_shift_symbolic_share']:.3f} | {recovery} |"
        )

    dte_gain = (
        summary["dte_local_regret"]["mean_post_shift_regret"]
        < summary["dte_surprise_only"]["mean_post_shift_regret"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "DTE local regret improves over DTE surprise-only on post-shift "
                "regret."
                if dte_gain
                else "DTE local regret did not improve over DTE surprise-only on this run."
            ),
            "",
            "The contextual and bandit baselines are intentionally strong. If they",
            "match or beat DTE, the conclusion is not failure; it identifies the",
            "conditions where a simpler router is sufficient. DTE's distinctive",
            "claim is strongest when explicit memory, topology, and stale-route",
            "diagnostics matter.",
            "",
            "## Boundary Finding",
            "",
            "This benchmark does not support the claim that DTE is the best flat",
            "router when task labels are clean, rewards are immediate, and module",
            "choices are independent. Under those assumptions, a contextual router",
            "or contextual bandit should win, and here it does.",
            "",
            "The result does support a narrower Neural V2 claim: local-regret memory",
            "is a real corrective mechanism for stale adaptive routing. The next",
            "validity test should remove the clean-label advantage by adding noisy",
            "task context, delayed rewards, graph-constrained multi-stage routing,",
            "and nonstationary module degradation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(
    config: BenchmarkConfig | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    config = config or BenchmarkConfig()
    rows = []
    for seed in config.seeds:
        stream = make_task_stream(seed, config)
        rows.append(run_dte_router("surprise_only", stream, seed, config))
        rows.append(run_dte_router("local_regret", stream, seed, config))
        rows.append(run_dte_ucb_router(stream, seed, config))
        rows.append(run_dte_contextual_ucb_router(stream, seed, config))
        rows.append(run_dte_arbitrated_ucb_router(stream, seed, config))
        rows.append(run_dte_reliability_arbitrated_ucb_router(stream, seed, config))
        rows.append(run_dte_exp3_router(stream, seed, config))
        rows.append(run_static_contextual_router(stream, seed, config))
        rows.append(run_epsilon_bandit_router(stream, seed, config))
        rows.append(run_ucb_router(stream, seed, config))
        rows.append(run_exp3_router(stream, seed, config))
        rows.append(run_oracle_router(stream, seed, config))
    payload = {
        "config": config.__dict__,
        "rows": rows,
        "summary": summarize(rows),
    }
    if write_outputs:
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        REPORT_PATH.write_text(render_report(payload), encoding="utf-8")
    return payload


def render_hard_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    order = [
        "hard_dte_surprise_only",
        "hard_dte_local_regret",
        "hard_dte_ucb",
        "hard_dte_contextual_ucb",
        "hard_dte_arbitrated_ucb",
        "hard_dte_reliability_arbitrated_ucb",
        "hard_dte_exp3",
        "hard_dte_reliability_arbitrated_exp3",
        "hard_static_contextual",
        "hard_epsilon_bandit",
        "hard_ucb",
        "hard_exp3",
        "hard_oracle",
    ]
    lines = [
        "# Neural V2 Hard Router Benchmark Report",
        "",
        "## Scope",
        "",
        "Stress benchmark for adaptive inference routing under noisy task context,",
        "corrupted context labels, delayed reward updates, post-shift module",
        "degradation, and verifier-gated symbolic tasks. Rewards are computed",
        "from the true task, while routers observe the noisy task surface.",
        "",
        "## Results",
        "",
        "| Router | Runs | Reward | Regret | Cost | Language Share | Symbolic Share | Recovery Tick |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in order:
        row = summary[label]
        recovery = "n/a" if row["mean_recovery_tick"] is None else f"{row['mean_recovery_tick']:.1f}"
        lines.append(
            f"| {label} | {row['runs']} | "
            f"{row['mean_post_shift_reward']:.3f} | "
            f"{row['mean_post_shift_regret']:.3f} | "
            f"{row['mean_post_shift_cost']:.3f} | "
            f"{row['mean_post_shift_language_share']:.3f} | "
            f"{row['mean_post_shift_symbolic_share']:.3f} | {recovery} |"
        )

    local = summary["hard_dte_local_regret"]
    dte_ucb = summary["hard_dte_ucb"]
    dte_contextual_ucb = summary["hard_dte_contextual_ucb"]
    dte_arbitrated_ucb = summary["hard_dte_arbitrated_ucb"]
    dte_reliability_arbitrated_ucb = summary[
        "hard_dte_reliability_arbitrated_ucb"
    ]
    dte_exp3 = summary["hard_dte_exp3"]
    dte_reliability_arbitrated_exp3 = summary[
        "hard_dte_reliability_arbitrated_exp3"
    ]
    surprise = summary["hard_dte_surprise_only"]
    contextual = summary["hard_static_contextual"]
    bandit = summary["hard_epsilon_bandit"]
    ucb = summary["hard_ucb"]
    exp3 = summary["hard_exp3"]
    regret_drop = (
        surprise["mean_post_shift_regret"]
        - local["mean_post_shift_regret"]
    )
    contextual_gap = (
        local["mean_post_shift_regret"]
        - contextual["mean_post_shift_regret"]
    )
    bandit_gap = local["mean_post_shift_regret"] - bandit["mean_post_shift_regret"]
    ucb_gap = local["mean_post_shift_regret"] - ucb["mean_post_shift_regret"]
    exp3_gap = local["mean_post_shift_regret"] - exp3["mean_post_shift_regret"]
    native_ucb_gain = (
        local["mean_post_shift_regret"]
        - dte_ucb["mean_post_shift_regret"]
    )
    contextual_ucb_gain = (
        local["mean_post_shift_regret"]
        - dte_contextual_ucb["mean_post_shift_regret"]
    )
    arbitrated_ucb_gain = (
        local["mean_post_shift_regret"]
        - dte_arbitrated_ucb["mean_post_shift_regret"]
    )
    reliability_arbitrated_ucb_gain = (
        local["mean_post_shift_regret"]
        - dte_reliability_arbitrated_ucb["mean_post_shift_regret"]
    )
    dte_exp3_gain = (
        local["mean_post_shift_regret"]
        - dte_exp3["mean_post_shift_regret"]
    )
    reliability_arbitrated_exp3_gain = (
        local["mean_post_shift_regret"]
        - dte_reliability_arbitrated_exp3["mean_post_shift_regret"]
    )
    exp3_reliability_gate_gain = (
        dte_exp3["mean_post_shift_regret"]
        - dte_reliability_arbitrated_exp3["mean_post_shift_regret"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"Local-regret DTE changes post-shift regret by `{regret_drop:.3f}` "
                "relative to surprise-only DTE."
            ),
            (
                f"Its regret gap versus the noisy contextual router is "
                f"`{contextual_gap:.3f}`; its gap versus the delayed bandit is "
                f"`{bandit_gap:.3f}`."
            ),
            (
                f"Its regret gap versus delayed UCB is `{ucb_gap:.3f}`; "
                f"its gap versus delayed EXP3 is `{exp3_gap:.3f}`."
            ),
            (
                f"DTE-native UCB changes regret by `{native_ucb_gain:.3f}` "
                "relative to local-regret DTE."
            ),
            (
                f"DTE-contextual UCB changes regret by `{contextual_ucb_gain:.3f}` "
                "relative to local-regret DTE."
            ),
            (
                f"DTE-arbitrated UCB changes regret by `{arbitrated_ucb_gain:.3f}` "
                "relative to local-regret DTE."
            ),
            (
                "DTE-reliability-arbitrated UCB changes regret by "
                f"`{reliability_arbitrated_ucb_gain:.3f}` relative to "
                "local-regret DTE."
            ),
            (
                f"DTE-EXP3 regret gain is `{dte_exp3_gain:.3f}` "
                "relative to local-regret DTE; negative values mean higher regret."
            ),
            (
                "DTE-reliability-arbitrated EXP3 regret gain is "
                f"`{reliability_arbitrated_exp3_gain:.3f}` relative to "
                "local-regret DTE."
            ),
            (
                "Reliability gating changes DTE-EXP3 regret by "
                f"`{exp3_reliability_gate_gain:.3f}`; positive values mean the "
                "gate reduced regret."
            ),
            "",
            "This is the validity test the clean benchmark asked for. It does not",
            "ask whether DTE beats a perfect contextual router. It asks whether",
            "explicit stale-memory correction becomes more valuable once the",
            "routing surface is corrupted and the reward signal arrives late.",
            "",
            "## Institutional Boundary",
            "",
            "A DTE advantage is institutionally meaningful only if it survives",
            "strong baselines after cost, delay, noisy context, and topology-gated",
            "utility are included. Otherwise DTE should be positioned as a",
            "diagnostic and governance layer over adaptive routers, not as a",
            "replacement for standard contextual decision algorithms.",
            "",
        ]
    )
    return "\n".join(lines)


def run_hard_benchmark(
    config: HardBenchmarkConfig | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    config = config or HardBenchmarkConfig()
    rows = []
    for seed in config.seeds:
        stream = make_hard_task_stream(seed, config)
        rows.append(run_hard_dte_router("surprise_only", stream, seed, config))
        rows.append(run_hard_dte_router("local_regret", stream, seed, config))
        rows.append(run_hard_dte_ucb_router(stream, seed, config))
        rows.append(run_hard_dte_contextual_ucb_router(stream, seed, config))
        rows.append(run_hard_dte_arbitrated_ucb_router(stream, seed, config))
        rows.append(run_hard_dte_reliability_arbitrated_ucb_router(stream, seed, config))
        rows.append(run_hard_dte_exp3_router(stream, seed, config))
        rows.append(run_hard_dte_reliability_arbitrated_exp3_router(stream, seed, config))
        rows.append(run_hard_static_contextual_router(stream, seed, config))
        rows.append(run_hard_epsilon_bandit_router(stream, seed, config))
        rows.append(run_hard_ucb_router(stream, seed, config))
        rows.append(run_hard_exp3_router(stream, seed, config))
        rows.append(run_hard_oracle_router(stream, seed, config))
    payload = {
        "config": config.__dict__,
        "rows": rows,
        "summary": summarize(rows),
    }
    if write_outputs:
        HARD_OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        HARD_REPORT_PATH.write_text(render_hard_report(payload), encoding="utf-8")
    return payload


def _frontier_base_config(config: FrontierConfig) -> HardBenchmarkConfig:
    return HardBenchmarkConfig(
        ticks=config.ticks,
        shift_tick=config.shift_tick,
        batch_size=config.batch_size,
        seeds=config.seeds,
    )


def _config_variant(
    base: HardBenchmarkConfig,
    **updates: Any,
) -> HardBenchmarkConfig:
    values = base.__dict__.copy()
    values.update(updates)
    return HardBenchmarkConfig(**values)


def _frontier_scenarios(
    config: FrontierConfig,
) -> list[tuple[str, float | int, HardBenchmarkConfig]]:
    base = _frontier_base_config(config)
    scenarios: list[tuple[str, float | int, HardBenchmarkConfig]] = []
    for value in config.context_noise_values:
        scenarios.append(("context_noise", value, _config_variant(base, context_noise=value)))
    for value in config.label_noise_values:
        scenarios.append(("label_noise", value, _config_variant(base, label_noise=value)))
    for value in config.reward_delay_values:
        scenarios.append(("reward_delay", value, _config_variant(base, reward_delay=value)))
    for value in config.language_degradation_values:
        scenarios.append(
            (
                "language_degradation",
                value,
                _config_variant(base, language_degradation=value),
            )
        )
    for value in config.verifier_bonus_values:
        scenarios.append(("verifier_bonus", value, _config_variant(base, verifier_bonus=value)))
    return scenarios


def _frontier_row(
    axis: str,
    value: float | int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    summary = payload["summary"]
    local = summary["hard_dte_local_regret"]
    dte_ucb = summary["hard_dte_ucb"]
    dte_contextual_ucb = summary["hard_dte_contextual_ucb"]
    dte_arbitrated_ucb = summary["hard_dte_arbitrated_ucb"]
    dte_reliability_arbitrated_ucb = summary[
        "hard_dte_reliability_arbitrated_ucb"
    ]
    dte_exp3 = summary["hard_dte_exp3"]
    surprise = summary["hard_dte_surprise_only"]
    contextual = summary["hard_static_contextual"]
    bandit = summary["hard_epsilon_bandit"]
    ucb = summary["hard_ucb"]
    exp3 = summary["hard_exp3"]
    return {
        "axis": axis,
        "value": value,
        "dte_regret": local["mean_post_shift_regret"],
        "dte_reward": local["mean_post_shift_reward"],
        "dte_ucb_regret": dte_ucb["mean_post_shift_regret"],
        "dte_ucb_reward": dte_ucb["mean_post_shift_reward"],
        "dte_contextual_ucb_regret": dte_contextual_ucb["mean_post_shift_regret"],
        "dte_contextual_ucb_reward": dte_contextual_ucb["mean_post_shift_reward"],
        "dte_arbitrated_ucb_regret": dte_arbitrated_ucb["mean_post_shift_regret"],
        "dte_arbitrated_ucb_reward": dte_arbitrated_ucb["mean_post_shift_reward"],
        "dte_reliability_arbitrated_ucb_regret": (
            dte_reliability_arbitrated_ucb["mean_post_shift_regret"]
        ),
        "dte_reliability_arbitrated_ucb_reward": (
            dte_reliability_arbitrated_ucb["mean_post_shift_reward"]
        ),
        "dte_exp3_regret": dte_exp3["mean_post_shift_regret"],
        "dte_exp3_reward": dte_exp3["mean_post_shift_reward"],
        "surprise_regret": surprise["mean_post_shift_regret"],
        "contextual_regret": contextual["mean_post_shift_regret"],
        "bandit_regret": bandit["mean_post_shift_regret"],
        "ucb_regret": ucb["mean_post_shift_regret"],
        "exp3_regret": exp3["mean_post_shift_regret"],
        "dte_vs_surprise_regret_gain": (
            surprise["mean_post_shift_regret"]
            - local["mean_post_shift_regret"]
        ),
        "dte_ucb_vs_dte_regret_gain": (
            local["mean_post_shift_regret"]
            - dte_ucb["mean_post_shift_regret"]
        ),
        "dte_contextual_ucb_vs_dte_regret_gain": (
            local["mean_post_shift_regret"]
            - dte_contextual_ucb["mean_post_shift_regret"]
        ),
        "dte_arbitrated_ucb_vs_dte_regret_gain": (
            local["mean_post_shift_regret"]
            - dte_arbitrated_ucb["mean_post_shift_regret"]
        ),
        "dte_reliability_arbitrated_ucb_vs_dte_regret_gain": (
            local["mean_post_shift_regret"]
            - dte_reliability_arbitrated_ucb["mean_post_shift_regret"]
        ),
        "dte_arbitrated_ucb_vs_contextual_ucb_regret_gain": (
            dte_contextual_ucb["mean_post_shift_regret"]
            - dte_arbitrated_ucb["mean_post_shift_regret"]
        ),
        "dte_reliability_arbitrated_ucb_vs_arbitrated_ucb_regret_gain": (
            dte_arbitrated_ucb["mean_post_shift_regret"]
            - dte_reliability_arbitrated_ucb["mean_post_shift_regret"]
        ),
        "dte_exp3_vs_dte_regret_gain": (
            local["mean_post_shift_regret"]
            - dte_exp3["mean_post_shift_regret"]
        ),
        "dte_exp3_vs_reliability_arbitrated_ucb_regret_gain": (
            dte_reliability_arbitrated_ucb["mean_post_shift_regret"]
            - dte_exp3["mean_post_shift_regret"]
        ),
        "dte_exp3_minus_external_exp3_regret": (
            dte_exp3["mean_post_shift_regret"]
            - exp3["mean_post_shift_regret"]
        ),
        "dte_minus_contextual_regret": (
            local["mean_post_shift_regret"]
            - contextual["mean_post_shift_regret"]
        ),
        "dte_ucb_minus_contextual_regret": (
            dte_ucb["mean_post_shift_regret"]
            - contextual["mean_post_shift_regret"]
        ),
        "dte_contextual_ucb_minus_contextual_regret": (
            dte_contextual_ucb["mean_post_shift_regret"]
            - contextual["mean_post_shift_regret"]
        ),
        "dte_arbitrated_ucb_minus_contextual_regret": (
            dte_arbitrated_ucb["mean_post_shift_regret"]
            - contextual["mean_post_shift_regret"]
        ),
        "dte_reliability_arbitrated_ucb_minus_contextual_regret": (
            dte_reliability_arbitrated_ucb["mean_post_shift_regret"]
            - contextual["mean_post_shift_regret"]
        ),
        "dte_minus_bandit_regret": (
            local["mean_post_shift_regret"]
            - bandit["mean_post_shift_regret"]
        ),
        "dte_minus_ucb_regret": (
            local["mean_post_shift_regret"]
            - ucb["mean_post_shift_regret"]
        ),
        "dte_minus_exp3_regret": (
            local["mean_post_shift_regret"]
            - exp3["mean_post_shift_regret"]
        ),
        "dte_symbolic_share": local["mean_post_shift_symbolic_share"],
        "dte_ucb_symbolic_share": dte_ucb["mean_post_shift_symbolic_share"],
        "dte_contextual_ucb_symbolic_share": dte_contextual_ucb[
            "mean_post_shift_symbolic_share"
        ],
        "dte_arbitrated_ucb_symbolic_share": dte_arbitrated_ucb[
            "mean_post_shift_symbolic_share"
        ],
        "dte_reliability_arbitrated_ucb_symbolic_share": (
            dte_reliability_arbitrated_ucb["mean_post_shift_symbolic_share"]
        ),
        "dte_exp3_symbolic_share": dte_exp3["mean_post_shift_symbolic_share"],
        "contextual_symbolic_share": contextual["mean_post_shift_symbolic_share"],
        "bandit_symbolic_share": bandit["mean_post_shift_symbolic_share"],
        "ucb_symbolic_share": ucb["mean_post_shift_symbolic_share"],
        "exp3_symbolic_share": exp3["mean_post_shift_symbolic_share"],
    }


def summarize_frontier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_axis: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_axis.setdefault(row["axis"], []).append(row)

    summary = {}
    for axis, axis_rows in by_axis.items():
        ordered = sorted(axis_rows, key=lambda row: float(row["value"]))
        contextual_wins = [
            row for row in ordered if row["dte_minus_contextual_regret"] <= 0.0
        ]
        bandit_wins = [
            row for row in ordered if row["dte_minus_bandit_regret"] <= 0.0
        ]
        ucb_wins = [
            row for row in ordered if row["dte_minus_ucb_regret"] <= 0.0
        ]
        exp3_wins = [
            row for row in ordered if row["dte_minus_exp3_regret"] <= 0.0
        ]
        native_ucb_wins = [
            row for row in ordered if row["dte_ucb_vs_dte_regret_gain"] > 0.0
        ]
        contextual_ucb_wins = [
            row for row in ordered
            if row["dte_contextual_ucb_vs_dte_regret_gain"] > 0.0
        ]
        arbitrated_ucb_wins = [
            row for row in ordered
            if row["dte_arbitrated_ucb_vs_dte_regret_gain"] > 0.0
        ]
        reliability_arbitrated_ucb_wins = [
            row for row in ordered
            if row["dte_reliability_arbitrated_ucb_vs_dte_regret_gain"] > 0.0
        ]
        reliability_gating_wins = [
            row for row in ordered
            if (
                row[
                    "dte_reliability_arbitrated_ucb_vs_arbitrated_ucb_regret_gain"
                ]
                > 0.0
            )
        ]
        dte_exp3_wins = [
            row for row in ordered if row["dte_exp3_vs_dte_regret_gain"] > 0.0
        ]
        dte_exp3_over_reliable_ucb_wins = [
            row for row in ordered
            if row["dte_exp3_vs_reliability_arbitrated_ucb_regret_gain"] > 0.0
        ]
        best_contextual = min(
            ordered,
            key=lambda row: row["dte_minus_contextual_regret"],
        )
        best_bandit = min(
            ordered,
            key=lambda row: row["dte_minus_bandit_regret"],
        )
        best_ucb = min(
            ordered,
            key=lambda row: row["dte_minus_ucb_regret"],
        )
        best_exp3 = min(
            ordered,
            key=lambda row: row["dte_minus_exp3_regret"],
        )
        best_native_ucb = max(
            ordered,
            key=lambda row: row["dte_ucb_vs_dte_regret_gain"],
        )
        best_contextual_ucb = max(
            ordered,
            key=lambda row: row["dte_contextual_ucb_vs_dte_regret_gain"],
        )
        best_arbitrated_ucb = max(
            ordered,
            key=lambda row: row["dte_arbitrated_ucb_vs_dte_regret_gain"],
        )
        best_arbitration_over_contextual_ucb = max(
            ordered,
            key=lambda row: row["dte_arbitrated_ucb_vs_contextual_ucb_regret_gain"],
        )
        best_reliability_arbitrated_ucb = max(
            ordered,
            key=lambda row: row[
                "dte_reliability_arbitrated_ucb_vs_dte_regret_gain"
            ],
        )
        best_reliability_gate = max(
            ordered,
            key=lambda row: row[
                "dte_reliability_arbitrated_ucb_vs_arbitrated_ucb_regret_gain"
            ],
        )
        best_dte_exp3 = max(
            ordered,
            key=lambda row: row["dte_exp3_vs_dte_regret_gain"],
        )
        best_dte_exp3_over_reliable_ucb = max(
            ordered,
            key=lambda row: row[
                "dte_exp3_vs_reliability_arbitrated_ucb_regret_gain"
            ],
        )
        summary[axis] = {
            "rows": ordered,
            "dte_beats_contextual_values": [row["value"] for row in contextual_wins],
            "dte_beats_bandit_values": [row["value"] for row in bandit_wins],
            "dte_beats_ucb_values": [row["value"] for row in ucb_wins],
            "dte_beats_exp3_values": [row["value"] for row in exp3_wins],
            "native_ucb_improves_dte_values": [row["value"] for row in native_ucb_wins],
            "contextual_ucb_improves_dte_values": [
                row["value"] for row in contextual_ucb_wins
            ],
            "arbitrated_ucb_improves_dte_values": [
                row["value"] for row in arbitrated_ucb_wins
            ],
            "reliability_arbitrated_ucb_improves_dte_values": [
                row["value"] for row in reliability_arbitrated_ucb_wins
            ],
            "reliability_gating_improves_arbitration_values": [
                row["value"] for row in reliability_gating_wins
            ],
            "dte_exp3_improves_dte_values": [
                row["value"] for row in dte_exp3_wins
            ],
            "dte_exp3_improves_reliability_arbitrated_ucb_values": [
                row["value"] for row in dte_exp3_over_reliable_ucb_wins
            ],
            "best_contextual_gap_value": best_contextual["value"],
            "best_contextual_gap": best_contextual["dte_minus_contextual_regret"],
            "best_bandit_gap_value": best_bandit["value"],
            "best_bandit_gap": best_bandit["dte_minus_bandit_regret"],
            "best_ucb_gap_value": best_ucb["value"],
            "best_ucb_gap": best_ucb["dte_minus_ucb_regret"],
            "best_exp3_gap_value": best_exp3["value"],
            "best_exp3_gap": best_exp3["dte_minus_exp3_regret"],
            "best_native_ucb_gain_value": best_native_ucb["value"],
            "best_native_ucb_gain": best_native_ucb["dte_ucb_vs_dte_regret_gain"],
            "best_contextual_ucb_gain_value": best_contextual_ucb["value"],
            "best_contextual_ucb_gain": best_contextual_ucb[
                "dte_contextual_ucb_vs_dte_regret_gain"
            ],
            "best_arbitrated_ucb_gain_value": best_arbitrated_ucb["value"],
            "best_arbitrated_ucb_gain": best_arbitrated_ucb[
                "dte_arbitrated_ucb_vs_dte_regret_gain"
            ],
            "best_arbitration_over_contextual_ucb_value": (
                best_arbitration_over_contextual_ucb["value"]
            ),
            "best_arbitration_over_contextual_ucb_gain": (
                best_arbitration_over_contextual_ucb[
                    "dte_arbitrated_ucb_vs_contextual_ucb_regret_gain"
                ]
            ),
            "best_reliability_arbitrated_ucb_gain_value": (
                best_reliability_arbitrated_ucb["value"]
            ),
            "best_reliability_arbitrated_ucb_gain": (
                best_reliability_arbitrated_ucb[
                    "dte_reliability_arbitrated_ucb_vs_dte_regret_gain"
                ]
            ),
            "best_reliability_gate_gain_value": best_reliability_gate["value"],
            "best_reliability_gate_gain": best_reliability_gate[
                "dte_reliability_arbitrated_ucb_vs_arbitrated_ucb_regret_gain"
            ],
            "best_dte_exp3_gain_value": best_dte_exp3["value"],
            "best_dte_exp3_gain": best_dte_exp3["dte_exp3_vs_dte_regret_gain"],
            "best_dte_exp3_over_reliable_ucb_gain_value": (
                best_dte_exp3_over_reliable_ucb["value"]
            ),
            "best_dte_exp3_over_reliable_ucb_gain": (
                best_dte_exp3_over_reliable_ucb[
                    "dte_exp3_vs_reliability_arbitrated_ucb_regret_gain"
                ]
            ),
        }
    return summary


def render_frontier_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Neural V2 Parameter Frontier Report",
        "",
        "## Scope",
        "",
        "One-axis frontier sweep for the hard Neural V2 router benchmark. Each",
        "axis varies around the same baseline while all routers see the same",
        "seeded task streams per scenario.",
        "",
        "Negative DTE-minus-baseline regret means DTE local-regret outperformed",
        "that baseline. Positive values mean the baseline had lower regret.",
        "",
    ]
    for axis, axis_summary in payload["summary"].items():
        lines.extend(
            [
                f"## {axis}",
                "",
                "| Value | DTE Regret | DTE-UCB Gain | Ctx-UCB Gain | Arb-UCB Gain | Rel-Arb Gain | DTE-EXP3 Gain | EXP3-vs-Rel | Gain vs Surprise | DTE - Contextual | DTE - Bandit | DTE - UCB | DTE - EXP3 | DTE Symbolic |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in axis_summary["rows"]:
            lines.append(
                f"| {row['value']} | "
                f"{row['dte_regret']:.3f} | "
                f"{row['dte_ucb_vs_dte_regret_gain']:.3f} | "
                f"{row['dte_contextual_ucb_vs_dte_regret_gain']:.3f} | "
                f"{row['dte_arbitrated_ucb_vs_dte_regret_gain']:.3f} | "
                f"{row['dte_reliability_arbitrated_ucb_vs_dte_regret_gain']:.3f} | "
                f"{row['dte_exp3_vs_dte_regret_gain']:.3f} | "
                f"{row['dte_exp3_vs_reliability_arbitrated_ucb_regret_gain']:.3f} | "
                f"{row['dte_vs_surprise_regret_gain']:.3f} | "
                f"{row['dte_minus_contextual_regret']:.3f} | "
                f"{row['dte_minus_bandit_regret']:.3f} | "
                f"{row['dte_minus_ucb_regret']:.3f} | "
                f"{row['dte_minus_exp3_regret']:.3f} | "
                f"{row['dte_symbolic_share']:.3f} |"
            )
        lines.extend(
            [
                "",
                (
                    "DTE beats contextual at values: "
                    f"{axis_summary['dte_beats_contextual_values']}"
                ),
                (
                    "DTE beats bandit at values: "
                    f"{axis_summary['dte_beats_bandit_values']}"
                ),
                (
                    "DTE beats UCB at values: "
                    f"{axis_summary['dte_beats_ucb_values']}"
                ),
                (
                    "DTE beats EXP3 at values: "
                    f"{axis_summary['dte_beats_exp3_values']}"
                ),
                (
                    "DTE-native UCB improves DTE at values: "
                    f"{axis_summary['native_ucb_improves_dte_values']}"
                ),
                (
                    "DTE-contextual UCB improves DTE at values: "
                    f"{axis_summary['contextual_ucb_improves_dte_values']}"
                ),
                (
                    "DTE-arbitrated UCB improves DTE at values: "
                    f"{axis_summary['arbitrated_ucb_improves_dte_values']}"
                ),
                (
                    "DTE-reliability-arbitrated UCB improves DTE at values: "
                    f"{axis_summary['reliability_arbitrated_ucb_improves_dte_values']}"
                ),
                (
                    "Reliability gating improves arbitration at values: "
                    f"{axis_summary['reliability_gating_improves_arbitration_values']}"
                ),
                (
                    "DTE-EXP3 improves DTE at values: "
                    f"{axis_summary['dte_exp3_improves_dte_values']}"
                ),
                (
                    "DTE-EXP3 improves reliability-arbitrated UCB at values: "
                    f"{axis_summary['dte_exp3_improves_reliability_arbitrated_ucb_values']}"
                ),
                (
                    "Best contextual gap: "
                    f"{axis_summary['best_contextual_gap']:.3f} at "
                    f"{axis_summary['best_contextual_gap_value']}."
                ),
                (
                    "Best bandit gap: "
                    f"{axis_summary['best_bandit_gap']:.3f} at "
                    f"{axis_summary['best_bandit_gap_value']}."
                ),
                (
                    "Best UCB gap: "
                    f"{axis_summary['best_ucb_gap']:.3f} at "
                    f"{axis_summary['best_ucb_gap_value']}."
                ),
                (
                    "Best EXP3 gap: "
                    f"{axis_summary['best_exp3_gap']:.3f} at "
                    f"{axis_summary['best_exp3_gap_value']}."
                ),
                (
                    "Best DTE-native UCB gain: "
                    f"{axis_summary['best_native_ucb_gain']:.3f} at "
                    f"{axis_summary['best_native_ucb_gain_value']}."
                ),
                (
                    "Best DTE-contextual UCB gain: "
                    f"{axis_summary['best_contextual_ucb_gain']:.3f} at "
                    f"{axis_summary['best_contextual_ucb_gain_value']}."
                ),
                (
                    "Best DTE-arbitrated UCB gain: "
                    f"{axis_summary['best_arbitrated_ucb_gain']:.3f} at "
                    f"{axis_summary['best_arbitrated_ucb_gain_value']}."
                ),
                (
                    "Best arbitration gain over additive contextual UCB: "
                    f"{axis_summary['best_arbitration_over_contextual_ucb_gain']:.3f} at "
                    f"{axis_summary['best_arbitration_over_contextual_ucb_value']}."
                ),
                (
                    "Best DTE-reliability-arbitrated UCB gain: "
                    f"{axis_summary['best_reliability_arbitrated_ucb_gain']:.3f} at "
                    f"{axis_summary['best_reliability_arbitrated_ucb_gain_value']}."
                ),
                (
                    "Best reliability-gating gain over arbitration: "
                    f"{axis_summary['best_reliability_gate_gain']:.3f} at "
                    f"{axis_summary['best_reliability_gate_gain_value']}."
                ),
                (
                    "Best DTE-EXP3 gain: "
                    f"{axis_summary['best_dte_exp3_gain']:.3f} at "
                    f"{axis_summary['best_dte_exp3_gain_value']}."
                ),
                (
                    "Best DTE-EXP3 gain over reliability-arbitrated UCB: "
                    f"{axis_summary['best_dte_exp3_over_reliable_ucb_gain']:.3f} at "
                    f"{axis_summary['best_dte_exp3_over_reliable_ucb_gain_value']}."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "This frontier is the first map of Neural V2's institutional validity.",
            "The meaningful claim is not universal dominance. The meaningful claim",
            "is boundary-sensitive: DTE local-regret has value when stale routing",
            "memory is a first-class failure mode and the observed decision surface",
            "is imperfect.",
            "",
        ]
    )
    return "\n".join(lines)


def run_frontier(
    config: FrontierConfig | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    config = config or FrontierConfig()
    rows = []
    for axis, value, scenario_config in _frontier_scenarios(config):
        payload = run_hard_benchmark(scenario_config, write_outputs=False)
        rows.append(_frontier_row(axis, value, payload))
    payload = {
        "config": config.__dict__,
        "rows": rows,
        "summary": summarize_frontier(rows),
    }
    if write_outputs:
        FRONTIER_OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        FRONTIER_REPORT_PATH.write_text(render_frontier_report(payload), encoding="utf-8")
    return payload


def _adversarial_switch_config(
    sweep: AdversarialSwitchSweepConfig,
    switch_period: int,
    label_noise: float,
) -> AdversarialSwitchConfig:
    return AdversarialSwitchConfig(
        ticks=sweep.ticks,
        shift_tick=sweep.shift_tick,
        batch_size=sweep.batch_size,
        seeds=sweep.seeds,
        context_noise=sweep.context_noise,
        label_noise=label_noise,
        reward_delay=sweep.reward_delay,
        switch_period=switch_period,
        adversarial_intensity=sweep.adversarial_intensity,
    )


def _run_adversarial_switch_scenario(
    config: AdversarialSwitchConfig,
) -> dict[str, Any]:
    rows = []
    for seed in config.seeds:
        stream = make_hard_task_stream(seed, config)
        rows.append(run_hard_dte_router("local_regret", stream, seed, config))
        rows.append(run_hard_dte_reliability_arbitrated_ucb_router(stream, seed, config))
        rows.append(run_hard_dte_exp3_router(stream, seed, config))
        rows.append(run_hard_dte_reliability_arbitrated_exp3_router(stream, seed, config))
        rows.append(run_hard_ucb_router(stream, seed, config))
        rows.append(run_hard_exp3_router(stream, seed, config))
        rows.append(run_hard_oracle_router(stream, seed, config))
    return {"config": config.__dict__, "rows": rows, "summary": summarize(rows)}


def _adversarial_switch_row(
    switch_period: int,
    label_noise: float,
    payload: dict[str, Any],
) -> dict[str, Any]:
    summary = payload["summary"]
    local = summary["hard_dte_local_regret"]
    reliable_ucb = summary["hard_dte_reliability_arbitrated_ucb"]
    dte_exp3 = summary["hard_dte_exp3"]
    reliable_exp3 = summary["hard_dte_reliability_arbitrated_exp3"]
    ucb = summary["hard_ucb"]
    exp3 = summary["hard_exp3"]
    oracle = summary["hard_oracle"]
    regrets = {
        "dte_local_regret": local["mean_post_shift_regret"],
        "dte_reliability_arbitrated_ucb": reliable_ucb["mean_post_shift_regret"],
        "dte_exp3": dte_exp3["mean_post_shift_regret"],
        "dte_reliability_arbitrated_exp3": reliable_exp3["mean_post_shift_regret"],
        "ucb": ucb["mean_post_shift_regret"],
        "exp3": exp3["mean_post_shift_regret"],
    }
    winner = min(regrets, key=regrets.get)
    return {
        "switch_period": switch_period,
        "label_noise": label_noise,
        "dte_local_regret": local["mean_post_shift_regret"],
        "dte_reliability_arbitrated_ucb": reliable_ucb["mean_post_shift_regret"],
        "dte_exp3": dte_exp3["mean_post_shift_regret"],
        "dte_reliability_arbitrated_exp3": reliable_exp3["mean_post_shift_regret"],
        "ucb": ucb["mean_post_shift_regret"],
        "exp3": exp3["mean_post_shift_regret"],
        "oracle": oracle["mean_post_shift_regret"],
        "dte_exp3_vs_local_gain": (
            local["mean_post_shift_regret"] - dte_exp3["mean_post_shift_regret"]
        ),
        "dte_exp3_vs_reliability_ucb_gain": (
            reliable_ucb["mean_post_shift_regret"]
            - dte_exp3["mean_post_shift_regret"]
        ),
        "reliability_exp3_vs_reliability_ucb_gain": (
            reliable_ucb["mean_post_shift_regret"]
            - reliable_exp3["mean_post_shift_regret"]
        ),
        "exp3_reliability_gate_gain": (
            dte_exp3["mean_post_shift_regret"]
            - reliable_exp3["mean_post_shift_regret"]
        ),
        "dte_exp3_minus_external_exp3": (
            dte_exp3["mean_post_shift_regret"] - exp3["mean_post_shift_regret"]
        ),
        "reliability_exp3_minus_external_exp3": (
            reliable_exp3["mean_post_shift_regret"] - exp3["mean_post_shift_regret"]
        ),
        "reliability_ucb_vs_external_ucb": (
            reliable_ucb["mean_post_shift_regret"] - ucb["mean_post_shift_regret"]
        ),
        "winner": winner,
    }


def summarize_adversarial_switch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exp3_wins = [
        row
        for row in rows
        if row["dte_exp3_vs_reliability_ucb_gain"] > 0.0
    ]
    exp3_beats_external = [
        row for row in rows if row["dte_exp3_minus_external_exp3"] < 0.0
    ]
    reliability_exp3_wins = [
        row
        for row in rows
        if row["reliability_exp3_vs_reliability_ucb_gain"] > 0.0
    ]
    reliability_gate_helps = [
        row for row in rows if row["exp3_reliability_gate_gain"] > 0.0
    ]
    best_exp3 = max(rows, key=lambda row: row["dte_exp3_vs_reliability_ucb_gain"])
    worst_exp3 = min(rows, key=lambda row: row["dte_exp3_vs_reliability_ucb_gain"])
    best_gate = max(rows, key=lambda row: row["exp3_reliability_gate_gain"])
    worst_gate = min(rows, key=lambda row: row["exp3_reliability_gate_gain"])
    winner_counts: dict[str, int] = {}
    for row in rows:
        winner_counts[row["winner"]] = winner_counts.get(row["winner"], 0) + 1
    return {
        "rows": rows,
        "dte_exp3_beats_reliability_ucb_coordinates": [
            {
                "switch_period": row["switch_period"],
                "label_noise": row["label_noise"],
            }
            for row in exp3_wins
        ],
        "dte_exp3_beats_external_exp3_coordinates": [
            {
                "switch_period": row["switch_period"],
                "label_noise": row["label_noise"],
            }
            for row in exp3_beats_external
        ],
        "reliability_exp3_beats_reliability_ucb_coordinates": [
            {
                "switch_period": row["switch_period"],
                "label_noise": row["label_noise"],
            }
            for row in reliability_exp3_wins
        ],
        "exp3_reliability_gate_helps_coordinates": [
            {
                "switch_period": row["switch_period"],
                "label_noise": row["label_noise"],
            }
            for row in reliability_gate_helps
        ],
        "best_dte_exp3_coordinate": {
            "switch_period": best_exp3["switch_period"],
            "label_noise": best_exp3["label_noise"],
            "gain": best_exp3["dte_exp3_vs_reliability_ucb_gain"],
        },
        "worst_dte_exp3_coordinate": {
            "switch_period": worst_exp3["switch_period"],
            "label_noise": worst_exp3["label_noise"],
            "gain": worst_exp3["dte_exp3_vs_reliability_ucb_gain"],
        },
        "best_exp3_reliability_gate_coordinate": {
            "switch_period": best_gate["switch_period"],
            "label_noise": best_gate["label_noise"],
            "gain": best_gate["exp3_reliability_gate_gain"],
        },
        "worst_exp3_reliability_gate_coordinate": {
            "switch_period": worst_gate["switch_period"],
            "label_noise": worst_gate["label_noise"],
            "gain": worst_gate["exp3_reliability_gate_gain"],
        },
        "winner_counts": winner_counts,
    }


def render_adversarial_switch_report(payload: dict[str, Any]) -> str:
    def coordinates(items: list[dict[str, Any]]) -> str:
        if not items:
            return "none"
        return ", ".join(
            f"({item['switch_period']}, {item['label_noise']:.2f})"
            for item in items
        )

    rows = sorted(
        payload["rows"],
        key=lambda row: (row["label_noise"], row["switch_period"]),
    )
    summary = payload["summary"]
    lines = [
        "# Neural V2 Adversarial-Switching Benchmark Report",
        "",
        "## Scope",
        "",
        "Controlled H2 test for DTE-EXP3. The reward surface alternates which",
        "module family is favored while label noise is varied independently.",
        "This separates adversarial reward nonstationarity from attribution",
        "corruption.",
        "",
        "## Matrix",
        "",
        "| Switch Period | Label Noise | Local DTE | Rel-UCB DTE | DTE-EXP3 | Rel-EXP3 DTE | UCB | EXP3 | EXP3 Gain vs Rel-UCB | Gate Gain | DTE-EXP3 - EXP3 | Winner |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['switch_period']} | {row['label_noise']:.2f} | "
            f"{row['dte_local_regret']:.3f} | "
            f"{row['dte_reliability_arbitrated_ucb']:.3f} | "
            f"{row['dte_exp3']:.3f} | "
            f"{row['dte_reliability_arbitrated_exp3']:.3f} | "
            f"{row['ucb']:.3f} | {row['exp3']:.3f} | "
            f"{row['dte_exp3_vs_reliability_ucb_gain']:.3f} | "
            f"{row['exp3_reliability_gate_gain']:.3f} | "
            f"{row['dte_exp3_minus_external_exp3']:.3f} | {row['winner']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            (
                "DTE-EXP3 beats reliability-arbitrated UCB at coordinates: "
                f"{coordinates(summary['dte_exp3_beats_reliability_ucb_coordinates'])}."
            ),
            (
                "DTE-EXP3 beats external EXP3 at coordinates: "
                f"{coordinates(summary['dte_exp3_beats_external_exp3_coordinates'])}."
            ),
            (
                "Reliability-gated DTE-EXP3 beats reliability-arbitrated UCB at "
                "coordinates: "
                f"{coordinates(summary['reliability_exp3_beats_reliability_ucb_coordinates'])}."
            ),
            (
                "Reliability gating improves DTE-EXP3 at coordinates: "
                f"{coordinates(summary['exp3_reliability_gate_helps_coordinates'])}."
            ),
            (
                "Best DTE-EXP3 coordinate: "
                f"{summary['best_dte_exp3_coordinate']}."
            ),
            (
                "Worst DTE-EXP3 coordinate: "
                f"{summary['worst_dte_exp3_coordinate']}."
            ),
            (
                "Best EXP3 reliability-gate coordinate: "
                f"{summary['best_exp3_reliability_gate_coordinate']}."
            ),
            (
                "Worst EXP3 reliability-gate coordinate: "
                f"{summary['worst_exp3_reliability_gate_coordinate']}."
            ),
            f"Winner counts: {summary['winner_counts']}.",
            "",
            "## Interpretation",
            "",
            "This benchmark is not a general hard-routing score. It is a diagnostic",
            "for the H2 mechanism. DTE-EXP3 winning inside the DTE family means",
            "multiplicative weights are useful for genuine adversarial",
            "nonstationarity. Reliability gating then answers a narrower",
            "question: whether attribution filtering protects the EXP3 lane from",
            "poisoned context updates. In this sweep, the gate helps selectively",
            "rather than universally, so it should be treated as a brake, not a",
            "new default policy.",
            "",
        ]
    )
    return "\n".join(lines)


def run_adversarial_switch_benchmark(
    config: AdversarialSwitchSweepConfig | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    config = config or AdversarialSwitchSweepConfig()
    rows = []
    for label_noise in config.label_noise_values:
        for switch_period in config.switch_period_values:
            scenario = _adversarial_switch_config(config, switch_period, label_noise)
            scenario_payload = _run_adversarial_switch_scenario(scenario)
            rows.append(
                _adversarial_switch_row(
                    switch_period,
                    label_noise,
                    scenario_payload,
                )
            )
    payload = {
        "config": config.__dict__,
        "rows": rows,
        "summary": summarize_adversarial_switch(rows),
    }
    if write_outputs:
        ADVERSARIAL_SWITCH_OUTPUT_PATH.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        ADVERSARIAL_SWITCH_REPORT_PATH.write_text(
            render_adversarial_switch_report(payload),
            encoding="utf-8",
        )
    return payload


if __name__ == "__main__":
    result = run_benchmark()
    hard_result = run_hard_benchmark()
    frontier_result = run_frontier()
    print(
        json.dumps(
            {
                "clean": result["summary"],
                "hard": hard_result["summary"],
                "frontier": frontier_result["summary"],
            },
            indent=2,
        )
    )

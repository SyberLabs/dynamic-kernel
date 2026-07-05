"""
Ant foraging prototype for the Dynamic Topology Engine.

The biological interpretation is deliberately narrow: ants route through a
small terrain graph, individual intent vectors drift after visits, and
successful food returns deposit a pheromone-like friction reduction on the
edges that produced the return. The point is not to mimic every ant behavior;
it is to test whether DTE's adaptive circulation kernel can distinguish
efficient collective routing from brittle trail lock-in.

Usage:
    .venv\\Scripts\\python.exe ant_foraging_dte.py --quick
    .venv\\Scripts\\python.exe ant_foraging_dte.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kernel import DynamicTopologyKernel, topology_from_edges


FEATURE_LABELS = [
    "Food Reward",
    "Pheromone Affinity",
    "Safety Preference",
    "Nest Pull",
    "Hazard Arousal",
    "Exploration Value",
    "Moisture Shelter",
]


NODES = {
    "Nest": [0.12, 0.10, 0.88, 1.00, 0.05, 0.42, 0.42],
    "Trail Fork": [0.32, 0.42, 0.54, 0.42, 0.20, 1.00, 0.36],
    "Short Trail": [0.56, 0.70, 0.34, 0.18, 0.26, 0.26, 0.22],
    "Long Trail": [0.46, 0.46, 0.62, 0.20, 0.14, 0.68, 0.60],
    "Risky Ridge": [0.78, 0.48, 0.14, 0.08, 1.00, 0.38, 0.10],
    "Shaded Detour": [0.38, 0.40, 0.82, 0.24, 0.08, 0.74, 0.98],
    "Rich Food Patch": [1.00, 0.68, 0.26, 0.05, 0.18, 0.08, 0.32],
    "Sparse Food Patch": [0.64, 0.40, 0.58, 0.06, 0.10, 0.46, 0.72],
    "Depleted Patch": [0.08, 0.22, 0.44, 0.04, 0.18, 0.82, 0.34],
    "Hazard Zone": [0.10, 0.04, 0.02, 0.02, 1.00, 0.24, 0.06],
    "Return Corridor": [0.20, 0.76, 0.78, 0.95, 0.04, 0.14, 0.38],
}


EDGES = [
    ("Nest", "Trail Fork", 1.2),
    ("Trail Fork", "Short Trail", 0.8),
    ("Trail Fork", "Long Trail", 1.5),
    ("Trail Fork", "Risky Ridge", 1.0),
    ("Trail Fork", "Shaded Detour", 1.9),
    ("Short Trail", "Rich Food Patch", 0.9),
    ("Long Trail", "Sparse Food Patch", 1.0),
    ("Risky Ridge", "Rich Food Patch", 0.7),
    ("Risky Ridge", "Hazard Zone", 0.45),
    ("Shaded Detour", "Sparse Food Patch", 0.8),
    ("Shaded Detour", "Rich Food Patch", 1.6),
    ("Rich Food Patch", "Return Corridor", 0.9),
    ("Sparse Food Patch", "Return Corridor", 1.0),
    ("Depleted Patch", "Return Corridor", 1.2),
    ("Hazard Zone", "Nest", 2.4),
    ("Return Corridor", "Nest", 0.8),
]


FOOD_NODES = ("Rich Food Patch", "Sparse Food Patch")
BRANCH_NODES = ("Short Trail", "Long Trail", "Risky Ridge", "Shaded Detour")


INTENTS = {
    "Scout": [0.42, 0.18, 0.58, 0.22, 0.12, 1.00, 0.62],
    "Exploiter": [1.00, 0.92, 0.42, 0.18, 0.10, 0.12, 0.24],
    "Risk Averse": [0.62, 0.40, 1.00, 0.28, 0.00, 0.42, 1.00],
    "Return Carrier": [0.10, 0.72, 0.78, 1.00, 0.00, 0.10, 0.42],
}


@dataclass(frozen=True)
class SimulationConfig:
    agents: int = 96
    steps: int = 110
    seed: int = 17
    alpha: float = 1.25
    beta_strength: float = 0.85
    feedback_rate: float = 0.16
    return_feedback_rate: float = 0.36
    max_path_memory: int = 18


@dataclass(frozen=True)
class ForagingScenario:
    name: str
    food_inventory: dict[str, int]
    initial_friction: dict[tuple[str, str], float] = field(default_factory=dict)
    shock_step: int | None = None
    shock_friction: dict[tuple[str, str], float] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class ForagingPolicy:
    name: str
    pheromone_deposit: float
    evaporation: float
    scout_share: float
    risk_averse_share: float
    temperature: float
    pheromone_cap: float = 2.6


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return arr / norms


def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    p = np.array([value / total for value in counts.values() if value > 0.0])
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / math.log(len(counts)))


def _sample_next(rng: np.random.Generator, transition_rows: np.ndarray) -> np.ndarray:
    u = rng.random(transition_rows.shape[0])
    cdf = np.cumsum(transition_rows, axis=1)
    sampled = [
        min(int(np.searchsorted(cdf[i], u[i], side="right")), transition_rows.shape[1] - 1)
        for i in range(transition_rows.shape[0])
    ]
    return np.array(sampled, dtype=int)


def topology():
    return topology_from_edges(
        nodes={label: np.array(vec, dtype=float) for label, vec in NODES.items()},
        edges=EDGES,
        undirected=False,
    )


def scenarios() -> list[ForagingScenario]:
    return [
        ForagingScenario(
            name="baseline_rich_short_path",
            food_inventory={"Rich Food Patch": 1500, "Sparse Food Patch": 1000},
            notes="A rich patch is reachable through the short path; a lower-yield reserve patch remains viable.",
        ),
        ForagingScenario(
            name="rich_patch_depletion",
            food_inventory={"Rich Food Patch": 40, "Sparse Food Patch": 1800},
            notes="The initially attractive rich patch is scarce, so path memory must release toward the sparse patch.",
        ),
        ForagingScenario(
            name="short_path_obstruction",
            food_inventory={"Rich Food Patch": 1500, "Sparse Food Patch": 1200},
            shock_step=46,
            shock_friction={
                ("Trail Fork", "Short Trail"): -1.35,
                ("Short Trail", "Rich Food Patch"): -2.65,
            },
            notes="A late obstruction penalizes the shortest route after pheromone trails have already formed.",
        ),
        ForagingScenario(
            name="predator_ridge",
            food_inventory={"Rich Food Patch": 1400, "Sparse Food Patch": 1000},
            initial_friction={
                ("Trail Fork", "Risky Ridge"): -0.75,
                ("Risky Ridge", "Rich Food Patch"): -0.95,
                ("Risky Ridge", "Hazard Zone"): 0.65,
            },
            notes="A risky shortcut carries high food reward but exposes ants to the hazard branch.",
        ),
    ]


def policies() -> list[ForagingPolicy]:
    return [
        ForagingPolicy(
            name="no_pheromone_control",
            pheromone_deposit=0.0,
            evaporation=0.0,
            scout_share=0.30,
            risk_averse_share=0.18,
            temperature=1.05,
        ),
        ForagingPolicy(
            name="balanced_pheromone",
            pheromone_deposit=0.16,
            evaporation=0.035,
            scout_share=0.24,
            risk_averse_share=0.16,
            temperature=0.95,
        ),
        ForagingPolicy(
            name="strong_pheromone_lockin",
            pheromone_deposit=0.42,
            evaporation=0.015,
            scout_share=0.10,
            risk_averse_share=0.12,
            temperature=0.80,
        ),
        ForagingPolicy(
            name="high_exploration",
            pheromone_deposit=0.10,
            evaporation=0.060,
            scout_share=0.48,
            risk_averse_share=0.18,
            temperature=1.25,
        ),
        ForagingPolicy(
            name="rapid_evaporation",
            pheromone_deposit=0.24,
            evaporation=0.120,
            scout_share=0.26,
            risk_averse_share=0.16,
            temperature=1.00,
        ),
    ]


def _friction_matrix(
    topo,
    edge_friction: dict[tuple[str, str], float],
) -> np.ndarray:
    idx = {label: i for i, label in enumerate(topo.labels)}
    matrix = np.zeros((topo.N, topo.N), dtype=float)
    for (src, dst), value in edge_friction.items():
        matrix[idx[src], idx[dst]] += float(value)
    return matrix


def _scenario_payload(scenario: ForagingScenario) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "food_inventory": dict(scenario.food_inventory),
        "initial_friction": {
            f"{src}->{dst}": value for (src, dst), value in scenario.initial_friction.items()
        },
        "shock_step": scenario.shock_step,
        "shock_friction": {
            f"{src}->{dst}": value for (src, dst), value in scenario.shock_friction.items()
        },
        "notes": scenario.notes,
    }


def _build_kernel(config: SimulationConfig, policy: ForagingPolicy) -> DynamicTopologyKernel:
    topo = topology()
    beta = np.full((topo.N, topo.N), config.beta_strength, dtype=float)
    node_bias = np.zeros(topo.N, dtype=float)
    node_bias[topo.labels.index("Nest")] = 0.18
    node_bias[topo.labels.index("Trail Fork")] = 0.06
    return DynamicTopologyKernel(
        topology=topo,
        alpha=config.alpha,
        beta=beta,
        feedback_rate=config.feedback_rate,
        temperature=policy.temperature,
        feedback_noise=0.0,
        node_bias=node_bias,
    )


def _assign_intents(
    rng: np.random.Generator,
    config: SimulationConfig,
    policy: ForagingPolicy,
) -> tuple[np.ndarray, list[str]]:
    exploiter_share = max(0.0, 1.0 - policy.scout_share - policy.risk_averse_share)
    names = ["Scout", "Exploiter", "Risk Averse"]
    probs = np.array([policy.scout_share, exploiter_share, policy.risk_averse_share], dtype=float)
    probs = probs / probs.sum()
    classes = rng.choice(names, size=config.agents, p=probs)
    telemetry = np.array([_normalize(np.array(INTENTS[name], dtype=float)) for name in classes])
    return telemetry, [str(name) for name in classes]


def _classify(row: dict[str, Any], baseline: dict[str, Any] | None = None) -> str:
    if row["hazard_rate"] >= 0.08:
        return "unsafe_shortcut"
    if row["empty_food_visit_rate"] >= 0.34 and row["food_completion_rate"] < 0.50:
        return "depletion_trap"
    if row["lock_in_index"] >= 0.78 and row["shock_recovery_steps"] is None and row["scenario"].endswith("obstruction"):
        return "fragile_lockin"
    if row["lock_in_index"] >= 0.78 and row["food_completion_rate"] < 0.52:
        return "wasteful_lockin"
    if baseline is not None:
        gain = row["food_completion_rate"] - baseline["food_completion_rate"]
        if gain >= 0.004 and row["lock_in_index"] <= 0.72:
            return "efficient_adaptation"
        if gain >= 0.004 and row["lock_in_index"] > 0.72:
            return "productive_lockin"
    if row["fork_entropy"] >= 0.72 and row["food_completion_rate"] < 0.25:
        return "exploratory_underuse"
    return "neutral_circulation"


def _memory_staleness_diagnostics(
    topo,
    telemetries: np.ndarray,
    pheromone: np.ndarray,
    edge_counts: np.ndarray,
    food_remaining: dict[str, int],
    visits_by_source: dict[str, int],
    empty_food_visits: int,
) -> dict[str, Any]:
    idx = {label: i for i, label in enumerate(topo.labels)}
    stale_food = [name for name in FOOD_NODES if food_remaining.get(name, 0) <= 0]
    active_food = [name for name in FOOD_NODES if food_remaining.get(name, 0) > 0]
    if not stale_food:
        return {
            "stale_nodes": [],
            "structural_stale_flow": 0.0,
            "preference_stale_concentration": 0.0,
            "state_stale_alignment": 0.0,
            "state_stale_alignment_gap": 0.0,
            "dominant_stale_memory_layer": "none",
        }

    stale_idx = [idx[name] for name in stale_food]
    active_idx = [idx[name] for name in active_food]
    total_food_visits = max(1, sum(visits_by_source.values()))
    structural_stale_flow = empty_food_visits / total_food_visits

    edge_mask = np.isfinite(topo.distance_matrix) & (topo.distance_matrix > 0.0)
    stale_incident = np.zeros_like(edge_mask, dtype=bool)
    for node_idx in stale_idx:
        stale_incident[:, node_idx] = edge_mask[:, node_idx]
        stale_incident[node_idx, :] = edge_mask[node_idx, :]
    total_pheromone = float(np.sum(pheromone[edge_mask]))
    stale_pheromone = float(np.sum(pheromone[stale_incident]))
    preference_stale_concentration = stale_pheromone / total_pheromone if total_pheromone > 0 else 0.0

    stale_alignment = telemetries @ topo.node_features[stale_idx].T
    stale_score = float(np.mean(np.max(stale_alignment, axis=1)))
    if active_idx:
        active_alignment = telemetries @ topo.node_features[active_idx].T
        active_score = float(np.mean(np.max(active_alignment, axis=1)))
    else:
        active_score = 0.0
    state_stale_alignment = stale_score / max(stale_score + active_score, 1e-9)
    state_stale_alignment_gap = stale_score - active_score

    layer_scores = {
        "structural_memory": structural_stale_flow,
        "preference_memory": preference_stale_concentration,
        "state_memory": max(0.0, state_stale_alignment - 0.5) * 2.0,
    }
    dominant = max(layer_scores, key=layer_scores.get)
    if layer_scores[dominant] <= 0.0:
        dominant = "none"

    return {
        "stale_nodes": stale_food,
        "structural_stale_flow": round(float(structural_stale_flow), 4),
        "preference_stale_concentration": round(float(preference_stale_concentration), 4),
        "state_stale_alignment": round(float(state_stale_alignment), 4),
        "state_stale_alignment_gap": round(float(state_stale_alignment_gap), 4),
        "dominant_stale_memory_layer": dominant,
    }


def simulate(
    config: SimulationConfig,
    scenario: ForagingScenario,
    policy: ForagingPolicy,
    seed_offset: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed + seed_offset)
    kernel = _build_kernel(config, policy)
    topo = kernel.topo
    idx = {label: i for i, label in enumerate(topo.labels)}

    nest = idx["Nest"]
    hazard = idx["Hazard Zone"]
    food_idx = {idx[name]: name for name in FOOD_NODES}
    branch_idx = {idx[name]: name for name in BRANCH_NODES}
    fork = idx["Trail Fork"]

    positions = np.full(config.agents, nest, dtype=int)
    telemetries, classes = _assign_intents(rng, config, policy)
    base_intents = np.array([_normalize(np.array(INTENTS[name], dtype=float)) for name in classes])
    return_intent = _normalize(np.array(INTENTS["Return Carrier"], dtype=float))

    carrying = np.zeros(config.agents, dtype=bool)
    carried_source: list[str | None] = [None for _ in range(config.agents)]
    paths: list[list[tuple[int, int]]] = [[] for _ in range(config.agents)]

    food_remaining = dict(scenario.food_inventory)
    initial_food_total = sum(food_remaining.values())
    static_friction = _friction_matrix(topo, scenario.initial_friction)
    shock_friction = _friction_matrix(topo, scenario.shock_friction)
    pheromone = np.zeros((topo.N, topo.N), dtype=float)

    edge_counts = np.zeros((topo.N, topo.N), dtype=int)
    fork_counts = {name: 0 for name in BRANCH_NODES}
    returns_by_source = {name: 0 for name in FOOD_NODES}
    visits_by_source = {name: 0 for name in FOOD_NODES}
    empty_food_visits = 0
    hazard_hits = 0
    food_returns = 0
    path_lengths_to_return: list[int] = []
    returns_by_step: list[int] = []
    shock_active = False

    for step in range(config.steps):
        if scenario.shock_step is not None and step >= scenario.shock_step:
            shock_active = True

        pheromone *= max(0.0, 1.0 - policy.evaporation)
        friction = static_friction + pheromone
        if shock_active:
            friction = friction + shock_friction
        kernel._sponsor_friction = friction

        p_all = kernel.transition_matrix_batch(telemetries, step=step)
        transition_rows = p_all[np.arange(config.agents), positions]
        next_positions = _sample_next(rng, transition_rows)

        step_returns = 0
        previous_positions = positions.copy()
        visited_features = topo.node_features[next_positions]
        telemetries = (1.0 - config.feedback_rate) * telemetries + config.feedback_rate * visited_features

        for ant in range(config.agents):
            src = int(previous_positions[ant])
            dst = int(next_positions[ant])
            if src != dst and np.isfinite(topo.distance_matrix[src, dst]):
                edge_counts[src, dst] += 1
                paths[ant].append((src, dst))
                if len(paths[ant]) > config.max_path_memory:
                    paths[ant] = paths[ant][-config.max_path_memory :]
                if src == fork and dst in branch_idx:
                    fork_counts[branch_idx[dst]] += 1

            if dst == hazard:
                hazard_hits += 1
                carrying[ant] = False
                carried_source[ant] = None
                paths[ant] = []
                next_positions[ant] = nest
                telemetries[ant] = base_intents[ant]
                continue

            if carrying[ant] and dst == nest:
                food_returns += 1
                step_returns += 1
                source = carried_source[ant]
                if source is not None:
                    returns_by_source[source] += 1
                path_lengths_to_return.append(len(paths[ant]))
                if policy.pheromone_deposit > 0.0 and paths[ant]:
                    deposit = policy.pheromone_deposit / math.sqrt(max(1, len(paths[ant])))
                    for i, j in paths[ant]:
                        pheromone[i, j] += deposit
                    np.clip(pheromone, 0.0, policy.pheromone_cap, out=pheromone)
                carrying[ant] = False
                carried_source[ant] = None
                paths[ant] = []
                telemetries[ant] = base_intents[ant]
                continue

            if not carrying[ant] and dst in food_idx:
                food_name = food_idx[dst]
                visits_by_source[food_name] += 1
                if food_remaining.get(food_name, 0) > 0:
                    food_remaining[food_name] -= 1
                    carrying[ant] = True
                    carried_source[ant] = food_name
                    telemetries[ant] = (
                        (1.0 - config.return_feedback_rate) * telemetries[ant]
                        + config.return_feedback_rate * return_intent
                    )
                else:
                    empty_food_visits += 1

            if carrying[ant]:
                telemetries[ant] = (
                    (1.0 - config.return_feedback_rate) * telemetries[ant]
                    + config.return_feedback_rate * return_intent
                )
            elif dst == nest:
                paths[ant] = []

        positions = next_positions
        telemetries = _normalize_rows(telemetries)
        returns_by_step.append(step_returns)

    fork_total = sum(fork_counts.values())
    lock_in_index = max(fork_counts.values()) / fork_total if fork_total else 0.0
    branch_winner = max(fork_counts, key=fork_counts.get) if fork_total else "none"
    rich_returns = returns_by_source["Rich Food Patch"]
    sparse_returns = returns_by_source["Sparse Food Patch"]
    return_total = max(1, rich_returns + sparse_returns)
    food_visit_total = max(1, sum(visits_by_source.values()))

    shock_recovery_steps: int | None = None
    if (
        scenario.shock_step is not None
        and scenario.shock_step >= 12
        and scenario.shock_step < len(returns_by_step)
    ):
        pre = np.mean(returns_by_step[max(0, scenario.shock_step - 12) : scenario.shock_step])
        target = 0.8 * pre
        if target > 0:
            for t in range(scenario.shock_step + 6, config.steps):
                window = returns_by_step[max(0, t - 8) : t + 1]
                if np.mean(window) >= target:
                    shock_recovery_steps = int(t - scenario.shock_step)
                    break

    mean_telemetry = _normalize(np.mean(telemetries, axis=0))
    flow = kernel.flow_diagnostic(mean_telemetry)
    memory_staleness = _memory_staleness_diagnostics(
        topo,
        telemetries,
        pheromone,
        edge_counts,
        food_remaining,
        visits_by_source,
        empty_food_visits,
    )

    row = {
        "scenario": scenario.name,
        "policy": policy.name,
        "agents": config.agents,
        "steps": config.steps,
        "food_returned": int(food_returns),
        "returns_per_100_ant_steps": round(100.0 * food_returns / (config.agents * config.steps), 4),
        "food_completion_rate": round(food_returns / max(1, initial_food_total), 4),
        "food_remaining": {key: int(value) for key, value in food_remaining.items()},
        "returns_by_source": {key: int(value) for key, value in returns_by_source.items()},
        "visits_by_source": {key: int(value) for key, value in visits_by_source.items()},
        "empty_food_visits": int(empty_food_visits),
        "empty_food_visit_rate": round(empty_food_visits / food_visit_total, 4),
        "rich_return_share": round(rich_returns / return_total, 4),
        "rich_visit_share": round(visits_by_source["Rich Food Patch"] / food_visit_total, 4),
        "sparse_visit_share": round(visits_by_source["Sparse Food Patch"] / food_visit_total, 4),
        "hazard_hits": int(hazard_hits),
        "hazard_rate": round(hazard_hits / (config.agents * config.steps), 4),
        "fork_counts": {key: int(value) for key, value in fork_counts.items()},
        "fork_entropy": round(_entropy(fork_counts), 4),
        "lock_in_index": round(lock_in_index, 4),
        "dominant_branch": branch_winner,
        "mean_return_path_length": round(float(np.mean(path_lengths_to_return)), 4)
        if path_lengths_to_return
        else 0.0,
        "pheromone_mass": round(float(np.sum(pheromone)), 4),
        "pheromone_max": round(float(np.max(pheromone)), 4),
        "shock_recovery_steps": shock_recovery_steps,
        "flow_entropy_production": round(float(flow["entropy_production"]), 6),
        "irreversible_flux": round(float(flow["irreversible_flux"]), 6),
        "memory_staleness": memory_staleness,
        "dominant_stale_memory_layer": memory_staleness["dominant_stale_memory_layer"],
    }
    return row


def run_pilot(config: SimulationConfig | None = None, quick: bool = False) -> dict[str, Any]:
    config = config or SimulationConfig()
    selected_scenarios = scenarios()
    selected_policies = policies()
    if quick:
        selected_scenarios = [selected_scenarios[0], selected_scenarios[1], selected_scenarios[2]]
        selected_policies = [
            selected_policies[0],
            selected_policies[1],
            selected_policies[2],
            selected_policies[4],
        ]

    rows: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, Any]] = {}
    seed_offset = 0
    for scenario in selected_scenarios:
        for policy in selected_policies:
            row = simulate(config, scenario, policy, seed_offset=seed_offset)
            if policy.name == "no_pheromone_control":
                baselines[scenario.name] = row
            rows.append(row)
            seed_offset += 997

    for row in rows:
        row["classification"] = _classify(row, baselines.get(row["scenario"]))

    summary = {
        "rows": len(rows),
        "best_food_completion": max(rows, key=lambda row: row["food_completion_rate"]),
        "lowest_hazard": min(rows, key=lambda row: row["hazard_rate"]),
        "highest_lock_in": max(rows, key=lambda row: row["lock_in_index"]),
        "class_counts": {},
    }
    for row in rows:
        summary["class_counts"][row["classification"]] = (
            summary["class_counts"].get(row["classification"], 0) + 1
        )
    return {
        "config": config.__dict__,
        "feature_labels": FEATURE_LABELS,
        "scenarios": [_scenario_payload(scenario) for scenario in selected_scenarios],
        "policies": [policy.__dict__ for policy in selected_policies],
        "rows": rows,
        "summary": summary,
    }


def render_report(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    summary = payload["summary"]
    lines = [
        "# Ant Foraging DTE Prototype",
        "",
        "## Scope",
        "",
        "Biological stress test of DTE as an adaptive collective routing kernel. "
        "Agents are ants; telemetry is forager state; successful food returns deposit "
        "pheromone as alignment-independent friction reduction; food patches deplete.",
        "",
        "## Apex Readout",
        "",
        f"- Runs: `{summary['rows']}`",
        f"- Best food completion: `{summary['best_food_completion']['food_completion_rate']}` "
        f"({summary['best_food_completion']['scenario']} / {summary['best_food_completion']['policy']})",
        f"- Highest lock-in: `{summary['highest_lock_in']['lock_in_index']}` "
        f"({summary['highest_lock_in']['scenario']} / {summary['highest_lock_in']['policy']})",
        f"- Lowest hazard rate: `{summary['lowest_hazard']['hazard_rate']}` "
        f"({summary['lowest_hazard']['scenario']} / {summary['lowest_hazard']['policy']})",
        "",
        "## Classification Counts",
        "",
    ]
    for key, value in sorted(summary["class_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Result Table",
            "",
            "| Scenario | Policy | Completion | Returns/100 ant-steps | Empty visits | Stale layer | Fork entropy | Lock-in | Dominant branch | Hazard | Recovery | Class |",
            "|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for row in rows:
        recovery = "n/a" if row["shock_recovery_steps"] is None else str(row["shock_recovery_steps"])
        lines.append(
            f"| {row['scenario']} | {row['policy']} | {row['food_completion_rate']:.3f} | "
            f"{row['returns_per_100_ant_steps']:.3f} | {row['empty_food_visit_rate']:.3f} | "
            f"{row['dominant_stale_memory_layer']} | "
            f"{row['fork_entropy']:.3f} | "
            f"{row['lock_in_index']:.3f} | {row['dominant_branch']} | "
            f"{row['hazard_rate']:.3f} | {recovery} | {row['classification']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The ant domain adds a missing biological regime to the DTE program: "
            "collective memory can improve circulation, but the same memory can become "
            "maladaptive after depletion or obstruction. The useful signal is not just "
            "food returned; it is the joint surface of completion, fork entropy, lock-in, "
            "hazard exposure, post-shock recovery, and layered memory staleness. "
            "The stale-layer field asks whether the dominant pathology is structural "
            "flow into dead ecology, preference memory around stale routes, or agent "
            "state alignment with obsolete targets. A standard shortest-path or static "
            "Markov model would miss that path quality is endogenous to prior success.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict[str, Any],
    json_path: Path,
    report_path: Path,
) -> None:
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ant foraging DTE prototype.")
    parser.add_argument("--quick", action="store_true", help="Run a reduced pilot grid.")
    parser.add_argument("--json", default="ant_foraging_output.json")
    parser.add_argument("--report", default="ANT_FORAGING_REPORT.md")
    args = parser.parse_args()

    config = SimulationConfig(agents=64, steps=80) if args.quick else SimulationConfig()
    payload = run_pilot(config=config, quick=args.quick)
    write_outputs(payload, Path(args.json), Path(args.report))
    print(render_report(payload))


if __name__ == "__main__":
    main()

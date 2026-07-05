"""
Public, non-sensitive Taiwan semiconductor logistics prototype.

This adapter is deliberately abstract. It models dependency categories and
logistics chokepoints using public, qualitative calibration ranges rather than
operational details. The purpose is to test whether DTE can separate:

1. Circulation: intent flow through the network.
2. Fab feasibility: required inputs arrive at the fab.
3. Exportable supply: wafer output, packaging, and review channels align.
4. Terminal fulfillment: exportable supply reaches demand/reserve nodes.

Usage:
    .venv\\Scripts\\python.exe taiwan_semiconductor_logistics.py --quick
    .venv\\Scripts\\python.exe taiwan_semiconductor_logistics.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from kernel import DynamicTopologyKernel, topology_from_edges


FEATURE_LABELS = [
    "Demand Urgency",
    "Reliability",
    "Speed",
    "Cost Efficiency",
    "Tech Criticality",
    "Capacity",
    "Policy Compatibility",
    "Strategic Buffer",
]


NODES = {
    "Allocation Desk": [0.90, 0.80, 0.55, 0.55, 0.85, 0.60, 0.80, 0.55],
    "Japan Materials": [0.65, 0.84, 0.45, 0.55, 0.78, 0.62, 0.84, 0.30],
    "Korea Packaging Inputs": [0.58, 0.78, 0.50, 0.58, 0.70, 0.58, 0.78, 0.28],
    "EU Lithography Tools": [0.55, 0.88, 0.25, 0.20, 0.98, 0.30, 0.86, 0.25],
    "US EDA IP": [0.70, 0.90, 0.65, 0.25, 0.96, 0.50, 0.90, 0.35],
    "Specialty Chemicals": [0.60, 0.78, 0.40, 0.50, 0.82, 0.55, 0.76, 0.22],
    "Taiwan Power Grid": [0.95, 0.76, 0.35, 0.40, 0.92, 0.50, 0.70, 0.30],
    "Taiwan Fab Cluster": [0.98, 0.88, 0.42, 0.28, 1.00, 0.56, 0.82, 0.22],
    "Taiwan OSAT Packaging": [0.84, 0.80, 0.52, 0.45, 0.88, 0.54, 0.78, 0.24],
    "Export Control Review": [0.82, 0.70, 0.30, 0.25, 0.92, 0.35, 0.96, 0.42],
    "Taiwan Port": [0.72, 0.66, 0.42, 0.76, 0.78, 0.74, 0.70, 0.20],
    "Taiwan Air Cargo": [0.94, 0.70, 0.96, 0.20, 0.84, 0.34, 0.72, 0.18],
    "Japan Reroute Hub": [0.75, 0.82, 0.58, 0.45, 0.74, 0.50, 0.86, 0.36],
    "Korea Reroute Hub": [0.72, 0.80, 0.60, 0.48, 0.72, 0.50, 0.82, 0.34],
    "Shipping Insurance": [0.62, 0.68, 0.35, 0.35, 0.78, 0.42, 0.74, 0.58],
    "Pacific Ocean Lane": [0.60, 0.62, 0.30, 0.84, 0.72, 0.86, 0.64, 0.16],
    "US West Coast Port": [0.78, 0.76, 0.55, 0.62, 0.82, 0.66, 0.84, 0.28],
    "China Regional Port": [0.80, 0.66, 0.52, 0.66, 0.78, 0.70, 0.50, 0.16],
    "US Data Center Demand": [1.00, 0.84, 0.90, 0.24, 0.98, 0.42, 0.92, 0.28],
    "China Electronics Demand": [0.88, 0.70, 0.68, 0.58, 0.84, 0.66, 0.46, 0.18],
    "Strategic Chip Reserve": [0.80, 0.92, 0.36, 0.22, 0.92, 0.32, 0.94, 1.00],
}


EDGES = [
    ("Allocation Desk", "Japan Materials", 2.1),
    ("Allocation Desk", "Korea Packaging Inputs", 2.2),
    ("Allocation Desk", "EU Lithography Tools", 3.2),
    ("Allocation Desk", "US EDA IP", 2.5),
    ("Allocation Desk", "Specialty Chemicals", 2.4),
    ("Allocation Desk", "Taiwan Power Grid", 2.0),
    ("Japan Materials", "Taiwan Fab Cluster", 2.2),
    ("EU Lithography Tools", "Taiwan Fab Cluster", 3.6),
    ("US EDA IP", "Taiwan Fab Cluster", 2.9),
    ("Specialty Chemicals", "Taiwan Fab Cluster", 2.6),
    ("Taiwan Power Grid", "Taiwan Fab Cluster", 1.8),
    ("Taiwan Fab Cluster", "Taiwan OSAT Packaging", 1.7),
    ("Korea Packaging Inputs", "Taiwan OSAT Packaging", 2.2),
    ("Taiwan OSAT Packaging", "Export Control Review", 1.9),
    ("Export Control Review", "Taiwan Port", 2.4),
    ("Export Control Review", "Taiwan Air Cargo", 3.1),
    ("Export Control Review", "Strategic Chip Reserve", 2.7),
    ("Taiwan Port", "Pacific Ocean Lane", 2.6),
    ("Taiwan Port", "China Regional Port", 2.2),
    ("Taiwan Port", "Japan Reroute Hub", 3.2),
    ("Taiwan Port", "Korea Reroute Hub", 3.0),
    ("Japan Reroute Hub", "Pacific Ocean Lane", 2.1),
    ("Korea Reroute Hub", "Pacific Ocean Lane", 2.2),
    ("Shipping Insurance", "Pacific Ocean Lane", 2.8),
    ("Pacific Ocean Lane", "US West Coast Port", 4.5),
    ("Taiwan Air Cargo", "US Data Center Demand", 2.2),
    ("Taiwan Air Cargo", "China Electronics Demand", 2.6),
    ("US West Coast Port", "US Data Center Demand", 1.7),
    ("China Regional Port", "China Electronics Demand", 1.7),
    ("Strategic Chip Reserve", "US Data Center Demand", 2.1),
    ("US Data Center Demand", "Allocation Desk", 5.8),
    ("China Electronics Demand", "Allocation Desk", 5.5),
]


INTENTS = {
    "US Strategic Compute": [1.00, 0.86, 0.88, 0.24, 1.00, 0.44, 0.92, 0.42],
    "China Manufacturing": [0.86, 0.70, 0.60, 0.70, 0.82, 0.70, 0.42, 0.20],
    "Allied Resilience": [0.80, 0.92, 0.45, 0.30, 0.92, 0.48, 0.96, 0.90],
    "Cost Sensitive Electronics": [0.60, 0.68, 0.42, 0.96, 0.70, 0.72, 0.50, 0.18],
}

DEFAULT_POPULATION = {
    "US Strategic Compute": 0.36,
    "China Manufacturing": 0.30,
    "Allied Resilience": 0.20,
    "Cost Sensitive Electronics": 0.14,
}

TERMINAL_NODES = ("US Data Center Demand", "China Electronics Demand", "Strategic Chip Reserve")
CRITICAL_NODE = "US Data Center Demand"


@dataclass(frozen=True)
class TaiwanConfig:
    agents: int = 256
    steps: int = 120
    feedback_rate: float = 0.16
    temperature: float = 0.78
    seed: int = 20260605
    gate_initial_inventory: int = 10


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    cost: float = 0.0
    friction_edges: tuple[tuple[str, str], ...] = ()
    friction_delta: float = 0.0
    beta_edges: tuple[tuple[str, str], ...] = ()
    beta_boost: float = 0.0
    failed_edges: tuple[tuple[str, str], ...] = ()
    edge_capacity_caps: dict[tuple[str, str], int] = field(default_factory=dict)
    node_capacity_caps: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GateSpec:
    name: str
    source: str
    target: str
    parts: tuple[str, ...]
    arrivals: dict[tuple[str, str], str]
    consumption: dict[str, int] = field(default_factory=dict)


GATES = [
    GateSpec(
        name="fab_input_gate",
        source="Taiwan Fab Cluster",
        target="Taiwan OSAT Packaging",
        parts=("materials", "chemicals", "lithography", "design_ip", "energy"),
        arrivals={
            ("Japan Materials", "Taiwan Fab Cluster"): "materials",
            ("Specialty Chemicals", "Taiwan Fab Cluster"): "chemicals",
            ("EU Lithography Tools", "Taiwan Fab Cluster"): "lithography",
            ("US EDA IP", "Taiwan Fab Cluster"): "design_ip",
            ("Taiwan Power Grid", "Taiwan Fab Cluster"): "energy",
        },
        consumption={
            "materials": 1,
            "chemicals": 1,
            "lithography": 0,
            "design_ip": 0,
            "energy": 1,
        },
    ),
    GateSpec(
        name="exportable_chip_gate",
        source="Taiwan OSAT Packaging",
        target="Export Control Review",
        parts=("wafer_output", "packaging"),
        arrivals={
            ("Taiwan Fab Cluster", "Taiwan OSAT Packaging"): "wafer_output",
            ("Korea Packaging Inputs", "Taiwan OSAT Packaging"): "packaging",
        },
        consumption={
            "wafer_output": 1,
            "packaging": 1,
        },
    ),
]


def _normalized(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def _stable_salt(*parts: str) -> int:
    text = "|".join(parts)
    total = 0
    for idx, char in enumerate(text):
        total += (idx + 1) * ord(char)
    return total % 65536


def _seed_for(config: TaiwanConfig, salt: int = 0) -> int:
    raw = (
        config.seed
        + 101 * config.agents
        + 1009 * config.steps
        + 9173 * int(round(config.feedback_rate * 1000))
        + 6113 * int(round(config.temperature * 1000))
        + salt
    )
    return int(raw % (2**32 - 1))


def build_kernel(config: TaiwanConfig) -> DynamicTopologyKernel:
    topology = topology_from_edges(
        nodes={label: np.array(features, dtype=np.float64) for label, features in NODES.items()},
        edges=EDGES,
        undirected=False,
    )
    n = topology.N
    node_bias = np.zeros(n, dtype=np.float64)
    for label, bias in {
        "Allocation Desk": 0.20,
        "Taiwan Fab Cluster": 0.12,
        "Export Control Review": -0.08,
        "Strategic Chip Reserve": 0.08,
    }.items():
        node_bias[topology.labels.index(label)] = bias
    return DynamicTopologyKernel(
        topology=topology,
        beta=np.full((n, n), 1.25, dtype=np.float64),
        feedback_rate=config.feedback_rate,
        feedback_noise=0.0,
        temperature=config.temperature,
        node_bias=node_bias,
        sponsor_decay=0.0,
    )


def _initial_telemetries(config: TaiwanConfig) -> np.ndarray:
    telemetries = np.zeros((config.agents, len(FEATURE_LABELS)), dtype=np.float64)
    start = 0
    items = list(DEFAULT_POPULATION.items())
    for idx, (intent, share) in enumerate(items):
        count = int(round(config.agents * share)) if idx < len(items) - 1 else config.agents - start
        end = min(config.agents, start + count)
        telemetries[start:end] = _normalized(INTENTS[intent])
        start = end
    return telemetries


def _idx(labels: list[str], names: tuple[str, ...] | list[str]) -> list[int]:
    return [labels.index(name) for name in names]


def _edge_cost(edge_names: tuple[tuple[str, str], ...], magnitude: float) -> float:
    return float(len(edge_names) * abs(magnitude))


def _apply_scenario(kernel: DynamicTopologyKernel, labels: list[str], scenario: Scenario) -> None:
    for source, target in scenario.friction_edges:
        i, j = labels.index(source), labels.index(target)
        if kernel.topo.adjacency_mask[i, j]:
            kernel.sponsor_edge_friction(i, j, scenario.friction_delta)
    for source, target in scenario.beta_edges:
        i, j = labels.index(source), labels.index(target)
        if kernel.topo.adjacency_mask[i, j]:
            kernel.sponsor_edge(i, j, scenario.beta_boost)


def shocks() -> list[Scenario]:
    port_edges = (
        ("Export Control Review", "Taiwan Port"),
        ("Taiwan Port", "Pacific Ocean Lane"),
        ("Taiwan Port", "China Regional Port"),
    )
    energy_edges = (("Taiwan Power Grid", "Taiwan Fab Cluster"),)
    export_edges = (
        ("Export Control Review", "Taiwan Port"),
        ("Export Control Review", "Taiwan Air Cargo"),
        ("Export Control Review", "Strategic Chip Reserve"),
    )
    return [
        Scenario("nominal", "nominal"),
        Scenario(
            "taiwan_port_disruption",
            "port",
            cost=_edge_cost(port_edges, 1.8),
            friction_edges=port_edges,
            friction_delta=-1.8,
        ),
        Scenario(
            "fab_capacity_shock",
            "fab_capacity",
            cost=2.0,
            node_capacity_caps={"Taiwan Fab Cluster": 18},
        ),
        Scenario(
            "energy_constraint",
            "energy",
            cost=_edge_cost(energy_edges, 2.4),
            friction_edges=energy_edges,
            friction_delta=-2.4,
        ),
        Scenario(
            "export_control_friction",
            "policy",
            cost=_edge_cost(export_edges, 1.7),
            friction_edges=export_edges,
            friction_delta=-1.7,
        ),
        Scenario(
            "air_freight_substitution_cap",
            "air_capacity",
            cost=2.0,
            edge_capacity_caps={
                ("Export Control Review", "Taiwan Air Cargo"): 16,
                ("Taiwan Air Cargo", "US Data Center Demand"): 16,
            },
        ),
    ]


def controls() -> list[Scenario]:
    reroute = (
        ("Taiwan Port", "Japan Reroute Hub"),
        ("Taiwan Port", "Korea Reroute Hub"),
        ("Japan Reroute Hub", "Pacific Ocean Lane"),
        ("Korea Reroute Hub", "Pacific Ocean Lane"),
    )
    air_bridge = (
        ("Export Control Review", "Taiwan Air Cargo"),
        ("Taiwan Air Cargo", "US Data Center Demand"),
    )
    reserve = (("Strategic Chip Reserve", "US Data Center Demand"),)
    review = (
        ("Taiwan OSAT Packaging", "Export Control Review"),
        ("Export Control Review", "Taiwan Port"),
        ("Export Control Review", "Taiwan Air Cargo"),
    )
    energy = (("Taiwan Power Grid", "Taiwan Fab Cluster"),)
    inputs = (
        ("Japan Materials", "Taiwan Fab Cluster"),
        ("Specialty Chemicals", "Taiwan Fab Cluster"),
        ("Korea Packaging Inputs", "Taiwan OSAT Packaging"),
    )
    tooling = (
        ("EU Lithography Tools", "Taiwan Fab Cluster"),
        ("US EDA IP", "Taiwan Fab Cluster"),
    )
    return [
        Scenario("no_control", "control"),
        Scenario(
            "allied_reroute",
            "control",
            cost=_edge_cost(reroute, 0.9),
            beta_edges=reroute,
            beta_boost=0.9,
        ),
        Scenario(
            "air_bridge",
            "control",
            cost=_edge_cost(air_bridge, 1.1),
            beta_edges=air_bridge,
            beta_boost=1.1,
        ),
        Scenario(
            "strategic_reserve_release",
            "control",
            cost=_edge_cost(reserve, 1.4),
            friction_edges=reserve,
            friction_delta=1.4,
        ),
        Scenario(
            "export_fast_lane",
            "control",
            cost=_edge_cost(review, 1.0),
            friction_edges=review,
            friction_delta=1.0,
        ),
        Scenario(
            "energy_stabilization",
            "control",
            cost=_edge_cost(energy, 1.3),
            friction_edges=energy,
            friction_delta=1.3,
        ),
        Scenario(
            "materials_diversification",
            "control",
            cost=_edge_cost(inputs, 0.8),
            beta_edges=inputs,
            beta_boost=0.8,
        ),
        Scenario(
            "tooling_continuity",
            "control",
            cost=_edge_cost(tooling, 1.0),
            friction_edges=tooling,
            friction_delta=1.0,
        ),
    ]


def _renormalize_rows(rows: np.ndarray) -> np.ndarray:
    row_sums = rows.sum(axis=1, keepdims=True)
    return np.divide(rows, row_sums, out=np.zeros_like(rows), where=row_sums > 1e-12)


def _init_gate_state(config: TaiwanConfig) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    inventory = {
        gate.name: {part: config.gate_initial_inventory for part in gate.parts}
        for gate in GATES
    }
    arrivals = {
        gate.name: {part: 0 for part in gate.parts}
        for gate in GATES
    }
    return inventory, arrivals


def simulate(
    config: TaiwanConfig,
    shock: Scenario | None = None,
    control: Scenario | None = None,
    enforce_gates: bool = True,
) -> dict:
    shock = shock or shocks()[0]
    control = control or controls()[0]
    kernel = build_kernel(config)
    labels = kernel.topo.labels
    _apply_scenario(kernel, labels, shock)
    _apply_scenario(kernel, labels, control)

    rng = np.random.default_rng(_seed_for(config, _stable_salt(shock.name, control.name, str(enforce_gates))))
    n = kernel.topo.N
    positions = np.full(config.agents, labels.index("Allocation Desk"), dtype=int)
    telemetries = _initial_telemetries(config)

    terminal_idx = np.array(_idx(labels, TERMINAL_NODES), dtype=int)
    critical_idx = labels.index(CRITICAL_NODE)
    failed_idx = [(labels.index(source), labels.index(target)) for source, target in shock.failed_edges]
    edge_capacity_idx = {
        (labels.index(source), labels.index(target)): int(capacity)
        for (source, target), capacity in shock.edge_capacity_caps.items()
    }
    node_capacity_idx = {
        labels.index(node): int(capacity)
        for node, capacity in shock.node_capacity_caps.items()
    }

    gate_source_target_idx = {
        gate.name: (labels.index(gate.source), labels.index(gate.target))
        for gate in GATES
    }
    gate_arrival_idx = {
        gate.name: {
            (labels.index(source), labels.index(target)): part
            for (source, target), part in gate.arrivals.items()
        }
        for gate in GATES
    }
    gate_inventory, gate_arrivals = _init_gate_state(config)
    gate_attempts = {gate.name: 0 for gate in GATES}
    gate_blocked = {gate.name: 0 for gate in GATES}
    gate_completions = {gate.name: 0 for gate in GATES}

    edge_counts = np.zeros((n, n), dtype=np.float64)
    terminal_hits = []
    critical_hits = []
    reserve_hits = []
    us_hits = []
    china_hits = []
    hard_blocked_mass = []
    edge_overflow_events = 0
    edge_capacity_attempts = 0
    node_overflow_events = 0
    node_capacity_attempts = 0

    for step in range(config.steps):
        P_all = kernel.transition_matrix_batch(telemetries, step=step)
        rows = P_all[np.arange(config.agents), positions, :].copy()

        block_mass = 0.0
        for i, j in failed_idx:
            source_mask = positions == i
            if np.any(source_mask):
                block_mass += float(rows[source_mask, j].sum() / config.agents)
                rows[source_mask, j] = 0.0
        if failed_idx:
            rows = _renormalize_rows(rows)
        hard_blocked_mass.append(block_mass)

        cdf = np.cumsum(rows, axis=1)
        draws = rng.random((config.agents, 1))
        next_positions = np.argmax(cdf >= draws, axis=1)
        row_sums = rows.sum(axis=1)
        next_positions[row_sums < 1e-12] = positions[row_sums < 1e-12]

        for (i, j), capacity in edge_capacity_idx.items():
            attempted = np.where((positions == i) & (next_positions == j))[0]
            edge_capacity_attempts += int(len(attempted))
            if len(attempted) > capacity:
                allowed = set(rng.choice(attempted, size=capacity, replace=False).tolist())
                overflow = [idx for idx in attempted.tolist() if idx not in allowed]
                edge_overflow_events += len(overflow)
                next_positions[np.array(overflow, dtype=int)] = positions[np.array(overflow, dtype=int)]

        for j, capacity in node_capacity_idx.items():
            attempted = np.where((positions != j) & (next_positions == j))[0]
            node_capacity_attempts += int(len(attempted))
            if len(attempted) > capacity:
                allowed = set(rng.choice(attempted, size=capacity, replace=False).tolist())
                overflow = [idx for idx in attempted.tolist() if idx not in allowed]
                node_overflow_events += len(overflow)
                next_positions[np.array(overflow, dtype=int)] = positions[np.array(overflow, dtype=int)]

        if enforce_gates:
            for gate in GATES:
                i, j = gate_source_target_idx[gate.name]
                attempted = np.where((positions == i) & (next_positions == j))[0]
                gate_attempts[gate.name] += int(len(attempted))
                feasible_limits = []
                feasible = int(len(attempted))
                for part in gate.parts:
                    consumption = gate.consumption.get(part, 1)
                    inventory = gate_inventory[gate.name][part]
                    if consumption <= 0:
                        if inventory <= 0:
                            feasible = 0
                            break
                    else:
                        feasible_limits.append(inventory // consumption)
                if feasible > 0 and feasible_limits:
                    feasible = min(feasible_limits)
                allowed_count = min(int(len(attempted)), int(feasible))
                if len(attempted) > allowed_count:
                    allowed = set(rng.choice(attempted, size=allowed_count, replace=False).tolist())
                    blocked = [idx for idx in attempted.tolist() if idx not in allowed]
                    gate_blocked[gate.name] += len(blocked)
                    next_positions[np.array(blocked, dtype=int)] = positions[np.array(blocked, dtype=int)]
                if allowed_count > 0:
                    gate_completions[gate.name] += allowed_count
                    for part in gate.parts:
                        consumption = gate.consumption.get(part, 1)
                        if consumption > 0:
                            gate_inventory[gate.name][part] -= allowed_count * consumption

        for gate in GATES:
            for (source_idx, target_idx), part in gate_arrival_idx[gate.name].items():
                arrivals = int(np.sum((positions == source_idx) & (next_positions == target_idx)))
                if arrivals > 0:
                    gate_arrivals[gate.name][part] += arrivals
                    gate_inventory[gate.name][part] += arrivals

        np.add.at(edge_counts, (positions, next_positions), 1.0)
        visited = kernel.topo.node_features[next_positions]
        lam = config.feedback_rate
        telemetries = (1.0 - lam) * telemetries + lam * visited
        norms = np.linalg.norm(telemetries, axis=1, keepdims=True)
        telemetries = np.where(norms > 0, telemetries / norms, telemetries)
        positions = next_positions

        terminal_hits.append(float(np.mean(np.isin(positions, terminal_idx))))
        critical_hits.append(float(np.mean(positions == critical_idx)))
        reserve_hits.append(float(np.mean(positions == labels.index("Strategic Chip Reserve"))))
        us_hits.append(float(np.mean(positions == labels.index("US Data Center Demand"))))
        china_hits.append(float(np.mean(positions == labels.index("China Electronics Demand"))))

    edge_flow = edge_counts / max(float(edge_counts.sum()), 1.0)
    edge_current = edge_flow - edge_flow.T
    reverse = edge_flow.T
    eps = 1e-12
    bidirectional = (edge_flow > eps) & (reverse > eps)
    entropy_production = 0.0
    if np.any(bidirectional):
        entropy_production = 0.5 * float(np.sum(edge_flow[bidirectional] * np.log(edge_flow[bidirectional] / reverse[bidirectional])))
    one_way = (edge_flow > eps) & (reverse <= eps)

    window = max(8, config.steps // 4)
    total_overflow = edge_overflow_events + node_overflow_events
    total_attempts = edge_capacity_attempts + node_capacity_attempts
    gate_total_attempts = sum(gate_attempts.values())
    gate_total_blocked = sum(gate_blocked.values())
    gate_total_completions = sum(gate_completions.values())
    limiting = _limiting_gate_part(gate_inventory, gate_arrivals)

    return {
        "shock": shock.name,
        "family": shock.family,
        "control": control.name,
        "control_cost": control.cost,
        "agents": config.agents,
        "steps": config.steps,
        "gates_enforced": bool(enforce_gates),
        "terminal_share": float(np.mean(terminal_hits[-window:])),
        "critical_service_share": float(np.mean(critical_hits[-window:])),
        "us_demand_share": float(np.mean(us_hits[-window:])),
        "china_demand_share": float(np.mean(china_hits[-window:])),
        "reserve_share": float(np.mean(reserve_hits[-window:])),
        "edge_current_norm": float(np.linalg.norm(edge_current, ord="fro")),
        "entropy_production": entropy_production,
        "irreversible_flux": float(np.sum(edge_flow[one_way])),
        "hard_blocked_mass": float(np.mean(hard_blocked_mass[-window:])),
        "capacity_overflow_rate": float(total_overflow / max(total_attempts, 1)),
        "edge_capacity_overflow_rate": float(edge_overflow_events / max(edge_capacity_attempts, 1)),
        "node_capacity_overflow_rate": float(node_overflow_events / max(node_capacity_attempts, 1)),
        "gate_attempts": gate_attempts,
        "gate_blocked": gate_blocked,
        "gate_completions": gate_completions,
        "gate_block_rate": float(gate_total_blocked / max(gate_total_attempts, 1)),
        "gate_completion_per_agent": float(gate_total_completions / config.agents),
        "gate_completion_rate": float(gate_total_completions / max(config.agents * config.steps, 1)),
        "gate_inventory_end": gate_inventory,
        "gate_arrivals": gate_arrivals,
        "limiting_gate": limiting["gate"],
        "limiting_part": limiting["part"],
        "limiting_inventory": limiting["inventory"],
    }


def _limiting_gate_part(
    gate_inventory: dict[str, dict[str, int]],
    gate_arrivals: dict[str, dict[str, int]],
) -> dict:
    candidates = []
    for gate_name, inventory in gate_inventory.items():
        for part, count in inventory.items():
            candidates.append((count, gate_arrivals[gate_name].get(part, 0), gate_name, part))
    count, _, gate_name, part = min(candidates)
    return {"gate": gate_name, "part": part, "inventory": int(count)}


def paired_run(config: TaiwanConfig, shock: Scenario, control: Scenario) -> dict:
    circulation = simulate(config, shock, control, enforce_gates=False)
    gated = simulate(config, shock, control, enforce_gates=True)
    row = {
        "shock": shock.name,
        "family": shock.family,
        "control": control.name,
        "control_cost": control.cost,
        "circulation_terminal_share": circulation["terminal_share"],
        "gated_terminal_share": gated["terminal_share"],
        "circulation_critical_service_share": circulation["critical_service_share"],
        "gated_critical_service_share": gated["critical_service_share"],
        "gated_us_demand_share": gated["us_demand_share"],
        "gated_china_demand_share": gated["china_demand_share"],
        "gated_reserve_share": gated["reserve_share"],
        "gate_block_rate": gated["gate_block_rate"],
        "gate_completion_per_agent": gated["gate_completion_per_agent"],
        "capacity_overflow_rate": gated["capacity_overflow_rate"],
        "limiting_gate": gated["limiting_gate"],
        "limiting_part": gated["limiting_part"],
        "limiting_inventory": gated["limiting_inventory"],
        "gate_completions": gated["gate_completions"],
    }
    row["feasibility_gap"] = row["circulation_terminal_share"] - row["gated_terminal_share"]
    row["critical_gap"] = row["circulation_critical_service_share"] - row["gated_critical_service_share"]
    row["institutional_score"] = (
        row["gated_terminal_share"]
        + 0.75 * row["gated_critical_service_share"]
        + 0.20 * row["gated_reserve_share"]
        + 0.20 * row["gate_completion_per_agent"]
        - 0.10 * row["gate_block_rate"]
        - 0.25 * row["capacity_overflow_rate"]
    )
    row["classification"] = classify(row)
    return row


def classify(row: dict) -> str:
    if row["capacity_overflow_rate"] >= 0.12:
        return "capacity_limited"
    if row["gate_completion_per_agent"] >= 0.35 and row["feasibility_gap"] >= 0.04:
        return "produced_but_delayed"
    if row["gate_block_rate"] >= 0.60:
        return "dependency_limited"
    if row["feasibility_gap"] >= 0.03:
        return "circulation_overstates_feasibility"
    return "feasible_flow"


def _with_baseline_deltas(rows: list[dict]) -> list[dict]:
    baselines = {
        row["shock"]: row
        for row in rows
        if row["control"] == "no_control"
    }
    enriched = []
    for row in rows:
        baseline = baselines[row["shock"]]
        item = dict(row)
        item["score_delta_vs_baseline"] = row["institutional_score"] - baseline["institutional_score"]
        item["terminal_delta_vs_baseline"] = row["gated_terminal_share"] - baseline["gated_terminal_share"]
        item["critical_delta_vs_baseline"] = row["gated_critical_service_share"] - baseline["gated_critical_service_share"]
        item["gap_delta_vs_baseline"] = row["feasibility_gap"] - baseline["feasibility_gap"]
        enriched.append(item)
    return enriched


def run_suite(config: TaiwanConfig | None = None, quick: bool = False) -> dict:
    config = config or TaiwanConfig()
    shock_list = shocks()[:3] if quick else shocks()
    rows = [
        paired_run(config, shock, control)
        for shock in shock_list
        for control in controls()
    ]
    rows = _with_baseline_deltas(rows)
    counts = {
        label: sum(1 for row in rows if row["classification"] == label)
        for label in sorted({row["classification"] for row in rows})
    }
    best_by_shock = {}
    for shock in sorted({row["shock"] for row in rows}):
        candidates = [row for row in rows if row["shock"] == shock and row["control"] != "no_control"]
        best = max(candidates, key=lambda row: (row["score_delta_vs_baseline"], row["terminal_delta_vs_baseline"]))
        best_by_shock[shock] = {
            "control": best["control"],
            "score_delta": best["score_delta_vs_baseline"],
            "terminal_delta": best["terminal_delta_vs_baseline"],
            "critical_delta": best["critical_delta_vs_baseline"],
            "classification": best["classification"],
            "limiting_gate": best["limiting_gate"],
            "limiting_part": best["limiting_part"],
        }
    return {
        "config": asdict(config) | {"quick": quick},
        "classification_counts": counts,
        "best_by_shock": best_by_shock,
        "rows": rows,
    }


def render_report(payload: dict) -> str:
    rows = payload["rows"]
    limiting_counts = {
        f"{row['limiting_gate']}::{row['limiting_part']}": sum(
            1
            for item in rows
            if item["limiting_gate"] == row["limiting_gate"] and item["limiting_part"] == row["limiting_part"]
        )
        for row in rows
    }
    worst_gaps = sorted(rows, key=lambda row: row["feasibility_gap"], reverse=True)[:10]
    lines = [
        "# Taiwan Semiconductor Logistics Prototype Report",
        "",
        "## Scope",
        "",
        (
            "Public, non-sensitive DTE adapter for semiconductor logistics. The model uses "
            "abstract dependency categories rather than operational data, and compares circulation "
            "against two gated feasibility layers: fab inputs and exportable packaged chips."
        ),
        "",
        f"- Agents per run: `{payload['config']['agents']}`",
        f"- Steps per run: `{payload['config']['steps']}`",
        f"- Gate initial inventory per part: `{payload['config']['gate_initial_inventory']}`",
        f"- Scenario-control rows: `{len(rows)}`",
        "",
        "## Classification Counts",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(payload["classification_counts"].items()):
        lines.append(f"| `{label}` | {count} |")

    lines.extend([
        "",
        "## Limiting Gate Parts",
        "",
        "| Gate Part | Rows Limited |",
        "|---|---:|",
    ])
    for key, count in sorted(limiting_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {count} |")

    lines.extend([
        "",
        "## Largest Circulation-Feasibility Gaps",
        "",
        "| Shock | Control | Gap | Circulation Terminal | Gated Terminal | Gate Completions / Agent | Gate Block Rate | Limiting Part |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in worst_gaps:
        lines.append(
            f"| `{row['shock']}` | `{row['control']}` | {row['feasibility_gap']:+.3f} | "
            f"{row['circulation_terminal_share']:.3f} | {row['gated_terminal_share']:.3f} | "
            f"{row['gate_completion_per_agent']:.2f} | {row['gate_block_rate']:.1%} | "
            f"`{row['limiting_gate']}::{row['limiting_part']}` |"
        )

    lines.extend([
        "",
        "## Best Control By Shock",
        "",
        "| Shock | Best Control | Score Delta | Terminal Delta | Critical Delta | Classification | Limiting Part |",
        "|---|---|---:|---:|---:|---|---|",
    ])
    for shock, row in payload["best_by_shock"].items():
        lines.append(
            f"| `{shock}` | `{row['control']}` | {row['score_delta']:+.4f} | "
            f"{row['terminal_delta']:+.3f} | {row['critical_delta']:+.3f} | "
            f"`{row['classification']}` | `{row['limiting_gate']}::{row['limiting_part']}` |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "This prototype is a dependency and logistics stress model, not a geopolitical "
            "prediction engine. Its value is diagnostic: it asks whether a proposed doctrine "
            "moves flow, clears dependency gates, or merely creates apparent circulation that "
            "cannot become exportable supply."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("taiwan_semiconductor_logistics_output.json"),
    output_md: Path = Path("TAIWAN_SEMICONDUCTOR_LOGISTICS_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Taiwan semiconductor logistics prototype.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--agents", type=int, default=TaiwanConfig.agents)
    parser.add_argument("--steps", type=int, default=TaiwanConfig.steps)
    parser.add_argument("--seed", type=int, default=TaiwanConfig.seed)
    parser.add_argument("--gate-initial-inventory", type=int, default=TaiwanConfig.gate_initial_inventory)
    parser.add_argument("--output-json", type=Path, default=Path("taiwan_semiconductor_logistics_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("TAIWAN_SEMICONDUCTOR_LOGISTICS_REPORT.md"))
    args = parser.parse_args()

    config = TaiwanConfig(
        agents=80 if args.quick and args.agents == TaiwanConfig.agents else args.agents,
        steps=40 if args.quick and args.steps == TaiwanConfig.steps else args.steps,
        seed=args.seed,
        gate_initial_inventory=args.gate_initial_inventory,
    )
    payload = run_suite(config, quick=args.quick)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "classification_counts": payload["classification_counts"],
        "rows": len(payload["rows"]),
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

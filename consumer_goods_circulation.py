"""
Consumer-goods cold-chain circulation pilot for DTE.

This module is intentionally narrow: refrigerated packaged goods under
cold-chain and promotion stress. It is a schema-first industrial pilot, not a
full consumer-goods enterprise model.

Usage:
    .venv\\Scripts\\python.exe consumer_goods_circulation.py --quick
    .venv\\Scripts\\python.exe consumer_goods_circulation.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kernel import DynamicTopologyKernel, topology_from_edges


FEATURE_LABELS = [
    "Demand Urgency",
    "Margin Priority",
    "Service Criticality",
    "Perishability",
    "Cost Sensitivity",
    "Substitution Tolerance",
    "Resilience Value",
    "Capacity Intensity",
    "Promotion Sensitivity",
]


NODES = {
    "Planning Desk": [0.86, 0.68, 0.70, 0.55, 0.60, 0.45, 0.72, 0.52, 0.52],
    "Primary Dairy Supplier": [0.62, 0.70, 0.72, 0.86, 0.58, 0.25, 0.30, 0.68, 0.20],
    "Alternate Supplier": [0.58, 0.62, 0.68, 0.80, 0.70, 0.40, 0.58, 0.48, 0.18],
    "Packaging Supplier": [0.42, 0.64, 0.58, 0.35, 0.78, 0.25, 0.32, 0.74, 0.12],
    "Regional Plant": [0.72, 0.74, 0.72, 0.82, 0.56, 0.28, 0.40, 0.74, 0.25],
    "Co-Packer": [0.68, 0.58, 0.66, 0.74, 0.70, 0.48, 0.50, 0.44, 0.24],
    "QA Release": [0.68, 0.78, 0.88, 0.86, 0.42, 0.22, 0.50, 0.36, 0.18],
    "Finished Goods Staging": [0.74, 0.76, 0.78, 0.88, 0.44, 0.22, 0.54, 0.50, 0.20],
    "Cold Capacity Pool": [0.76, 0.62, 0.82, 1.00, 0.36, 0.18, 0.40, 0.42, 0.16],
    "Cold Chain Carrier": [0.82, 0.62, 0.92, 1.00, 0.34, 0.16, 0.32, 0.36, 0.18],
    "Dry Carrier": [0.44, 0.56, 0.52, 0.10, 0.90, 0.60, 0.16, 0.82, 0.10],
    "Regional DC": [0.78, 0.76, 0.76, 0.82, 0.54, 0.32, 0.58, 0.62, 0.36],
    "Cross-Dock": [0.88, 0.60, 0.94, 0.88, 0.36, 0.36, 0.42, 0.38, 0.34],
    "Safety Stock": [0.76, 0.72, 0.88, 0.80, 0.34, 0.22, 1.00, 0.30, 0.16],
    "Promotion Demand": [0.92, 0.88, 0.58, 0.64, 0.46, 0.66, 0.22, 0.58, 1.00],
    "Priority Retail Accounts": [1.00, 0.86, 1.00, 0.90, 0.30, 0.20, 0.58, 0.34, 0.46],
    "Standard Retail Accounts": [0.78, 0.70, 0.62, 0.72, 0.66, 0.58, 0.30, 0.54, 0.64],
    "Lost Demand": [0.20, 0.10, 0.10, 0.20, 0.92, 0.82, 0.02, 0.90, 0.20],
}


EDGES = [
    ("Planning Desk", "Primary Dairy Supplier", 2.0),
    ("Planning Desk", "Alternate Supplier", 3.2),
    ("Planning Desk", "Packaging Supplier", 2.2),
    ("Planning Desk", "Regional Plant", 2.6),
    ("Planning Desk", "Co-Packer", 3.0),
    ("Planning Desk", "Cold Capacity Pool", 2.7),
    ("Planning Desk", "Safety Stock", 3.2),
    ("Planning Desk", "Promotion Demand", 2.3),
    ("Primary Dairy Supplier", "Regional Plant", 1.8),
    ("Alternate Supplier", "Regional Plant", 2.6),
    ("Alternate Supplier", "Co-Packer", 2.4),
    ("Packaging Supplier", "Regional Plant", 1.9),
    ("Packaging Supplier", "Co-Packer", 2.1),
    ("Regional Plant", "QA Release", 1.6),
    ("Co-Packer", "QA Release", 1.9),
    ("QA Release", "Finished Goods Staging", 1.4),
    ("Finished Goods Staging", "Cold Chain Carrier", 1.7),
    ("Finished Goods Staging", "Dry Carrier", 3.8),
    ("Cold Capacity Pool", "Cold Chain Carrier", 1.5),
    ("Cold Chain Carrier", "Regional DC", 1.9),
    ("Cold Chain Carrier", "Cross-Dock", 2.4),
    ("Dry Carrier", "Lost Demand", 2.2),
    ("Regional DC", "Priority Retail Accounts", 1.6),
    ("Regional DC", "Standard Retail Accounts", 1.7),
    ("Cross-Dock", "Priority Retail Accounts", 1.5),
    ("Cross-Dock", "Standard Retail Accounts", 2.0),
    ("Safety Stock", "Regional DC", 1.9),
    ("Safety Stock", "Priority Retail Accounts", 1.5),
    ("Promotion Demand", "Standard Retail Accounts", 1.5),
    ("Promotion Demand", "Lost Demand", 2.8),
    ("Priority Retail Accounts", "Planning Desk", 5.5),
    ("Standard Retail Accounts", "Planning Desk", 5.7),
    ("Lost Demand", "Planning Desk", 6.5),
]


INTENTS = {
    "Standard Replenishment": [0.74, 0.66, 0.60, 0.70, 0.62, 0.48, 0.28, 0.58, 0.42],
    "Promotion Surge": [0.95, 0.92, 0.62, 0.68, 0.45, 0.72, 0.18, 0.62, 1.00],
    "Priority Account": [1.00, 0.82, 1.00, 0.92, 0.28, 0.18, 0.54, 0.36, 0.38],
    "Perishability Constrained": [0.88, 0.66, 0.88, 1.00, 0.36, 0.22, 0.42, 0.42, 0.34],
    "Resilience Stock": [0.60, 0.70, 0.78, 0.72, 0.38, 0.26, 1.00, 0.34, 0.18],
}


DEFAULT_POPULATION = {
    "Standard Replenishment": 0.40,
    "Promotion Surge": 0.22,
    "Priority Account": 0.20,
    "Perishability Constrained": 0.12,
    "Resilience Stock": 0.06,
}


TERMINAL_NODES = ("Priority Retail Accounts", "Standard Retail Accounts", "Lost Demand")
SERVICE_NODES = ("Priority Retail Accounts", "Standard Retail Accounts")
PRIORITY_NODE = "Priority Retail Accounts"
LOST_NODE = "Lost Demand"


@dataclass(frozen=True)
class SimulationConfig:
    agents: int = 160
    steps: int = 48
    feedback_rate: float = 0.16
    temperature: float = 0.78
    seed: int = 20260611
    gate_initial_inventory: int = 16
    randomization_key: str | None = "consumer_goods_cold_chain"


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    cost: float = 0.0
    friction_edges: tuple[tuple[str, str], ...] = ()
    friction_delta: float = 0.0
    friction_edge_deltas: dict[tuple[str, str], float] = field(default_factory=dict)
    beta_edges: tuple[tuple[str, str], ...] = ()
    beta_boost: float = 0.0
    beta_edge_boosts: dict[tuple[str, str], float] = field(default_factory=dict)
    node_capacity_caps: dict[str, int] = field(default_factory=dict)
    edge_capacity_caps: dict[tuple[str, str], int] = field(default_factory=dict)
    gate_replenishment: dict[tuple[str, str], int] = field(default_factory=dict)
    gate_capacity_caps: dict[str, int] = field(default_factory=dict)


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
        name="production_gate",
        source="Regional Plant",
        target="QA Release",
        parts=("product_input", "packaging_input", "plant_labor"),
        arrivals={
            ("Primary Dairy Supplier", "Regional Plant"): "product_input",
            ("Alternate Supplier", "Regional Plant"): "product_input",
            ("Packaging Supplier", "Regional Plant"): "packaging_input",
            ("Planning Desk", "Regional Plant"): "plant_labor",
        },
        consumption={"product_input": 1, "packaging_input": 1, "plant_labor": 1},
    ),
    GateSpec(
        name="copacker_gate",
        source="Co-Packer",
        target="QA Release",
        parts=("copack_product", "copack_packaging"),
        arrivals={
            ("Alternate Supplier", "Co-Packer"): "copack_product",
            ("Packaging Supplier", "Co-Packer"): "copack_packaging",
        },
        consumption={"copack_product": 1, "copack_packaging": 1},
    ),
    GateSpec(
        name="cold_chain_gate",
        source="Finished Goods Staging",
        target="Cold Chain Carrier",
        parts=("finished_lot", "cold_slot"),
        arrivals={
            ("QA Release", "Finished Goods Staging"): "finished_lot",
            ("Cold Capacity Pool", "Cold Chain Carrier"): "cold_slot",
        },
        consumption={"finished_lot": 1, "cold_slot": 1},
    ),
]


BASE_GATE_REPLENISHMENT = {
    ("production_gate", "product_input"): 3,
    ("production_gate", "packaging_input"): 3,
    ("production_gate", "plant_labor"): 3,
    ("copacker_gate", "copack_product"): 1,
    ("copacker_gate", "copack_packaging"): 1,
    ("cold_chain_gate", "cold_slot"): 4,
}


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


def _seed_for(config: SimulationConfig, salt: int = 0) -> int:
    raw = (
        config.seed
        + 101 * config.agents
        + 1009 * config.steps
        + 9173 * int(round(config.feedback_rate * 1000))
        + 6113 * int(round(config.temperature * 1000))
        + salt
    )
    return int(raw % (2**32 - 1))


def build_kernel(config: SimulationConfig) -> DynamicTopologyKernel:
    topology = topology_from_edges(
        nodes={label: np.array(features, dtype=np.float64) for label, features in NODES.items()},
        edges=EDGES,
        undirected=False,
    )
    n = topology.N
    node_bias = np.zeros(n, dtype=np.float64)
    for label, bias in {
        "Planning Desk": 0.20,
        "Regional DC": 0.12,
        "Priority Retail Accounts": 0.08,
        "Safety Stock": 0.08,
    }.items():
        node_bias[topology.labels.index(label)] = bias
    return DynamicTopologyKernel(
        topology=topology,
        beta=np.full((n, n), 1.30, dtype=np.float64),
        feedback_rate=config.feedback_rate,
        feedback_noise=0.0,
        temperature=config.temperature,
        node_bias=node_bias,
        sponsor_decay=0.0,
    )


def _edge_cost(edges: tuple[tuple[str, str], ...], magnitude: float) -> float:
    return float(len(edges) * abs(magnitude))


def _apply_scenario(kernel: DynamicTopologyKernel, labels: list[str], scenario: Scenario) -> None:
    for source, target in scenario.friction_edges:
        i, j = labels.index(source), labels.index(target)
        if kernel.topo.adjacency_mask[i, j]:
            delta = scenario.friction_edge_deltas.get((source, target), scenario.friction_delta)
            kernel.sponsor_edge_friction(i, j, delta)
    for source, target in scenario.beta_edges:
        i, j = labels.index(source), labels.index(target)
        if kernel.topo.adjacency_mask[i, j]:
            boost = scenario.beta_edge_boosts.get((source, target), scenario.beta_boost)
            kernel.sponsor_edge(i, j, boost)


def _initial_telemetries(config: SimulationConfig) -> np.ndarray:
    telemetries = np.zeros((config.agents, len(FEATURE_LABELS)), dtype=np.float64)
    start = 0
    items = list(DEFAULT_POPULATION.items())
    for idx, (intent, share) in enumerate(items):
        count = int(round(config.agents * share)) if idx < len(items) - 1 else config.agents - start
        end = min(config.agents, start + count)
        telemetries[start:end] = _normalized(INTENTS[intent])
        start = end
    return telemetries


def _combined_gate_replenishment(*items: Scenario) -> dict[tuple[str, str], int]:
    valid = {(gate.name, part) for gate in GATES for part in gate.parts}
    replenishment: dict[tuple[str, str], int] = dict(BASE_GATE_REPLENISHMENT)
    for item in items:
        for key, amount in item.gate_replenishment.items():
            if key in valid and amount > 0:
                replenishment[key] = replenishment.get(key, 0) + int(amount)
    return replenishment


def _combined_gate_capacity_caps(*items: Scenario) -> dict[str, int]:
    caps: dict[str, int] = {}
    valid = {gate.name for gate in GATES}
    for item in items:
        for gate_name, capacity in item.gate_capacity_caps.items():
            if gate_name in valid and capacity > 0:
                caps[gate_name] = min(caps.get(gate_name, int(capacity)), int(capacity))
    return caps


def _init_gate_state(config: SimulationConfig) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    inventory = {
        gate.name: {part: config.gate_initial_inventory for part in gate.parts}
        for gate in GATES
    }
    arrivals = {
        gate.name: {part: 0 for part in gate.parts}
        for gate in GATES
    }
    return inventory, arrivals


def _limiting_part(inventory: dict[str, dict[str, int]]) -> dict:
    candidates = []
    for gate_name, parts in inventory.items():
        for part, count in parts.items():
            candidates.append((count, gate_name, part))
    count, gate_name, part = min(candidates)
    return {"gate": gate_name, "part": part, "inventory": int(count)}


def scenarios() -> list[Scenario]:
    cold_edges = (
        ("Finished Goods Staging", "Cold Chain Carrier"),
        ("Cold Chain Carrier", "Regional DC"),
    )
    promotion_edges = (
        ("Planning Desk", "Promotion Demand"),
        ("Promotion Demand", "Standard Retail Accounts"),
    )
    supplier_edges = (
        ("Planning Desk", "Primary Dairy Supplier"),
        ("Primary Dairy Supplier", "Regional Plant"),
    )
    return [
        Scenario("nominal", "baseline"),
        Scenario(
            "cold_chain_capacity_stress",
            "capacity",
            cost=2.0,
            friction_edges=cold_edges,
            friction_delta=-0.8,
            gate_capacity_caps={"cold_chain_gate": 4},
            edge_capacity_caps={("Cold Chain Carrier", "Regional DC"): 6},
        ),
        Scenario(
            "promotion_surge",
            "demand",
            cost=_edge_cost(promotion_edges, 1.4),
            beta_edges=promotion_edges,
            beta_boost=1.4,
            node_capacity_caps={"Standard Retail Accounts": 10},
        ),
        Scenario(
            "combined_surge_cold_chain",
            "combined",
            cost=_edge_cost(promotion_edges + cold_edges, 1.3),
            beta_edges=promotion_edges,
            beta_boost=1.4,
            friction_edges=cold_edges,
            friction_delta=-0.8,
            gate_capacity_caps={"cold_chain_gate": 4},
            edge_capacity_caps={("Cold Chain Carrier", "Regional DC"): 6},
            node_capacity_caps={"Standard Retail Accounts": 10},
        ),
        Scenario(
            "supplier_delay",
            "supplier",
            cost=_edge_cost(supplier_edges, 1.8),
            friction_edges=supplier_edges,
            friction_delta=-1.8,
        ),
        Scenario(
            "regional_dc_receiving_constraint",
            "capacity",
            cost=2.0,
            node_capacity_caps={"Regional DC": 7},
        ),
    ]


def controls() -> list[Scenario]:
    carrier_edges = (
        ("Finished Goods Staging", "Cold Chain Carrier"),
        ("Cold Chain Carrier", "Regional DC"),
    )
    throttle_edges = (
        ("Planning Desk", "Promotion Demand"),
        ("Promotion Demand", "Standard Retail Accounts"),
    )
    reserve_edges = (
        ("Planning Desk", "Safety Stock"),
        ("Safety Stock", "Regional DC"),
        ("Safety Stock", "Priority Retail Accounts"),
    )
    crossdock_edges = (
        ("Cold Chain Carrier", "Cross-Dock"),
        ("Cross-Dock", "Priority Retail Accounts"),
        ("Cross-Dock", "Standard Retail Accounts"),
    )
    alternate_edges = (
        ("Planning Desk", "Alternate Supplier"),
        ("Alternate Supplier", "Regional Plant"),
        ("Alternate Supplier", "Co-Packer"),
    )
    return [
        Scenario("no_control", "control"),
        Scenario(
            "carrier_priority",
            "control",
            cost=_edge_cost(carrier_edges, 1.2),
            friction_edges=carrier_edges,
            friction_delta=1.2,
            gate_replenishment={("cold_chain_gate", "cold_slot"): 3},
        ),
        Scenario(
            "promotion_throttle",
            "control",
            cost=_edge_cost(throttle_edges, 1.0),
            friction_edges=throttle_edges,
            friction_delta=-1.0,
        ),
        Scenario(
            "safety_stock_release",
            "control",
            cost=_edge_cost(reserve_edges, 1.1),
            friction_edges=reserve_edges,
            friction_delta=1.1,
            gate_replenishment={("cold_chain_gate", "finished_lot"): 2},
        ),
        Scenario(
            "crossdock_activation",
            "control",
            cost=_edge_cost(crossdock_edges, 1.0),
            beta_edges=crossdock_edges,
            beta_boost=1.0,
            node_capacity_caps={"Cross-Dock": 8},
        ),
        Scenario(
            "alternate_supplier_pull",
            "control",
            cost=_edge_cost(alternate_edges, 1.0),
            beta_edges=alternate_edges,
            beta_boost=1.0,
            gate_replenishment={
                ("production_gate", "product_input"): 2,
                ("copacker_gate", "copack_product"): 2,
            },
        ),
    ]


def _classify(row: dict, baseline: dict | None = None) -> str:
    baseline_completion = baseline["service_completion_rate"] if baseline else 0.0
    if row["capacity_overflow_rate"] > 0.12:
        return "capacity_backfire"
    if row["gate_pressure_rate"] > 0.95:
        return "gate_blocked"
    if (
        row["service_completion_rate"] >= max(0.35, baseline_completion + 0.03)
        and row["priority_service_rate"] >= 0.12
        and row["lost_demand_rate"] <= 0.18
    ):
        return "viable_service_recovery"
    if row["service_completion_rate"] > baseline_completion:
        return "partial_recovery"
    return "no_recovery"


def _choice_point_inventory() -> list[dict]:
    outgoing: dict[str, list[str]] = {}
    for source, target, _ in EDGES:
        outgoing.setdefault(source, []).append(target)
    inventory = []
    for source, targets in sorted(outgoing.items()):
        inventory.append({
            "node": source,
            "outdegree": len(targets),
            "choice_type": "choice_point" if len(targets) > 1 else "serial_corridor",
            "targets": targets,
        })
    return inventory


def simulate(
    config: SimulationConfig,
    scenario: Scenario | None = None,
    control: Scenario | None = None,
) -> dict:
    scenario = scenario or scenarios()[0]
    control = control or controls()[0]
    kernel = build_kernel(config)
    labels = kernel.topo.labels
    _apply_scenario(kernel, labels, scenario)
    _apply_scenario(kernel, labels, control)

    randomization_parts = (
        (config.randomization_key, scenario.name)
        if config.randomization_key is not None
        else (scenario.name, control.name)
    )
    rng = np.random.default_rng(_seed_for(config, _stable_salt(*randomization_parts)))
    n = kernel.topo.N
    positions = np.full(config.agents, labels.index("Planning Desk"), dtype=int)
    telemetries = _initial_telemetries(config)

    node_capacity_idx = {
        labels.index(node): int(capacity)
        for item in (scenario, control)
        for node, capacity in item.node_capacity_caps.items()
    }
    edge_capacity_idx = {
        (labels.index(source), labels.index(target)): int(capacity)
        for item in (scenario, control)
        for (source, target), capacity in item.edge_capacity_caps.items()
    }
    gate_replenishment = _combined_gate_replenishment(scenario, control)
    gate_capacity_caps = _combined_gate_capacity_caps(scenario, control)
    gate_inventory, gate_arrivals = _init_gate_state(config)
    gate_attempts = {gate.name: 0 for gate in GATES}
    gate_blocked = {gate.name: 0 for gate in GATES}
    gate_inventory_blocked = {gate.name: 0 for gate in GATES}
    gate_capacity_blocked = {gate.name: 0 for gate in GATES}
    gate_completions = {gate.name: 0 for gate in GATES}
    gate_backlog_series = {gate.name: [] for gate in GATES}
    gate_replenished = {gate.name: {part: 0 for part in gate.parts} for gate in GATES}

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

    terminal_idx = np.array([labels.index(name) for name in TERMINAL_NODES], dtype=int)
    service_idx = np.array([labels.index(name) for name in SERVICE_NODES], dtype=int)
    priority_idx = labels.index(PRIORITY_NODE)
    standard_idx = labels.index("Standard Retail Accounts")
    lost_idx = labels.index(LOST_NODE)
    regional_dc_idx = labels.index("Regional DC")
    crossdock_idx = labels.index("Cross-Dock")
    safety_idx = labels.index("Safety Stock")
    completed_source = np.zeros(config.agents, dtype=np.int8)
    edge_counts = np.zeros((n, n), dtype=np.float64)

    edge_overflow_events = 0
    edge_capacity_attempts = 0
    node_overflow_events = 0
    node_capacity_attempts = 0
    terminal_hits = []
    service_hits = []
    priority_hits = []
    lost_hits = []

    for step in range(config.steps):
        for (gate_name, part), amount in gate_replenishment.items():
            gate_inventory[gate_name][part] += amount
            gate_arrivals[gate_name][part] += amount
            gate_replenished[gate_name][part] += amount

        P_all = kernel.transition_matrix_batch(telemetries, step=step)
        rows = P_all[np.arange(config.agents), positions, :].copy()
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
                blocked = [idx for idx in attempted.tolist() if idx not in allowed]
                edge_overflow_events += len(blocked)
                next_positions[np.array(blocked, dtype=int)] = positions[np.array(blocked, dtype=int)]

        for j, capacity in node_capacity_idx.items():
            attempted = np.where((positions != j) & (next_positions == j))[0]
            node_capacity_attempts += int(len(attempted))
            if len(attempted) > capacity:
                allowed = set(rng.choice(attempted, size=capacity, replace=False).tolist())
                blocked = [idx for idx in attempted.tolist() if idx not in allowed]
                node_overflow_events += len(blocked)
                next_positions[np.array(blocked, dtype=int)] = positions[np.array(blocked, dtype=int)]

        for gate in GATES:
            i, j = gate_source_target_idx[gate.name]
            attempted = np.where((positions == i) & (next_positions == j))[0]
            gate_attempts[gate.name] += int(len(attempted))
            feasible = int(len(attempted))
            feasible_limits = []
            for part in gate.parts:
                consumption = gate.consumption.get(part, 1)
                inventory = gate_inventory[gate.name][part]
                if consumption <= 0:
                    continue
                feasible_limits.append(inventory // consumption)
            if feasible_limits:
                feasible = min(feasible, min(feasible_limits))
            inventory_feasible = min(int(len(attempted)), int(feasible))
            service_capacity = gate_capacity_caps.get(gate.name, inventory_feasible)
            allowed_count = min(inventory_feasible, service_capacity)
            gate_inventory_blocked[gate.name] += max(0, int(len(attempted)) - inventory_feasible)
            gate_capacity_blocked[gate.name] += max(0, inventory_feasible - allowed_count)
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
            i, _ = gate_source_target_idx[gate.name]
            gate_backlog_series[gate.name].append(float(np.mean(next_positions == i)))

        for gate in GATES:
            for (source_idx, target_idx), part in gate_arrival_idx[gate.name].items():
                arrivals = int(np.sum((positions == source_idx) & (next_positions == target_idx)))
                if arrivals > 0:
                    gate_arrivals[gate.name][part] += arrivals
                    gate_inventory[gate.name][part] += arrivals

        uncompleted = completed_source == 0
        priority_arrival = next_positions == priority_idx
        standard_arrival = next_positions == standard_idx
        lost_arrival = next_positions == lost_idx
        completed_source[uncompleted & ((positions == regional_dc_idx) | (positions == crossdock_idx)) & priority_arrival] = 1
        completed_source[uncompleted & ((positions == regional_dc_idx) | (positions == crossdock_idx)) & standard_arrival] = 2
        completed_source[uncompleted & (positions == safety_idx) & priority_arrival] = 3
        completed_source[uncompleted & lost_arrival] = 4

        np.add.at(edge_counts, (positions, next_positions), 1.0)
        visited = kernel.topo.node_features[next_positions]
        lam = config.feedback_rate
        telemetries = (1.0 - lam) * telemetries + lam * visited
        norms = np.linalg.norm(telemetries, axis=1, keepdims=True)
        telemetries = np.where(norms > 0, telemetries / norms, telemetries)
        positions = next_positions

        terminal_hits.append(float(np.mean(np.isin(positions, terminal_idx))))
        service_hits.append(float(np.mean(np.isin(positions, service_idx))))
        priority_hits.append(float(np.mean(positions == priority_idx)))
        lost_hits.append(float(np.mean(positions == lost_idx)))

    total_capacity_attempts = edge_capacity_attempts + node_capacity_attempts
    total_overflow = edge_overflow_events + node_overflow_events
    gate_total_attempts = sum(gate_attempts.values())
    gate_total_blocked = sum(gate_blocked.values())
    gate_total_inventory_blocked = sum(gate_inventory_blocked.values())
    gate_total_capacity_blocked = sum(gate_capacity_blocked.values())
    gate_total_completions = sum(gate_completions.values())
    window = max(8, config.steps // 4)
    gate_block_rate = float(gate_total_blocked / max(gate_total_attempts, 1))
    gate_starvation_rate = float(gate_total_inventory_blocked / max(gate_total_attempts, 1))
    gate_service_capacity_block_rate = float(gate_total_capacity_blocked / max(gate_total_attempts, 1))
    gate_contention_pass_rate = float(gate_total_completions / max(gate_total_attempts, 1))
    gate_contention_rate = float(gate_total_attempts / max(config.agents * config.steps, 1))
    gate_backlog_pressure = float(max(
        (np.mean(series[-window:]) if series else 0.0)
        for series in gate_backlog_series.values()
    ))
    gate_starvation_share_of_blocked = float(gate_total_inventory_blocked / max(gate_total_blocked, 1))
    gate_capacity_share_of_blocked = float(gate_total_capacity_blocked / max(gate_total_blocked, 1))
    if gate_total_blocked == 0:
        gate_primary_pressure = "ordinary_contention" if gate_total_attempts > 0 else "none"
    elif gate_total_inventory_blocked >= gate_total_capacity_blocked:
        gate_primary_pressure = "inventory_starvation"
    else:
        gate_primary_pressure = "service_capacity"
    service_completed = int(np.sum((completed_source == 1) | (completed_source == 2) | (completed_source == 3)))
    priority_completed = int(np.sum((completed_source == 1) | (completed_source == 3)))
    standard_completed = int(np.sum(completed_source == 2))
    lost_completed = int(np.sum(completed_source == 4))
    limiting = _limiting_part(gate_inventory)

    edge_flow = edge_counts / max(float(edge_counts.sum()), 1.0)
    edge_current = edge_flow - edge_flow.T
    return {
        "scenario": scenario.name,
        "control": control.name,
        "agents": config.agents,
        "steps": config.steps,
        "service_completion_rate": float(service_completed / max(config.agents, 1)),
        "priority_service_rate": float(priority_completed / max(config.agents, 1)),
        "standard_service_rate": float(standard_completed / max(config.agents, 1)),
        "lost_demand_rate": float(lost_completed / max(config.agents, 1)),
        "terminal_share": float(np.mean(terminal_hits[-window:])),
        "service_share": float(np.mean(service_hits[-window:])),
        "priority_share": float(np.mean(priority_hits[-window:])),
        "lost_share": float(np.mean(lost_hits[-window:])),
        "capacity_overflow_rate": float(total_overflow / max(total_capacity_attempts, 1)),
        "gate_pressure_rate": float(gate_total_blocked / max(gate_total_blocked + gate_total_completions, 1)),
        "gate_block_rate": gate_block_rate,
        "gate_starvation_rate": gate_starvation_rate,
        "gate_service_capacity_block_rate": gate_service_capacity_block_rate,
        "gate_contention_pass_rate": gate_contention_pass_rate,
        "gate_contention_rate": gate_contention_rate,
        "gate_backlog_pressure": gate_backlog_pressure,
        "gate_starvation_share_of_blocked": gate_starvation_share_of_blocked,
        "gate_capacity_share_of_blocked": gate_capacity_share_of_blocked,
        "gate_primary_pressure": gate_primary_pressure,
        "gate_completion_ratio": float(gate_total_completions / max(gate_total_attempts, 1)),
        "gate_attempts": gate_attempts,
        "gate_blocked": gate_blocked,
        "gate_inventory_blocked": gate_inventory_blocked,
        "gate_capacity_blocked": gate_capacity_blocked,
        "gate_capacity_caps": gate_capacity_caps,
        "gate_completions": gate_completions,
        "gate_replenished": gate_replenished,
        "gate_inventory_end": gate_inventory,
        "gate_arrivals": gate_arrivals,
        "limiting_gate": limiting["gate"],
        "limiting_part": limiting["part"],
        "limiting_inventory": limiting["inventory"],
        "edge_current_norm": float(np.linalg.norm(edge_current, ord="fro")),
        "control_cost": control.cost,
    }


def static_expected_service_share(config: SimulationConfig, scenario: Scenario, control: Scenario) -> float:
    kernel = build_kernel(config)
    labels = kernel.topo.labels
    _apply_scenario(kernel, labels, scenario)
    _apply_scenario(kernel, labels, control)
    telemetry = _initial_telemetries(config).mean(axis=0)
    telemetry /= max(float(np.linalg.norm(telemetry)), 1e-12)
    transition = kernel.transition_matrix(telemetry, step=0)
    occupancy = np.zeros(kernel.topo.N, dtype=np.float64)
    occupancy[labels.index("Planning Desk")] = 1.0
    service_entry_mass = 0.0
    service_idx = np.array([labels.index(name) for name in SERVICE_NODES], dtype=int)
    for _ in range(config.steps):
        flow = occupancy[:, np.newaxis] * transition
        service_entry_mass += float(flow[:, service_idx].sum())
        occupancy = occupancy @ transition
    return min(service_entry_mass, 1.0)


def evaluate_cell(config: SimulationConfig, scenario: Scenario, control: Scenario) -> dict:
    baseline = simulate(config, scenario, controls()[0])
    row = simulate(config, scenario, control)
    frozen_config = SimulationConfig(
        agents=config.agents,
        steps=config.steps,
        feedback_rate=0.0,
        temperature=config.temperature,
        seed=config.seed,
        gate_initial_inventory=config.gate_initial_inventory,
        randomization_key=config.randomization_key,
    )
    frozen = simulate(frozen_config, scenario, control)
    row["classification"] = _classify(row, baseline)
    row["service_delta_vs_baseline"] = row["service_completion_rate"] - baseline["service_completion_rate"]
    row["priority_delta_vs_baseline"] = row["priority_service_rate"] - baseline["priority_service_rate"]
    row["lost_delta_vs_baseline"] = row["lost_demand_rate"] - baseline["lost_demand_rate"]
    row["static_expected_service_share"] = static_expected_service_share(config, scenario, control)
    row["frozen_service_completion_rate"] = frozen["service_completion_rate"]
    row["frozen_classification"] = _classify(frozen, simulate(frozen_config, scenario, controls()[0]))
    row["resilience_score"] = (
        row["service_delta_vs_baseline"]
        + 0.70 * row["priority_delta_vs_baseline"]
        - 0.50 * max(row["lost_delta_vs_baseline"], 0.0)
        - 0.40 * row["capacity_overflow_rate"]
        - 0.05 * row["gate_pressure_rate"]
        - 0.002 * row["control_cost"]
    )
    return row


def run_pilot(config: SimulationConfig | None = None, quick: bool = False) -> dict:
    config = config or SimulationConfig()
    scenario_list = scenarios()[:3] if quick else scenarios()
    control_list = controls()[:4] if quick else controls()
    rows = [
        evaluate_cell(config, scenario, control)
        for scenario in scenario_list
        for control in control_list
    ]
    grouped = {}
    best_by_scenario = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        subset = [row for row in rows if row["scenario"] == scenario]
        best = max(subset, key=lambda row: row["resilience_score"])
        best_by_scenario[scenario] = best
        grouped[scenario] = {
            "best_control": best["control"],
            "best_classification": best["classification"],
            "best_score": best["resilience_score"],
            "baseline_service": next(row["service_completion_rate"] for row in subset if row["control"] == "no_control"),
            "best_service": best["service_completion_rate"],
            "best_priority": best["priority_service_rate"],
            "best_overflow": best["capacity_overflow_rate"],
            "best_gate_pressure": best["gate_pressure_rate"],
            "best_gate_starvation": best["gate_starvation_rate"],
            "best_gate_capacity_block": best["gate_service_capacity_block_rate"],
            "best_gate_contention": best["gate_contention_rate"],
            "best_gate_primary_pressure": best["gate_primary_pressure"],
        }
    return {
        "config": {
            "agents": config.agents,
            "steps": config.steps,
            "feedback_rate": config.feedback_rate,
            "temperature": config.temperature,
            "seed": config.seed,
            "quick": quick,
        },
        "choice_point_inventory": _choice_point_inventory(),
        "rows": rows,
        "summary": grouped,
        "best_by_scenario": best_by_scenario,
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Consumer Goods Cold-Chain Circulation Pilot",
        "",
        "## Scope",
        "",
        (
            "Narrow DTE industrial module for refrigerated packaged goods under "
            "cold-chain, promotion, supplier, and regional receiving stress."
        ),
        "",
        f"- Agents: `{payload['config']['agents']}`",
        f"- Steps: `{payload['config']['steps']}`",
        f"- Feedback rate: `{payload['config']['feedback_rate']}`",
        f"- Scenario-control runs: `{len(payload['rows'])}`",
        "",
        "## Choice-Point Inventory",
        "",
        "| Node | Outdegree | Type |",
        "|---|---:|---|",
    ]
    for item in payload["choice_point_inventory"]:
        lines.append(f"| `{item['node']}` | {item['outdegree']} | `{item['choice_type']}` |")

    lines.extend([
        "",
        "## Best Control By Scenario",
        "",
        "| Scenario | Best Control | Classification | Baseline Service | Best Service | Priority Service | Overflow | Gate Starvation | Gate Capacity Block | Gate Load | Primary Gate Pressure |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for scenario, row in payload["summary"].items():
        lines.append(
            f"| `{scenario}` | `{row['best_control']}` | `{row['best_classification']}` | "
            f"{row['baseline_service']:.1%} | {row['best_service']:.1%} | "
            f"{row['best_priority']:.1%} | {row['best_overflow']:.1%} | "
            f"{row['best_gate_starvation']:.1%} | {row['best_gate_capacity_block']:.1%} | "
            f"{row['best_gate_contention']:.1%} | `{row['best_gate_primary_pressure']}` |"
        )

    lines.extend([
        "",
        "## Full Results",
        "",
        "| Scenario | Control | Class | Service | Priority | Lost | Overflow | Starvation | Gate Capacity | Gate Load | Static Service | Frozen Service | Score |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["rows"]:
        lines.append(
            f"| `{row['scenario']}` | `{row['control']}` | `{row['classification']}` | "
            f"{row['service_completion_rate']:.1%} | {row['priority_service_rate']:.1%} | "
            f"{row['lost_demand_rate']:.1%} | {row['capacity_overflow_rate']:.1%} | "
            f"{row['gate_starvation_rate']:.1%} | {row['gate_service_capacity_block_rate']:.1%} | "
            f"{row['gate_contention_rate']:.1%} | {row['static_expected_service_share']:.1%} | "
            f"{row['frozen_service_completion_rate']:.1%} | {row['resilience_score']:.4f} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "This pilot should be read as a module-validation slice. A useful control "
            "must improve completed service while avoiding capacity overflow and gate "
            "pressure. Static service share is included as a false-positive check, not "
            "as a recommendation criterion."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("consumer_goods_circulation_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("CONSUMER_GOODS_PILOT_REPORT.md"))
    args = parser.parse_args()
    config = SimulationConfig(agents=64, steps=24) if args.quick else SimulationConfig()
    payload = run_pilot(config, quick=args.quick)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": len(payload["rows"]),
        "summary": payload["summary"],
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

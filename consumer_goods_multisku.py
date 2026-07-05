"""
Controlled multi-SKU consumer-goods topology for DTE.

This module extends the first cold-chain pilot into a larger but still
interpretable industrial slice:

- 3 SKU demand classes
- 2 regional DCs
- reserved and spot cold-chain carriers
- promotion calendar pressure
- SKU-level safety stock and substitution accounting
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
    "Regional Specificity",
]


NODES = {
    "Demand Control Tower": [0.82, 0.70, 0.74, 0.55, 0.55, 0.45, 0.78, 0.48, 0.46, 0.50],
    "Core Dairy Supplier": [0.62, 0.70, 0.76, 0.86, 0.58, 0.24, 0.34, 0.62, 0.20, 0.44],
    "Cultured Dairy Supplier": [0.70, 0.78, 0.66, 0.82, 0.48, 0.45, 0.28, 0.56, 0.72, 0.48],
    "Frozen Ingredient Supplier": [0.54, 0.74, 0.72, 1.00, 0.50, 0.20, 0.28, 0.74, 0.26, 0.44],
    "Packaging Supplier": [0.48, 0.60, 0.58, 0.32, 0.82, 0.28, 0.34, 0.70, 0.22, 0.38],
    "Plant Line A": [0.72, 0.72, 0.74, 0.76, 0.55, 0.30, 0.42, 0.72, 0.38, 0.46],
    "Plant Line B": [0.66, 0.70, 0.74, 0.94, 0.50, 0.24, 0.38, 0.82, 0.26, 0.42],
    "Core Finished Staging": [0.72, 0.70, 0.78, 0.82, 0.42, 0.24, 0.52, 0.54, 0.26, 0.46],
    "Yogurt Finished Staging": [0.86, 0.86, 0.66, 0.78, 0.42, 0.56, 0.36, 0.56, 1.00, 0.48],
    "Frozen Finished Staging": [0.68, 0.78, 0.82, 1.00, 0.36, 0.18, 0.42, 0.66, 0.30, 0.42],
    "Reserved Cold Carrier": [0.84, 0.64, 0.94, 1.00, 0.30, 0.18, 0.48, 0.38, 0.26, 0.52],
    "Spot Cold Carrier": [0.88, 0.50, 0.72, 0.92, 0.82, 0.36, 0.18, 0.74, 0.50, 0.42],
    "North DC": [0.78, 0.72, 0.76, 0.78, 0.52, 0.34, 0.58, 0.62, 0.42, 0.88],
    "South DC": [0.78, 0.72, 0.76, 0.78, 0.52, 0.34, 0.58, 0.62, 0.42, 0.12],
    "North Priority Accounts": [1.00, 0.88, 1.00, 0.88, 0.30, 0.18, 0.62, 0.34, 0.48, 0.92],
    "North Standard Accounts": [0.78, 0.70, 0.62, 0.72, 0.62, 0.56, 0.30, 0.54, 0.66, 0.90],
    "South Priority Accounts": [1.00, 0.88, 1.00, 0.88, 0.30, 0.18, 0.62, 0.34, 0.48, 0.10],
    "South Standard Accounts": [0.78, 0.70, 0.62, 0.72, 0.62, 0.56, 0.30, 0.54, 0.66, 0.08],
    "Promo Calendar": [0.96, 0.92, 0.58, 0.66, 0.42, 0.68, 0.18, 0.64, 1.00, 0.52],
    "Core Safety Stock": [0.70, 0.74, 0.88, 0.78, 0.34, 0.24, 1.00, 0.34, 0.18, 0.50],
    "Yogurt Safety Stock": [0.88, 0.84, 0.78, 0.78, 0.34, 0.54, 0.92, 0.36, 0.92, 0.50],
    "Frozen Safety Stock": [0.76, 0.80, 0.90, 1.00, 0.32, 0.18, 0.94, 0.40, 0.22, 0.50],
    "Substitute Shelf": [0.62, 0.58, 0.52, 0.64, 0.54, 1.00, 0.24, 0.42, 0.48, 0.50],
    "Lost Demand": [0.18, 0.10, 0.10, 0.20, 0.92, 0.82, 0.02, 0.90, 0.24, 0.50],
}


EDGES = [
    ("Demand Control Tower", "Core Dairy Supplier", 2.0),
    ("Demand Control Tower", "Cultured Dairy Supplier", 2.2),
    ("Demand Control Tower", "Frozen Ingredient Supplier", 2.5),
    ("Demand Control Tower", "Packaging Supplier", 2.1),
    ("Demand Control Tower", "Plant Line A", 2.8),
    ("Demand Control Tower", "Plant Line B", 3.0),
    ("Demand Control Tower", "Promo Calendar", 2.2),
    ("Demand Control Tower", "Core Safety Stock", 3.0),
    ("Demand Control Tower", "Yogurt Safety Stock", 3.0),
    ("Demand Control Tower", "Frozen Safety Stock", 3.2),
    ("Core Dairy Supplier", "Plant Line A", 1.7),
    ("Cultured Dairy Supplier", "Plant Line A", 1.8),
    ("Frozen Ingredient Supplier", "Plant Line B", 1.8),
    ("Packaging Supplier", "Plant Line A", 1.9),
    ("Packaging Supplier", "Plant Line B", 2.1),
    ("Plant Line A", "Core Finished Staging", 1.5),
    ("Plant Line A", "Yogurt Finished Staging", 1.6),
    ("Plant Line B", "Frozen Finished Staging", 1.6),
    ("Core Finished Staging", "Reserved Cold Carrier", 1.7),
    ("Yogurt Finished Staging", "Reserved Cold Carrier", 1.6),
    ("Frozen Finished Staging", "Reserved Cold Carrier", 1.8),
    ("Core Finished Staging", "Spot Cold Carrier", 2.5),
    ("Yogurt Finished Staging", "Spot Cold Carrier", 2.3),
    ("Frozen Finished Staging", "Spot Cold Carrier", 2.8),
    ("Reserved Cold Carrier", "North DC", 1.8),
    ("Reserved Cold Carrier", "South DC", 1.8),
    ("Spot Cold Carrier", "North DC", 2.4),
    ("Spot Cold Carrier", "South DC", 2.6),
    ("North DC", "North Priority Accounts", 1.5),
    ("North DC", "North Standard Accounts", 1.7),
    ("South DC", "South Priority Accounts", 1.5),
    ("South DC", "South Standard Accounts", 1.7),
    ("North DC", "South DC", 2.4),
    ("South DC", "North DC", 2.4),
    ("Promo Calendar", "Yogurt Finished Staging", 1.4),
    ("Promo Calendar", "North Standard Accounts", 2.0),
    ("Promo Calendar", "South Standard Accounts", 2.1),
    ("Core Safety Stock", "North Priority Accounts", 1.5),
    ("Core Safety Stock", "South Priority Accounts", 1.6),
    ("Yogurt Safety Stock", "North Standard Accounts", 1.6),
    ("Yogurt Safety Stock", "South Standard Accounts", 1.7),
    ("Frozen Safety Stock", "North Priority Accounts", 1.8),
    ("Frozen Safety Stock", "South Priority Accounts", 1.8),
    ("Yogurt Safety Stock", "Substitute Shelf", 2.0),
    ("Core Safety Stock", "Substitute Shelf", 2.3),
    ("North Standard Accounts", "Demand Control Tower", 5.6),
    ("South Standard Accounts", "Demand Control Tower", 5.6),
    ("North Priority Accounts", "Demand Control Tower", 5.4),
    ("South Priority Accounts", "Demand Control Tower", 5.4),
    ("Substitute Shelf", "Demand Control Tower", 5.8),
    ("Lost Demand", "Demand Control Tower", 6.5),
    ("Promo Calendar", "Lost Demand", 3.0),
    ("Spot Cold Carrier", "Lost Demand", 3.2),
]


INTENTS = {
    "Core Dairy": [0.76, 0.68, 0.72, 0.82, 0.56, 0.34, 0.32, 0.58, 0.32, 0.48],
    "Promo Yogurt": [0.96, 0.92, 0.62, 0.78, 0.42, 0.74, 0.22, 0.62, 1.00, 0.54],
    "Frozen Dessert": [0.82, 0.86, 0.84, 1.00, 0.38, 0.22, 0.38, 0.70, 0.38, 0.46],
    "Priority Replenishment": [1.00, 0.82, 1.00, 0.90, 0.28, 0.18, 0.58, 0.38, 0.42, 0.50],
}


DEFAULT_POPULATION = {
    "Core Dairy": 0.36,
    "Promo Yogurt": 0.30,
    "Frozen Dessert": 0.22,
    "Priority Replenishment": 0.12,
}


TERMINAL_NODES = (
    "North Priority Accounts",
    "North Standard Accounts",
    "South Priority Accounts",
    "South Standard Accounts",
    "Substitute Shelf",
    "Lost Demand",
)
SERVICE_NODES = (
    "North Priority Accounts",
    "North Standard Accounts",
    "South Priority Accounts",
    "South Standard Accounts",
    "Substitute Shelf",
)
PRIORITY_NODES = ("North Priority Accounts", "South Priority Accounts")
STANDARD_NODES = ("North Standard Accounts", "South Standard Accounts")
SUBSTITUTE_NODE = "Substitute Shelf"
LOST_NODE = "Lost Demand"


@dataclass(frozen=True)
class MultiSKUConfig:
    agents: int = 220
    steps: int = 56
    feedback_rate: float = 0.15
    temperature: float = 0.80
    seed: int = 20260617
    gate_initial_inventory: int = 18
    randomization_key: str | None = "consumer_goods_multisku"


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
        "core_production_gate",
        "Plant Line A",
        "Core Finished Staging",
        ("core_input", "packaging", "line_a_labor"),
        {
            ("Core Dairy Supplier", "Plant Line A"): "core_input",
            ("Packaging Supplier", "Plant Line A"): "packaging",
            ("Demand Control Tower", "Plant Line A"): "line_a_labor",
        },
        {"core_input": 1, "packaging": 1, "line_a_labor": 1},
    ),
    GateSpec(
        "yogurt_production_gate",
        "Plant Line A",
        "Yogurt Finished Staging",
        ("yogurt_input", "packaging", "line_a_labor"),
        {
            ("Cultured Dairy Supplier", "Plant Line A"): "yogurt_input",
            ("Packaging Supplier", "Plant Line A"): "packaging",
            ("Demand Control Tower", "Plant Line A"): "line_a_labor",
        },
        {"yogurt_input": 1, "packaging": 1, "line_a_labor": 1},
    ),
    GateSpec(
        "frozen_production_gate",
        "Plant Line B",
        "Frozen Finished Staging",
        ("frozen_input", "packaging", "line_b_labor"),
        {
            ("Frozen Ingredient Supplier", "Plant Line B"): "frozen_input",
            ("Packaging Supplier", "Plant Line B"): "packaging",
            ("Demand Control Tower", "Plant Line B"): "line_b_labor",
        },
        {"frozen_input": 1, "packaging": 1, "line_b_labor": 1},
    ),
    GateSpec("core_reserved_gate", "Core Finished Staging", "Reserved Cold Carrier", ("reserved_slot",), {}, {"reserved_slot": 1}),
    GateSpec("yogurt_reserved_gate", "Yogurt Finished Staging", "Reserved Cold Carrier", ("reserved_slot",), {}, {"reserved_slot": 1}),
    GateSpec("frozen_reserved_gate", "Frozen Finished Staging", "Reserved Cold Carrier", ("reserved_slot",), {}, {"reserved_slot": 1}),
    GateSpec("core_spot_gate", "Core Finished Staging", "Spot Cold Carrier", ("spot_slot",), {}, {"spot_slot": 1}),
    GateSpec("yogurt_spot_gate", "Yogurt Finished Staging", "Spot Cold Carrier", ("spot_slot",), {}, {"spot_slot": 1}),
    GateSpec("frozen_spot_gate", "Frozen Finished Staging", "Spot Cold Carrier", ("spot_slot",), {}, {"spot_slot": 1}),
]


BASE_GATE_REPLENISHMENT = {
    ("core_production_gate", "core_input"): 3,
    ("core_production_gate", "packaging"): 3,
    ("core_production_gate", "line_a_labor"): 3,
    ("yogurt_production_gate", "yogurt_input"): 3,
    ("yogurt_production_gate", "packaging"): 3,
    ("yogurt_production_gate", "line_a_labor"): 3,
    ("frozen_production_gate", "frozen_input"): 2,
    ("frozen_production_gate", "packaging"): 2,
    ("frozen_production_gate", "line_b_labor"): 2,
    ("core_reserved_gate", "reserved_slot"): 3,
    ("yogurt_reserved_gate", "reserved_slot"): 3,
    ("frozen_reserved_gate", "reserved_slot"): 2,
    ("core_spot_gate", "spot_slot"): 1,
    ("yogurt_spot_gate", "spot_slot"): 1,
    ("frozen_spot_gate", "spot_slot"): 1,
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


def _seed_for(config: MultiSKUConfig, salt: int = 0) -> int:
    raw = (
        config.seed
        + 101 * config.agents
        + 1009 * config.steps
        + 9173 * int(round(config.feedback_rate * 1000))
        + 6113 * int(round(config.temperature * 1000))
        + salt
    )
    return int(raw % (2**32 - 1))


def _edge_cost(edges: tuple[tuple[str, str], ...], magnitude: float) -> float:
    return float(len(edges) * abs(magnitude))


def build_kernel(config: MultiSKUConfig) -> DynamicTopologyKernel:
    topology = topology_from_edges(
        nodes={label: np.array(features, dtype=np.float64) for label, features in NODES.items()},
        edges=EDGES,
        undirected=False,
    )
    node_bias = np.zeros(topology.N, dtype=np.float64)
    for label, bias in {
        "Demand Control Tower": 0.22,
        "North DC": 0.08,
        "South DC": 0.08,
        "North Priority Accounts": 0.06,
        "South Priority Accounts": 0.06,
    }.items():
        node_bias[topology.labels.index(label)] = bias
    return DynamicTopologyKernel(
        topology=topology,
        beta=np.full((topology.N, topology.N), 1.25, dtype=np.float64),
        feedback_rate=config.feedback_rate,
        feedback_noise=0.0,
        temperature=config.temperature,
        node_bias=node_bias,
        sponsor_decay=0.0,
    )


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


def _initial_agents(config: MultiSKUConfig) -> tuple[np.ndarray, list[str]]:
    telemetries = np.zeros((config.agents, len(FEATURE_LABELS)), dtype=np.float64)
    classes: list[str] = []
    start = 0
    items = list(DEFAULT_POPULATION.items())
    for idx, (intent, share) in enumerate(items):
        count = int(round(config.agents * share)) if idx < len(items) - 1 else config.agents - start
        end = min(config.agents, start + count)
        telemetries[start:end] = _normalized(INTENTS[intent])
        classes.extend([intent] * max(0, end - start))
        start = end
    return telemetries, classes


def _combined_gate_replenishment(*items: Scenario) -> dict[tuple[str, str], int]:
    valid = {(gate.name, part) for gate in GATES for part in gate.parts}
    replenishment = dict(BASE_GATE_REPLENISHMENT)
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


def _init_gate_state(config: MultiSKUConfig) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    inventory = {gate.name: {part: config.gate_initial_inventory for part in gate.parts} for gate in GATES}
    arrivals = {gate.name: {part: 0 for part in gate.parts} for gate in GATES}
    return inventory, arrivals


def _limiting_part(inventory: dict[str, dict[str, int]]) -> dict:
    candidates = []
    for gate_name, parts in inventory.items():
        for part, count in parts.items():
            candidates.append((count, gate_name, part))
    count, gate_name, part = min(candidates)
    return {"gate": gate_name, "part": part, "inventory": int(count)}


def scenarios() -> list[Scenario]:
    promo_edges = (
        ("Demand Control Tower", "Promo Calendar"),
        ("Promo Calendar", "Yogurt Finished Staging"),
        ("Promo Calendar", "North Standard Accounts"),
        ("Promo Calendar", "South Standard Accounts"),
    )
    reserved_edges = (
        ("Core Finished Staging", "Reserved Cold Carrier"),
        ("Yogurt Finished Staging", "Reserved Cold Carrier"),
        ("Frozen Finished Staging", "Reserved Cold Carrier"),
        ("Reserved Cold Carrier", "North DC"),
        ("Reserved Cold Carrier", "South DC"),
    )
    spot_edges = (
        ("Core Finished Staging", "Spot Cold Carrier"),
        ("Yogurt Finished Staging", "Spot Cold Carrier"),
        ("Frozen Finished Staging", "Spot Cold Carrier"),
        ("Spot Cold Carrier", "North DC"),
        ("Spot Cold Carrier", "South DC"),
    )
    return [
        Scenario("nominal_multisku", "baseline"),
        Scenario(
            "north_promotion_calendar",
            "promotion",
            cost=_edge_cost(promo_edges, 1.3),
            beta_edges=promo_edges,
            beta_boost=1.3,
            node_capacity_caps={"North Standard Accounts": 13},
        ),
        Scenario(
            "reserved_carrier_shortage",
            "capacity",
            cost=_edge_cost(reserved_edges, 1.0),
            friction_edges=reserved_edges,
            friction_delta=-0.9,
            gate_capacity_caps={
                "core_reserved_gate": 4,
                "yogurt_reserved_gate": 4,
                "frozen_reserved_gate": 3,
            },
            edge_capacity_caps={
                ("Reserved Cold Carrier", "North DC"): 8,
                ("Reserved Cold Carrier", "South DC"): 8,
            },
        ),
        Scenario(
            "spot_market_failure",
            "capacity",
            cost=_edge_cost(spot_edges, 1.0),
            friction_edges=spot_edges,
            friction_delta=-1.2,
            gate_capacity_caps={
                "core_spot_gate": 1,
                "yogurt_spot_gate": 1,
                "frozen_spot_gate": 1,
            },
            edge_capacity_caps={
                ("Spot Cold Carrier", "North DC"): 3,
                ("Spot Cold Carrier", "South DC"): 3,
            },
        ),
        Scenario(
            "south_receiving_constraint",
            "receiving",
            cost=3.0,
            node_capacity_caps={"South DC": 8, "South Standard Accounts": 8},
        ),
        Scenario(
            "combined_promo_reserved_shortage",
            "combined",
            cost=_edge_cost(promo_edges + reserved_edges, 1.2),
            beta_edges=promo_edges,
            beta_boost=1.3,
            friction_edges=reserved_edges,
            friction_delta=-0.9,
            gate_capacity_caps={
                "core_reserved_gate": 4,
                "yogurt_reserved_gate": 4,
                "frozen_reserved_gate": 3,
            },
            node_capacity_caps={"North Standard Accounts": 12, "South Standard Accounts": 12},
            edge_capacity_caps={
                ("Reserved Cold Carrier", "North DC"): 7,
                ("Reserved Cold Carrier", "South DC"): 7,
            },
        ),
    ]


def controls() -> list[Scenario]:
    reserved_edges = (
        ("Core Finished Staging", "Reserved Cold Carrier"),
        ("Yogurt Finished Staging", "Reserved Cold Carrier"),
        ("Frozen Finished Staging", "Reserved Cold Carrier"),
        ("Reserved Cold Carrier", "North DC"),
        ("Reserved Cold Carrier", "South DC"),
    )
    spot_edges = (
        ("Core Finished Staging", "Spot Cold Carrier"),
        ("Yogurt Finished Staging", "Spot Cold Carrier"),
        ("Frozen Finished Staging", "Spot Cold Carrier"),
        ("Spot Cold Carrier", "North DC"),
        ("Spot Cold Carrier", "South DC"),
    )
    safety_edges = (
        ("Core Safety Stock", "North Priority Accounts"),
        ("Core Safety Stock", "South Priority Accounts"),
        ("Yogurt Safety Stock", "North Standard Accounts"),
        ("Yogurt Safety Stock", "South Standard Accounts"),
        ("Frozen Safety Stock", "North Priority Accounts"),
        ("Frozen Safety Stock", "South Priority Accounts"),
    )
    transfer_edges = (
        ("North DC", "South DC"),
        ("South DC", "North DC"),
    )
    promo_edges = (
        ("Demand Control Tower", "Promo Calendar"),
        ("Promo Calendar", "Yogurt Finished Staging"),
        ("Promo Calendar", "North Standard Accounts"),
        ("Promo Calendar", "South Standard Accounts"),
    )
    return [
        Scenario("no_control", "control"),
        Scenario(
            "reserved_cold_priority",
            "control",
            cost=_edge_cost(reserved_edges, 1.2),
            friction_edges=reserved_edges,
            friction_delta=1.2,
            gate_replenishment={
                ("core_reserved_gate", "reserved_slot"): 3,
                ("yogurt_reserved_gate", "reserved_slot"): 3,
                ("frozen_reserved_gate", "reserved_slot"): 2,
            },
        ),
        Scenario(
            "spot_market_buy",
            "control",
            cost=_edge_cost(spot_edges, 1.4),
            friction_edges=spot_edges,
            friction_delta=1.1,
            gate_replenishment={
                ("core_spot_gate", "spot_slot"): 3,
                ("yogurt_spot_gate", "spot_slot"): 3,
                ("frozen_spot_gate", "spot_slot"): 2,
            },
        ),
        Scenario(
            "sku_safety_release",
            "control",
            cost=_edge_cost(safety_edges, 1.0),
            friction_edges=safety_edges,
            friction_delta=1.0,
            gate_replenishment={
                ("core_production_gate", "core_input"): 2,
                ("yogurt_production_gate", "yogurt_input"): 2,
                ("frozen_production_gate", "frozen_input"): 1,
            },
        ),
        Scenario(
            "north_south_rebalance",
            "control",
            cost=_edge_cost(transfer_edges, 1.1),
            beta_edges=transfer_edges,
            beta_boost=1.1,
            node_capacity_caps={"North DC": 14, "South DC": 14},
        ),
        Scenario(
            "promotion_sequence",
            "control",
            cost=_edge_cost(promo_edges, 0.9),
            friction_edges=promo_edges,
            friction_delta=-0.9,
        ),
    ]


def _classify(row: dict, baseline: dict | None = None) -> str:
    baseline_completion = baseline["service_completion_rate"] if baseline else 0.0
    if row["capacity_overflow_rate"] > 0.14:
        return "capacity_backfire"
    if row["gate_pressure_rate"] > 0.96:
        return "gate_blocked"
    if (
        row["service_completion_rate"] >= max(0.35, baseline_completion + 0.035)
        and row["priority_service_rate"] >= 0.16
        and row["lost_demand_rate"] <= 0.20
    ):
        return "viable_service_recovery"
    if row["service_completion_rate"] > baseline_completion:
        return "partial_recovery"
    return "no_recovery"


def simulate(config: MultiSKUConfig, scenario: Scenario | None = None, control: Scenario | None = None) -> dict:
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
    positions = np.full(config.agents, labels.index("Demand Control Tower"), dtype=int)
    telemetries, agent_classes = _initial_agents(config)

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

    gate_source_target_idx = {gate.name: (labels.index(gate.source), labels.index(gate.target)) for gate in GATES}
    gate_arrival_idx = {
        gate.name: {
            (labels.index(source), labels.index(target)): part
            for (source, target), part in gate.arrivals.items()
        }
        for gate in GATES
    }

    terminal_idx = np.array([labels.index(name) for name in TERMINAL_NODES], dtype=int)
    service_idx = np.array([labels.index(name) for name in SERVICE_NODES], dtype=int)
    priority_idx = np.array([labels.index(name) for name in PRIORITY_NODES], dtype=int)
    standard_idx = np.array([labels.index(name) for name in STANDARD_NODES], dtype=int)
    substitute_idx = labels.index(SUBSTITUTE_NODE)
    lost_idx = labels.index(LOST_NODE)

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
        service_arrival = np.isin(next_positions, service_idx)
        priority_arrival = np.isin(next_positions, priority_idx)
        standard_arrival = np.isin(next_positions, standard_idx)
        substitute_arrival = next_positions == substitute_idx
        lost_arrival = next_positions == lost_idx
        completed_source[uncompleted & priority_arrival] = 1
        completed_source[uncompleted & standard_arrival] = 2
        completed_source[uncompleted & substitute_arrival] = 3
        completed_source[uncompleted & lost_arrival] = 4

        np.add.at(edge_counts, (positions, next_positions), 1.0)
        visited = kernel.topo.node_features[next_positions]
        lam = config.feedback_rate
        telemetries = (1.0 - lam) * telemetries + lam * visited
        norms = np.linalg.norm(telemetries, axis=1, keepdims=True)
        telemetries = np.where(norms > 0, telemetries / norms, telemetries)
        positions = next_positions

        terminal_hits.append(float(np.mean(np.isin(positions, terminal_idx))))
        service_hits.append(float(np.mean(service_arrival | np.isin(positions, service_idx))))
        priority_hits.append(float(np.mean(np.isin(positions, priority_idx))))
        lost_hits.append(float(np.mean(positions == lost_idx)))

    total_capacity_attempts = edge_capacity_attempts + node_capacity_attempts
    total_overflow = edge_overflow_events + node_overflow_events
    gate_total_attempts = sum(gate_attempts.values())
    gate_total_blocked = sum(gate_blocked.values())
    gate_total_inventory_blocked = sum(gate_inventory_blocked.values())
    gate_total_capacity_blocked = sum(gate_capacity_blocked.values())
    gate_total_completions = sum(gate_completions.values())
    window = max(8, config.steps // 4)
    limiting = _limiting_part(gate_inventory)
    if gate_total_blocked == 0:
        gate_primary_pressure = "ordinary_contention" if gate_total_attempts > 0 else "none"
    elif gate_total_inventory_blocked >= gate_total_capacity_blocked:
        gate_primary_pressure = "inventory_starvation"
    else:
        gate_primary_pressure = "service_capacity"

    service_completed = int(np.sum((completed_source == 1) | (completed_source == 2) | (completed_source == 3)))
    priority_completed = int(np.sum(completed_source == 1))
    standard_completed = int(np.sum(completed_source == 2))
    substitute_completed = int(np.sum(completed_source == 3))
    lost_completed = int(np.sum(completed_source == 4))
    sku_service_rates = {}
    sku_lost_rates = {}
    for sku in sorted(set(agent_classes)):
        mask = np.array([item == sku for item in agent_classes], dtype=bool)
        count = int(np.sum(mask))
        sku_service_rates[sku] = float(np.sum(mask & ((completed_source == 1) | (completed_source == 2) | (completed_source == 3))) / max(count, 1))
        sku_lost_rates[sku] = float(np.sum(mask & (completed_source == 4)) / max(count, 1))

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
        "substitution_rate": float(substitute_completed / max(config.agents, 1)),
        "lost_demand_rate": float(lost_completed / max(config.agents, 1)),
        "sku_service_rates": sku_service_rates,
        "sku_lost_rates": sku_lost_rates,
        "terminal_share": float(np.mean(terminal_hits[-window:])),
        "service_share": float(np.mean(service_hits[-window:])),
        "priority_share": float(np.mean(priority_hits[-window:])),
        "lost_share": float(np.mean(lost_hits[-window:])),
        "capacity_overflow_rate": float(total_overflow / max(total_capacity_attempts, 1)),
        "gate_pressure_rate": float(gate_total_blocked / max(gate_total_blocked + gate_total_completions, 1)),
        "gate_block_rate": float(gate_total_blocked / max(gate_total_attempts, 1)),
        "gate_starvation_rate": float(gate_total_inventory_blocked / max(gate_total_attempts, 1)),
        "gate_service_capacity_block_rate": float(gate_total_capacity_blocked / max(gate_total_attempts, 1)),
        "gate_contention_rate": float(gate_total_attempts / max(config.agents * config.steps, 1)),
        "gate_backlog_pressure": float(max(
            (np.mean(series[-window:]) if series else 0.0)
            for series in gate_backlog_series.values()
        )),
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


def evaluate_cell(config: MultiSKUConfig, scenario: Scenario, control: Scenario) -> dict:
    baseline = simulate(config, scenario, controls()[0])
    row = simulate(config, scenario, control)
    row["classification"] = _classify(row, baseline)
    row["service_delta_vs_baseline"] = row["service_completion_rate"] - baseline["service_completion_rate"]
    row["priority_delta_vs_baseline"] = row["priority_service_rate"] - baseline["priority_service_rate"]
    row["lost_delta_vs_baseline"] = row["lost_demand_rate"] - baseline["lost_demand_rate"]
    row["substitution_delta_vs_baseline"] = row["substitution_rate"] - baseline["substitution_rate"]
    row["resilience_score"] = (
        row["service_delta_vs_baseline"]
        + 0.70 * row["priority_delta_vs_baseline"]
        + 0.20 * row["substitution_delta_vs_baseline"]
        - 0.50 * max(row["lost_delta_vs_baseline"], 0.0)
        - 0.40 * row["capacity_overflow_rate"]
        - 0.05 * row["gate_pressure_rate"]
        - 0.002 * row["control_cost"]
    )
    return row


def run_pilot(config: MultiSKUConfig | None = None, quick: bool = False) -> dict:
    config = config or MultiSKUConfig()
    scenario_list = scenarios()[:3] if quick else scenarios()
    control_list = controls()[:4] if quick else controls()
    rows = [
        evaluate_cell(config, scenario, control)
        for scenario in scenario_list
        for control in control_list
    ]
    summary = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        subset = [row for row in rows if row["scenario"] == scenario]
        best = max(subset, key=lambda row: row["resilience_score"])
        baseline = next(row for row in subset if row["control"] == "no_control")
        summary[scenario] = {
            "best_control": best["control"],
            "best_classification": best["classification"],
            "baseline_service": baseline["service_completion_rate"],
            "best_service": best["service_completion_rate"],
            "best_priority": best["priority_service_rate"],
            "best_substitution": best["substitution_rate"],
            "best_lost": best["lost_demand_rate"],
            "best_overflow": best["capacity_overflow_rate"],
            "best_gate_starvation": best["gate_starvation_rate"],
            "best_gate_capacity_block": best["gate_service_capacity_block_rate"],
            "best_gate_primary_pressure": best["gate_primary_pressure"],
            "best_score": best["resilience_score"],
            "best_sku_service_rates": best["sku_service_rates"],
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
        "rows": rows,
        "summary": summary,
    }


def render_report(payload: dict) -> str:
    lines = [
        "# Consumer Goods Multi-SKU Controlled Topology",
        "",
        "## Scope",
        "",
        (
            "Larger controlled consumer-goods topology with three SKU classes, two DCs, "
            "reserved/spot cold carriers, promotion calendar pressure, SKU safety stock, "
            "and substitution accounting."
        ),
        "",
        f"- Agents: `{payload['config']['agents']}`",
        f"- Steps: `{payload['config']['steps']}`",
        f"- Scenario-control runs: `{len(payload['rows'])}`",
        "",
        "## Best Control By Scenario",
        "",
        "| Scenario | Best Control | Class | Baseline Service | Best Service | Priority | Substitute | Lost | Overflow | Starvation | Gate Capacity | Primary Gate Pressure | Score |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for scenario, row in payload["summary"].items():
        lines.append(
            f"| `{scenario}` | `{row['best_control']}` | `{row['best_classification']}` | "
            f"{row['baseline_service']:.1%} | {row['best_service']:.1%} | "
            f"{row['best_priority']:.1%} | {row['best_substitution']:.1%} | "
            f"{row['best_lost']:.1%} | {row['best_overflow']:.1%} | "
            f"{row['best_gate_starvation']:.1%} | {row['best_gate_capacity_block']:.1%} | "
            f"`{row['best_gate_primary_pressure']}` | {row['best_score']:.4f} |"
        )

    lines.extend([
        "",
        "## SKU Service Rates For Best Controls",
        "",
        "| Scenario | Control | Core Dairy | Frozen Dessert | Priority Replenishment | Promo Yogurt |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for scenario, row in payload["summary"].items():
        rates = row["best_sku_service_rates"]
        lines.append(
            f"| `{scenario}` | `{row['best_control']}` | "
            f"{rates.get('Core Dairy', 0.0):.1%} | {rates.get('Frozen Dessert', 0.0):.1%} | "
            f"{rates.get('Priority Replenishment', 0.0):.1%} | {rates.get('Promo Yogurt', 0.0):.1%} |"
        )

    lines.extend([
        "",
        "## Full Results",
        "",
        "| Scenario | Control | Class | Service | Priority | Substitute | Lost | Overflow | Starvation | Gate Capacity | Score |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["rows"]:
        lines.append(
            f"| `{row['scenario']}` | `{row['control']}` | `{row['classification']}` | "
            f"{row['service_completion_rate']:.1%} | {row['priority_service_rate']:.1%} | "
            f"{row['substitution_rate']:.1%} | {row['lost_demand_rate']:.1%} | "
            f"{row['capacity_overflow_rate']:.1%} | {row['gate_starvation_rate']:.1%} | "
            f"{row['gate_service_capacity_block_rate']:.1%} | {row['resilience_score']:.4f} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "This topology tests whether the first cold-chain findings survive SKU, "
            "regional, carrier-class, and substitution heterogeneity. The decisive "
            "question is no longer only total service; it is which SKU class and account "
            "tier is preserved when promotion calendars and cold-chain contracts collide."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("consumer_goods_multisku_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("CONSUMER_GOODS_MULTISKU_REPORT.md"))
    args = parser.parse_args()
    config = MultiSKUConfig(agents=72, steps=24) if args.quick else MultiSKUConfig()
    payload = run_pilot(config, quick=args.quick)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": len(payload["rows"]),
        "report": str(args.output_md),
        "best_controls": {
            scenario: row["best_control"]
            for scenario, row in payload["summary"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()

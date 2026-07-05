"""
Public semiconductor onshoring phase-transition prototype.

This adapter models the U.S., Taiwan, and China semiconductor sector as a
public, non-sensitive DTE topology. Firms are represented as role-bearing
industrial nodes, not as predictions about private operations.

The central observable is onshore share:

    US-produced finished flow into US demand
    ---------------------------------------
    total finished flow into US demand

The suite asks when tariff/subsidy/capacity/dependency conditions make
U.S.-based production attractive and feasible rather than merely desired.

Usage:
    .venv\\Scripts\\python.exe semiconductor_onshoring.py --quick
    .venv\\Scripts\\python.exe semiconductor_onshoring.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from kernel import DynamicTopologyKernel, topology_from_edges


FEATURE_LABELS = [
    "AI Demand Urgency",
    "Reliability",
    "Speed",
    "Cost Efficiency",
    "Advanced Node Criticality",
    "Capacity",
    "US Policy Compatibility",
    "China Exposure",
    "Strategic Buffer",
]


NODES = {
    "Market Allocation Desk": [0.90, 0.82, 0.62, 0.54, 0.90, 0.60, 0.82, 0.30, 0.45],
    "NVIDIA AI Accelerator Demand": [1.00, 0.84, 0.88, 0.28, 1.00, 0.44, 0.86, 0.38, 0.30],
    "AMD AI Accelerator Demand": [0.94, 0.82, 0.82, 0.34, 0.94, 0.44, 0.84, 0.34, 0.28],
    "Intel IDM Demand": [0.82, 0.86, 0.58, 0.44, 0.88, 0.56, 0.92, 0.22, 0.42],
    "US Hyperscaler Demand": [1.00, 0.84, 0.86, 0.26, 0.96, 0.50, 0.90, 0.32, 0.35],
    "US Defense Demand": [1.00, 0.94, 0.74, 0.18, 0.96, 0.36, 0.96, 0.10, 0.80],
    "China Electronics Demand": [0.82, 0.70, 0.62, 0.74, 0.70, 0.72, 0.36, 0.92, 0.18],
    "US EDA IP": [0.82, 0.94, 0.76, 0.28, 0.98, 0.55, 0.94, 0.20, 0.36],
    "EU Lithography Tools": [0.72, 0.90, 0.30, 0.20, 0.98, 0.30, 0.84, 0.24, 0.30],
    "Japan Chemicals Materials": [0.66, 0.84, 0.42, 0.52, 0.82, 0.60, 0.78, 0.26, 0.24],
    "Korea Packaging Inputs": [0.62, 0.80, 0.54, 0.56, 0.76, 0.58, 0.76, 0.28, 0.24],
    "US Packaging Inputs": [0.78, 0.78, 0.58, 0.38, 0.86, 0.46, 0.94, 0.08, 0.42],
    "US Power Labor": [0.86, 0.72, 0.36, 0.34, 0.80, 0.46, 0.88, 0.10, 0.42],
    "CHIPS Subsidy Credit": [0.78, 0.78, 0.32, 0.18, 0.88, 0.38, 0.98, 0.05, 0.78],
    "Section 232 Tariff Offset": [0.78, 0.74, 0.36, 0.22, 0.88, 0.35, 0.98, 0.08, 0.72],
    "Intel US Fabs": [0.82, 0.82, 0.46, 0.38, 0.88, 0.56, 0.94, 0.08, 0.44],
    "TSMC Arizona Fabs": [0.92, 0.84, 0.44, 0.32, 0.98, 0.46, 0.94, 0.12, 0.44],
    "Samsung Texas Fabs": [0.86, 0.82, 0.46, 0.36, 0.90, 0.48, 0.90, 0.14, 0.38],
    "US Wafer Fabrication": [0.94, 0.82, 0.44, 0.34, 0.98, 0.52, 0.96, 0.08, 0.45],
    "US Advanced Packaging": [0.88, 0.80, 0.54, 0.42, 0.92, 0.44, 0.94, 0.08, 0.42],
    "US Finished Packaged Chips": [0.92, 0.84, 0.66, 0.40, 0.94, 0.50, 0.96, 0.08, 0.46],
    "TSMC Taiwan Fabs": [0.98, 0.90, 0.48, 0.34, 1.00, 0.78, 0.70, 0.32, 0.22],
    "Taiwan OSAT Packaging": [0.88, 0.82, 0.58, 0.46, 0.90, 0.62, 0.68, 0.34, 0.22],
    "Taiwan Export Logistics": [0.78, 0.68, 0.48, 0.70, 0.84, 0.72, 0.58, 0.42, 0.18],
    "China Foundry Cluster": [0.70, 0.66, 0.50, 0.76, 0.58, 0.76, 0.28, 0.98, 0.18],
    "Export Control Review": [0.82, 0.74, 0.34, 0.24, 0.92, 0.36, 0.96, 0.16, 0.62],
    "Pacific Shipping Lane": [0.62, 0.64, 0.32, 0.84, 0.72, 0.86, 0.58, 0.42, 0.16],
    "US West Coast Port": [0.78, 0.76, 0.56, 0.62, 0.82, 0.68, 0.86, 0.20, 0.28],
    "Strategic Chip Reserve": [0.82, 0.94, 0.36, 0.20, 0.92, 0.34, 0.98, 0.05, 1.00],
}


EDGES = [
    ("Market Allocation Desk", "NVIDIA AI Accelerator Demand", 1.8),
    ("Market Allocation Desk", "AMD AI Accelerator Demand", 2.0),
    ("Market Allocation Desk", "Intel IDM Demand", 2.2),
    ("Market Allocation Desk", "US Hyperscaler Demand", 1.9),
    ("Market Allocation Desk", "US Defense Demand", 2.3),
    ("Market Allocation Desk", "China Electronics Demand", 2.7),
    ("Market Allocation Desk", "US EDA IP", 2.4),
    ("Market Allocation Desk", "EU Lithography Tools", 3.1),
    ("Market Allocation Desk", "Japan Chemicals Materials", 2.3),
    ("Market Allocation Desk", "Korea Packaging Inputs", 2.4),
    ("Market Allocation Desk", "US Packaging Inputs", 2.2),
    ("Market Allocation Desk", "US Power Labor", 2.2),
    ("Market Allocation Desk", "CHIPS Subsidy Credit", 2.6),
    ("Market Allocation Desk", "Section 232 Tariff Offset", 2.7),
    ("Market Allocation Desk", "Strategic Chip Reserve", 2.8),
    ("NVIDIA AI Accelerator Demand", "TSMC Taiwan Fabs", 1.7),
    ("NVIDIA AI Accelerator Demand", "TSMC Arizona Fabs", 2.8),
    ("NVIDIA AI Accelerator Demand", "Intel US Fabs", 3.2),
    ("AMD AI Accelerator Demand", "TSMC Taiwan Fabs", 1.9),
    ("AMD AI Accelerator Demand", "TSMC Arizona Fabs", 2.9),
    ("AMD AI Accelerator Demand", "Samsung Texas Fabs", 3.1),
    ("Intel IDM Demand", "Intel US Fabs", 1.8),
    ("US Hyperscaler Demand", "TSMC Taiwan Fabs", 2.0),
    ("US Hyperscaler Demand", "TSMC Arizona Fabs", 2.6),
    ("US Defense Demand", "Intel US Fabs", 2.0),
    ("US Defense Demand", "TSMC Arizona Fabs", 2.4),
    ("China Electronics Demand", "China Foundry Cluster", 1.8),
    ("China Electronics Demand", "TSMC Taiwan Fabs", 2.5),
    ("Intel US Fabs", "US Wafer Fabrication", 1.6),
    ("TSMC Arizona Fabs", "US Wafer Fabrication", 1.7),
    ("Samsung Texas Fabs", "US Wafer Fabrication", 1.9),
    ("US EDA IP", "US Wafer Fabrication", 1.9),
    ("EU Lithography Tools", "US Wafer Fabrication", 3.1),
    ("Japan Chemicals Materials", "US Wafer Fabrication", 2.2),
    ("US Power Labor", "US Wafer Fabrication", 2.0),
    ("US Wafer Fabrication", "US Advanced Packaging", 1.7),
    ("US Packaging Inputs", "US Advanced Packaging", 1.8),
    ("US Advanced Packaging", "US Finished Packaged Chips", 1.5),
    ("US Finished Packaged Chips", "NVIDIA AI Accelerator Demand", 1.8),
    ("US Finished Packaged Chips", "AMD AI Accelerator Demand", 1.9),
    ("US Finished Packaged Chips", "US Hyperscaler Demand", 1.7),
    ("US Finished Packaged Chips", "US Defense Demand", 1.6),
    ("US Finished Packaged Chips", "Strategic Chip Reserve", 2.2),
    ("US EDA IP", "TSMC Taiwan Fabs", 2.2),
    ("EU Lithography Tools", "TSMC Taiwan Fabs", 3.0),
    ("Japan Chemicals Materials", "TSMC Taiwan Fabs", 1.9),
    ("TSMC Taiwan Fabs", "Taiwan OSAT Packaging", 1.5),
    ("Korea Packaging Inputs", "Taiwan OSAT Packaging", 1.9),
    ("Taiwan OSAT Packaging", "Export Control Review", 1.8),
    ("Export Control Review", "Taiwan Export Logistics", 2.2),
    ("Taiwan Export Logistics", "Pacific Shipping Lane", 2.4),
    ("Pacific Shipping Lane", "US West Coast Port", 4.3),
    ("US West Coast Port", "NVIDIA AI Accelerator Demand", 1.9),
    ("US West Coast Port", "AMD AI Accelerator Demand", 2.0),
    ("US West Coast Port", "US Hyperscaler Demand", 1.8),
    ("US West Coast Port", "US Defense Demand", 2.2),
    ("China Foundry Cluster", "China Electronics Demand", 1.6),
    ("China Foundry Cluster", "Export Control Review", 4.0),
    ("CHIPS Subsidy Credit", "Intel US Fabs", 1.8),
    ("CHIPS Subsidy Credit", "TSMC Arizona Fabs", 1.8),
    ("CHIPS Subsidy Credit", "Samsung Texas Fabs", 1.9),
    ("Section 232 Tariff Offset", "TSMC Arizona Fabs", 1.9),
    ("Section 232 Tariff Offset", "US Advanced Packaging", 2.0),
    ("Strategic Chip Reserve", "US Defense Demand", 1.5),
    ("Strategic Chip Reserve", "US Hyperscaler Demand", 2.0),
]


INTENTS = {
    "AI Accelerator": [1.00, 0.84, 0.90, 0.24, 1.00, 0.46, 0.84, 0.32, 0.30],
    "Defense Critical": [1.00, 0.96, 0.70, 0.16, 0.96, 0.36, 0.98, 0.06, 0.88],
    "China Electronics": [0.76, 0.68, 0.56, 0.86, 0.62, 0.72, 0.34, 0.96, 0.18],
    "Industrial Resilience": [0.82, 0.90, 0.42, 0.30, 0.90, 0.50, 0.96, 0.08, 0.94],
}

DEFAULT_POPULATION = {
    "AI Accelerator": 0.48,
    "Defense Critical": 0.18,
    "China Electronics": 0.18,
    "Industrial Resilience": 0.16,
}

US_DEMAND_NODES = (
    "NVIDIA AI Accelerator Demand",
    "AMD AI Accelerator Demand",
    "US Hyperscaler Demand",
    "US Defense Demand",
)
TERMINAL_NODES = US_DEMAND_NODES + ("China Electronics Demand", "Strategic Chip Reserve")


@dataclass(frozen=True)
class OnshoringConfig:
    agents: int = 256
    steps: int = 120
    feedback_rate: float = 0.15
    temperature: float = 0.80
    seed: int = 20260606
    gate_initial_inventory: int = 12
    randomization_key: str | None = None
    additional_edges: tuple[tuple[str, str, float], ...] = ()
    removed_edges: tuple[tuple[str, str], ...] = ()


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
        name="us_fab_gate",
        source="US Wafer Fabrication",
        target="US Advanced Packaging",
        parts=("domestic_wafers", "materials", "eda_ip", "tooling", "power_labor"),
        arrivals={
            ("Intel US Fabs", "US Wafer Fabrication"): "domestic_wafers",
            ("TSMC Arizona Fabs", "US Wafer Fabrication"): "domestic_wafers",
            ("Samsung Texas Fabs", "US Wafer Fabrication"): "domestic_wafers",
            ("Japan Chemicals Materials", "US Wafer Fabrication"): "materials",
            ("US EDA IP", "US Wafer Fabrication"): "eda_ip",
            ("EU Lithography Tools", "US Wafer Fabrication"): "tooling",
            ("US Power Labor", "US Wafer Fabrication"): "power_labor",
        },
        consumption={
            "domestic_wafers": 1,
            "materials": 1,
            "eda_ip": 0,
            "tooling": 0,
            "power_labor": 1,
        },
    ),
    GateSpec(
        name="us_advanced_packaging_gate",
        source="US Advanced Packaging",
        target="US Finished Packaged Chips",
        parts=("packaged_us_wafers", "us_packaging_inputs"),
        arrivals={
            ("US Wafer Fabrication", "US Advanced Packaging"): "packaged_us_wafers",
            ("US Packaging Inputs", "US Advanced Packaging"): "us_packaging_inputs",
        },
        consumption={
            "packaged_us_wafers": 1,
            "us_packaging_inputs": 1,
        },
    ),
    GateSpec(
        name="taiwan_export_gate",
        source="Taiwan OSAT Packaging",
        target="Export Control Review",
        parts=("taiwan_wafers", "taiwan_packaging_inputs"),
        arrivals={
            ("TSMC Taiwan Fabs", "Taiwan OSAT Packaging"): "taiwan_wafers",
            ("Korea Packaging Inputs", "Taiwan OSAT Packaging"): "taiwan_packaging_inputs",
        },
        consumption={
            "taiwan_wafers": 1,
            "taiwan_packaging_inputs": 1,
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


def _seed_for(config: OnshoringConfig, salt: int = 0) -> int:
    raw = (
        config.seed
        + 101 * config.agents
        + 1009 * config.steps
        + 9173 * int(round(config.feedback_rate * 1000))
        + 6113 * int(round(config.temperature * 1000))
        + salt
    )
    return int(raw % (2**32 - 1))


def build_kernel(config: OnshoringConfig) -> DynamicTopologyKernel:
    removed = set(config.removed_edges)
    edges = [edge for edge in EDGES if (edge[0], edge[1]) not in removed]
    edges.extend(config.additional_edges)
    topology = topology_from_edges(
        nodes={label: np.array(features, dtype=np.float64) for label, features in NODES.items()},
        edges=edges,
        undirected=False,
    )
    n = topology.N
    node_bias = np.zeros(n, dtype=np.float64)
    for label, bias in {
        "Market Allocation Desk": 0.20,
        "TSMC Taiwan Fabs": 0.14,
        "US Wafer Fabrication": -0.04,
        "US Advanced Packaging": -0.04,
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


def _initial_telemetries(config: OnshoringConfig) -> np.ndarray:
    telemetries = np.zeros((config.agents, len(FEATURE_LABELS)), dtype=np.float64)
    start = 0
    items = list(DEFAULT_POPULATION.items())
    for idx, (intent, share) in enumerate(items):
        count = int(round(config.agents * share)) if idx < len(items) - 1 else config.agents - start
        end = min(config.agents, start + count)
        telemetries[start:end] = _normalized(INTENTS[intent])
        start = end
    return telemetries


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


def _combined_gate_replenishment(*items: Scenario) -> dict[tuple[str, str], int]:
    replenishment: dict[tuple[str, str], int] = {}
    valid = {(gate.name, part) for gate in GATES for part in gate.parts}
    for item in items:
        for key, amount in item.gate_replenishment.items():
            if key not in valid:
                continue
            replenishment[key] = replenishment.get(key, 0) + int(amount)
    return {key: amount for key, amount in replenishment.items() if amount > 0}


def _combined_gate_capacity_caps(*items: Scenario) -> dict[str, int]:
    valid = {gate.name for gate in GATES}
    caps: dict[str, int] = {}
    for item in items:
        for gate_name, capacity in item.gate_capacity_caps.items():
            if gate_name not in valid or capacity <= 0:
                continue
            caps[gate_name] = min(caps.get(gate_name, int(capacity)), int(capacity))
    return caps


def scenarios() -> list[Scenario]:
    offshore_import_edges = (
        ("Taiwan OSAT Packaging", "Export Control Review"),
        ("Export Control Review", "Taiwan Export Logistics"),
        ("Taiwan Export Logistics", "Pacific Shipping Lane"),
        ("Pacific Shipping Lane", "US West Coast Port"),
    )
    china_edges = (
        ("China Foundry Cluster", "Export Control Review"),
        ("China Electronics Demand", "TSMC Taiwan Fabs"),
    )
    domestic_capacity_caps = {
        "Intel US Fabs": 12,
        "TSMC Arizona Fabs": 10,
        "Samsung Texas Fabs": 8,
        "US Advanced Packaging": 16,
    }
    return [
        Scenario("nominal", "nominal"),
        Scenario(
            "section_232_tariff_25",
            "tariff",
            cost=_edge_cost(offshore_import_edges, 1.0),
            friction_edges=offshore_import_edges,
            friction_delta=-1.0,
        ),
        Scenario(
            "broad_semiconductor_tariff_50",
            "tariff",
            cost=_edge_cost(offshore_import_edges + china_edges, 2.0),
            friction_edges=offshore_import_edges + china_edges,
            friction_delta=-2.0,
        ),
        Scenario(
            "tariff_with_domestic_capacity_constraint",
            "capacity",
            cost=_edge_cost(offshore_import_edges, 1.0),
            friction_edges=offshore_import_edges,
            friction_delta=-1.0,
            node_capacity_caps=domestic_capacity_caps,
        ),
        Scenario(
            "ai_demand_surge_capacity",
            "capacity",
            cost=2.0,
            node_capacity_caps={
                "TSMC Arizona Fabs": 12,
                "US Advanced Packaging": 14,
                "TSMC Taiwan Fabs": 24,
            },
        ),
        Scenario(
            "export_control_tightening_china",
            "export_control",
            cost=_edge_cost(china_edges, 2.2),
            friction_edges=china_edges,
            friction_delta=-2.2,
        ),
    ]


def controls() -> list[Scenario]:
    subsidy_edges = (
        ("CHIPS Subsidy Credit", "Intel US Fabs"),
        ("CHIPS Subsidy Credit", "TSMC Arizona Fabs"),
        ("CHIPS Subsidy Credit", "Samsung Texas Fabs"),
    )
    offset_edges = (
        ("Section 232 Tariff Offset", "TSMC Arizona Fabs"),
        ("Section 232 Tariff Offset", "US Advanced Packaging"),
    )
    tsmc_ramp = (
        ("TSMC Arizona Fabs", "US Wafer Fabrication"),
        ("US Wafer Fabrication", "US Advanced Packaging"),
    )
    intel_ramp = (
        ("Intel US Fabs", "US Wafer Fabrication"),
        ("US Wafer Fabrication", "US Advanced Packaging"),
    )
    packaging_ramp = (
        ("Market Allocation Desk", "US Packaging Inputs"),
        ("US Packaging Inputs", "US Advanced Packaging"),
        ("US Advanced Packaging", "US Finished Packaged Chips"),
        ("US Finished Packaged Chips", "NVIDIA AI Accelerator Demand"),
        ("US Finished Packaged Chips", "US Hyperscaler Demand"),
    )
    materials = (
        ("Market Allocation Desk", "Japan Chemicals Materials"),
        ("Market Allocation Desk", "EU Lithography Tools"),
        ("Market Allocation Desk", "US EDA IP"),
        ("Market Allocation Desk", "US Power Labor"),
        ("Market Allocation Desk", "Korea Packaging Inputs"),
        ("Market Allocation Desk", "US Packaging Inputs"),
        ("Japan Chemicals Materials", "US Wafer Fabrication"),
        ("EU Lithography Tools", "US Wafer Fabrication"),
        ("US EDA IP", "US Wafer Fabrication"),
        ("US Power Labor", "US Wafer Fabrication"),
    )
    reserve = (
        ("Strategic Chip Reserve", "US Defense Demand"),
        ("Strategic Chip Reserve", "US Hyperscaler Demand"),
    )
    return [
        Scenario("no_control", "control"),
        Scenario(
            "chips_subsidy_boost",
            "control",
            cost=_edge_cost(subsidy_edges, 1.0),
            beta_edges=subsidy_edges,
            beta_boost=1.0,
        ),
        Scenario(
            "tariff_offset_buildout",
            "control",
            cost=_edge_cost(offset_edges, 1.2),
            friction_edges=offset_edges,
            friction_delta=1.2,
        ),
        Scenario(
            "tsmc_arizona_ramp",
            "control",
            cost=_edge_cost(tsmc_ramp, 1.1),
            beta_edges=tsmc_ramp,
            beta_boost=1.1,
        ),
        Scenario(
            "intel_foundry_ramp",
            "control",
            cost=_edge_cost(intel_ramp, 1.1),
            beta_edges=intel_ramp,
            beta_boost=1.1,
        ),
        Scenario(
            "advanced_packaging_ramp",
            "control",
            cost=_edge_cost(packaging_ramp, 1.0),
            beta_edges=packaging_ramp,
            beta_boost=1.0,
        ),
        Scenario(
            "materials_and_tools_continuity",
            "control",
            cost=_edge_cost(materials, 0.9),
            friction_edges=materials,
            friction_delta=0.9,
        ),
        Scenario(
            "strategic_reserve_release",
            "control",
            cost=_edge_cost(reserve, 1.3),
            friction_edges=reserve,
            friction_delta=1.3,
        ),
    ]


def _renormalize_rows(rows: np.ndarray) -> np.ndarray:
    row_sums = rows.sum(axis=1, keepdims=True)
    return np.divide(rows, row_sums, out=np.zeros_like(rows), where=row_sums > 1e-12)


def _init_gate_state(config: OnshoringConfig) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    inventory = {
        gate.name: {part: config.gate_initial_inventory for part in gate.parts}
        for gate in GATES
    }
    arrivals = {
        gate.name: {part: 0 for part in gate.parts}
        for gate in GATES
    }
    return inventory, arrivals


def _limiting_gate_part(inventory: dict[str, dict[str, int]], arrivals: dict[str, dict[str, int]]) -> dict:
    candidates = []
    for gate_name, parts in inventory.items():
        for part, count in parts.items():
            candidates.append((count, arrivals[gate_name].get(part, 0), gate_name, part))
    count, _, gate_name, part = min(candidates)
    return {"gate": gate_name, "part": part, "inventory": int(count)}


def simulate(
    config: OnshoringConfig,
    scenario: Scenario | None = None,
    control: Scenario | None = None,
    enforce_gates: bool = True,
) -> dict:
    scenario = scenario or scenarios()[0]
    control = control or controls()[0]
    kernel = build_kernel(config)
    labels = kernel.topo.labels
    _apply_scenario(kernel, labels, scenario)
    _apply_scenario(kernel, labels, control)
    gate_replenishment = _combined_gate_replenishment(scenario, control)
    gate_capacity_caps = _combined_gate_capacity_caps(scenario, control)

    randomization_parts = (
        (config.randomization_key, str(enforce_gates))
        if config.randomization_key is not None
        else (scenario.name, control.name, str(enforce_gates))
    )
    rng = np.random.default_rng(_seed_for(config, _stable_salt(*randomization_parts)))
    n = kernel.topo.N
    positions = np.full(config.agents, labels.index("Market Allocation Desk"), dtype=int)
    telemetries = _initial_telemetries(config)

    terminal_idx = np.array([labels.index(name) for name in TERMINAL_NODES], dtype=int)
    us_demand_idx = np.array([labels.index(name) for name in US_DEMAND_NODES], dtype=int)
    us_finished_idx = labels.index("US Finished Packaged Chips")
    us_port_idx = labels.index("US West Coast Port")
    reserve_idx = labels.index("Strategic Chip Reserve")
    edge_capacity_idx = {
        (labels.index(source), labels.index(target)): int(capacity)
        for (source, target), capacity in scenario.edge_capacity_caps.items()
    }
    node_capacity_idx = {
        labels.index(node): int(capacity)
        for node, capacity in scenario.node_capacity_caps.items()
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
    gate_capacity_blocked = {gate.name: 0 for gate in GATES}
    gate_completions = {gate.name: 0 for gate in GATES}
    gate_backlog_series = {gate.name: [] for gate in GATES}
    gate_replenished = {gate.name: {part: 0 for part in gate.parts} for gate in GATES}
    completed_lot_source = np.zeros(config.agents, dtype=np.int8)

    edge_counts = np.zeros((n, n), dtype=np.float64)
    terminal_hits = []
    us_demand_hits = []
    edge_overflow_events = 0
    edge_capacity_attempts = 0
    node_overflow_events = 0
    node_capacity_attempts = 0

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
                feasible = int(len(attempted))
                feasible_limits = []
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
                inventory_feasible = min(int(len(attempted)), int(feasible))
                service_capacity = gate_capacity_caps.get(gate.name, inventory_feasible)
                allowed_count = min(inventory_feasible, service_capacity)
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

        demand_arrival = np.isin(next_positions, us_demand_idx)
        uncompleted = completed_lot_source == 0
        completed_lot_source[
            uncompleted & (positions == us_finished_idx) & demand_arrival
        ] = 1
        completed_lot_source[
            uncompleted & (positions == us_port_idx) & demand_arrival
        ] = 2
        completed_lot_source[
            uncompleted & (positions == reserve_idx) & demand_arrival
        ] = 3

        np.add.at(edge_counts, (positions, next_positions), 1.0)
        visited = kernel.topo.node_features[next_positions]
        lam = config.feedback_rate
        telemetries = (1.0 - lam) * telemetries + lam * visited
        norms = np.linalg.norm(telemetries, axis=1, keepdims=True)
        telemetries = np.where(norms > 0, telemetries / norms, telemetries)
        positions = next_positions

        terminal_hits.append(float(np.mean(np.isin(positions, terminal_idx))))
        us_demand_hits.append(float(np.mean(np.isin(positions, us_demand_idx))))

    window = max(8, config.steps // 4)
    total_overflow = edge_overflow_events + node_overflow_events
    total_capacity_attempts = edge_capacity_attempts + node_capacity_attempts
    gate_total_attempts = sum(gate_attempts.values())
    gate_total_blocked = sum(gate_blocked.values())
    gate_total_completions = sum(gate_completions.values())
    limiting = _limiting_gate_part(gate_inventory, gate_arrivals)
    gate_block_rate = float(gate_total_blocked / max(gate_total_attempts, 1))
    gate_completion_ratio = float(gate_total_completions / max(gate_total_attempts, 1))
    gate_pressure_rate = float(gate_total_blocked / max(gate_total_blocked + gate_total_completions, 1))
    gate_backlog_pressure = float(max(
        (np.mean(series[-window:]) if series else 0.0)
        for series in gate_backlog_series.values()
    ))
    starvation_values = []
    for gate in GATES:
        arrivals = gate_arrivals[gate.name]
        backlog_agents = (
            gate_backlog_series[gate.name][-1] * config.agents
            if gate_backlog_series[gate.name]
            else 0.0
        )
        for part in gate.parts:
            consumption = gate.consumption.get(part, 1)
            if consumption <= 0:
                continue
            observed_need = (gate_completions[gate.name] + backlog_agents) * consumption
            available_supply = config.gate_initial_inventory + arrivals.get(part, 0)
            if observed_need > 0:
                starvation_values.append(max(0.0, 1.0 - available_supply / observed_need))
    gate_starvation_index = float(max(starvation_values, default=0.0))

    us_demand_indices = [labels.index(name) for name in US_DEMAND_NODES]
    onshore_finished = float(edge_counts[us_finished_idx, us_demand_indices].sum())
    imported_finished = float(edge_counts[us_port_idx, us_demand_indices].sum())
    reserve_finished = float(edge_counts[reserve_idx, us_demand_indices].sum())
    total_us_finished = onshore_finished + imported_finished + reserve_finished
    flow_onshore_share = onshore_finished / max(total_us_finished, 1.0)
    lot_onshore_finished = int(np.sum(completed_lot_source == 1))
    lot_imported_finished = int(np.sum(completed_lot_source == 2))
    lot_reserve_finished = int(np.sum(completed_lot_source == 3))
    lot_total_us_finished = lot_onshore_finished + lot_imported_finished + lot_reserve_finished
    lot_onshore_share = lot_onshore_finished / max(lot_total_us_finished, 1)

    edge_flow = edge_counts / max(float(edge_counts.sum()), 1.0)
    edge_current = edge_flow - edge_flow.T
    return {
        "scenario": scenario.name,
        "family": scenario.family,
        "control": control.name,
        "control_cost": control.cost,
        "agents": config.agents,
        "steps": config.steps,
        "terminal_share": float(np.mean(terminal_hits[-window:])),
        "us_demand_share": float(np.mean(us_demand_hits[-window:])),
        "onshore_finished_flow": onshore_finished,
        "imported_finished_flow": imported_finished,
        "reserve_finished_flow": reserve_finished,
        "total_us_finished_flow": total_us_finished,
        "flow_onshore_share": float(flow_onshore_share),
        "lot_onshore_finished": lot_onshore_finished,
        "lot_imported_finished": lot_imported_finished,
        "lot_reserve_finished": lot_reserve_finished,
        "lot_total_us_finished": lot_total_us_finished,
        "lot_completion_rate": float(lot_total_us_finished / max(config.agents, 1)),
        "onshore_share": float(lot_onshore_share),
        "capacity_overflow_rate": float(total_overflow / max(total_capacity_attempts, 1)),
        "gate_block_rate": gate_block_rate,
        "gate_completion_ratio": gate_completion_ratio,
        "gate_pressure_rate": gate_pressure_rate,
        "gate_backlog_pressure": gate_backlog_pressure,
        "gate_starvation_index": gate_starvation_index,
        "gate_completion_per_agent": float(gate_total_completions / config.agents),
        "gate_attempts": gate_attempts,
        "gate_blocked": gate_blocked,
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
        "gates_enforced": bool(enforce_gates),
    }


def classify(row: dict) -> str:
    dependency_pressure = max(row["gate_backlog_pressure"], row["gate_starvation_index"])
    if row["capacity_overflow_rate"] >= 0.10:
        return "capacity_blocked"
    if row["onshore_share"] >= 0.50 and row["capacity_overflow_rate"] < 0.10 and dependency_pressure < 0.65:
        return "onshoring_transition"
    if row["onshore_share"] >= 0.35:
        return "partial_onshoring"
    if dependency_pressure >= 0.65:
        return "dependency_blocked"
    return "import_dominant"


def _with_baseline_deltas(rows: list[dict]) -> list[dict]:
    baselines = {
        row["scenario"]: row
        for row in rows
        if row["control"] == "no_control"
    }
    enriched = []
    for row in rows:
        baseline = baselines[row["scenario"]]
        item = dict(row)
        item["onshore_delta_vs_baseline"] = row["onshore_share"] - baseline["onshore_share"]
        item["us_finished_delta_vs_baseline"] = row["lot_total_us_finished"] - baseline["lot_total_us_finished"]
        item["overflow_delta_vs_baseline"] = row["capacity_overflow_rate"] - baseline["capacity_overflow_rate"]
        item["institutional_score"] = (
            row["onshore_share"]
            + 0.15 * row["us_demand_share"]
            + 0.10 * row["gate_completion_per_agent"]
            - 0.20 * row["capacity_overflow_rate"]
            - 0.05 * max(row["gate_backlog_pressure"], row["gate_starvation_index"])
        )
        baseline_score = (
            baseline["onshore_share"]
            + 0.15 * baseline["us_demand_share"]
            + 0.10 * baseline["gate_completion_per_agent"]
            - 0.20 * baseline["capacity_overflow_rate"]
            - 0.05 * max(baseline["gate_backlog_pressure"], baseline["gate_starvation_index"])
        )
        item["score_delta_vs_baseline"] = item["institutional_score"] - baseline_score
        item["classification"] = classify(item)
        enriched.append(item)
    return enriched


def run_suite(config: OnshoringConfig | None = None, quick: bool = False) -> dict:
    config = config or OnshoringConfig()
    scenario_list = scenarios()[:3] if quick else scenarios()
    rows = [
        simulate(config, scenario, control, enforce_gates=True)
        for scenario in scenario_list
        for control in controls()
    ]
    rows = _with_baseline_deltas(rows)
    counts = {
        label: sum(1 for row in rows if row["classification"] == label)
        for label in sorted({row["classification"] for row in rows})
    }
    best_by_scenario = {}
    for scenario_name in sorted({row["scenario"] for row in rows}):
        candidates = [row for row in rows if row["scenario"] == scenario_name and row["control"] != "no_control"]
        best = max(candidates, key=lambda row: (row["score_delta_vs_baseline"], row["onshore_delta_vs_baseline"]))
        best_by_scenario[scenario_name] = {
            "control": best["control"],
            "classification": best["classification"],
            "onshore_share": best["onshore_share"],
            "onshore_delta": best["onshore_delta_vs_baseline"],
            "us_finished_delta": best["us_finished_delta_vs_baseline"],
            "score_delta": best["score_delta_vs_baseline"],
            "limiting_gate": best["limiting_gate"],
            "limiting_part": best["limiting_part"],
        }
    return {
        "config": asdict(config) | {"quick": quick},
        "classification_counts": counts,
        "best_by_scenario": best_by_scenario,
        "rows": rows,
    }


def render_report(payload: dict) -> str:
    rows = payload["rows"]
    best = payload["best_by_scenario"]
    frontier = sorted(
        [row for row in rows if row["control"] != "no_control"],
        key=lambda row: (row["onshore_share"], row["score_delta_vs_baseline"]),
        reverse=True,
    )[:10]
    limiting_counts = {
        f"{row['limiting_gate']}::{row['limiting_part']}": sum(
            1 for item in rows
            if item["limiting_gate"] == row["limiting_gate"] and item["limiting_part"] == row["limiting_part"]
        )
        for row in rows
    }
    lines = [
        "# Semiconductor Onshoring Report",
        "",
        "## Scope",
        "",
        (
            "Public, non-sensitive sectoral DTE model of U.S., Taiwan, and China semiconductor "
            "industrial reallocation under tariff, subsidy, capacity, and dependency constraints."
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
        "## Onshoring Frontier",
        "",
        "| Scenario | Control | Onshore Share | Onshore Delta | US Finished Delta | Overflow | Classification | Limiting Part |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ])
    for row in frontier:
        lines.append(
            f"| `{row['scenario']}` | `{row['control']}` | {row['onshore_share']:.3f} | "
            f"{row['onshore_delta_vs_baseline']:+.3f} | {row['us_finished_delta_vs_baseline']:+.1f} | "
            f"{row['capacity_overflow_rate']:.1%} | `{row['classification']}` | "
            f"`{row['limiting_gate']}::{row['limiting_part']}` |"
        )

    lines.extend([
        "",
        "## Best Control By Scenario",
        "",
        "| Scenario | Best Control | Classification | Onshore Share | Onshore Delta | US Finished Delta | Score Delta | Limiting Part |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ])
    for scenario_name, row in best.items():
        lines.append(
            f"| `{scenario_name}` | `{row['control']}` | `{row['classification']}` | "
            f"{row['onshore_share']:.3f} | {row['onshore_delta']:+.3f} | "
            f"{row['us_finished_delta']:+.1f} | {row['score_delta']:+.4f} | "
            f"`{row['limiting_gate']}::{row['limiting_part']}` |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "Onshoring is treated as a phase transition, not a slogan. A tariff can tilt routing "
            "away from imports, but onshoring only becomes real when domestic wafer fabrication, "
            "advanced packaging, tooling, materials, power/labor, and demand absorption clear their "
            "gates without saturating. This report should be read as a model scaffold; public "
            "calibration and seed-robust sweeps are required before policy interpretation."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("semiconductor_onshoring_output.json"),
    output_md: Path = Path("SEMICONDUCTOR_ONSHORING_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semiconductor onshoring phase-transition suite.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--agents", type=int, default=OnshoringConfig.agents)
    parser.add_argument("--steps", type=int, default=OnshoringConfig.steps)
    parser.add_argument("--seed", type=int, default=OnshoringConfig.seed)
    parser.add_argument("--gate-initial-inventory", type=int, default=OnshoringConfig.gate_initial_inventory)
    parser.add_argument("--output-json", type=Path, default=Path("semiconductor_onshoring_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SEMICONDUCTOR_ONSHORING_REPORT.md"))
    args = parser.parse_args()

    config = OnshoringConfig(
        agents=80 if args.quick and args.agents == OnshoringConfig.agents else args.agents,
        steps=40 if args.quick and args.steps == OnshoringConfig.steps else args.steps,
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

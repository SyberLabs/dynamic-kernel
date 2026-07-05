"""
Supply-chain resilience prototype for the Dynamic Topology Engine.

This harness treats DTE as a cybernetic circulation model for material,
contractual, and information flow. Shipment-intent agents route through a
directed procurement/logistics graph; shocks increase route friction, while
controls express supplier substitution, expedited logistics, and buffer release.

Usage:
    .venv\\Scripts\\python.exe supply_chain_resilience.py
    .venv\\Scripts\\python.exe supply_chain_resilience.py --quick
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from kernel import DynamicTopologyKernel, topology_from_edges


FEATURE_LABELS = [
    "Demand Urgency",
    "Reliability",
    "Speed",
    "Cost Efficiency",
    "Perishability",
    "Capacity",
    "Strategic Buffer",
]


NODES = {
    "Planning Desk": [0.90, 0.70, 0.60, 0.70, 0.30, 0.55, 0.70],
    "Lithium Supplier": [0.30, 0.70, 0.25, 0.65, 0.10, 0.75, 0.20],
    "Copper Supplier": [0.25, 0.75, 0.30, 0.70, 0.10, 0.80, 0.20],
    "Chip Fab A": [0.70, 0.80, 0.35, 0.35, 0.05, 0.55, 0.10],
    "Chip Fab B": [0.65, 0.72, 0.45, 0.42, 0.05, 0.45, 0.25],
    "Packaging Supplier": [0.40, 0.82, 0.55, 0.75, 0.15, 0.70, 0.15],
    "Cold Chain Supplier": [0.60, 0.76, 0.72, 0.35, 0.95, 0.50, 0.10],
    "Battery Plant": [0.45, 0.74, 0.40, 0.55, 0.20, 0.75, 0.20],
    "Electronics Assembly": [0.72, 0.78, 0.52, 0.48, 0.10, 0.62, 0.20],
    "Chassis Plant": [0.38, 0.80, 0.45, 0.68, 0.10, 0.78, 0.18],
    "Final Assembly": [0.82, 0.78, 0.58, 0.52, 0.20, 0.70, 0.25],
    "QA Buffer": [0.55, 0.88, 0.42, 0.46, 0.18, 0.45, 0.82],
    "Asia Port": [0.62, 0.58, 0.38, 0.78, 0.25, 0.88, 0.12],
    "Air Freight": [0.88, 0.66, 0.96, 0.20, 0.70, 0.35, 0.08],
    "Ocean Carrier": [0.40, 0.62, 0.30, 0.90, 0.20, 0.92, 0.10],
    "West Coast Port": [0.72, 0.64, 0.48, 0.72, 0.25, 0.78, 0.16],
    "Rail Hub": [0.55, 0.72, 0.55, 0.75, 0.22, 0.82, 0.15],
    "Inland DC": [0.68, 0.82, 0.62, 0.62, 0.35, 0.72, 0.42],
    "Regional DC": [0.76, 0.84, 0.68, 0.56, 0.48, 0.66, 0.50],
    "Retail Demand": [0.88, 0.70, 0.72, 0.52, 0.30, 0.45, 0.25],
    "Critical Care Demand": [1.00, 0.86, 0.95, 0.20, 0.98, 0.35, 0.20],
    "Government Reserve": [0.50, 0.90, 0.35, 0.30, 0.40, 0.30, 1.00],
}


EDGES = [
    ("Planning Desk", "Lithium Supplier", 2.6),
    ("Planning Desk", "Copper Supplier", 2.4),
    ("Planning Desk", "Chip Fab A", 2.2),
    ("Planning Desk", "Chip Fab B", 3.0),
    ("Planning Desk", "Packaging Supplier", 2.0),
    ("Planning Desk", "Cold Chain Supplier", 2.8),
    ("Lithium Supplier", "Battery Plant", 3.2),
    ("Copper Supplier", "Battery Plant", 2.8),
    ("Chip Fab A", "Electronics Assembly", 3.4),
    ("Chip Fab B", "Electronics Assembly", 4.0),
    ("Packaging Supplier", "Final Assembly", 2.5),
    ("Cold Chain Supplier", "Regional DC", 4.2),
    ("Battery Plant", "Final Assembly", 3.0),
    ("Electronics Assembly", "Final Assembly", 2.6),
    ("Chassis Plant", "Final Assembly", 2.7),
    ("Planning Desk", "Chassis Plant", 2.3),
    ("Final Assembly", "QA Buffer", 1.6),
    ("QA Buffer", "Asia Port", 2.4),
    ("QA Buffer", "Air Freight", 4.2),
    ("Asia Port", "Ocean Carrier", 2.2),
    ("Asia Port", "Air Freight", 3.8),
    ("Ocean Carrier", "West Coast Port", 5.2),
    ("Air Freight", "West Coast Port", 2.1),
    ("West Coast Port", "Rail Hub", 2.4),
    ("Rail Hub", "Inland DC", 2.8),
    ("Inland DC", "Regional DC", 2.0),
    ("Regional DC", "Retail Demand", 1.8),
    ("Regional DC", "Critical Care Demand", 2.0),
    ("Government Reserve", "Regional DC", 1.7),
    ("Government Reserve", "Critical Care Demand", 1.5),
    ("Retail Demand", "Planning Desk", 6.0),
    ("Critical Care Demand", "Planning Desk", 5.5),
    ("Regional DC", "Government Reserve", 3.5),
]


INTENTS = {
    "Standard Retail": [0.80, 0.70, 0.55, 0.75, 0.20, 0.65, 0.25],
    "Critical Medical": [1.00, 0.86, 0.95, 0.25, 0.95, 0.45, 0.30],
    "Cost Sensitive": [0.55, 0.70, 0.35, 0.98, 0.10, 0.70, 0.20],
    "Resilience Stock": [0.50, 0.92, 0.35, 0.45, 0.30, 0.55, 1.00],
}

DEFAULT_POPULATION = {
    "Standard Retail": 0.45,
    "Critical Medical": 0.25,
    "Cost Sensitive": 0.20,
    "Resilience Stock": 0.10,
}

TERMINAL_NODES = ("Retail Demand", "Critical Care Demand")
CRITICAL_NODE = "Critical Care Demand"
SUPPLIER_NODES = (
    "Lithium Supplier",
    "Copper Supplier",
    "Chip Fab A",
    "Chip Fab B",
    "Packaging Supplier",
    "Cold Chain Supplier",
)
BUFFER_NODES = ("QA Buffer", "Government Reserve")


@dataclass(frozen=True)
class SimulationConfig:
    agents: int = 256
    steps: int = 96
    feedback_rate: float = 0.18
    temperature: float = 0.75
    seed: int = 20260604


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    cost: float
    apply: Callable


def _idx(labels: list[str], names: tuple[str, ...] | list[str]) -> list[int]:
    return [labels.index(name) for name in names]


def _normalized(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


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


def _stable_salt(*parts: str) -> int:
    text = "|".join(parts)
    total = 0
    for idx, char in enumerate(text):
        total += (idx + 1) * ord(char)
    return total % 65536


def build_kernel(config: SimulationConfig):
    topology = topology_from_edges(
        nodes={label: np.array(features, dtype=np.float64) for label, features in NODES.items()},
        edges=EDGES,
        undirected=False,
    )
    n = topology.N
    node_bias = np.zeros(n, dtype=np.float64)
    for label, bias in {
        "Planning Desk": 0.25,
        "Regional DC": 0.15,
        "Government Reserve": 0.10,
    }.items():
        node_bias[topology.labels.index(label)] = bias
    return DynamicTopologyKernel(
        topology=topology,
        beta=np.full((n, n), 1.35, dtype=np.float64),
        feedback_rate=config.feedback_rate,
        feedback_noise=0.0,
        temperature=config.temperature,
        node_bias=node_bias,
        sponsor_decay=0.0,
    )


def _edge_cost(edge_names: list[tuple[str, str]], magnitude: float) -> float:
    return float(len(edge_names) * abs(magnitude))


def _apply_friction(labels: list[str], edge_names: list[tuple[str, str]], reduction: float):
    def apply(kernel):
        for source, target in edge_names:
            i, j = labels.index(source), labels.index(target)
            if kernel.topo.adjacency_mask[i, j]:
                kernel.sponsor_edge_friction(i, j, reduction)
    return apply


def _apply_beta(labels: list[str], edge_names: list[tuple[str, str]], boost: float):
    def apply(kernel):
        for source, target in edge_names:
            i, j = labels.index(source), labels.index(target)
            if kernel.topo.adjacency_mask[i, j]:
                kernel.sponsor_edge(i, j, boost)
    return apply


def shocks(labels: list[str]) -> list[Scenario]:
    port_edges = [
        ("Asia Port", "Ocean Carrier"),
        ("Ocean Carrier", "West Coast Port"),
        ("West Coast Port", "Rail Hub"),
    ]
    chip_edges = [
        ("Planning Desk", "Chip Fab A"),
        ("Chip Fab A", "Electronics Assembly"),
    ]
    cold_edges = [
        ("Planning Desk", "Cold Chain Supplier"),
        ("Cold Chain Supplier", "Regional DC"),
        ("Regional DC", "Critical Care Demand"),
    ]
    return [
        Scenario("none", "shock", 0.0, lambda kernel: None),
        Scenario("port_congestion", "shock", _edge_cost(port_edges, 2.0), _apply_friction(labels, port_edges, -2.0)),
        Scenario("chip_fab_outage", "shock", _edge_cost(chip_edges, 2.5), _apply_friction(labels, chip_edges, -2.5)),
        Scenario("cold_chain_disruption", "shock", _edge_cost(cold_edges, 1.8), _apply_friction(labels, cold_edges, -1.8)),
    ]


def controls(labels: list[str]) -> list[Scenario]:
    chip_substitution = [
        ("Planning Desk", "Chip Fab B"),
        ("Chip Fab B", "Electronics Assembly"),
    ]
    expedited = [
        ("QA Buffer", "Air Freight"),
        ("Asia Port", "Air Freight"),
        ("Air Freight", "West Coast Port"),
    ]
    buffer_release = [
        ("Government Reserve", "Regional DC"),
        ("Government Reserve", "Critical Care Demand"),
    ]
    port_reroute = [
        ("Asia Port", "Air Freight"),
        ("QA Buffer", "Air Freight"),
        ("Air Freight", "West Coast Port"),
    ]
    generic = chip_substitution + expedited + buffer_release
    return [
        Scenario("no_control", "control", 0.0, lambda kernel: None),
        Scenario(
            "dual_source_chips",
            "control",
            _edge_cost(chip_substitution, 1.1),
            _apply_beta(labels, chip_substitution, 1.1),
        ),
        Scenario(
            "expedite_air",
            "control",
            _edge_cost(expedited, 1.0),
            _apply_friction(labels, expedited, 1.0),
        ),
        Scenario(
            "buffer_release",
            "control",
            _edge_cost(buffer_release, 1.2),
            _apply_friction(labels, buffer_release, 1.2),
        ),
        Scenario(
            "port_reroute",
            "control",
            _edge_cost(port_reroute, 0.8),
            _apply_beta(labels, port_reroute, 0.8),
        ),
        Scenario(
            "generic_resilience",
            "control",
            _edge_cost(generic, 0.35),
            _apply_beta(labels, generic, 0.35),
        ),
    ]


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


def simulate(
    config: SimulationConfig,
    shock: Scenario | None = None,
    control: Scenario | None = None,
) -> dict:
    kernel = build_kernel(config)
    labels = kernel.topo.labels
    if shock is not None:
        shock.apply(kernel)
    if control is not None:
        control.apply(kernel)

    rng = np.random.default_rng(_seed_for(
        config,
        salt=_stable_salt(shock.name if shock else "", control.name if control else ""),
    ))
    n = kernel.topo.N
    start = labels.index("Planning Desk")
    positions = np.full(config.agents, start, dtype=int)
    telemetries = _initial_telemetries(config)

    terminal_idx = np.array(_idx(labels, TERMINAL_NODES), dtype=int)
    critical_idx = labels.index(CRITICAL_NODE)
    supplier_idx = np.array(_idx(labels, SUPPLIER_NODES), dtype=int)
    buffer_idx = np.array(_idx(labels, BUFFER_NODES), dtype=int)
    edge_counts = np.zeros((n, n), dtype=np.float64)

    terminal_hits = []
    critical_hits = []
    buffer_hits = []
    supplier_hits = []
    edge_entropy_series = []

    for step in range(config.steps):
        P_all = kernel.transition_matrix_batch(telemetries, step=step)
        rows = P_all[np.arange(config.agents), positions, :]
        cdf = np.cumsum(rows, axis=1)
        draws = rng.random((config.agents, 1))
        next_positions = np.argmax(cdf >= draws, axis=1)
        row_sums = rows.sum(axis=1)
        next_positions[row_sums < 1e-12] = positions[row_sums < 1e-12]

        np.add.at(edge_counts, (positions, next_positions), 1.0)
        visited = kernel.topo.node_features[next_positions]
        lam = config.feedback_rate
        telemetries = (1.0 - lam) * telemetries + lam * visited
        norms = np.linalg.norm(telemetries, axis=1, keepdims=True)
        telemetries = np.where(norms > 0, telemetries / norms, telemetries)
        positions = next_positions

        terminal_hits.append(float(np.mean(np.isin(positions, terminal_idx))))
        critical_hits.append(float(np.mean(positions == critical_idx)))
        buffer_hits.append(float(np.mean(np.isin(positions, buffer_idx))))
        supplier_hits.append(float(np.mean(np.isin(positions, supplier_idx))))

        mean_rows = rows.mean(axis=0)
        positive = mean_rows > 0
        edge_entropy_series.append(float(-np.sum(mean_rows[positive] * np.log(mean_rows[positive]))))

    edge_flow = edge_counts / max(float(edge_counts.sum()), 1.0)
    edge_current = edge_flow - edge_flow.T
    reverse = edge_flow.T
    eps = 1e-12
    bidirectional = (edge_flow > eps) & (reverse > eps)
    entropy_production = 0.0
    if np.any(bidirectional):
        entropy_production = 0.5 * float(np.sum(edge_flow[bidirectional] * np.log(edge_flow[bidirectional] / reverse[bidirectional])))
    one_way = (edge_flow > eps) & (reverse <= eps)
    irreversible_flux = float(np.sum(edge_flow[one_way]))

    supplier_visits = edge_counts[:, supplier_idx].sum(axis=0)
    supplier_distribution = supplier_visits / max(float(supplier_visits.sum()), 1.0)
    supplier_positive = supplier_distribution > 0
    supplier_diversity = 0.0
    if np.any(supplier_positive):
        supplier_diversity = float(-np.sum(supplier_distribution[supplier_positive] * np.log(supplier_distribution[supplier_positive])))

    window = max(8, config.steps // 4)
    fulfillment = float(np.mean(terminal_hits[-window:]))
    critical_service = float(np.mean(critical_hits[-window:]))
    buffer_use = float(np.mean(buffer_hits[-window:]))
    supplier_load = float(np.mean(supplier_hits[-window:]))
    mean_edge_entropy = float(np.mean(edge_entropy_series[-window:]))

    return {
        "shock": shock.name if shock else "none",
        "control": control.name if control else "no_control",
        "agents": config.agents,
        "steps": config.steps,
        "fulfillment_share": fulfillment,
        "critical_service_share": critical_service,
        "buffer_use_share": buffer_use,
        "supplier_load_share": supplier_load,
        "supplier_diversity": supplier_diversity,
        "edge_current_norm": float(np.linalg.norm(edge_current, ord="fro")),
        "entropy_production": entropy_production,
        "irreversible_flux": irreversible_flux,
        "mean_edge_entropy": mean_edge_entropy,
        "terminal_series": terminal_hits,
        "critical_series": critical_hits,
    }


def run_scenarios(config: SimulationConfig | None = None) -> list[dict]:
    config = config or SimulationConfig()
    labels = build_kernel(config).topo.labels
    rows = []
    for shock in shocks(labels):
        shocked_baseline = None
        for control in controls(labels):
            row = simulate(config, shock=shock, control=control)
            row["shock_cost"] = shock.cost
            row["control_cost"] = control.cost
            if control.name == "no_control":
                shocked_baseline = row
            if shocked_baseline is None:
                row["fulfillment_delta_vs_shock"] = 0.0
                row["critical_delta_vs_shock"] = 0.0
                row["resilience_roi"] = 0.0
            else:
                row["fulfillment_delta_vs_shock"] = row["fulfillment_share"] - shocked_baseline["fulfillment_share"]
                row["critical_delta_vs_shock"] = row["critical_service_share"] - shocked_baseline["critical_service_share"]
                row["resilience_roi"] = (
                    (row["fulfillment_delta_vs_shock"] + 0.75 * row["critical_delta_vs_shock"]) / control.cost
                    if control.cost > 0
                    else 0.0
                )
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    by_shock = {}
    best_controls = {}
    for shock in sorted({row["shock"] for row in rows}):
        shock_rows = [row for row in rows if row["shock"] == shock]
        baseline = next(row for row in shock_rows if row["control"] == "no_control")
        ranked = sorted(
            [row for row in shock_rows if row["control"] != "no_control"],
            key=lambda row: (row["resilience_roi"], row["fulfillment_delta_vs_shock"], row["critical_delta_vs_shock"]),
            reverse=True,
        )
        by_shock[shock] = {
            "baseline_fulfillment": baseline["fulfillment_share"],
            "baseline_critical_service": baseline["critical_service_share"],
            "baseline_supplier_diversity": baseline["supplier_diversity"],
            "baseline_edge_current_norm": baseline["edge_current_norm"],
            "best_control": ranked[0]["control"] if ranked else None,
            "best_control_roi": ranked[0]["resilience_roi"] if ranked else 0.0,
        }
        best_controls[shock] = ranked[0] if ranked else None
    all_control_rows = [row for row in rows if row["control"] != "no_control"]
    return {
        "scenario_count": len(rows),
        "shocks": by_shock,
        "best_controls": {
            shock: {key: value for key, value in row.items() if key not in {"terminal_series", "critical_series"}}
            for shock, row in best_controls.items()
            if row is not None
        },
        "mean_resilience_roi": float(np.mean([row["resilience_roi"] for row in all_control_rows])) if all_control_rows else 0.0,
    }


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    rows = payload["rows"]
    lines = [
        "# Supply Chain Resilience Report",
        "",
        "## Scope",
        "",
        (
            "A 22-node directed supply-chain topology with procurement, production, logistics, "
            "distribution, demand, and reserve nodes. Shocks increase route friction; controls "
            "use beta preference and friction reduction to model contracts, expedited shipping, "
            "rerouting, and buffer release."
        ),
        "",
        "## Summary",
        "",
        f"- Scenarios: `{summary['scenario_count']}`",
        f"- Mean resilience ROI: `{summary['mean_resilience_roi']:.4f}`",
        "",
        "## Shock Baselines",
        "",
        "| Shock | Baseline Fulfillment | Critical Service | Supplier Diversity | Edge Current | Best Control | Best ROI |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for shock, row in summary["shocks"].items():
        lines.append(
            f"| {shock} | {row['baseline_fulfillment']:.3f} | {row['baseline_critical_service']:.3f} | "
            f"{row['baseline_supplier_diversity']:.3f} | {row['baseline_edge_current_norm']:.3f} | "
            f"{row['best_control']} | {row['best_control_roi']:.4f} |"
        )

    lines.extend([
        "",
        "## Control Matrix",
        "",
        "| Shock | Control | Fulfillment | Fulfillment Delta | Critical Delta | ROI | Buffer Use | Supplier Diversity |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        if row["control"] == "no_control":
            continue
        lines.append(
            f"| {row['shock']} | {row['control']} | {row['fulfillment_share']:.3f} | "
            f"{row['fulfillment_delta_vs_shock']:+.3f} | {row['critical_delta_vs_shock']:+.3f} | "
            f"{row['resilience_roi']:.4f} | {row['buffer_use_share']:.3f} | {row['supplier_diversity']:.3f} |"
        )
    lines.extend([
        "",
        "## Research Reading",
        "",
        (
            "This first harness is a topology-control instrument, not a demand-forecasting model. "
            "The strongest DTE claim is not that it predicts exact inventory levels; it exposes "
            "which cybernetic controls reroute circulation after shocks and where controls saturate "
            "or backfire."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("supply_chain_resilience_output.json"),
    output_md: Path = Path("SUPPLY_CHAIN_RESILIENCE_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DTE supply-chain resilience scenarios.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("supply_chain_resilience_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SUPPLY_CHAIN_RESILIENCE_REPORT.md"))
    args = parser.parse_args()

    config = SimulationConfig(
        agents=args.agents if args.agents is not None else (96 if args.quick else 256),
        steps=args.steps if args.steps is not None else (32 if args.quick else 96),
    )
    rows = run_scenarios(config)
    payload = {"summary": summarize(rows), "rows": rows}
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

"""
Supply-chain shock severity, hard-failure, and capacity stress tests.

This extends the first supply-chain resilience prototype with two missing
mechanics:

1. Hard edge failures: selected routes are removed from the local transition
   rows before sampling.
2. Capacity caps: selected routes can attract flow, but only a fixed number of
   agents may traverse them per tick; selected nodes can also admit only a fixed
   number of arrivals per tick. Overflow remains upstream.

Usage:
    .venv\\Scripts\\python.exe supply_chain_stress.py
    .venv\\Scripts\\python.exe supply_chain_stress.py --quick
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from supply_chain_resilience import (
    BUFFER_NODES,
    CRITICAL_NODE,
    SUPPLIER_NODES,
    TERMINAL_NODES,
    SimulationConfig,
    _initial_telemetries,
    _seed_for,
    _stable_salt,
    build_kernel,
    controls,
)


PORT_EDGES = [
    ("Asia Port", "Ocean Carrier"),
    ("Ocean Carrier", "West Coast Port"),
    ("West Coast Port", "Rail Hub"),
]
CHIP_EDGES = [
    ("Planning Desk", "Chip Fab A"),
    ("Chip Fab A", "Electronics Assembly"),
]
COLD_CHAIN_EDGES = [
    ("Planning Desk", "Cold Chain Supplier"),
    ("Cold Chain Supplier", "Regional DC"),
    ("Regional DC", "Critical Care Demand"),
]
AIR_EDGES = [
    ("QA Buffer", "Air Freight"),
    ("Asia Port", "Air Freight"),
    ("Air Freight", "West Coast Port"),
]


@dataclass(frozen=True)
class StressShock:
    name: str
    family: str
    severity: float = 0.0
    friction_edges: tuple[tuple[str, str], ...] = ()
    failed_edges: tuple[tuple[str, str], ...] = ()
    capacity_caps: dict[tuple[str, str], int] = field(default_factory=dict)
    node_capacity_caps: dict[str, int] = field(default_factory=dict)


def _idx(labels: list[str], names: tuple[str, ...] | list[str]) -> list[int]:
    return [labels.index(name) for name in names]


def _apply_negative_friction(kernel, labels: list[str], edges: tuple[tuple[str, str], ...], severity: float) -> None:
    if severity <= 0:
        return
    for source, target in edges:
        i, j = labels.index(source), labels.index(target)
        if kernel.topo.adjacency_mask[i, j]:
            kernel.sponsor_edge_friction(i, j, -severity)


def _renormalize_rows(rows: np.ndarray) -> np.ndarray:
    row_sums = rows.sum(axis=1, keepdims=True)
    return np.divide(rows, row_sums, out=np.zeros_like(rows), where=row_sums > 1e-12)


def severity_shocks(severities: list[float]) -> list[StressShock]:
    shocks = []
    for severity in severities:
        shocks.extend([
            StressShock(
                name=f"port_congestion_s{severity:.2f}",
                family="port_congestion",
                severity=severity,
                friction_edges=tuple(PORT_EDGES),
            ),
            StressShock(
                name=f"chip_outage_s{severity:.2f}",
                family="chip_outage",
                severity=severity,
                friction_edges=tuple(CHIP_EDGES),
            ),
            StressShock(
                name=f"cold_chain_s{severity:.2f}",
                family="cold_chain",
                severity=severity,
                friction_edges=tuple(COLD_CHAIN_EDGES),
            ),
        ])
    return shocks


def hard_capacity_shocks() -> list[StressShock]:
    return [
        StressShock(
            name="chip_fab_a_hard_failure",
            family="hard_failure",
            failed_edges=tuple(CHIP_EDGES),
        ),
        StressShock(
            name="ocean_route_hard_failure",
            family="hard_failure",
            failed_edges=tuple(PORT_EDGES[:2]),
        ),
        StressShock(
            name="air_freight_capacity_08",
            family="capacity",
            capacity_caps={
                ("QA Buffer", "Air Freight"): 8,
                ("Asia Port", "Air Freight"): 8,
                ("Air Freight", "West Coast Port"): 8,
            },
        ),
        StressShock(
            name="air_freight_capacity_16",
            family="capacity",
            capacity_caps={
                ("QA Buffer", "Air Freight"): 16,
                ("Asia Port", "Air Freight"): 16,
                ("Air Freight", "West Coast Port"): 16,
            },
        ),
        StressShock(
            name="port_and_air_capacity",
            family="capacity",
            capacity_caps={
                ("Asia Port", "Ocean Carrier"): 20,
                ("Ocean Carrier", "West Coast Port"): 20,
                ("QA Buffer", "Air Freight"): 10,
                ("Asia Port", "Air Freight"): 10,
                ("Air Freight", "West Coast Port"): 10,
            },
        ),
        StressShock(
            name="chip_fab_b_capacity_04",
            family="node_capacity",
            node_capacity_caps={"Chip Fab B": 4},
        ),
        StressShock(
            name="regional_dc_capacity_18",
            family="node_capacity",
            node_capacity_caps={"Regional DC": 18},
        ),
        StressShock(
            name="government_reserve_capacity_04",
            family="node_capacity",
            node_capacity_caps={"Government Reserve": 4},
        ),
        StressShock(
            name="fab_dc_reserve_capacity",
            family="node_capacity",
            node_capacity_caps={
                "Chip Fab B": 5,
                "Regional DC": 16,
                "Government Reserve": 4,
            },
        ),
    ]


def simulate_stress(
    config: SimulationConfig,
    shock: StressShock,
    control_name: str = "no_control",
    enforce_bom: bool = False,
    bom_initial_inventory: int = 0,
) -> dict:
    kernel = build_kernel(config)
    labels = kernel.topo.labels
    _apply_negative_friction(kernel, labels, shock.friction_edges, shock.severity)
    control = next(item for item in controls(labels) if item.name == control_name)
    control.apply(kernel)

    rng = np.random.default_rng(_seed_for(config, _stable_salt(shock.name, control_name)))
    n = kernel.topo.N
    start = labels.index("Planning Desk")
    positions = np.full(config.agents, start, dtype=int)
    telemetries = _initial_telemetries(config)

    terminal_idx = np.array(_idx(labels, TERMINAL_NODES), dtype=int)
    critical_idx = labels.index(CRITICAL_NODE)
    supplier_idx = np.array(_idx(labels, SUPPLIER_NODES), dtype=int)
    buffer_idx = np.array(_idx(labels, BUFFER_NODES), dtype=int)
    failed_idx = [(labels.index(source), labels.index(target)) for source, target in shock.failed_edges]
    capacity_idx = {
        (labels.index(source), labels.index(target)): int(capacity)
        for (source, target), capacity in shock.capacity_caps.items()
    }
    node_capacity_idx = {
        labels.index(node): int(capacity)
        for node, capacity in shock.node_capacity_caps.items()
    }
    final_assembly_idx = labels.index("Final Assembly")
    qa_buffer_idx = labels.index("QA Buffer")
    bom_sources = {
        labels.index("Battery Plant"): "battery",
        labels.index("Electronics Assembly"): "electronics",
        labels.index("Chassis Plant"): "chassis",
        labels.index("Packaging Supplier"): "packaging",
    }
    bom_parts = tuple(bom_sources.values())
    bom_inventory = {
        part: int(bom_initial_inventory)
        for part in bom_parts
    }
    bom_arrivals = {part: 0 for part in bom_parts}

    edge_counts = np.zeros((n, n), dtype=np.float64)
    terminal_hits = []
    critical_hits = []
    supplier_hits = []
    buffer_hits = []
    blocked_mass = []
    edge_overflow_events = 0
    edge_capacity_attempts = 0
    node_overflow_events = 0
    node_capacity_attempts = 0
    bom_attempts = 0
    bom_blocked_events = 0
    bom_completion_events = 0
    bom_completion_series = []
    bom_blocked_series = []

    for step in range(config.steps):
        P_all = kernel.transition_matrix_batch(telemetries, step=step)
        rows = P_all[np.arange(config.agents), positions, :].copy()

        hard_block_mass = 0.0
        for i, j in failed_idx:
            source_mask = positions == i
            if np.any(source_mask):
                hard_block_mass += float(rows[source_mask, j].sum() / config.agents)
                rows[source_mask, j] = 0.0
        if failed_idx:
            rows = _renormalize_rows(rows)
        blocked_mass.append(hard_block_mass)

        cdf = np.cumsum(rows, axis=1)
        draws = rng.random((config.agents, 1))
        next_positions = np.argmax(cdf >= draws, axis=1)
        row_sums = rows.sum(axis=1)
        next_positions[row_sums < 1e-12] = positions[row_sums < 1e-12]

        for (i, j), capacity in capacity_idx.items():
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

        step_bom_completions = 0
        step_bom_blocked = 0
        if enforce_bom:
            attempted = np.where((positions == final_assembly_idx) & (next_positions == qa_buffer_idx))[0]
            bom_attempts += int(len(attempted))
            feasible_units = min(bom_inventory.values()) if bom_inventory else 0
            allowed_count = min(int(len(attempted)), int(feasible_units))
            if len(attempted) > allowed_count:
                allowed = set(rng.choice(attempted, size=allowed_count, replace=False).tolist())
                blocked = [idx for idx in attempted.tolist() if idx not in allowed]
                step_bom_blocked = len(blocked)
                bom_blocked_events += step_bom_blocked
                next_positions[np.array(blocked, dtype=int)] = positions[np.array(blocked, dtype=int)]
            if allowed_count > 0:
                step_bom_completions = allowed_count
                bom_completion_events += allowed_count
                for part in bom_parts:
                    bom_inventory[part] -= allowed_count

            for source_idx, part in bom_sources.items():
                arrivals = int(np.sum((positions == source_idx) & (next_positions == final_assembly_idx)))
                if arrivals > 0:
                    bom_arrivals[part] += arrivals
                    bom_inventory[part] += arrivals
        bom_completion_series.append(float(step_bom_completions / config.agents))
        bom_blocked_series.append(float(step_bom_blocked / config.agents))

        np.add.at(edge_counts, (positions, next_positions), 1.0)
        visited = kernel.topo.node_features[next_positions]
        lam = config.feedback_rate
        telemetries = (1.0 - lam) * telemetries + lam * visited
        norms = np.linalg.norm(telemetries, axis=1, keepdims=True)
        telemetries = np.where(norms > 0, telemetries / norms, telemetries)
        positions = next_positions

        terminal_hits.append(float(np.mean(np.isin(positions, terminal_idx))))
        critical_hits.append(float(np.mean(positions == critical_idx)))
        supplier_hits.append(float(np.mean(np.isin(positions, supplier_idx))))
        buffer_hits.append(float(np.mean(np.isin(positions, buffer_idx))))

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
    total_overflow_events = edge_overflow_events + node_overflow_events
    total_capacity_attempts = edge_capacity_attempts + node_capacity_attempts
    attempted = max(total_capacity_attempts, 1)
    edge_attempted = max(edge_capacity_attempts, 1)
    node_attempted = max(node_capacity_attempts, 1)
    return {
        "shock": shock.name,
        "family": shock.family,
        "severity": shock.severity,
        "control": control_name,
        "control_cost": control.cost,
        "agents": config.agents,
        "steps": config.steps,
        "fulfillment_share": float(np.mean(terminal_hits[-window:])),
        "critical_service_share": float(np.mean(critical_hits[-window:])),
        "supplier_load_share": float(np.mean(supplier_hits[-window:])),
        "buffer_use_share": float(np.mean(buffer_hits[-window:])),
        "edge_current_norm": float(np.linalg.norm(edge_current, ord="fro")),
        "entropy_production": entropy_production,
        "irreversible_flux": float(np.sum(edge_flow[one_way])),
        "hard_blocked_mass": float(np.mean(blocked_mass[-window:])),
        "capacity_overflow_rate": float(total_overflow_events / attempted),
        "capacity_attempts": int(total_capacity_attempts),
        "edge_capacity_overflow_rate": float(edge_overflow_events / edge_attempted),
        "edge_capacity_attempts": int(edge_capacity_attempts),
        "node_capacity_overflow_rate": float(node_overflow_events / node_attempted),
        "node_capacity_attempts": int(node_capacity_attempts),
        "bom_enforced": bool(enforce_bom),
        "bom_attempts": int(bom_attempts),
        "bom_blocked_events": int(bom_blocked_events),
        "bom_completion_events": int(bom_completion_events),
        "bom_completion_per_agent": float(bom_completion_events / config.agents),
        "bom_completion_rate": float(bom_completion_events / max(config.agents * config.steps, 1)),
        "bom_blocked_per_agent": float(bom_blocked_events / config.agents),
        "bom_block_rate": float(bom_blocked_events / max(bom_attempts, 1)),
        "bom_completion_flow_share": float(np.mean(bom_completion_series[-window:])),
        "bom_blocked_flow_share": float(np.mean(bom_blocked_series[-window:])),
        "bom_inventory_end": {part: int(count) for part, count in bom_inventory.items()},
        "bom_arrivals": {part: int(count) for part, count in bom_arrivals.items()},
        "bom_min_inventory_end": int(min(bom_inventory.values())) if bom_inventory else 0,
    }


def run_severity_sweep(
    severities: list[float] | None = None,
    control_names: list[str] | None = None,
    config: SimulationConfig | None = None,
) -> dict:
    severities = severities or [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    control_names = control_names or ["no_control", "port_reroute", "dual_source_chips", "buffer_release", "generic_resilience"]
    config = config or SimulationConfig()
    rows = [
        simulate_stress(config, shock, control_name)
        for shock in severity_shocks(severities)
        for control_name in control_names
    ]
    return {"summary": summarize_severity(rows), "rows": rows, "config": {"severities": severities, "controls": control_names}}


def run_hard_capacity_cases(
    control_names: list[str] | None = None,
    config: SimulationConfig | None = None,
) -> dict:
    control_names = control_names or ["no_control", "port_reroute", "expedite_air", "buffer_release", "generic_resilience", "dual_source_chips"]
    config = config or SimulationConfig()
    rows = [
        simulate_stress(config, shock, control_name)
        for shock in hard_capacity_shocks()
        for control_name in control_names
    ]
    return {"summary": summarize_hard_capacity(rows), "rows": rows, "config": {"controls": control_names}}


def _with_deltas(rows: list[dict]) -> list[dict]:
    baselines = {
        row["shock"]: row
        for row in rows
        if row["control"] == "no_control"
    }
    enriched = []
    for row in rows:
        item = dict(row)
        baseline = baselines[row["shock"]]
        item["fulfillment_delta_vs_baseline"] = row["fulfillment_share"] - baseline["fulfillment_share"]
        item["critical_delta_vs_baseline"] = row["critical_service_share"] - baseline["critical_service_share"]
        item["stress_delta_vs_baseline"] = row["edge_current_norm"] - baseline["edge_current_norm"]
        item["resilience_roi"] = (
            (item["fulfillment_delta_vs_baseline"] + 0.75 * item["critical_delta_vs_baseline"]) / row["control_cost"]
            if row["control_cost"] > 0
            else 0.0
        )
        enriched.append(item)
    return enriched


def summarize_severity(rows: list[dict]) -> dict:
    rows = _with_deltas(rows)
    families = {}
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        controls = {}
        for control in sorted({row["control"] for row in family_rows}):
            control_rows = [row for row in family_rows if row["control"] == control]
            controls[control] = {
                "mean_fulfillment": float(np.mean([row["fulfillment_share"] for row in control_rows])),
                "mean_critical_service": float(np.mean([row["critical_service_share"] for row in control_rows])),
                "mean_roi": float(np.mean([row["resilience_roi"] for row in control_rows])),
                "max_edge_current": float(max(row["edge_current_norm"] for row in control_rows)),
            }
        best_by_severity = {}
        for severity in sorted({row["severity"] for row in family_rows}):
            candidates = [
                row for row in family_rows
                if row["severity"] == severity and row["control"] != "no_control"
            ]
            best = max(candidates, key=lambda row: (row["resilience_roi"], row["fulfillment_delta_vs_baseline"]))
            best_by_severity[f"{severity:.2f}"] = {
                "control": best["control"],
                "roi": best["resilience_roi"],
                "fulfillment_delta": best["fulfillment_delta_vs_baseline"],
                "critical_delta": best["critical_delta_vs_baseline"],
            }
        families[family] = {"controls": controls, "best_by_severity": best_by_severity}
    return {"families": families, "rows": rows}


def summarize_hard_capacity(rows: list[dict]) -> dict:
    rows = _with_deltas(rows)
    cases = {}
    for shock in sorted({row["shock"] for row in rows}):
        case_rows = [row for row in rows if row["shock"] == shock]
        baseline = next(row for row in case_rows if row["control"] == "no_control")
        best = max(
            [row for row in case_rows if row["control"] != "no_control"],
            key=lambda row: (row["resilience_roi"], row["fulfillment_delta_vs_baseline"]),
        )
        cases[shock] = {
            "family": baseline["family"],
            "baseline_fulfillment": baseline["fulfillment_share"],
            "baseline_critical_service": baseline["critical_service_share"],
            "baseline_hard_blocked_mass": baseline["hard_blocked_mass"],
            "baseline_capacity_overflow_rate": baseline["capacity_overflow_rate"],
            "baseline_edge_capacity_overflow_rate": baseline["edge_capacity_overflow_rate"],
            "baseline_node_capacity_overflow_rate": baseline["node_capacity_overflow_rate"],
            "best_control": best["control"],
            "best_roi": best["resilience_roi"],
            "best_fulfillment_delta": best["fulfillment_delta_vs_baseline"],
            "best_critical_delta": best["critical_delta_vs_baseline"],
            "best_overflow_rate": best["capacity_overflow_rate"],
            "best_edge_overflow_rate": best["edge_capacity_overflow_rate"],
            "best_node_overflow_rate": best["node_capacity_overflow_rate"],
        }
    return {"cases": cases, "rows": rows}


def run_stress_suite(config: SimulationConfig | None = None) -> dict:
    config = config or SimulationConfig()
    return {
        "severity": run_severity_sweep(config=config),
        "hard_capacity": run_hard_capacity_cases(config=config),
    }


def render_report(payload: dict) -> str:
    severity = payload["severity"]["summary"]
    hard = payload["hard_capacity"]["summary"]
    lines = [
        "# Supply Chain Stress Report",
        "",
        "## Scope",
        "",
        (
            "Shock severity sweep plus hard-edge failure and capacity constraints. "
            "Hard failures remove local transition probability; capacity caps create upstream overflow."
        ),
        "",
        "## Severity Sweep: Best Control By Severity",
        "",
        "| Family | Severity | Best Control | ROI | Fulfillment Delta | Critical Delta |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for family, family_summary in severity["families"].items():
        for severity_key, row in family_summary["best_by_severity"].items():
            lines.append(
                f"| {family} | {severity_key} | {row['control']} | {row['roi']:.4f} | "
                f"{row['fulfillment_delta']:+.3f} | {row['critical_delta']:+.3f} |"
            )

    lines.extend([
        "",
        "## Hard Failure And Capacity Cases",
        "",
        "| Shock | Family | Baseline Fulfillment | Blocked Mass | Overflow | Node Overflow | Best Control | Best ROI | Fulfillment Delta | Critical Delta | Best Overflow | Best Node Overflow |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for shock, row in hard["cases"].items():
        lines.append(
            f"| {shock} | {row['family']} | {row['baseline_fulfillment']:.3f} | "
            f"{row['baseline_hard_blocked_mass']:.3f} | {row['baseline_capacity_overflow_rate']:.1%} | "
            f"{row['baseline_node_capacity_overflow_rate']:.1%} | "
            f"{row['best_control']} | {row['best_roi']:.4f} | "
            f"{row['best_fulfillment_delta']:+.3f} | {row['best_critical_delta']:+.3f} | "
            f"{row['best_overflow_rate']:.1%} | {row['best_node_overflow_rate']:.1%} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "Severity sweeps ask when a friction shock crosses from absorbable to structurally important. "
            "Hard failures test true route removal. Capacity cases test a different mechanism: a route may "
            "remain attractive while its throughput cap forces upstream waiting. Node-capacity cases test "
            "facility admission constraints: substitution may fail because the backup supplier, distribution "
            "center, or reserve has no spare receiving capacity. These experiments separate preference, "
            "availability, route capacity, and node capacity."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("supply_chain_stress_output.json"),
    output_md: Path = Path("SUPPLY_CHAIN_STRESS_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supply-chain shock stress suite.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("supply_chain_stress_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SUPPLY_CHAIN_STRESS_REPORT.md"))
    args = parser.parse_args()

    config = SimulationConfig(
        agents=args.agents if args.agents is not None else (96 if args.quick else 256),
        steps=args.steps if args.steps is not None else (32 if args.quick else 96),
    )
    if args.quick:
        payload = {
            "severity": run_severity_sweep(severities=[0.0, 2.0], config=config),
            "hard_capacity": run_hard_capacity_cases(config=config),
        }
    else:
        payload = run_stress_suite(config)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "severity_families": list(payload["severity"]["summary"]["families"]),
        "hard_capacity_cases": list(payload["hard_capacity"]["summary"]["cases"]),
    }, indent=2))


if __name__ == "__main__":
    main()

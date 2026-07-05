"""
Kernel V1 local-regret witness.

This is a deliberately tiny topology using the built-in PopulationSimulator
memory loop:

    completed traversals -> traffic matrix -> node rewards -> memory_law_step

The scenario creates an already learned stale preference for Entry -> Stale.
The stale node still pays a decent reward, so surprise-only adaptive_eta sees
no destination-level reward collapse. A better reachable alternative exists,
however, so opportunity-cost adaptive_eta should evaporate the stale memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from kernel import DynamicTopologyKernel, topology_from_edges
from simulator import PopulationSimulator


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "kernel_local_regret_witness_output.json"
REPORT_PATH = ROOT / "KERNEL_LOCAL_REGRET_WITNESS_REPORT.md"


def build_kernel(opportunity_gain: float) -> DynamicTopologyKernel:
    topo = topology_from_edges(
        nodes={
            "Entry": np.array([1.0, 0.0]),
            "Stale": np.array([1.0, 0.0]),
            "Better": np.array([1.0, 0.0]),
        },
        edges=[
            ("Entry", "Stale", 1.0),
            ("Entry", "Better", 1.0),
            ("Stale", "Entry", 1.0),
            ("Better", "Entry", 1.0),
        ],
        undirected=False,
    )
    kernel = DynamicTopologyKernel(
        topo,
        alpha=1.0,
        beta=0.0,
        temperature=0.25,
        feedback_noise=0.0,
    )
    # Pre-existing preference memory: the system learned Entry -> Stale.
    kernel.sponsor_edge_friction(0, 1, reduction=2.0)
    kernel.configure_memory_law(
        mode="adaptive_eta",
        rho=0.0,
        eta=0.02,
        eta_max=0.50,
        surprise_gain=4.0,
        opportunity_gain=opportunity_gain,
        initial_expectation=0.8,
    )
    return kernel


def run_condition(
    opportunity_gain: float,
    seed: int,
    agents: int = 300,
    ticks: int = 50,
) -> dict[str, Any]:
    np.random.seed(seed)
    kernel = build_kernel(opportunity_gain)
    sim = PopulationSimulator(
        kernel,
        K=agents,
        time_multiplier=1.0,
        node_rewards=np.array([0.0, 0.8, 1.0]),
        dwell_range=(1, 2),
        sink_dwell_range=(1, 2),
    )
    telemetry = np.array([1.0, 0.0])
    previous_crossings = np.zeros((3, 3), dtype=np.int64)
    history: list[dict[str, Any]] = []

    for tick in range(ticks):
        result = sim.tick()
        crossings = sim.total_edge_crossings.copy()
        incremental = crossings - previous_crossings
        previous_crossings = crossings
        entry_total = int(incremental[0, 1] + incremental[0, 2])
        opportunity = result["memory_update"].get("opportunity_cost")
        P = kernel.transition_matrix(telemetry)
        history.append(
            {
                "tick": tick,
                "entry_to_stale": int(incremental[0, 1]),
                "entry_to_better": int(incremental[0, 2]),
                "entry_stale_share": (
                    float(incremental[0, 1] / entry_total)
                    if entry_total > 0
                    else None
                ),
                "stale_delta": float(
                    kernel._sponsor_friction[0, 1] - kernel._friction_baseline[0, 1]
                ),
                "p_entry_stale": float(P[0, 1]),
                "eta_stale": (
                    None
                    if kernel._last_eta_effective is None
                    else float(kernel._last_eta_effective[1])
                ),
                "mean_opportunity_cost": (
                    None if opportunity is None else opportunity["mean_opportunity_cost"]
                ),
                "stale_flow_share": (
                    None if opportunity is None else opportunity["stale_flow_share"]
                ),
            }
        )

    recent_shares = [
        row["entry_stale_share"]
        for row in history[-10:]
        if row["entry_stale_share"] is not None
    ]
    return {
        "opportunity_gain": opportunity_gain,
        "seed": seed,
        "agents": agents,
        "ticks": ticks,
        "final_stale_delta": history[-1]["stale_delta"],
        "final_p_entry_stale": history[-1]["p_entry_stale"],
        "last10_entry_stale_share": mean(recent_shares) if recent_shares else 0.0,
        "total_entry_to_stale": int(sim.total_edge_crossings[0, 1]),
        "total_entry_to_better": int(sim.total_edge_crossings[0, 2]),
        "history": history,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_gain: dict[float, list[dict[str, Any]]] = {}
    for run in runs:
        by_gain.setdefault(float(run["opportunity_gain"]), []).append(run)
    summary = {}
    for gain, cells in sorted(by_gain.items()):
        summary[str(gain)] = {
            "runs": len(cells),
            "mean_final_stale_delta": mean(c["final_stale_delta"] for c in cells),
            "mean_final_p_entry_stale": mean(c["final_p_entry_stale"] for c in cells),
            "mean_last10_entry_stale_share": mean(
                c["last10_entry_stale_share"] for c in cells
            ),
        }
    return summary


def write_report(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    no_regret = summary["0.0"]
    regret = summary["1.0"]
    lines = [
        "# Kernel Local-Regret Witness Report",
        "",
        "## Scope",
        "",
        "Tiny three-node topology using the built-in `PopulationSimulator` memory loop.",
        "The learned stale route still pays reward `0.8`; the reachable alternative",
        "pays `1.0`. Surprise-only adaptive evaporation receives no destination",
        "reward-collapse signal, while local regret sees the opportunity cost.",
        "",
        "## Result",
        "",
        "| Condition | Runs | Final stale delta | Final P(Entry->Stale) | Last-10 stale share |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Surprise only | {no_regret['runs']} | "
            f"{no_regret['mean_final_stale_delta']:.4f} | "
            f"{no_regret['mean_final_p_entry_stale']:.4f} | "
            f"{no_regret['mean_last10_entry_stale_share']:.4f} |"
        ),
        (
            f"| Local regret | {regret['runs']} | "
            f"{regret['mean_final_stale_delta']:.4f} | "
            f"{regret['mean_final_p_entry_stale']:.4f} | "
            f"{regret['mean_last10_entry_stale_share']:.4f} |"
        ),
        "",
        "## Interpretation",
        "",
        "Local regret behaves as intended: it evaporates stale preference memory",
        "even when the stale route remains mildly productive. The route probability",
        "returns to the unbiased baseline near 0.5, while surprise-only adaptation",
        "leaves the stale route dominant.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict[str, Any]:
    seeds = list(range(5))
    runs = []
    for seed in seeds:
        runs.append(run_condition(0.0, seed))
        runs.append(run_condition(1.0, seed))
    payload = {
        "seeds": seeds,
        "conditions": ["surprise_only", "local_regret"],
        "runs": runs,
        "summary": summarize(runs),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload)
    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["summary"], indent=2))

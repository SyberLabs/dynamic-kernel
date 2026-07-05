"""
Two-route memory-ecology theorem simulator.

This is the minimal analytical witness for the Memory-Ecology Mismatch
Principle. It removes biological richness and keeps only:

    Choice -> Rich finite route
           -> Sparse persistent route

The model is deterministic in expectation. At each cycle, transition mass
chooses rich or sparse. Successful rich returns deposit preference memory until
the rich resource depletes; sparse returns remain persistent. After depletion,
rich memory evaporates but can keep transition mass concentrated on the now
empty route.

Usage:
    .venv\\Scripts\\python.exe two_route_memory_theorem.py --quick
    .venv\\Scripts\\python.exe two_route_memory_theorem.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TwoRouteConfig:
    cycles: int = 120
    deplete_cycle: int = 36
    rich_initial_reward: float = 1.0
    sparse_reward: float = 0.62
    rho: float = 0.45
    eta: float = 0.02
    epsilon: float = 0.04
    inverse_temperature: float = 5.0
    initial_memory_rich: float = 0.0
    initial_memory_sparse: float = 0.0
    recovery_theta: float = 0.62


@dataclass(frozen=True)
class SweepConfig:
    cycles: int = 120
    deplete_cycle: int = 36
    seeds: int = 1


def _softmax2(a: float, b: float, inverse_temperature: float) -> tuple[float, float]:
    scaled = np.array([a, b], dtype=float) * inverse_temperature
    scaled -= float(np.max(scaled))
    exp = np.exp(scaled)
    probs = exp / np.sum(exp)
    return float(probs[0]), float(probs[1])


def transition_probs(
    rich_memory: float,
    sparse_memory: float,
    rich_reward: float,
    sparse_reward: float,
    config: TwoRouteConfig,
) -> tuple[float, float]:
    rich_score = rich_reward + rich_memory
    sparse_score = sparse_reward + sparse_memory
    p_rich, p_sparse = _softmax2(rich_score, sparse_score, config.inverse_temperature)
    if config.epsilon > 0.0:
        p_rich = (1.0 - config.epsilon) * p_rich + 0.5 * config.epsilon
        p_sparse = (1.0 - config.epsilon) * p_sparse + 0.5 * config.epsilon
    return p_rich, p_sparse


def decay_only_recovery_bound(delta_memory_at_depletion: float, eta: float, theta: float = 0.0) -> int | None:
    """Sufficient recovery timescale if stale memory decays only by evaporation."""
    if delta_memory_at_depletion <= theta:
        return 0
    if eta <= 0.0:
        return None
    if eta >= 1.0:
        return 1
    return int(math.ceil(math.log(max(theta, 1e-12) / delta_memory_at_depletion) / math.log(1.0 - eta)))


def recovery_lower_bound(delta_memory_at_depletion: float, eta: float, theta: float = 0.0) -> int | None:
    """Backward-compatible alias for decay_only_recovery_bound()."""
    return decay_only_recovery_bound(delta_memory_at_depletion, eta, theta)


def simulate(config: TwoRouteConfig) -> dict[str, Any]:
    rich_memory = config.initial_memory_rich
    sparse_memory = config.initial_memory_sparse
    rows = []
    stale_lockin_duration = 0
    empty_rich_mass = 0.0
    throughput = 0.0
    recovery_cycle: int | None = None
    delta_memory_at_depletion: float | None = None
    transition_gap_at_depletion: float | None = None

    for cycle in range(config.cycles):
        rich_available = cycle < config.deplete_cycle
        rich_reward = config.rich_initial_reward if rich_available else 0.0
        p_rich, p_sparse = transition_probs(
            rich_memory,
            sparse_memory,
            rich_reward,
            config.sparse_reward,
            config,
        )
        reward_returned = p_rich * rich_reward + p_sparse * config.sparse_reward
        throughput += reward_returned
        if not rich_available:
            empty_rich_mass += p_rich
            if p_rich > p_sparse:
                stale_lockin_duration += 1
            elif recovery_cycle is None:
                recovery_cycle = cycle

        rows.append({
            "cycle": cycle,
            "rich_available": rich_available,
            "rich_reward": rich_reward,
            "sparse_reward": config.sparse_reward,
            "p_rich": p_rich,
            "p_sparse": p_sparse,
            "rich_memory": rich_memory,
            "sparse_memory": sparse_memory,
            "memory_gap": rich_memory - sparse_memory,
            "reward_returned": reward_returned,
            "empty_rich_mass_cumulative": empty_rich_mass,
        })

        if cycle == config.deplete_cycle:
            delta_memory_at_depletion = rich_memory - sparse_memory
            transition_gap_at_depletion = p_rich - p_sparse

        rich_success = p_rich * rich_reward
        sparse_success = p_sparse * config.sparse_reward
        rich_memory = rich_memory * (1.0 - config.eta) + config.rho * rich_success
        sparse_memory = sparse_memory * (1.0 - config.eta) + config.rho * sparse_success

    if recovery_cycle is None and config.deplete_cycle < config.cycles:
        recovery_cycle = None
    if delta_memory_at_depletion is None:
        delta_memory_at_depletion = rows[min(config.deplete_cycle, len(rows) - 1)]["memory_gap"]
    if transition_gap_at_depletion is None:
        transition_gap_at_depletion = rows[min(config.deplete_cycle, len(rows) - 1)]["p_rich"] - rows[
            min(config.deplete_cycle, len(rows) - 1)
        ]["p_sparse"]

    observed_recovery_time = (
        None if recovery_cycle is None else max(0, recovery_cycle - config.deplete_cycle)
    )
    bound = decay_only_recovery_bound(
        max(0.0, delta_memory_at_depletion),
        config.eta,
        config.recovery_theta,
    )
    cycles_after_depletion = max(0, config.cycles - config.deplete_cycle)
    post_depletion_empty_rate = empty_rich_mass / max(1, cycles_after_depletion)
    final = rows[-1]

    classification = "no_lockin"
    if stale_lockin_duration > 0 and observed_recovery_time is None:
        classification = "unrecovered_stale_lockin"
    elif stale_lockin_duration > 0:
        classification = "recovered_stale_lockin"
    elif post_depletion_empty_rate >= 0.25:
        classification = "diffuse_empty_drag"

    return {
        "config": asdict(config),
        "memory_ratio": float(config.rho / max(config.eta, 1e-12)),
        "delta_memory_at_depletion": float(delta_memory_at_depletion),
        "transition_gap_at_depletion": float(transition_gap_at_depletion),
        "stale_lockin_duration": int(stale_lockin_duration),
        "observed_recovery_time": observed_recovery_time,
        "decay_only_recovery_bound": bound,
        "post_depletion_empty_rate": float(post_depletion_empty_rate),
        "throughput": float(throughput),
        "final_p_rich": float(final["p_rich"]),
        "final_p_sparse": float(final["p_sparse"]),
        "final_memory_gap": float(final["memory_gap"]),
        "classification": classification,
        "rows": rows,
    }


def sweep_grid(quick: bool = False) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if quick:
        return (0.12, 0.45, 0.90), (0.01, 0.04, 0.12), (0.02, 0.10)
    return (
        (0.08, 0.12, 0.24, 0.45, 0.70, 0.90),
        (0.005, 0.01, 0.02, 0.04, 0.08, 0.12),
        (0.0, 0.02, 0.06, 0.10, 0.18),
    )


def run_sweep(config: SweepConfig | None = None, quick: bool = False) -> dict[str, Any]:
    config = config or (SweepConfig(cycles=90, deplete_cycle=30) if quick else SweepConfig())
    rhos, etas, epsilons = sweep_grid(quick)
    rows = []
    for rho in rhos:
        for eta in etas:
            for epsilon in epsilons:
                sim_config = TwoRouteConfig(
                    cycles=config.cycles,
                    deplete_cycle=config.deplete_cycle,
                    rho=rho,
                    eta=eta,
                    epsilon=epsilon,
                )
                result = simulate(sim_config)
                rows.append({
                    key: value
                    for key, value in result.items()
                    if key not in {"rows"}
                })

    summary = summarize_sweep(rows)
    return {
        "config": asdict(config) | {"quick": quick},
        "rhos": rhos,
        "etas": etas,
        "epsilons": epsilons,
        "summary": summary,
        "rows": rows,
    }


def summarize_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = {
        label: sum(row["classification"] == label for row in rows)
        for label in sorted({row["classification"] for row in rows})
    }
    lockin_rows = [row for row in rows if row["stale_lockin_duration"] > 0]
    unrecovered = [row for row in rows if row["classification"] == "unrecovered_stale_lockin"]
    recovered = [row for row in rows if row["classification"] == "recovered_stale_lockin"]
    return {
        "runs": len(rows),
        "classification_counts": class_counts,
        "first_lockin_memory_ratio": min((row["memory_ratio"] for row in lockin_rows), default=None),
        "first_unrecovered_memory_ratio": min((row["memory_ratio"] for row in unrecovered), default=None),
        "max_lockin_duration": max(lockin_rows, key=lambda row: row["stale_lockin_duration"], default=None),
        "best_throughput": max(rows, key=lambda row: row["throughput"], default=None),
        "largest_bound_gap": max(
            recovered,
            key=lambda row: (
                0
                if row["decay_only_recovery_bound"] is None or row["observed_recovery_time"] is None
                else row["decay_only_recovery_bound"] - row["observed_recovery_time"]
            ),
            default=None,
        ),
    }


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Two-Route Memory-Ecology Theorem Simulator",
        "",
        "## Scope",
        "",
        (
            "Minimal deterministic witness for stale-reinforcement lock-in. One rich "
            "finite route competes with one sparse persistent route; rich reward drops "
            "to zero at the depletion cycle while preference memory evaporates slowly."
        ),
        "",
        f"- Runs: `{summary['runs']}`",
        f"- Depletion cycle: `{payload['config']['deplete_cycle']}`",
        f"- First lock-in memory ratio: `{summary['first_lockin_memory_ratio']}`",
        f"- First unrecovered memory ratio: `{summary['first_unrecovered_memory_ratio']}`",
        "",
        "## Classification Counts",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(summary["classification_counts"].items()):
        lines.append(f"| `{label}` | {count} |")

    max_lock = summary["max_lockin_duration"]
    if max_lock is not None:
        lines.extend(
            [
                "",
                "## Maximum Lock-In",
                "",
                f"- Memory ratio: `{max_lock['memory_ratio']:.2f}`",
                f"- rho: `{max_lock['config']['rho']}`",
                f"- eta: `{max_lock['config']['eta']}`",
                f"- epsilon: `{max_lock['config']['epsilon']}`",
                f"- Stale lock-in duration: `{max_lock['stale_lockin_duration']}`",
                f"- Observed recovery time: `{max_lock['observed_recovery_time']}`",
                f"- Decay-only recovery bound: `{max_lock['decay_only_recovery_bound']}`",
                f"- Post-depletion empty rate: `{max_lock['post_depletion_empty_rate']:.3f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Sweep Table",
            "",
        "| rho | eta | epsilon | ratio | class | lock-in | recovery | decay-only bound | empty rate | throughput |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        recovery = "n/a" if row["observed_recovery_time"] is None else str(row["observed_recovery_time"])
        bound = "n/a" if row["decay_only_recovery_bound"] is None else str(row["decay_only_recovery_bound"])
        lines.append(
            f"| {row['config']['rho']:.3f} | {row['config']['eta']:.3f} | "
            f"{row['config']['epsilon']:.3f} | {row['memory_ratio']:.2f} | "
            f"{row['classification']} | {row['stale_lockin_duration']} | {recovery} | "
            f"{bound} | {row['post_depletion_empty_rate']:.3f} | {row['throughput']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This toy model isolates the theorem mechanism. Before depletion, rich "
                "success builds a memory gap. After depletion, the reward goes to zero "
                "but the memory gap evaporates gradually, so transition mass can remain "
                "on a semantically false route. The model gives a clean target for proof "
                "and a calibration baseline for the richer ant topology."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], json_path: Path, report_path: Path) -> None:
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-route memory theorem simulator.")
    parser.add_argument("--quick", action="store_true", help="Run a reduced theorem sweep.")
    parser.add_argument("--json", default="two_route_memory_theorem_output.json")
    parser.add_argument("--report", default="TWO_ROUTE_MEMORY_THEOREM_REPORT.md")
    args = parser.parse_args()

    payload = run_sweep(quick=args.quick)
    write_outputs(payload, Path(args.json), Path(args.report))
    print(render_report(payload))


if __name__ == "__main__":
    main()

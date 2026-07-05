"""
Seed-robust Neural V2 validation table.

This script is the publication-facing aggregation layer for the Neural V2
benchmarks. It reports seed-level confidence intervals and paired deltas
instead of only benchmark means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import numpy as np

from neural_v2_router_benchmark import (
    AdversarialSwitchConfig,
    BenchmarkConfig,
    HardBenchmarkConfig,
    make_hard_task_stream,
    make_task_stream,
    run_dte_exp3_router,
    run_dte_reliability_arbitrated_ucb_router,
    run_dte_router,
    run_exp3_router,
    run_hard_dte_exp3_router,
    run_hard_dte_reliability_arbitrated_exp3_router,
    run_hard_dte_reliability_arbitrated_ucb_router,
    run_hard_dte_router,
    run_hard_exp3_router,
    run_hard_ucb_router,
    run_ucb_router,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "neural_v2_seed_validation_output.json"
REPORT_PATH = ROOT / "NEURAL_V2_SEED_VALIDATION_REPORT.md"


@dataclass(frozen=True)
class SeedValidationConfig:
    ticks: int = 90
    shift_tick: int = 30
    batch_size: int = 100
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    hard_context_noise: float = 0.22
    hard_label_noise: float = 0.28
    hard_reward_delay: int = 8
    adversarial_switch_period: int = 8
    adversarial_label_noise: float = 0.0
    adversarial_reward_delay: int = 4
    adversarial_intensity: float = 0.65


RouterFn = Callable[[Any, int, Any], dict[str, Any]]


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / sqrt(len(values))


def _paired_delta(
    rows: list[dict[str, Any]],
    router: str,
    baseline: str = "dte_local_regret",
) -> list[float]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["router"])] = row
    deltas = []
    for seed_rows in by_seed.values():
        if baseline not in seed_rows or router not in seed_rows:
            continue
        deltas.append(
            seed_rows[baseline]["post_shift_mean_regret"]
            - seed_rows[router]["post_shift_mean_regret"]
        )
    return deltas


def _summarize_regime(
    regime: str,
    rows: list[dict[str, Any]],
    baseline: str,
) -> list[dict[str, Any]]:
    by_router: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_router.setdefault(str(row["router"]), []).append(row)
    summary = []
    for router, group in sorted(by_router.items()):
        regrets = [row["post_shift_mean_regret"] for row in group]
        rewards = [row["post_shift_mean_reward"] for row in group]
        deltas = _paired_delta(rows, router, baseline=baseline)
        summary.append(
            {
                "regime": regime,
                "router": router,
                "runs": len(group),
                "mean_regret": mean(regrets),
                "ci95_regret": _ci95(regrets),
                "mean_reward": mean(rewards),
                "ci95_reward": _ci95(rewards),
                "paired_delta_vs_local": mean(deltas) if deltas else 0.0,
                "ci95_delta_vs_local": _ci95(deltas),
            }
        )
    return summary


def _clean_rows(config: SeedValidationConfig) -> list[dict[str, Any]]:
    clean_config = BenchmarkConfig(
        ticks=config.ticks,
        shift_tick=config.shift_tick,
        batch_size=config.batch_size,
        seeds=config.seeds,
    )
    rows = []
    for seed in config.seeds:
        stream = make_task_stream(seed, clean_config)
        rows.append(run_dte_router("local_regret", stream, seed, clean_config))
        rows.append(run_dte_reliability_arbitrated_ucb_router(stream, seed, clean_config))
        rows.append(run_dte_exp3_router(stream, seed, clean_config))
        rows.append(run_ucb_router(stream, seed, clean_config))
        rows.append(run_exp3_router(stream, seed, clean_config))
    return rows


def _hard_rows(config: SeedValidationConfig) -> list[dict[str, Any]]:
    hard_config = HardBenchmarkConfig(
        ticks=config.ticks,
        shift_tick=config.shift_tick,
        batch_size=config.batch_size,
        seeds=config.seeds,
        context_noise=config.hard_context_noise,
        label_noise=config.hard_label_noise,
        reward_delay=config.hard_reward_delay,
    )
    rows = []
    for seed in config.seeds:
        stream = make_hard_task_stream(seed, hard_config)
        rows.append(run_hard_dte_router("local_regret", stream, seed, hard_config))
        rows.append(run_hard_dte_reliability_arbitrated_ucb_router(stream, seed, hard_config))
        rows.append(run_hard_dte_exp3_router(stream, seed, hard_config))
        rows.append(run_hard_dte_reliability_arbitrated_exp3_router(stream, seed, hard_config))
        rows.append(run_hard_ucb_router(stream, seed, hard_config))
        rows.append(run_hard_exp3_router(stream, seed, hard_config))
    return rows


def _adversarial_rows(config: SeedValidationConfig) -> list[dict[str, Any]]:
    switch_config = AdversarialSwitchConfig(
        ticks=config.ticks,
        shift_tick=config.shift_tick,
        batch_size=config.batch_size,
        seeds=config.seeds,
        context_noise=0.0,
        label_noise=config.adversarial_label_noise,
        reward_delay=config.adversarial_reward_delay,
        switch_period=config.adversarial_switch_period,
        adversarial_intensity=config.adversarial_intensity,
    )
    rows = []
    for seed in config.seeds:
        stream = make_hard_task_stream(seed, switch_config)
        rows.append(run_hard_dte_router("local_regret", stream, seed, switch_config))
        rows.append(run_hard_dte_reliability_arbitrated_ucb_router(stream, seed, switch_config))
        rows.append(run_hard_dte_exp3_router(stream, seed, switch_config))
        rows.append(run_hard_dte_reliability_arbitrated_exp3_router(stream, seed, switch_config))
        rows.append(run_hard_ucb_router(stream, seed, switch_config))
        rows.append(run_hard_exp3_router(stream, seed, switch_config))
    return rows


def run_seed_validation(
    config: SeedValidationConfig | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    config = config or SeedValidationConfig()
    clean = _clean_rows(config)
    hard = _hard_rows(config)
    adversarial = _adversarial_rows(config)
    summary = (
        _summarize_regime("clean", clean, baseline="dte_local_regret")
        + _summarize_regime("hard", hard, baseline="hard_dte_local_regret")
        + _summarize_regime(
            "adversarial_switch",
            adversarial,
            baseline="hard_dte_local_regret",
        )
    )
    payload = {
        "config": config.__dict__,
        "summary": summary,
        "rows": {
            "clean": clean,
            "hard": hard,
            "adversarial_switch": adversarial,
        },
    }
    if write_outputs:
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        REPORT_PATH.write_text(render_seed_validation_report(payload), encoding="utf-8")
    return payload


def render_seed_validation_report(payload: dict[str, Any]) -> str:
    config = payload["config"]
    lines = [
        "# Neural V2 Seed Validation Report",
        "",
        "## Scope",
        "",
        "Paired-seed validation for the Neural V2 controlled benchmark. Regret",
        "deltas are paired against local-regret DTE within the same task stream",
        "and seed.",
        "",
        "## Configuration",
        "",
        f"- Seeds: `{config['seeds']}`",
        f"- Ticks: `{config['ticks']}`",
        f"- Batch size: `{config['batch_size']}`",
        f"- Hard label noise: `{config['hard_label_noise']}`",
        f"- Hard reward delay: `{config['hard_reward_delay']}`",
        (
            "- Adversarial switch cell: "
            f"period `{config['adversarial_switch_period']}`, "
            f"label noise `{config['adversarial_label_noise']}`"
        ),
        "",
        "## Results",
        "",
        "| Regime | Router | Runs | Regret | CI95 | Reward | Delta vs Local | Delta CI95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['regime']} | {row['router']} | {row['runs']} | "
            f"{row['mean_regret']:.4f} | {row['ci95_regret']:.4f} | "
            f"{row['mean_reward']:.4f} | "
            f"{row['paired_delta_vs_local']:.4f} | "
            f"{row['ci95_delta_vs_local']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "Positive paired deltas mean the router reduced regret relative to",
            "local-regret DTE on the same seed. The table is designed to make the",
            "policy-arbitration boundary visible: DTE-native lanes should improve",
            "inside the DTE family in memory-sensitive regimes, while external",
            "policy owners may still dominate when the task behaves like a clean",
            "contextual bandit.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_seed_validation()
    print(json.dumps({"summary": result["summary"]}, indent=2))

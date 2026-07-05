"""
Dense budget sweep for DTE Phase 2 influence wargames.

Sweeps defender budget from 0.0 to 3.0 in steps of 0.25 across random seeds
to isolate the precise phase-transition boundary where the platform
moves from vulnerable to safe under a Q-learning adversary.

Usage:
    .venv\\Scripts\\python.exe influence_dense_sweep.py
    .venv\\Scripts\\python.exe influence_dense_sweep.py --quick
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from influence_game import GameConfig
from influence_qlearn import train_q_adversary


def _mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    array = np.array(values, dtype=np.float64)
    mean = float(array.mean())
    half_width = float(z * array.std(ddof=1) / np.sqrt(len(array)))
    return mean - half_width, mean + half_width


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _game_config(
    seed: int,
    defender_budget: float,
    agents: int,
    rounds: int,
    steps_per_round: int,
    adversary_budget: float,
) -> GameConfig:
    return GameConfig(
        rounds=rounds,
        agents=agents,
        steps_per_round=steps_per_round,
        adversary_budget=adversary_budget,
        defender_budget=defender_budget,
        seed=seed,
        dynamic_defense=True,
        trust_hysteresis=True,
        max_defender_budget=max(12.0, defender_budget + 5.0),
    )


def run_dense_sweep(
    budgets: list[float],
    seeds: int = 5,
    episodes: int = 8,
    agents: int = 96,
    rounds: int = 8,
    steps_per_round: int = 12,
    adversary_budget: float = 10.0,
    seed_base: int = 20260610,
    verbose: bool = False,
) -> dict:
    rows = []
    total_runs = seeds * len(budgets)
    current_run = 0
    
    for seed_index in range(seeds):
        seed = seed_base + seed_index
        for budget in budgets:
            current_run += 1
            if verbose:
                print(f"[{current_run}/{total_runs}] Running seed={seed}, defender_budget={budget:.2f}...")
            config = _game_config(
                seed=seed,
                defender_budget=budget,
                agents=agents,
                rounds=rounds,
                steps_per_round=steps_per_round,
                adversary_budget=adversary_budget,
            )
            payload = train_q_adversary(config, episodes=episodes)
            summary = payload["summary"]
            rows.append({
                "seed": seed,
                "defender_budget": budget,
                "episodes": episodes,
                "q_eval_peak_risk": summary["q_eval_peak_risk"],
                "q_eval_final_risk": summary["q_eval_final_risk"],
                "baseline_peak_risk": summary["baseline_peak_risk"],
                "baseline_final_risk": summary["baseline_final_risk"],
                "q_minus_baseline_final_risk": summary["q_minus_baseline_final_risk"],
                "q_threshold_crossed": summary["q_threshold_crossed"],
                "q_pause_rate": summary["q_pause_rate"],
            })
            
    return {
        "summary": summarize_dense_sweep(rows),
        "rows": rows,
        "config": {
            "budgets": budgets,
            "seeds": seeds,
            "episodes": episodes,
            "agents": agents,
            "rounds": rounds,
            "steps_per_round": steps_per_round,
            "adversary_budget": adversary_budget,
            "seed_base": seed_base,
        },
    }


def summarize_dense_sweep(rows: list[dict], risk_threshold: float = 0.18) -> dict:
    by_budget: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_budget[row["defender_budget"]].append(row)

    budget_summary = {}
    for budget, items in sorted(by_budget.items()):
        peaks = [row["q_eval_peak_risk"] for row in items]
        finals = [row["q_eval_final_risk"] for row in items]
        crossed = sum(row["q_threshold_crossed"] for row in items)
        crossing_low, crossing_high = _wilson_interval(crossed, len(items))
        peak_low, peak_high = _mean_ci(peaks)
        budget_summary[f"{budget:.2f}"] = {
            "runs": len(items),
            "mean_q_peak_risk": float(np.mean(peaks)),
            "mean_q_peak_risk_ci95": [peak_low, peak_high],
            "max_q_peak_risk": float(np.max(peaks)),
            "mean_q_final_risk": float(np.mean(finals)),
            "threshold_crossing_rate": float(crossed / max(len(items), 1)),
            "threshold_crossing_rate_ci95": [crossing_low, crossing_high],
            "mean_q_pause_rate": float(np.mean([row["q_pause_rate"] for row in items])),
        }

    ordered_budgets = sorted(by_budget)
    all_seed_safe_budget = None
    empirical_robust_safe_budget = None
    mean_safe_budget = None
    conservative_mean_ci_safe_budget = None

    for budget in ordered_budgets:
        if all(not row["q_threshold_crossed"] for row in by_budget[budget]):
            all_seed_safe_budget = budget
            break

    for idx, budget in enumerate(ordered_budgets):
        future_budgets = ordered_budgets[idx:]
        if all(not row["q_threshold_crossed"] for b in future_budgets for row in by_budget[b]):
            empirical_robust_safe_budget = budget
            break
            
    for idx, budget in enumerate(ordered_budgets):
        future_summaries = [budget_summary[f"{b:.2f}"] for b in ordered_budgets[idx:]]
        if all(row["mean_q_peak_risk"] < risk_threshold for row in future_summaries):
            mean_safe_budget = budget
            break
            
    for idx, budget in enumerate(ordered_budgets):
        future_summaries = [budget_summary[f"{b:.2f}"] for b in ordered_budgets[idx:]]
        if all(row["mean_q_peak_risk_ci95"][1] < risk_threshold for row in future_summaries):
            conservative_mean_ci_safe_budget = budget
            break

    monotonicity_violations = []
    for prev, curr in zip(ordered_budgets, ordered_budgets[1:]):
        prev_mean = budget_summary[f"{prev:.2f}"]["mean_q_peak_risk"]
        curr_mean = budget_summary[f"{curr:.2f}"]["mean_q_peak_risk"]
        if curr_mean > prev_mean + 0.005:
            monotonicity_violations.append({
                "from_budget": prev,
                "to_budget": curr,
                "previous_mean_peak": prev_mean,
                "current_mean_peak": curr_mean,
            })

    return {
        "budgets_tested": ordered_budgets,
        "seeds": len({row["seed"] for row in rows}),
        "runs": len(rows),
        "risk_threshold": risk_threshold,
        "all_seed_safe_budget": all_seed_safe_budget,
        "empirical_robust_safe_budget": empirical_robust_safe_budget,
        "mean_safe_budget": mean_safe_budget,
        "conservative_mean_ci_safe_budget": conservative_mean_ci_safe_budget,
        "monotonicity_violation_count": len(monotonicity_violations),
        "monotonicity_violations": monotonicity_violations,
        "budget_summary": budget_summary,
    }


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Dense Defender Budget Frontier Report",
        "",
        "## Scope",
        "",
        (
            "High-resolution wargame sweep of defender budget from 0.0 to 3.0 in steps of 0.25 "
            "under a Q-learning adversary. Resolves the precise phase transition where the system "
            "moves from vulnerable (threshold crossings) to safe."
        ),
        "",
        "## Summary",
        "",
        f"- Seeds: `{summary['seeds']}`",
        f"- Runs: `{summary['runs']}`",
        f"- Risk threshold: `{summary['risk_threshold']:.3f}`",
        f"- All-seed safe budget: `{summary['all_seed_safe_budget']}`",
        f"- Empirical robust safe budget: `{summary['empirical_robust_safe_budget']}`",
        f"- Mean safe budget: `{summary['mean_safe_budget']}`",
        f"- Conservative mean-CI safe budget: `{summary['conservative_mean_ci_safe_budget']}`",
        f"- Monotonicity violations: `{summary['monotonicity_violation_count']}`",
        "",
        "## Budget Frontier",
        "",
        "| Budget | Runs | Mean Peak | CI95 Peak | Max Peak | Crossing Rate | Mean Final | Mean Pause |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in summary["budgets_tested"]:
        row = summary["budget_summary"][f"{budget:.2f}"]
        lines.append(
            f"| {budget:.2f} | {row['runs']} | {row['mean_q_peak_risk']:.3f} | "
            f"[{row['mean_q_peak_risk_ci95'][0]:.3f}, {row['mean_q_peak_risk_ci95'][1]:.3f}] | "
            f"{row['max_q_peak_risk']:.3f} | {row['threshold_crossing_rate']:.1%} | "
            f"{row['mean_q_final_risk']:.3f} | {row['mean_q_pause_rate']:.1%} |"
        )
    lines.extend([
        "",
        "## Interpretation Guardrail",
        "",
        (
            "The `0.25` threshold is a local seeded frontier, not a universal procurement constant. "
            "Because defender actions can spend fractional remaining budget and dynamic immune response "
            "can replenish later rounds, the first nonzero budget may unlock a qualitatively different "
            "trajectory. The flat plateau above `0.25` should be treated as intervention saturation "
            "under the current action-unit and immune-cap settings."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("influence_dense_sweep_output.json"),
    output_md: Path = Path("INFLUENCE_DENSE_SWEEP_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense defender-budget sweep.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("influence_dense_sweep_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("INFLUENCE_DENSE_SWEEP_REPORT.md"))
    args = parser.parse_args()

    if args.quick:
        budgets = [0.0, 1.0, 2.0, 3.0]
        seeds = args.seeds if args.seeds is not None else 2
        episodes = args.episodes if args.episodes is not None else 3
        agents = 48
        rounds = 4
        steps_per_round = 6
    else:
        # High resolution sweep from 0.0 to 3.0 in steps of 0.25
        budgets = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
        seeds = args.seeds if args.seeds is not None else 5
        episodes = args.episodes if args.episodes is not None else 8
        agents = 96
        rounds = 8
        steps_per_round = 12

    payload = run_dense_sweep(
        budgets=budgets,
        seeds=seeds,
        episodes=episodes,
        agents=agents,
        rounds=rounds,
        steps_per_round=steps_per_round,
        verbose=args.verbose,
    )
    write_outputs(payload, args.output_json, args.output_md)
    
    summary = payload["summary"]
    print("\n=== SWEEP COMPLETED ===")
    print(json.dumps({
        "empirical_robust_safe_budget": summary["empirical_robust_safe_budget"],
        "mean_safe_budget": summary["mean_safe_budget"],
        "conservative_mean_ci_safe_budget": summary["conservative_mean_ci_safe_budget"],
    }, indent=2))


if __name__ == "__main__":
    main()

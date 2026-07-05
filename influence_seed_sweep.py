"""
Seed-robust Phase 2 influence-game sweeps.

This harness separates institutional evidence from single-seed anecdotes. It
tests the Q-adversary defender-budget frontier across random seeds and compares
fixed defender doctrines against the Exp3 no-regret defender.

Usage:
    .venv\\Scripts\\python.exe influence_seed_sweep.py
    .venv\\Scripts\\python.exe influence_seed_sweep.py --quick
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from influence_budget_sweep import DEFAULT_BUDGETS
from influence_defender_learning import run_exp3_defender_game
from influence_game import GameConfig, run_game
from influence_qlearn import train_q_adversary


DEFAULT_POLICIES = [
    "none",
    "risk_threshold",
    "structural_warning",
    "off_ramp_first",
    "combined_first",
    "exp3_defender",
]


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


def run_q_budget_seed_sweep(
    budgets: list[float] | None = None,
    seeds: int = 5,
    episodes: int = 8,
    agents: int = 96,
    rounds: int = 8,
    steps_per_round: int = 12,
    adversary_budget: float = 10.0,
    seed_base: int = 20260610,
) -> dict:
    budgets = budgets or DEFAULT_BUDGETS
    rows = []
    for seed_index in range(seeds):
        seed = seed_base + seed_index
        for budget in budgets:
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
                "q_action_counts": summary["q_action_counts"],
            })
    return {
        "summary": summarize_q_budget_seed_sweep(rows),
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


def summarize_q_budget_seed_sweep(rows: list[dict], risk_threshold: float = 0.18) -> dict:
    by_budget: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_budget[row["defender_budget"]].append(row)

    budget_summary = {}
    action_counter: Counter[str] = Counter()
    for budget, items in sorted(by_budget.items()):
        peaks = [row["q_eval_peak_risk"] for row in items]
        finals = [row["q_eval_final_risk"] for row in items]
        baseline_peaks = [row["baseline_peak_risk"] for row in items]
        crossed = sum(row["q_threshold_crossed"] for row in items)
        crossing_low, crossing_high = _wilson_interval(crossed, len(items))
        peak_low, peak_high = _mean_ci(peaks)
        for row in items:
            action_counter.update(row["q_action_counts"])
        budget_summary[str(budget)] = {
            "runs": len(items),
            "mean_q_peak_risk": float(np.mean(peaks)),
            "mean_q_peak_risk_ci95": [peak_low, peak_high],
            "max_q_peak_risk": float(np.max(peaks)),
            "mean_q_final_risk": float(np.mean(finals)),
            "mean_baseline_peak_risk": float(np.mean(baseline_peaks)),
            "threshold_crossing_rate": float(crossed / max(len(items), 1)),
            "threshold_crossing_rate_ci95": [crossing_low, crossing_high],
            "mean_q_pause_rate": float(np.mean([row["q_pause_rate"] for row in items])),
        }

    ordered_budgets = sorted(by_budget)
    all_seed_safe_budget = None
    empirical_robust_safe_budget = None
    mean_safe_budget = None
    conservative_mean_ci_safe_budget = None
    for idx, budget in enumerate(ordered_budgets):
        future_budgets = ordered_budgets[idx:]
        if all(not row["q_threshold_crossed"] for row in by_budget[budget]):
            all_seed_safe_budget = budget if all_seed_safe_budget is None else all_seed_safe_budget
        if all(not row["q_threshold_crossed"] for b in future_budgets for row in by_budget[b]):
            empirical_robust_safe_budget = budget
            break
    for idx, budget in enumerate(ordered_budgets):
        future_summaries = [budget_summary[str(b)] for b in ordered_budgets[idx:]]
        if all(row["mean_q_peak_risk"] < risk_threshold for row in future_summaries):
            mean_safe_budget = budget
            break
    for idx, budget in enumerate(ordered_budgets):
        future_summaries = [budget_summary[str(b)] for b in ordered_budgets[idx:]]
        if all(row["mean_q_peak_risk_ci95"][1] < risk_threshold for row in future_summaries):
            conservative_mean_ci_safe_budget = budget
            break

    monotonicity_violations = []
    for prev, curr in zip(ordered_budgets, ordered_budgets[1:]):
        prev_mean = budget_summary[str(prev)]["mean_q_peak_risk"]
        curr_mean = budget_summary[str(curr)]["mean_q_peak_risk"]
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
        "aggregate_q_action_counts": dict(action_counter),
        "budget_summary": budget_summary,
    }


def run_policy_seed_sweep(
    policies: list[str] | None = None,
    seeds: int = 5,
    agents: int = 96,
    rounds: int = 8,
    steps_per_round: int = 12,
    adversary_budget: float = 10.0,
    defender_budget: float = 5.0,
    seed_base: int = 20260710,
) -> dict:
    policies = policies or DEFAULT_POLICIES
    rows = []
    for seed_index in range(seeds):
        seed = seed_base + seed_index
        config = _game_config(
            seed=seed,
            defender_budget=defender_budget,
            agents=agents,
            rounds=rounds,
            steps_per_round=steps_per_round,
            adversary_budget=adversary_budget,
        )
        for policy in policies:
            if policy == "exp3_defender":
                payload = run_exp3_defender_game(config, adversary_policy_name="escalating")
            else:
                payload = run_game(
                    config,
                    adversary_policy_name="escalating",
                    defender_policy_name=policy,
                )
            summary = payload["summary"]
            rows.append({
                "seed": seed,
                "policy": policy,
                "threshold_crossed": summary["threshold_crossed"],
                "peak_risk": summary["peak_risk"],
                "final_risk": summary["final_risk"],
                "mean_risk": summary["mean_risk"],
                "final_escape_probability": summary["final_escape_probability"],
                "total_defender_cost": summary["total_defender_cost"],
                "immune_budget_added": summary.get("total_immune_budget_added", 0.0),
            })
    return {
        "summary": summarize_policy_seed_sweep(rows),
        "rows": rows,
        "config": {
            "policies": policies,
            "seeds": seeds,
            "agents": agents,
            "rounds": rounds,
            "steps_per_round": steps_per_round,
            "adversary_budget": adversary_budget,
            "defender_budget": defender_budget,
            "seed_base": seed_base,
        },
    }


def summarize_policy_seed_sweep(rows: list[dict]) -> dict:
    by_policy: dict[str, list[dict]] = defaultdict(list)
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)
        by_seed[row["seed"]].append(row)

    policy_summary = {}
    for policy, items in sorted(by_policy.items()):
        peak_low, peak_high = _mean_ci([row["peak_risk"] for row in items])
        crossed = sum(row["threshold_crossed"] for row in items)
        crossing_low, crossing_high = _wilson_interval(crossed, len(items))
        policy_summary[policy] = {
            "runs": len(items),
            "mean_peak_risk": float(np.mean([row["peak_risk"] for row in items])),
            "mean_peak_risk_ci95": [peak_low, peak_high],
            "max_peak_risk": float(np.max([row["peak_risk"] for row in items])),
            "mean_final_risk": float(np.mean([row["final_risk"] for row in items])),
            "mean_escape": float(np.mean([row["final_escape_probability"] for row in items])),
            "mean_defender_cost": float(np.mean([row["total_defender_cost"] for row in items])),
            "threshold_crossing_rate": float(crossed / max(len(items), 1)),
            "threshold_crossing_rate_ci95": [crossing_low, crossing_high],
        }

    best_counts: Counter[str] = Counter()
    for items in by_seed.values():
        best = min(
            items,
            key=lambda row: (
                row["threshold_crossed"],
                row["final_risk"],
                row["mean_risk"],
                row["total_defender_cost"],
            ),
        )
        best_counts[best["policy"]] += 1

    fixed_policies = [policy for policy in by_policy if policy != "exp3_defender"]
    exp3 = policy_summary.get("exp3_defender")
    best_fixed_final = min(
        (policy_summary[policy]["mean_final_risk"] for policy in fixed_policies),
        default=None,
    )
    exp3_minus_best_fixed_final = (
        exp3["mean_final_risk"] - best_fixed_final
        if exp3 is not None and best_fixed_final is not None
        else None
    )

    return {
        "seeds": len(by_seed),
        "runs": len(rows),
        "policy_summary": policy_summary,
        "best_policy_counts": dict(best_counts),
        "exp3_minus_best_fixed_final_risk": exp3_minus_best_fixed_final,
    }


def run_seed_robust_sweeps(
    budgets: list[float] | None = None,
    policies: list[str] | None = None,
    seeds: int = 5,
    episodes: int = 8,
    agents: int = 96,
    rounds: int = 8,
    steps_per_round: int = 12,
) -> dict:
    return {
        "q_budget": run_q_budget_seed_sweep(
            budgets=budgets,
            seeds=seeds,
            episodes=episodes,
            agents=agents,
            rounds=rounds,
            steps_per_round=steps_per_round,
        ),
        "policy": run_policy_seed_sweep(
            policies=policies,
            seeds=seeds,
            agents=agents,
            rounds=rounds,
            steps_per_round=steps_per_round,
        ),
    }


def render_report(payload: dict) -> str:
    q_summary = payload["q_budget"]["summary"]
    policy_summary = payload["policy"]["summary"]
    lines = [
        "# Seed-Robust Influence Sweep Report",
        "",
        "## Q-Adversary Budget Robustness",
        "",
        f"- Seeds: `{q_summary['seeds']}`",
        f"- Runs: `{q_summary['runs']}`",
        f"- Risk threshold: `{q_summary['risk_threshold']:.3f}`",
        f"- All-seed safe budget: `{q_summary['all_seed_safe_budget']}`",
        f"- Empirical robust safe budget: `{q_summary['empirical_robust_safe_budget']}`",
        f"- Mean-safe budget: `{q_summary['mean_safe_budget']}`",
        f"- Conservative mean-CI safe budget: `{q_summary['conservative_mean_ci_safe_budget']}`",
        f"- Monotonicity violations: `{q_summary['monotonicity_violation_count']}`",
        f"- Aggregate Q action counts: `{q_summary['aggregate_q_action_counts']}`",
        "",
        "## Budget Frontier",
        "",
        "| Budget | Runs | Mean Peak | CI95 Peak | Max Peak | Crossing Rate | Mean Final | Mean Pause |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in q_summary["budgets_tested"]:
        row = q_summary["budget_summary"][str(budget)]
        lines.append(
            f"| {budget:.1f} | {row['runs']} | {row['mean_q_peak_risk']:.3f} | "
            f"[{row['mean_q_peak_risk_ci95'][0]:.3f}, {row['mean_q_peak_risk_ci95'][1]:.3f}] | "
            f"{row['max_q_peak_risk']:.3f} | {row['threshold_crossing_rate']:.1%} | "
            f"{row['mean_q_final_risk']:.3f} | {row['mean_q_pause_rate']:.1%} |"
        )

    lines.extend([
        "",
        "## Defender Doctrine Robustness",
        "",
        f"- Seeds: `{policy_summary['seeds']}`",
        f"- Runs: `{policy_summary['runs']}`",
        f"- Exp3 minus best fixed final risk: `{policy_summary['exp3_minus_best_fixed_final_risk']}`",
        "",
        "| Policy | Runs | Mean Peak | CI95 Peak | Max Peak | Crossing Rate | Mean Final | Mean Escape | Mean Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for policy, row in policy_summary["policy_summary"].items():
        lines.append(
            f"| {policy} | {row['runs']} | {row['mean_peak_risk']:.3f} | "
            f"[{row['mean_peak_risk_ci95'][0]:.3f}, {row['mean_peak_risk_ci95'][1]:.3f}] | "
            f"{row['max_peak_risk']:.3f} | {row['threshold_crossing_rate']:.1%} | "
            f"{row['mean_final_risk']:.3f} | {row['mean_escape']:.3f} | "
            f"{row['mean_defender_cost']:.2f} |"
        )

    lines.extend([
        "",
        "## Best Doctrine Counts",
        "",
        "| Policy | Seeds Won |",
        "|---|---:|",
    ])
    for policy, count in sorted(
        policy_summary["best_policy_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"| {policy} | {count} |")
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("influence_seed_sweep_output.json"),
    output_md: Path = Path("INFLUENCE_SEED_SWEEP_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seed-robust Phase 2 influence sweeps.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--steps-per-round", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("influence_seed_sweep_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("INFLUENCE_SEED_SWEEP_REPORT.md"))
    args = parser.parse_args()

    payload = run_seed_robust_sweeps(
        budgets=[0.0, 5.0, 10.0] if args.quick else DEFAULT_BUDGETS,
        seeds=args.seeds if args.seeds is not None else (2 if args.quick else 5),
        episodes=args.episodes if args.episodes is not None else (2 if args.quick else 8),
        agents=args.agents if args.agents is not None else (48 if args.quick else 96),
        rounds=args.rounds if args.rounds is not None else (4 if args.quick else 8),
        steps_per_round=args.steps_per_round if args.steps_per_round is not None else (6 if args.quick else 12),
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "q_budget": {
            "runs": payload["q_budget"]["summary"]["runs"],
            "empirical_robust_safe_budget": payload["q_budget"]["summary"]["empirical_robust_safe_budget"],
            "mean_safe_budget": payload["q_budget"]["summary"]["mean_safe_budget"],
            "monotonicity_violations": payload["q_budget"]["summary"]["monotonicity_violation_count"],
        },
        "policy": {
            "runs": payload["policy"]["summary"]["runs"],
            "best_policy_counts": payload["policy"]["summary"]["best_policy_counts"],
            "exp3_minus_best_fixed_final_risk": payload["policy"]["summary"]["exp3_minus_best_fixed_final_risk"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()

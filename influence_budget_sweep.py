"""
Minimum defender-budget sweep under a Q-learning adversary.

This estimates the resilience threshold: the smallest starting defender
budget that keeps Q-adversary peak risk below the civic-risk threshold, plus
the plateau point where additional defense gives diminishing returns.

Usage:
    .venv\\Scripts\\python.exe influence_budget_sweep.py
    .venv\\Scripts\\python.exe influence_budget_sweep.py --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from influence_game import GameConfig
from influence_qlearn import train_q_adversary


DEFAULT_BUDGETS = [0.0, 2.5, 5.0, 7.5, 10.0]


def _plateau_budget(rows: list[dict], epsilon: float = 0.01) -> float | None:
    ordered = sorted(rows, key=lambda row: row["defender_budget"])
    for idx, row in enumerate(ordered):
        future = ordered[idx + 1 :]
        if any(item["q_threshold_crossed"] for item in ordered[idx:]):
            continue
        if not future:
            return row["defender_budget"]
        improvement = row["q_eval_peak_risk"] - min(item["q_eval_peak_risk"] for item in future)
        if improvement <= epsilon:
            return row["defender_budget"]
    return None


def run_budget_sweep(
    budgets: list[float] | None = None,
    episodes: int = 12,
    agents: int = 96,
    rounds: int = 8,
    steps_per_round: int = 12,
) -> dict:
    budgets = budgets or DEFAULT_BUDGETS
    rows = []
    for idx, budget in enumerate(budgets):
        config = GameConfig(
            rounds=rounds,
            agents=agents,
            steps_per_round=steps_per_round,
            adversary_budget=10.0,
            defender_budget=budget,
            seed=20260603 + 1000 * idx,
            dynamic_defense=True,
            trust_hysteresis=True,
            max_defender_budget=max(12.0, budget + 5.0),
        )
        payload = train_q_adversary(config, episodes=episodes)
        summary = payload["summary"]
        rows.append({
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
        "summary": summarize_budget_sweep(rows),
        "rows": rows,
        "config": {
            "budgets": budgets,
            "episodes": episodes,
            "agents": agents,
            "rounds": rounds,
            "steps_per_round": steps_per_round,
        },
    }


def summarize_budget_sweep(rows: list[dict], risk_threshold: float = 0.18) -> dict:
    ordered = sorted(rows, key=lambda row: row["defender_budget"])
    safe_rows = [row for row in rows if row["q_eval_peak_risk"] < risk_threshold]
    min_safe_budget = min((row["defender_budget"] for row in safe_rows), default=None)
    robust_safe_budget = None
    for idx, row in enumerate(ordered):
        if all(item["q_eval_peak_risk"] < risk_threshold for item in ordered[idx:]):
            robust_safe_budget = row["defender_budget"]
            break
    return {
        "budgets_tested": [row["defender_budget"] for row in rows],
        "min_safe_budget": min_safe_budget,
        "robust_safe_budget": robust_safe_budget,
        "plateau_budget": _plateau_budget(rows),
        "max_q_peak_risk": max((row["q_eval_peak_risk"] for row in rows), default=0.0),
        "min_q_peak_risk": min((row["q_eval_peak_risk"] for row in rows), default=0.0),
        "threshold_crossing_budgets": [
            row["defender_budget"] for row in rows if row["q_threshold_crossed"]
        ],
    }


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Minimum Defender Budget Report",
        "",
        "## Summary",
        "",
        f"- Budgets tested: `{summary['budgets_tested']}`",
        f"- Minimum safe budget: `{summary['min_safe_budget']}`",
        f"- Robust safe budget: `{summary['robust_safe_budget']}`",
        f"- Plateau budget: `{summary['plateau_budget']}`",
        f"- Max Q peak risk: `{summary['max_q_peak_risk']:.3f}`",
        f"- Min Q peak risk: `{summary['min_q_peak_risk']:.3f}`",
        "",
        "## Budget Curve",
        "",
        "| Defender Budget | Q Peak Risk | Q Final Risk | Baseline Peak | Q-Baseline Final | Q Pause Rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['defender_budget']:.1f} | {row['q_eval_peak_risk']:.3f} | "
            f"{row['q_eval_final_risk']:.3f} | {row['baseline_peak_risk']:.3f} | "
            f"{row['q_minus_baseline_final_risk']:+.3f} | {row['q_pause_rate']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("influence_budget_sweep_output.json"),
    output_md: Path = Path("INFLUENCE_BUDGET_SWEEP_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run min defender-budget sweep.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("influence_budget_sweep_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("INFLUENCE_BUDGET_SWEEP_REPORT.md"))
    args = parser.parse_args()

    payload = run_budget_sweep(
        budgets=[0.0, 5.0, 10.0] if args.quick else DEFAULT_BUDGETS,
        episodes=args.episodes if args.episodes is not None else (3 if args.quick else 12),
        agents=48 if args.quick else 96,
        rounds=4 if args.quick else 8,
        steps_per_round=6 if args.quick else 12,
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

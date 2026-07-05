"""
Legitimacy war-game sweeps for the repeated DTE influence game.

The harness searches for failure regimes, compares defender doctrines, and
estimates where structural warning is actually useful rather than merely
plausible in a single scenario.

Usage:
    .venv\\Scripts\\python.exe influence_wargame.py
    .venv\\Scripts\\python.exe influence_wargame.py --quick
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from influence_game import GameConfig, run_game


DEFAULT_LAMBDAS = [0.20, 0.35, 0.50]
DEFAULT_TAUS = [0.45, 0.65, 0.90]
DEFAULT_ADVERSARY_BUDGETS = [3.0, 6.0, 8.0]
DEFAULT_DEFENDER_BUDGETS = [0.0, 3.0, 5.0, 7.0]
DEFAULT_DEFENDER_POLICIES = [
    "none",
    "risk_threshold",
    "structural_warning",
    "off_ramp_first",
    "prebunk_first",
    "throttle_first",
    "combined_first",
]


@dataclass(frozen=True)
class WarGameCell:
    feedback_rate: float
    temperature: float
    adversary_budget: float
    defender_budget: float
    seed: int

    def key(self) -> tuple:
        return (
            self.feedback_rate,
            self.temperature,
            self.adversary_budget,
            self.defender_budget,
            self.seed,
        )


def _game_config(
    cell: WarGameCell,
    agents: int,
    rounds: int,
    steps_per_round: int,
    dynamic_defense: bool,
    trust_hysteresis: bool,
) -> GameConfig:
    return GameConfig(
        rounds=rounds,
        steps_per_round=steps_per_round,
        agents=agents,
        feedback_rate=cell.feedback_rate,
        temperature=cell.temperature,
        adversary_budget=cell.adversary_budget,
        defender_budget=cell.defender_budget,
        seed=cell.seed,
        dynamic_defense=dynamic_defense,
        trust_hysteresis=trust_hysteresis,
    )


def _row_from_summary(cell: WarGameCell, policy: str, summary: dict) -> dict:
    return {
        "lambda": cell.feedback_rate,
        "tau": cell.temperature,
        "adversary_budget": cell.adversary_budget,
        "defender_budget": cell.defender_budget,
        "seed": cell.seed,
        "defender_policy": policy,
        "threshold_crossed": bool(summary["threshold_crossed"]),
        "institutional_result": summary["institutional_result"],
        "first_warning_round": summary["first_warning_round"],
        "first_defensive_round": summary["first_defensive_round"],
        "first_risk_alarm_round": summary["first_risk_alarm_round"],
        "peak_risk": float(summary["peak_risk"]),
        "mean_risk": float(summary["mean_risk"]),
        "final_risk": float(summary["final_risk"]),
        "final_escape_probability": float(summary["final_escape_probability"]),
        "final_detection_pressure": float(summary["final_detection_pressure"]),
        "total_adversary_cost": float(summary["total_adversary_cost"]),
        "total_defender_cost": float(summary["total_defender_cost"]),
        "adversary_budget_left": float(summary["adversary_budget_left"]),
        "defender_budget_left": float(summary["defender_budget_left"]),
        "cumulative_adversary_payoff": float(summary["cumulative_adversary_payoff"]),
        "cumulative_defender_payoff": float(summary["cumulative_defender_payoff"]),
    }


def run_wargame(
    lambdas: list[float] | None = None,
    taus: list[float] | None = None,
    adversary_budgets: list[float] | None = None,
    defender_budgets: list[float] | None = None,
    defender_policies: list[str] | None = None,
    seeds: int = 1,
    agents: int = 128,
    rounds: int = 8,
    steps_per_round: int = 16,
    dynamic_defense: bool = False,
    trust_hysteresis: bool = False,
    seed_base: int = 20260602,
) -> dict:
    lambdas = lambdas or DEFAULT_LAMBDAS
    taus = taus or DEFAULT_TAUS
    adversary_budgets = adversary_budgets or DEFAULT_ADVERSARY_BUDGETS
    defender_budgets = defender_budgets or DEFAULT_DEFENDER_BUDGETS
    defender_policies = defender_policies or DEFAULT_DEFENDER_POLICIES

    rows = []
    for seed_index in range(seeds):
        for lam in lambdas:
            for tau in taus:
                for adv_budget in adversary_budgets:
                    for def_budget in defender_budgets:
                        cell = WarGameCell(
                            feedback_rate=lam,
                            temperature=tau,
                            adversary_budget=adv_budget,
                            defender_budget=def_budget,
                            seed=seed_base + seed_index,
                        )
                        config = _game_config(
                            cell,
                            agents,
                            rounds,
                            steps_per_round,
                            dynamic_defense,
                            trust_hysteresis,
                        )
                        for policy in defender_policies:
                            summary = run_game(
                                config,
                                adversary_policy_name="escalating",
                                defender_policy_name=policy,
                            )["summary"]
                            rows.append(_row_from_summary(cell, policy, summary))

    return {
        "summary": summarize_wargame(rows),
        "runs": rows,
        "config": {
            "lambdas": lambdas,
            "taus": taus,
            "adversary_budgets": adversary_budgets,
            "defender_budgets": defender_budgets,
            "defender_policies": defender_policies,
            "seeds": seeds,
            "agents": agents,
            "rounds": rounds,
            "steps_per_round": steps_per_round,
            "dynamic_defense": dynamic_defense,
            "trust_hysteresis": trust_hysteresis,
        },
    }


def _cell_key(row: dict) -> tuple:
    return (
        row["lambda"],
        row["tau"],
        row["adversary_budget"],
        row["defender_budget"],
        row["seed"],
    )


def _best_policy(rows: list[dict]) -> dict:
    return min(
        rows,
        key=lambda row: (
            row["threshold_crossed"],
            row["final_risk"],
            row["mean_risk"],
            row["peak_risk"],
            row["total_defender_cost"],
        ),
    )


def summarize_wargame(rows: list[dict]) -> dict:
    by_policy = defaultdict(list)
    by_cell = defaultdict(list)
    for row in rows:
        by_policy[row["defender_policy"]].append(row)
        by_cell[_cell_key(row)].append(row)

    policy_summary = {}
    for policy, items in sorted(by_policy.items()):
        policy_summary[policy] = {
            "runs": len(items),
            "threshold_crossing_rate": float(np.mean([row["threshold_crossed"] for row in items])),
            "preemptive_rate": float(np.mean([row["institutional_result"] == "defense_preempted" for row in items])),
            "mean_peak_risk": float(np.mean([row["peak_risk"] for row in items])),
            "mean_final_risk": float(np.mean([row["final_risk"] for row in items])),
            "mean_defender_cost": float(np.mean([row["total_defender_cost"] for row in items])),
            "mean_escape": float(np.mean([row["final_escape_probability"] for row in items])),
        }

    best_counts = Counter()
    collapse_cases = []
    structural_failures = []
    structural_advantages = []
    dangerous_cases = []

    for key, items in by_cell.items():
        best = _best_policy(items)
        best_counts[best["defender_policy"]] += 1
        all_cross = all(row["threshold_crossed"] for row in items)
        if all_cross:
            collapse_cases.append({
                "lambda": key[0],
                "tau": key[1],
                "adversary_budget": key[2],
                "defender_budget": key[3],
                "seed": key[4],
                "best_policy": best["defender_policy"],
                "best_final_risk": best["final_risk"],
            })

        by_name = {row["defender_policy"]: row for row in items}
        none = by_name.get("none")
        risk_threshold = by_name.get("risk_threshold")
        structural = by_name.get("structural_warning")
        if none and none["peak_risk"] >= 0.16:
            dangerous_cases.append({
                "lambda": key[0],
                "tau": key[1],
                "adversary_budget": key[2],
                "defender_budget": key[3],
                "seed": key[4],
                "no_defense_peak_risk": none["peak_risk"],
                "no_defense_final_risk": none["final_risk"],
                "best_policy": best["defender_policy"],
                "best_final_risk": best["final_risk"],
            })
        if structural and risk_threshold:
            structural_advantages.append(risk_threshold["final_risk"] - structural["final_risk"])
            if structural["threshold_crossed"] or structural["final_risk"] > risk_threshold["final_risk"]:
                structural_failures.append({
                    "lambda": key[0],
                    "tau": key[1],
                    "adversary_budget": key[2],
                    "defender_budget": key[3],
                    "seed": key[4],
                    "structural_final_risk": structural["final_risk"],
                    "risk_threshold_final_risk": risk_threshold["final_risk"],
                    "structural_threshold_crossed": structural["threshold_crossed"],
                })

    return {
        "total_runs": len(rows),
        "cells": len(by_cell),
        "policy_summary": policy_summary,
        "best_policy_counts": dict(best_counts),
        "collapse_case_count": len(collapse_cases),
        "funded_collapse_case_count": len([
            row for row in collapse_cases if row["defender_budget"] > 0
        ]),
        "collapse_cases": collapse_cases[:20],
        "dangerous_case_count": len(dangerous_cases),
        "dangerous_cases": sorted(
            dangerous_cases,
            key=lambda row: row["no_defense_peak_risk"],
            reverse=True,
        )[:20],
        "structural_failure_count": len(structural_failures),
        "structural_failures": structural_failures[:20],
        "mean_structural_final_risk_advantage_vs_risk_threshold": (
            float(np.mean(structural_advantages)) if structural_advantages else 0.0
        ),
    }


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    config = payload["config"]
    lines = [
        "# Influence War Game Report",
        "",
        "## Scope",
        "",
        (
            "Regime sweep over repeated DTE influence games. The goal is to find "
            "where structural warning succeeds, where it fails, and which defender "
            "doctrines are cost-effective under adversarial topology pressure."
        ),
        "",
        "## Grid",
        "",
        f"- Runs: `{summary['total_runs']}`",
        f"- Cells: `{summary['cells']}`",
        f"- Lambdas: `{config['lambdas']}`",
        f"- Temperatures: `{config['taus']}`",
        f"- Adversary budgets: `{config['adversary_budgets']}`",
        f"- Defender budgets: `{config['defender_budgets']}`",
        f"- Defender policies: `{config['defender_policies']}`",
        f"- Dynamic defense: `{config['dynamic_defense']}`",
        f"- Trust hysteresis: `{config['trust_hysteresis']}`",
        "",
        "## Policy Summary",
        "",
        "| Policy | Runs | Crossing Rate | Preemptive Rate | Mean Peak Risk | Mean Final Risk | Mean Def Cost | Mean Escape |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, row in summary["policy_summary"].items():
        lines.append(
            f"| {policy} | {row['runs']} | {row['threshold_crossing_rate']:.1%} | "
            f"{row['preemptive_rate']:.1%} | {row['mean_peak_risk']:.3f} | "
            f"{row['mean_final_risk']:.3f} | {row['mean_defender_cost']:.2f} | "
            f"{row['mean_escape']:.3f} |"
        )

    lines.extend([
        "",
        "## Doctrine Wins",
        "",
        "| Policy | Best-Cell Count |",
        "|---|---:|",
    ])
    for policy, count in sorted(summary["best_policy_counts"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {policy} | {count} |")

    lines.extend([
        "",
        "## Failure Search",
        "",
        f"- Collapse cases where every policy crosses threshold: `{summary['collapse_case_count']}`",
        f"- Funded collapse cases where every policy crosses threshold: `{summary['funded_collapse_case_count']}`",
        f"- Dangerous no-defense cases with peak risk >= 0.16: `{summary['dangerous_case_count']}`",
        f"- Structural-warning failures vs risk-threshold policy: `{summary['structural_failure_count']}`",
        (
            "- Mean structural final-risk advantage vs risk-threshold policy: "
            f"`{summary['mean_structural_final_risk_advantage_vs_risk_threshold']:.4f}`"
        ),
    ])

    if summary["dangerous_cases"]:
        lines.extend([
            "",
            "## Top Dangerous Cases",
            "",
            "| Lambda | Tau | Adv Budget | Def Budget | No-Def Peak | Best Policy | Best Final Risk |",
            "|---:|---:|---:|---:|---:|---|---:|",
        ])
        for row in summary["dangerous_cases"][:10]:
            lines.append(
                f"| {row['lambda']:.2f} | {row['tau']:.2f} | {row['adversary_budget']:.1f} | "
                f"{row['defender_budget']:.1f} | {row['no_defense_peak_risk']:.3f} | "
                f"{row['best_policy']} | {row['best_final_risk']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("influence_wargame_output.json"),
    output_md: Path = Path("INFLUENCE_WARGAME_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DTE influence war-game sweeps.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--steps-per-round", type=int, default=None)
    parser.add_argument("--dynamic-defense", action="store_true")
    parser.add_argument("--trust-hysteresis", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("influence_wargame_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("INFLUENCE_WARGAME_REPORT.md"))
    args = parser.parse_args()

    if args.quick:
        payload = run_wargame(
            lambdas=[0.35, 0.50],
            taus=[0.45, 0.65],
            adversary_budgets=[3.0, 5.0],
            defender_budgets=[0.0, 5.0],
            defender_policies=["none", "risk_threshold", "structural_warning", "off_ramp_first"],
            seeds=args.seeds,
            agents=args.agents if args.agents is not None else 64,
            rounds=args.rounds if args.rounds is not None else 5,
            steps_per_round=args.steps_per_round if args.steps_per_round is not None else 8,
            dynamic_defense=args.dynamic_defense,
            trust_hysteresis=args.trust_hysteresis,
        )
    elif args.stress:
        payload = run_wargame(
            lambdas=[0.50, 0.65],
            taus=[0.25, 0.45],
            adversary_budgets=[4.0, 8.0, 12.0],
            defender_budgets=[0.0, 3.0, 7.0],
            defender_policies=[
                "none",
                "risk_threshold",
                "structural_warning",
                "off_ramp_first",
                "combined_first",
            ],
            seeds=args.seeds,
            agents=args.agents if args.agents is not None else 128,
            rounds=args.rounds if args.rounds is not None else 12,
            steps_per_round=args.steps_per_round if args.steps_per_round is not None else 16,
            dynamic_defense=args.dynamic_defense,
            trust_hysteresis=args.trust_hysteresis,
        )
    else:
        payload = run_wargame(
            seeds=args.seeds,
            agents=args.agents if args.agents is not None else 128,
            rounds=args.rounds if args.rounds is not None else 8,
            steps_per_round=args.steps_per_round if args.steps_per_round is not None else 16,
            dynamic_defense=args.dynamic_defense,
            trust_hysteresis=args.trust_hysteresis,
        )

    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": payload["summary"]["total_runs"],
        "cells": payload["summary"]["cells"],
        "best_policy_counts": payload["summary"]["best_policy_counts"],
        "collapse_case_count": payload["summary"]["collapse_case_count"],
        "structural_failure_count": payload["summary"]["structural_failure_count"],
        "structural_advantage": payload["summary"]["mean_structural_final_risk_advantage_vs_risk_threshold"],
    }, indent=2))


if __name__ == "__main__":
    main()

"""
No-regret defender harness for the DTE influence game.

The defender uses Exp3-style multiplicative weights over abstract defense
doctrines. This is a defensive platform-learning model: it adapts topology
repair choices from realized aggregate outcomes.

Usage:
    .venv\\Scripts\\python.exe influence_defender_learning.py
    .venv\\Scripts\\python.exe influence_defender_learning.py --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from influence_game import (
    GameConfig,
    adversary_budget_spend,
    adversary_detection_increment,
    adversary_policy,
    apply_adversary_action,
    apply_defender_action,
    apply_immune_response,
    build_kernel,
    initialize_state,
    simulate_round,
    summarize_game,
)


LEARNER_ACTIONS = ("observe", "off_ramp", "prebunk", "throttle", "combined")


def _probabilities(weights: np.ndarray, gamma: float) -> np.ndarray:
    normalized = weights / weights.sum()
    return (1.0 - gamma) * normalized + gamma / len(weights)


def _defender_reward(metrics: dict, defender_cost: float) -> float:
    raw = (
        0.45 * (1.0 - metrics["risk_share"])
        + 0.35 * metrics["escape_probability"]
        + 0.20 * metrics["protective_share"]
        - 0.03 * defender_cost
    )
    return float(np.clip(raw, 0.0, 1.0))


def run_exp3_defender_game(
    config: GameConfig | None = None,
    adversary_policy_name: str = "escalating",
    gamma: float = 0.15,
    eta: float = 0.35,
) -> dict:
    config = config or GameConfig(dynamic_defense=True, trust_hysteresis=True)
    kernel = build_kernel(config)
    labels = kernel.topo.labels
    rng = np.random.default_rng(config.seed)
    state = initialize_state(config)
    weights = np.ones(len(LEARNER_ACTIONS), dtype=np.float64)
    rounds = []

    for round_index in range(config.rounds):
        state.round_index = round_index
        immune_budget_added = apply_immune_response(state, config)
        adv_action = adversary_policy(state, config, adversary_policy_name)
        probs = _probabilities(weights, gamma)
        action_idx = int(rng.choice(len(LEARNER_ACTIONS), p=probs))
        def_action = LEARNER_ACTIONS[action_idx]
        if state.defender_budget_left <= 0:
            def_action = "observe"

        adv_unit = min(adversary_budget_spend(adv_action, config.action_unit), state.adversary_budget_left)
        def_unit = min(config.action_unit, state.defender_budget_left)
        adv_cost = apply_adversary_action(kernel, labels, adv_action, adv_unit, state.detection_pressure)
        def_cost = apply_defender_action(kernel, labels, def_action, def_unit)
        if adv_action != "pause":
            state.adversary_budget_left = max(0.0, state.adversary_budget_left - adv_unit)
        if def_action != "observe":
            state.defender_budget_left = max(0.0, state.defender_budget_left - def_unit)

        metrics = simulate_round(kernel, labels, state, config, rng)
        reward = _defender_reward(metrics, def_cost)
        estimated_reward = reward / max(probs[action_idx], 1e-9)
        weights[action_idx] *= np.exp(eta * estimated_reward / len(LEARNER_ACTIONS))
        weights = np.clip(weights, 1e-9, 1e9)

        anomaly = (
            0.45 * metrics["warning_score"]
            + 0.35 * max(0.0, metrics["risk_share"] - 0.10) / max(config.risk_threshold - 0.10, 1e-9)
            + 0.20 * (1.0 if def_action in {"throttle", "combined"} else 0.0)
        )
        state.detection_pressure = (
            config.detection_decay * state.detection_pressure
            + anomaly
            + adversary_detection_increment(adv_action)
        )

        adv_payoff = (
            metrics["risk_share"]
            + 0.60 * metrics["target_share"]
            + 0.35 * metrics["polarization_gap"]
            - 0.03 * adv_cost
            - 0.04 * state.detection_pressure
        )
        def_payoff = (
            -metrics["risk_share"]
            + 0.35 * metrics["protective_share"]
            + 0.25 * metrics["escape_probability"]
            - 0.025 * def_cost
        )
        row = {
            "round": round_index,
            "adversary_action": adv_action,
            "defender_action": def_action,
            "action_probabilities": {
                action: float(prob) for action, prob in zip(LEARNER_ACTIONS, probs)
            },
            "weights": {
                action: float(weight) for action, weight in zip(LEARNER_ACTIONS, weights)
            },
            "defender_reward": reward,
            "adversary_cost": float(adv_cost),
            "defender_cost": float(def_cost),
            "immune_budget_added": float(immune_budget_added),
            "adversary_budget_left": float(state.adversary_budget_left),
            "defender_budget_left": float(state.defender_budget_left),
            "detection_pressure": float(state.detection_pressure),
            "adversary_payoff": float(adv_payoff),
            "defender_payoff": float(def_payoff),
            **metrics,
        }
        rounds.append(row)
        state.previous_metrics = metrics

    summary = summarize_game(rounds, state, config, adversary_policy_name, "exp3_defender")
    summary["final_action_probabilities"] = rounds[-1]["action_probabilities"] if rounds else {}
    summary["final_weights"] = rounds[-1]["weights"] if rounds else {}
    summary["mean_defender_reward"] = float(np.mean([row["defender_reward"] for row in rounds])) if rounds else 0.0
    return {"summary": summary, "rounds": rounds}


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# No-Regret Defender Report",
        "",
        "## Summary",
        "",
        f"- Peak risk: `{summary['peak_risk']:.3f}`",
        f"- Final risk: `{summary['final_risk']:.3f}`",
        f"- Final escape probability: `{summary['final_escape_probability']:.3f}`",
        f"- Mean defender reward: `{summary['mean_defender_reward']:.3f}`",
        f"- Total defender cost: `{summary['total_defender_cost']:.2f}`",
        f"- Final action probabilities: `{summary['final_action_probabilities']}`",
        "",
        "## Rounds",
        "",
        "| Round | Adv | Defender | Risk | Escape | Reward | Def Cost |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["rounds"]:
        lines.append(
            f"| {row['round']} | {row['adversary_action']} | {row['defender_action']} | "
            f"{row['risk_share']:.3f} | {row['escape_probability']:.3f} | "
            f"{row['defender_reward']:.3f} | {row['defender_cost']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("influence_defender_learning_output.json"),
    output_md: Path = Path("INFLUENCE_DEFENDER_LEARNING_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp3 no-regret defender game.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--steps-per-round", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("influence_defender_learning_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("INFLUENCE_DEFENDER_LEARNING_REPORT.md"))
    args = parser.parse_args()

    config = GameConfig(
        rounds=args.rounds if args.rounds is not None else (6 if args.quick else 12),
        agents=args.agents if args.agents is not None else (64 if args.quick else 160),
        steps_per_round=args.steps_per_round if args.steps_per_round is not None else (8 if args.quick else 16),
        adversary_budget=6.0 if args.quick else 10.0,
        defender_budget=5.0,
        dynamic_defense=True,
        trust_hysteresis=True,
    )
    payload = run_exp3_defender_game(config)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

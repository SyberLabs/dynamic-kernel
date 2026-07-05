"""
Tabular Q-learning adversary for the defensive DTE influence game.

The learner operates only over abstract campaign actions:
pause, amplify, suppress_escape, polarize. It is a stress-testing tool for
defender policy evaluation, not an operational influence system.

Usage:
    .venv\\Scripts\\python.exe influence_qlearn.py
    .venv\\Scripts\\python.exe influence_qlearn.py --quick
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from influence_game import (
    ADVERSARY_ACTIONS,
    GameConfig,
    adversary_budget_spend,
    adversary_detection_increment,
    apply_adversary_action,
    apply_defender_action,
    apply_immune_response,
    build_kernel,
    defender_policy,
    initialize_state,
    run_game,
    simulate_round,
    summarize_game,
)


QTable = dict[str, dict[str, float]]


def _bin(value: float, cuts: tuple[float, ...]) -> int:
    for idx, cut in enumerate(cuts):
        if value < cut:
            return idx
    return len(cuts)


def q_state_key(metrics: dict | None, adversary_budget_left: float, detection_pressure: float) -> str:
    metrics = metrics or {}
    risk_bin = _bin(float(metrics.get("risk_share", 0.0)), (0.10, 0.16, 0.18))
    escape_bin = _bin(float(metrics.get("escape_probability", 1.0)), (0.45, 0.55, 0.65))
    warning_bin = _bin(float(metrics.get("warning_score", 0.0)), (0.45, 0.50, 0.60))
    detection_bin = _bin(float(detection_pressure), (0.75, 1.50, 2.50))
    budget_bin = _bin(float(adversary_budget_left), (1.0, 4.0, 8.0))
    return f"r{risk_bin}:e{escape_bin}:w{warning_bin}:d{detection_bin}:b{budget_bin}"


def _ensure_state(q_table: QTable, key: str) -> dict[str, float]:
    if key not in q_table:
        q_table[key] = {action: 0.0 for action in ADVERSARY_ACTIONS}
    return q_table[key]


def choose_action(q_table: QTable, key: str, rng: np.random.Generator, epsilon: float) -> str:
    values = _ensure_state(q_table, key)
    if rng.random() < epsilon:
        return str(rng.choice(ADVERSARY_ACTIONS))
    best_value = max(values.values())
    best_actions = [action for action, value in values.items() if value == best_value]
    return str(rng.choice(best_actions))


def _adversary_reward(
    metrics: dict,
    adv_cost: float,
    detection_pressure: float,
    detection_delta: float,
    risk_threshold: float,
    is_terminal: bool,
    terminal_bonus: float,
    detection_tax_mode: str,
) -> float:
    detection_tax = detection_delta if detection_tax_mode == "delta" else detection_pressure
    reward = (
        metrics["risk_share"]
        + 0.60 * metrics["target_share"]
        + 0.35 * metrics["polarization_gap"]
        - 0.03 * adv_cost
        - 0.04 * detection_tax
    )
    if is_terminal and metrics["risk_share"] >= risk_threshold:
        reward += terminal_bonus
    return float(reward)


def run_q_episode(
    config: GameConfig,
    q_table: QTable,
    epsilon: float = 0.10,
    alpha: float = 0.25,
    gamma: float = 0.90,
    defender_policy_name: str = "structural_warning",
    terminal_bonus: float = 10.0,
    detection_tax_mode: str = "delta",
    train: bool = True,
    seed_offset: int = 0,
) -> dict:
    episode_config = GameConfig(
        rounds=config.rounds,
        steps_per_round=config.steps_per_round,
        agents=config.agents,
        feedback_rate=config.feedback_rate,
        temperature=config.temperature,
        noise_sigma=config.noise_sigma,
        seed=config.seed + seed_offset,
        adversary_budget=config.adversary_budget,
        defender_budget=config.defender_budget,
        action_unit=config.action_unit,
        risk_threshold=config.risk_threshold,
        warning_threshold=config.warning_threshold,
        detection_decay=config.detection_decay,
        dynamic_defense=config.dynamic_defense,
        immune_response_gain=config.immune_response_gain,
        immune_response_cap=config.immune_response_cap,
        max_defender_budget=config.max_defender_budget,
        trust_hysteresis=config.trust_hysteresis,
        trust_floor=config.trust_floor,
        trust_recovery_slowdown=config.trust_recovery_slowdown,
        cohorts=config.cohorts,
    )
    kernel = build_kernel(episode_config)
    labels = kernel.topo.labels
    rng = np.random.default_rng(episode_config.seed)
    state = initialize_state(episode_config)
    rounds = []

    for round_index in range(episode_config.rounds):
        state.round_index = round_index
        immune_budget_added = apply_immune_response(state, episode_config)
        key = q_state_key(state.previous_metrics, state.adversary_budget_left, state.detection_pressure)
        adv_action = choose_action(q_table, key, rng, epsilon if train else 0.0)
        if state.adversary_budget_left <= 0:
            adv_action = "pause"
        def_action = defender_policy(state, episode_config, defender_policy_name)

        adv_unit = min(adversary_budget_spend(adv_action, episode_config.action_unit), state.adversary_budget_left)
        def_unit = min(episode_config.action_unit, state.defender_budget_left)
        adv_cost = apply_adversary_action(kernel, labels, adv_action, adv_unit, state.detection_pressure)
        def_cost = apply_defender_action(kernel, labels, def_action, def_unit)
        if adv_action != "pause":
            state.adversary_budget_left = max(0.0, state.adversary_budget_left - adv_unit)
        if def_action != "observe":
            state.defender_budget_left = max(0.0, state.defender_budget_left - def_unit)

        metrics = simulate_round(kernel, labels, state, episode_config, rng)
        anomaly = (
            0.45 * metrics["warning_score"]
            + 0.35 * max(0.0, metrics["risk_share"] - 0.10) / max(episode_config.risk_threshold - 0.10, 1e-9)
            + 0.20 * (1.0 if def_action in {"throttle", "combined"} else 0.0)
        )
        previous_detection = state.detection_pressure
        state.detection_pressure = (
            episode_config.detection_decay * state.detection_pressure
            + anomaly
            + adversary_detection_increment(adv_action)
        )
        detection_delta = max(0.0, state.detection_pressure - previous_detection)
        reward = _adversary_reward(
            metrics,
            adv_cost,
            state.detection_pressure,
            detection_delta,
            episode_config.risk_threshold,
            round_index == episode_config.rounds - 1,
            terminal_bonus,
            detection_tax_mode,
        )
        next_key = q_state_key(metrics, state.adversary_budget_left, state.detection_pressure)
        if train:
            current = _ensure_state(q_table, key)[adv_action]
            next_best = max(_ensure_state(q_table, next_key).values())
            q_table[key][adv_action] = current + alpha * (reward + gamma * next_best - current)

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
            "adversary_cost": float(adv_cost),
            "defender_cost": float(def_cost),
            "immune_budget_added": float(immune_budget_added),
            "adversary_budget_left": float(state.adversary_budget_left),
            "defender_budget_left": float(state.defender_budget_left),
            "detection_pressure": float(state.detection_pressure),
            "adversary_payoff": float(reward),
            "defender_payoff": float(def_payoff),
            **metrics,
        }
        rounds.append(row)
        state.previous_metrics = metrics

    return {
        "summary": summarize_game(rounds, state, episode_config, "q_learning", defender_policy_name),
        "rounds": rounds,
    }


def train_q_adversary(
    config: GameConfig | None = None,
    episodes: int = 50,
    epsilon: float = 0.20,
    alpha: float = 0.25,
    gamma: float = 0.90,
    defender_policy_name: str = "structural_warning",
    terminal_bonus: float = 10.0,
    detection_tax_mode: str = "delta",
) -> dict:
    config = config or GameConfig(dynamic_defense=True, trust_hysteresis=True)
    q_table: QTable = {}
    training_summaries = []
    for episode in range(episodes):
        result = run_q_episode(
            config,
            q_table,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            defender_policy_name=defender_policy_name,
            terminal_bonus=terminal_bonus,
            detection_tax_mode=detection_tax_mode,
            train=True,
            seed_offset=episode,
        )
        training_summaries.append(result["summary"])

    evaluation = run_q_episode(
        config,
        q_table,
        epsilon=0.0,
        defender_policy_name=defender_policy_name,
        terminal_bonus=terminal_bonus,
        detection_tax_mode=detection_tax_mode,
        train=False,
        seed_offset=episodes + 1,
    )
    baseline = run_game(config, adversary_policy_name="escalating", defender_policy_name=defender_policy_name)
    return {
        "q_table": q_table,
        "training": training_summaries,
        "evaluation": evaluation,
        "baseline": baseline,
        "summary": summarize_training(training_summaries, evaluation, baseline, q_table),
    }


def summarize_training(training: list[dict], evaluation: dict, baseline: dict, q_table: QTable) -> dict:
    final_window = training[-10:] if len(training) >= 10 else training
    actions = Counter(row["adversary_action"] for row in evaluation["rounds"])
    return {
        "episodes": len(training),
        "visited_states": len(q_table),
        "training_mean_peak_risk_last_window": float(np.mean([row["peak_risk"] for row in final_window])) if final_window else 0.0,
        "training_mean_final_risk_last_window": float(np.mean([row["final_risk"] for row in final_window])) if final_window else 0.0,
        "q_eval_peak_risk": float(evaluation["summary"]["peak_risk"]),
        "q_eval_final_risk": float(evaluation["summary"]["final_risk"]),
        "baseline_peak_risk": float(baseline["summary"]["peak_risk"]),
        "baseline_final_risk": float(baseline["summary"]["final_risk"]),
        "q_minus_baseline_final_risk": float(
            evaluation["summary"]["final_risk"] - baseline["summary"]["final_risk"]
        ),
        "q_threshold_crossed": bool(evaluation["summary"]["threshold_crossed"]),
        "baseline_threshold_crossed": bool(baseline["summary"]["threshold_crossed"]),
        "q_action_counts": dict(actions),
        "q_pause_rate": float(actions.get("pause", 0) / max(len(evaluation["rounds"]), 1)),
    }


def render_report(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Q-Learning Influence Adversary Report",
        "",
        "## Scope",
        "",
        (
            "Tabular Q-learning over abstract campaign actions. This is a defensive "
            "stress test for DTE policy evaluation, not an operational influence model."
        ),
        "",
        "## Summary",
        "",
        f"- Episodes: `{summary['episodes']}`",
        f"- Q eval peak risk: `{summary['q_eval_peak_risk']:.3f}`",
        f"- Q eval final risk: `{summary['q_eval_final_risk']:.3f}`",
        f"- Baseline peak risk: `{summary['baseline_peak_risk']:.3f}`",
        f"- Baseline final risk: `{summary['baseline_final_risk']:.3f}`",
        f"- Q minus baseline final risk: `{summary['q_minus_baseline_final_risk']:.3f}`",
        f"- Q threshold crossed: `{summary['q_threshold_crossed']}`",
        f"- Q pause rate: `{summary['q_pause_rate']:.1%}`",
        f"- Q action counts: `{summary['q_action_counts']}`",
        "",
        "## Evaluation Rounds",
        "",
        "| Round | Action | Defense | Risk | Escape | Warning | Detection |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["evaluation"]["rounds"]:
        lines.append(
            f"| {row['round']} | {row['adversary_action']} | {row['defender_action']} | "
            f"{row['risk_share']:.3f} | {row['escape_probability']:.3f} | "
            f"{row['warning_score']:.3f} | {row['detection_pressure']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("influence_qlearn_output.json"),
    output_md: Path = Path("INFLUENCE_QLEARN_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning adversary.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--steps-per-round", type=int, default=None)
    parser.add_argument("--terminal-bonus", type=float, default=10.0)
    parser.add_argument("--detection-tax-mode", choices=["delta", "absolute"], default="delta")
    parser.add_argument("--output-json", type=Path, default=Path("influence_qlearn_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("INFLUENCE_QLEARN_REPORT.md"))
    args = parser.parse_args()

    config = GameConfig(
        rounds=args.rounds if args.rounds is not None else (5 if args.quick else 10),
        agents=args.agents if args.agents is not None else (64 if args.quick else 160),
        steps_per_round=args.steps_per_round if args.steps_per_round is not None else (8 if args.quick else 16),
        adversary_budget=6.0 if args.quick else 10.0,
        defender_budget=5.0,
        dynamic_defense=True,
        trust_hysteresis=True,
    )
    payload = train_q_adversary(
        config,
        episodes=args.episodes if args.episodes is not None else (5 if args.quick else 40),
        terminal_bonus=args.terminal_bonus,
        detection_tax_mode=args.detection_tax_mode,
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

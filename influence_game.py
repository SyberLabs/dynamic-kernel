"""
Repeated defensive influence game on the Social Media DTE topology.

This is a strategic abstraction for resilience analysis. It models actor
choices as topology perturbations, not operational content or account tactics.

Usage:
    .venv\\Scripts\\python.exe influence_game.py
    .venv\\Scripts\\python.exe influence_game.py --quick
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adapters import SOCIAL_MEDIA
from geopolitical_influence import (
    ADVERSARY_BETA_EDGES,
    ADVERSARY_FRICTION_EDGES,
    ADVERSARY_SUPPRESSION_EDGES,
    Cohort,
    FOREIGN_INFLUENCE_TARGET,
    OFF_RAMP_EDGES,
    PREBUNK_EDGES,
    PROTECTIVE_NODES,
    RISK_NODES,
    THROTTLE_EDGES,
    _apply_beta,
    _apply_friction,
    _edge_cost,
    _effective_budget,
    _feature_index,
    _idx,
    _normalized_intent,
)
from legitimacy_ontology import (
    feature_index as legitimacy_feature_index,
    project_social_telemetry_to_legitimacy,
)


ADVERSARY_ACTIONS = (
    "pause",
    "amplify",
    "suppress_escape",
    "polarize",
    "stealth_amplify",
    "stealth_suppress_escape",
    "stealth_polarize",
)
DEFENDER_ACTIONS = ("observe", "off_ramp", "prebunk", "throttle", "combined")


@dataclass(frozen=True)
class GameConfig:
    rounds: int = 12
    steps_per_round: int = 24
    agents: int = 320
    feedback_rate: float = 0.35
    temperature: float = 0.65
    noise_sigma: float = 0.02
    seed: int = 20260602
    adversary_budget: float = 12.0
    defender_budget: float = 7.0
    action_unit: float = 1.0
    risk_threshold: float = 0.18
    warning_threshold: float = 0.50
    detection_decay: float = 0.88
    dynamic_defense: bool = False
    immune_response_gain: float = 0.75
    immune_response_cap: float = 3.0
    max_defender_budget: float = 12.0
    trust_hysteresis: bool = False
    trust_floor: float = 0.42
    trust_recovery_slowdown: float = 0.25
    cohorts: tuple[Cohort, ...] = (
        Cohort("civic_learners", "Civic Learner", 0.25),
        Cohort("high_arousal", "High-Arousal Scroll", 0.35),
        Cohort("casual_scrollers", "Casual Scroller", 0.25),
        Cohort("deep_researchers", "Deep Research", 0.15),
    )


@dataclass
class GameState:
    round_index: int
    positions: np.ndarray
    telemetries: np.ndarray
    cohort_ids: np.ndarray
    cohort_names: list[str]
    adversary_budget_left: float
    defender_budget_left: float
    detection_pressure: float = 0.0
    previous_metrics: dict | None = None
    first_warning_round: int | None = None
    first_risk_alarm_round: int | None = None
    immune_budget_added: float = 0.0


def build_kernel(config: GameConfig):
    return SOCIAL_MEDIA.build_kernel(
        feedback_rate=config.feedback_rate,
        feedback_noise=0.0,
        temperature=config.temperature,
    )


def _cohort_assignments(config: GameConfig) -> tuple[np.ndarray, list[str], np.ndarray]:
    shares = np.array([cohort.share for cohort in config.cohorts], dtype=np.float64)
    shares = shares / shares.sum()
    raw_counts = shares * config.agents
    counts = np.floor(raw_counts).astype(int)
    remainder = config.agents - int(counts.sum())
    if remainder > 0:
        for idx in np.argsort(raw_counts - counts)[::-1][:remainder]:
            counts[idx] += 1

    cohort_ids = np.concatenate([
        np.full(count, idx, dtype=int) for idx, count in enumerate(counts)
    ])
    cohort_names = [cohort.name for cohort in config.cohorts]
    telemetries = np.vstack([
        np.tile(_normalized_intent(config.cohorts[idx].intent), (count, 1))
        for idx, count in enumerate(counts)
        if count > 0
    ])
    return cohort_ids, cohort_names, telemetries


def initialize_state(config: GameConfig) -> GameState:
    labels = SOCIAL_MEDIA.build_topology().labels
    start = labels.index("Onboarding")
    cohort_ids, cohort_names, telemetries = _cohort_assignments(config)
    return GameState(
        round_index=0,
        positions=np.full(config.agents, start, dtype=int),
        telemetries=telemetries,
        cohort_ids=cohort_ids,
        cohort_names=cohort_names,
        adversary_budget_left=config.adversary_budget,
        defender_budget_left=config.defender_budget,
    )


def adversary_policy(state: GameState, config: GameConfig, policy: str = "escalating") -> str:
    if state.adversary_budget_left <= 0:
        return "pause"
    metrics = state.previous_metrics or {}
    risk = metrics.get("risk_share", 0.0)
    escape = metrics.get("escape_probability", 1.0)

    if policy == "fixed":
        return "polarize"
    if policy == "greedy":
        return "suppress_escape" if escape > 0.50 else "amplify"
    if policy == "stealth":
        return "pause" if state.detection_pressure > 1.0 else "stealth_suppress_escape"

    if state.round_index < 2:
        return "amplify"
    if escape > 0.50:
        return "suppress_escape"
    if risk < config.risk_threshold:
        return "polarize"
    return "pause"


def adversary_budget_spend(action: str, action_unit: float) -> float:
    if action == "pause":
        return 0.0
    if action.startswith("stealth_"):
        return 0.50 * action_unit
    return action_unit


def adversary_detection_increment(action: str) -> float:
    if action == "pause":
        return 0.0
    if action.startswith("stealth_"):
        return 0.025
    return 0.10


def defender_policy(state: GameState, config: GameConfig, policy: str = "structural_warning") -> str:
    if state.defender_budget_left <= 0:
        return "observe"
    metrics = state.previous_metrics or {}
    risk = metrics.get("risk_share", 0.0)
    escape = metrics.get("escape_probability", 1.0)
    current = metrics.get("edge_current_norm", 0.0)
    production = metrics.get("entropy_production", 0.0)
    warning = metrics.get("warning_score", 0.0)

    if policy == "none":
        return "observe"
    if policy == "off_ramp_first":
        return "off_ramp"
    if policy == "prebunk_first":
        return "prebunk"
    if policy == "throttle_first":
        return "throttle"
    if policy == "combined_first":
        return "combined"
    if policy == "risk_threshold":
        return "combined" if risk >= config.risk_threshold else "observe"
    if policy == "throttle_only":
        return "throttle" if risk >= 0.14 or escape < 0.50 else "observe"

    if warning >= config.warning_threshold:
        if escape < 0.55:
            return "off_ramp"
        if current > 0.14 or production > 0.08:
            return "combined"
        return "prebunk"
    return "observe"


def apply_immune_response(state: GameState, config: GameConfig) -> float:
    """
    Add defender budget from prior non-equilibrium stress.

    This models an automatic platform/institutional response: entropy
    production and edge-current anomalies increase available moderation,
    prebunking, or topology-repair capacity in the next round.
    """
    if not config.dynamic_defense or not state.previous_metrics:
        return 0.0
    current = float(state.previous_metrics.get("edge_current_norm", 0.0))
    production = float(state.previous_metrics.get("entropy_production", 0.0))
    stress = (
        0.55 * min(1.0, production / 0.10)
        + 0.45 * min(1.0, current / 0.18)
    )
    delta = min(config.immune_response_cap, config.immune_response_gain * stress)
    capacity = max(0.0, config.max_defender_budget - state.defender_budget_left)
    added = min(delta, capacity)
    state.defender_budget_left += added
    state.immune_budget_added += added
    return float(added)


def apply_adversary_action(kernel, labels: list[str], action: str, unit: float, detection_pressure: float) -> float:
    effective = _effective_budget(unit, detection_pressure)
    if action == "pause" or effective <= 0:
        return 0.0
    if action == "amplify":
        beta_boost = 0.60 * effective
        _apply_beta(kernel, labels, ADVERSARY_BETA_EDGES, beta_boost)
        return _edge_cost(ADVERSARY_BETA_EDGES, beta_boost)
    if action == "stealth_amplify":
        beta_boost = 0.40 * effective
        _apply_beta(kernel, labels, ADVERSARY_BETA_EDGES, beta_boost)
        return _edge_cost(ADVERSARY_BETA_EDGES, beta_boost)
    if action == "suppress_escape":
        friction_increase = 0.50 * effective
        _apply_friction(kernel, labels, ADVERSARY_SUPPRESSION_EDGES, -friction_increase)
        return _edge_cost(ADVERSARY_SUPPRESSION_EDGES, friction_increase)
    if action == "stealth_suppress_escape":
        friction_increase = 0.28 * effective
        _apply_friction(kernel, labels, ADVERSARY_SUPPRESSION_EDGES, -friction_increase)
        return _edge_cost(ADVERSARY_SUPPRESSION_EDGES, friction_increase)
    if action == "polarize":
        beta_boost = 0.35 * effective
        friction_reduction = 0.30 * effective
        _apply_beta(kernel, labels, ADVERSARY_BETA_EDGES, beta_boost)
        _apply_friction(kernel, labels, ADVERSARY_FRICTION_EDGES, friction_reduction)
        target = labels.index(FOREIGN_INFLUENCE_TARGET)
        neutral = np.zeros(kernel.topo.F, dtype=np.float64)
        bias = kernel.get_diagnostic(neutral)["node_bias"][target]
        kernel.set_node_bias(target, bias + 0.06 * effective)
        return (
            _edge_cost(ADVERSARY_BETA_EDGES, beta_boost)
            + _edge_cost(ADVERSARY_FRICTION_EDGES, friction_reduction)
            + 0.06 * effective
        )
    if action == "stealth_polarize":
        beta_boost = 0.22 * effective
        friction_reduction = 0.18 * effective
        _apply_beta(kernel, labels, ADVERSARY_BETA_EDGES, beta_boost)
        _apply_friction(kernel, labels, ADVERSARY_FRICTION_EDGES, friction_reduction)
        target = labels.index(FOREIGN_INFLUENCE_TARGET)
        neutral = np.zeros(kernel.topo.F, dtype=np.float64)
        bias = kernel.get_diagnostic(neutral)["node_bias"][target]
        kernel.set_node_bias(target, bias + 0.03 * effective)
        return (
            _edge_cost(ADVERSARY_BETA_EDGES, beta_boost)
            + _edge_cost(ADVERSARY_FRICTION_EDGES, friction_reduction)
            + 0.03 * effective
        )
    raise ValueError(f"Unknown adversary action: {action}")


def apply_defender_action(kernel, labels: list[str], action: str, unit: float) -> float:
    if action == "observe" or unit <= 0:
        return 0.0
    if action == "off_ramp":
        friction_reduction = 0.45 * unit
        beta_boost = 0.25 * unit
        _apply_friction(kernel, labels, OFF_RAMP_EDGES, friction_reduction)
        _apply_beta(kernel, labels, OFF_RAMP_EDGES, beta_boost)
        return _edge_cost(OFF_RAMP_EDGES, friction_reduction + beta_boost)
    if action == "prebunk":
        beta_boost = 0.45 * unit
        _apply_beta(kernel, labels, PREBUNK_EDGES, beta_boost)
        return _edge_cost(PREBUNK_EDGES, beta_boost)
    if action == "throttle":
        friction_increase = 0.45 * unit
        _apply_friction(kernel, labels, THROTTLE_EDGES, -friction_increase)
        return _edge_cost(THROTTLE_EDGES, friction_increase)
    if action == "combined":
        return (
            apply_defender_action(kernel, labels, "off_ramp", 0.45 * unit)
            + apply_defender_action(kernel, labels, "prebunk", 0.35 * unit)
            + apply_defender_action(kernel, labels, "throttle", 0.20 * unit)
        )
    raise ValueError(f"Unknown defender action: {action}")


def _round_metrics(
    labels: list[str],
    positions: np.ndarray,
    telemetries: np.ndarray,
    cohort_ids: np.ndarray,
    cohort_names: list[str],
    edge_counts: np.ndarray,
    escape_values: list[float],
) -> dict:
    risk_idx = np.array(_idx(labels, RISK_NODES), dtype=int)
    protective_idx = np.array(_idx(labels, PROTECTIVE_NODES), dtype=int)
    target_idx = labels.index(FOREIGN_INFLUENCE_TARGET)
    credibility_i = _feature_index("Credibility")
    conflict_i = _feature_index("Conflict")

    risk_share = float(np.mean(np.isin(positions, risk_idx)))
    protective_share = float(np.mean(np.isin(positions, protective_idx)))
    target_share = float(np.mean(positions == target_idx))
    escape_probability = float(np.mean(escape_values)) if escape_values else 1.0
    conflict_credibility_drift = float(np.mean(telemetries[:, conflict_i] - telemetries[:, credibility_i]))
    credibility_state = float(np.mean(telemetries[:, credibility_i]))

    cohort_risk = {}
    for idx, name in enumerate(cohort_names):
        mask = cohort_ids == idx
        cohort_risk[name] = float(np.mean(np.isin(positions[mask], risk_idx)))
    polarization_gap = float(max(cohort_risk.values()) - min(cohort_risk.values()))

    node_counts = np.bincount(positions, minlength=len(labels)).astype(np.float64)
    occupancy = node_counts / max(float(node_counts.sum()), 1.0)
    occupancy_entropy = -float(np.sum(occupancy[occupancy > 0] * np.log(occupancy[occupancy > 0])))
    population_entropy = occupancy_entropy / np.log(len(labels))

    edge_flow = edge_counts / max(float(edge_counts.sum()), 1.0)
    edge_current = edge_flow - edge_flow.T
    eps = 1e-12
    rev = edge_flow.T
    mask = (edge_flow > eps) & (rev > eps)
    entropy_production = 0.0
    if np.any(mask):
        entropy_production = 0.5 * float(np.sum(edge_flow[mask] * np.log(edge_flow[mask] / rev[mask])))
    edge_current_norm = float(np.linalg.norm(edge_current, ord="fro"))

    warning_score = float(
        0.45 * max(0.0, 1.0 - escape_probability)
        + 0.35 * min(1.0, edge_current_norm / 0.18)
        + 0.20 * min(1.0, entropy_production / 0.10)
    )
    return {
        "risk_share": risk_share,
        "target_share": target_share,
        "protective_share": protective_share,
        "escape_probability": escape_probability,
        "credibility_state": credibility_state,
        "conflict_credibility_drift": conflict_credibility_drift,
        "population_entropy": population_entropy,
        "polarization_gap": polarization_gap,
        "cohort_risk": cohort_risk,
        "edge_current_norm": edge_current_norm,
        "entropy_production": entropy_production,
        "warning_score": warning_score,
    }


def _apply_trust_hysteresis(
    previous: np.ndarray,
    updated: np.ndarray,
    config: GameConfig,
) -> np.ndarray:
    if not config.trust_hysteresis:
        return updated
    prev_legitimacy = project_social_telemetry_to_legitimacy(previous, SOCIAL_MEDIA.feature_labels)
    next_legitimacy = project_social_telemetry_to_legitimacy(updated, SOCIAL_MEDIA.feature_labels)
    trust_i = legitimacy_feature_index("institutional_trust")
    fairness_i = legitimacy_feature_index("procedural_fairness")
    prev_trust = prev_legitimacy[:, trust_i]
    next_trust = next_legitimacy[:, trust_i]
    prev_fairness = prev_legitimacy[:, fairness_i]
    next_fairness = next_legitimacy[:, fairness_i]
    recovering = (
        (prev_trust < config.trust_floor)
        & (next_trust > prev_trust)
        & (next_fairness >= prev_fairness)
    )
    if np.any(recovering):
        updated = updated.copy()
        social_trust_i = _feature_index("Credibility")
        previous_social_trust = previous[:, social_trust_i]
        updated_social_trust = updated[:, social_trust_i]
        updated[recovering, social_trust_i] = (
            previous_social_trust[recovering]
            + config.trust_recovery_slowdown
            * (updated_social_trust[recovering] - previous_social_trust[recovering])
        )
    return updated


def simulate_round(kernel, labels: list[str], state: GameState, config: GameConfig, rng: np.random.Generator) -> dict:
    n = kernel.topo.N
    risk_idx = np.array(_idx(labels, RISK_NODES), dtype=int)
    edge_counts = np.zeros((n, n), dtype=np.float64)
    escape_values = []

    for step in range(config.steps_per_round):
        P_all = kernel.transition_matrix_batch(state.telemetries, step=state.round_index * config.steps_per_round + step)
        rows = P_all[np.arange(config.agents), state.positions, :]

        in_risk = np.isin(state.positions, risk_idx)
        if np.any(in_risk):
            escape_values.append(float(np.mean(1.0 - rows[in_risk][:, risk_idx].sum(axis=1))))
        else:
            escape_values.append(1.0)

        cdf = np.cumsum(rows, axis=1)
        draws = rng.random((config.agents, 1))
        next_positions = np.argmax(cdf >= draws, axis=1)
        row_sums = rows.sum(axis=1)
        next_positions[row_sums < 1e-12] = state.positions[row_sums < 1e-12]
        np.add.at(edge_counts, (state.positions, next_positions), 1.0)

        visited = kernel.topo.node_features[next_positions]
        lam = config.feedback_rate
        previous_telemetries = state.telemetries
        updated_telemetries = (1.0 - lam) * state.telemetries + lam * visited
        state.telemetries = _apply_trust_hysteresis(previous_telemetries, updated_telemetries, config)
        if config.noise_sigma > 0:
            state.telemetries += rng.normal(scale=config.noise_sigma, size=state.telemetries.shape)
        norms = np.linalg.norm(state.telemetries, axis=1, keepdims=True)
        state.telemetries = np.where(norms > 0, state.telemetries / norms, state.telemetries)
        state.positions = next_positions

    return _round_metrics(
        labels=labels,
        positions=state.positions,
        telemetries=state.telemetries,
        cohort_ids=state.cohort_ids,
        cohort_names=state.cohort_names,
        edge_counts=edge_counts,
        escape_values=escape_values,
    )


def run_game(
    config: GameConfig | None = None,
    adversary_policy_name: str = "escalating",
    defender_policy_name: str = "structural_warning",
) -> dict:
    config = config or GameConfig()
    kernel = build_kernel(config)
    labels = kernel.topo.labels
    rng = np.random.default_rng(config.seed)
    state = initialize_state(config)
    rounds = []

    for round_index in range(config.rounds):
        state.round_index = round_index
        immune_budget_added = apply_immune_response(state, config)
        adv_action = adversary_policy(state, config, adversary_policy_name)
        def_action = defender_policy(state, config, defender_policy_name)

        adv_unit = min(adversary_budget_spend(adv_action, config.action_unit), state.adversary_budget_left)
        def_unit = min(config.action_unit, state.defender_budget_left)
        adv_cost = apply_adversary_action(kernel, labels, adv_action, adv_unit, state.detection_pressure)
        def_cost = apply_defender_action(kernel, labels, def_action, def_unit)
        if adv_action != "pause":
            state.adversary_budget_left = max(0.0, state.adversary_budget_left - adv_unit)
        if def_action != "observe":
            state.defender_budget_left = max(0.0, state.defender_budget_left - def_unit)

        metrics = simulate_round(kernel, labels, state, config, rng)
        if state.first_warning_round is None and metrics["warning_score"] >= config.warning_threshold:
            state.first_warning_round = round_index
        if state.first_risk_alarm_round is None and metrics["risk_share"] >= config.risk_threshold:
            state.first_risk_alarm_round = round_index

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

    summary = summarize_game(rounds, state, config, adversary_policy_name, defender_policy_name)
    return {"summary": summary, "rounds": rounds}


def summarize_game(
    rounds: list[dict],
    state: GameState,
    config: GameConfig,
    adversary_policy_name: str,
    defender_policy_name: str,
) -> dict:
    risk_crossings = [row["round"] for row in rounds if row["risk_share"] >= config.risk_threshold]
    warning_rounds = [row["round"] for row in rounds if row["warning_score"] >= config.warning_threshold]
    defensive_rounds = [row["round"] for row in rounds if row["defender_action"] != "observe"]
    final = rounds[-1] if rounds else {}
    peak_risk = max((row["risk_share"] for row in rounds), default=0.0)
    mean_risk = float(np.mean([row["risk_share"] for row in rounds])) if rounds else 0.0
    total_adv_cost = float(sum(row["adversary_cost"] for row in rounds))
    total_def_cost = float(sum(row["defender_cost"] for row in rounds))
    warning_lead = None
    if warning_rounds and risk_crossings:
        warning_lead = risk_crossings[0] - warning_rounds[0]

    return {
        "rounds": len(rounds),
        "adversary_policy": adversary_policy_name,
        "defender_policy": defender_policy_name,
        "risk_threshold": config.risk_threshold,
        "warning_threshold": config.warning_threshold,
        "first_warning_round": warning_rounds[0] if warning_rounds else None,
        "first_defensive_round": defensive_rounds[0] if defensive_rounds else None,
        "first_risk_alarm_round": risk_crossings[0] if risk_crossings else None,
        "warning_lead_rounds": warning_lead,
        "threshold_crossed": bool(risk_crossings),
        "peak_risk": peak_risk,
        "mean_risk": mean_risk,
        "final_risk": float(final.get("risk_share", 0.0)),
        "final_escape_probability": float(final.get("escape_probability", 0.0)),
        "final_detection_pressure": float(final.get("detection_pressure", 0.0)),
        "total_adversary_cost": total_adv_cost,
        "total_defender_cost": total_def_cost,
        "total_immune_budget_added": float(sum(row.get("immune_budget_added", 0.0) for row in rounds)),
        "adversary_budget_left": float(state.adversary_budget_left),
        "defender_budget_left": float(state.defender_budget_left),
        "cumulative_adversary_payoff": float(sum(row["adversary_payoff"] for row in rounds)),
        "cumulative_defender_payoff": float(sum(row["defender_payoff"] for row in rounds)),
        "institutional_result": (
            "defense_preempted"
            if defensive_rounds and (not risk_crossings or defensive_rounds[0] < risk_crossings[0])
            else "reactive_or_failed"
        ),
    }


def compare_defender_policies(config: GameConfig | None = None) -> dict:
    config = config or GameConfig()
    policies = ["none", "risk_threshold", "structural_warning"]
    runs = {
        policy: run_game(config, adversary_policy_name="escalating", defender_policy_name=policy)["summary"]
        for policy in policies
    }
    best = min(
        runs.items(),
        key=lambda item: (
            item[1]["final_risk"],
            item[1]["mean_risk"],
            item[1]["peak_risk"],
            -item[1]["defender_budget_left"],
        ),
    )
    return {"policies": runs, "best_policy": best[0]}


def render_report(payload: dict, comparison: dict | None = None) -> str:
    summary = payload["summary"]
    lines = [
        "# Influence Game DTE Report",
        "",
        "## Scope",
        "",
        (
            "Repeated defensive influence game. Actors choose abstract topology "
            "pressures and defenses; the population state carries across rounds."
        ),
        "",
        "## Summary",
        "",
        f"- Adversary policy: `{summary['adversary_policy']}`",
        f"- Defender policy: `{summary['defender_policy']}`",
        f"- Peak risk: `{summary['peak_risk']:.3f}`",
        f"- Final risk: `{summary['final_risk']:.3f}`",
        f"- First warning round: `{summary['first_warning_round']}`",
        f"- First defensive round: `{summary['first_defensive_round']}`",
        f"- First risk alarm round: `{summary['first_risk_alarm_round']}`",
        f"- Immune budget added: `{summary['total_immune_budget_added']:.2f}`",
        f"- Institutional result: `{summary['institutional_result']}`",
        "",
        "## Rounds",
        "",
        "| Round | Adv | Def | Risk | Escape | Warning | Detection | Immune | Adv Budget | Def Budget |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rounds"]:
        lines.append(
            f"| {row['round']} | {row['adversary_action']} | {row['defender_action']} | "
            f"{row['risk_share']:.3f} | {row['escape_probability']:.3f} | "
            f"{row['warning_score']:.3f} | {row['detection_pressure']:.3f} | "
            f"{row['immune_budget_added']:.2f} | {row['adversary_budget_left']:.1f} | "
            f"{row['defender_budget_left']:.1f} |"
        )

    if comparison:
        lines.extend([
            "",
            "## Defender Policy Comparison",
            "",
            "| Policy | Peak Risk | Final Risk | First Defense | First Risk Alarm | Result |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for policy, row in comparison["policies"].items():
            lines.append(
                f"| {policy} | {row['peak_risk']:.3f} | {row['final_risk']:.3f} | "
                f"{row['first_defensive_round']} | {row['first_risk_alarm_round']} | "
                f"{row['institutional_result']} |"
            )
        lines.append("")
        lines.append(f"Best policy by final-risk minimization: `{comparison['best_policy']}`.")
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    comparison: dict | None = None,
    output_json: Path = Path("influence_game_output.json"),
    output_md: Path = Path("INFLUENCE_GAME_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    combined = {"game": payload, "comparison": comparison}
    output_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload, comparison), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated defensive influence game.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--steps-per-round", type=int, default=None)
    parser.add_argument("--adversary-policy", default="escalating")
    parser.add_argument("--defender-policy", default="structural_warning")
    parser.add_argument("--dynamic-defense", action="store_true")
    parser.add_argument("--trust-hysteresis", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("influence_game_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("INFLUENCE_GAME_REPORT.md"))
    args = parser.parse_args()

    config = GameConfig(
        rounds=args.rounds if args.rounds is not None else (6 if args.quick else 12),
        agents=args.agents if args.agents is not None else (96 if args.quick else 320),
        steps_per_round=args.steps_per_round if args.steps_per_round is not None else (12 if args.quick else 24),
        dynamic_defense=args.dynamic_defense,
        trust_hysteresis=args.trust_hysteresis,
    )
    payload = run_game(config, args.adversary_policy, args.defender_policy)
    comparison = compare_defender_policies(config)
    write_outputs(payload, comparison, args.output_json, args.output_md)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

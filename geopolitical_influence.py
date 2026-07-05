"""
Defensive geopolitical influence simulation on top of the Social Media DTE.

This module models contested information routing at the level of abstract
campaign pressures. It does not simulate account creation, posting strategy,
or operational bot tactics. Actors are represented as budgeted perturbations
to recommendation topology, and the outputs are institutional risk metrics.

Usage:
    .venv\\Scripts\\python.exe geopolitical_influence.py
    .venv\\Scripts\\python.exe geopolitical_influence.py --quick
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from adapters import SOCIAL_MEDIA


RISK_NODES = ("Conflict Commentary", "Conspiracy/Rumor")
PROTECTIVE_NODES = ("Local News", "Science Explainers", "Longform Off-Ramp")
FOREIGN_INFLUENCE_TARGET = "Conspiracy/Rumor"

ADVERSARY_BETA_EDGES = [
    ("Local News", "Conflict Commentary"),
    ("Civic Debate", "Conflict Commentary"),
    ("Finance Advice", "Conflict Commentary"),
    ("Conflict Commentary", "Conspiracy/Rumor"),
    ("Conspiracy/Rumor", "Conflict Commentary"),
]

ADVERSARY_FRICTION_EDGES = [
    ("Civic Debate", "Conflict Commentary"),
    ("Conflict Commentary", "Conspiracy/Rumor"),
    ("Conspiracy/Rumor", "Conflict Commentary"),
]

ADVERSARY_SUPPRESSION_EDGES = [
    ("Conflict Commentary", "Local News"),
    ("Conflict Commentary", "Longform Off-Ramp"),
    ("Conspiracy/Rumor", "Civic Debate"),
    ("Conspiracy/Rumor", "Science Explainers"),
    ("Conspiracy/Rumor", "Longform Off-Ramp"),
]

OFF_RAMP_EDGES = [
    ("Local News", "Longform Off-Ramp"),
    ("Civic Debate", "Longform Off-Ramp"),
    ("Conflict Commentary", "Longform Off-Ramp"),
    ("Conspiracy/Rumor", "Longform Off-Ramp"),
    ("Music & Culture", "Longform Off-Ramp"),
]

PREBUNK_EDGES = [
    ("Onboarding", "Local News"),
    ("Local News", "Science Explainers"),
    ("Civic Debate", "Science Explainers"),
    ("Conspiracy/Rumor", "Science Explainers"),
    ("Longform Off-Ramp", "Science Explainers"),
]

THROTTLE_EDGES = [
    ("Conflict Commentary", "Conspiracy/Rumor"),
    ("Conspiracy/Rumor", "Conflict Commentary"),
]


@dataclass(frozen=True)
class Cohort:
    name: str
    intent: str
    share: float


@dataclass(frozen=True)
class InfluenceConfig:
    feedback_rate: float = 0.35
    temperature: float = 0.65
    noise_sigma: float = 0.02
    agents: int = 320
    steps: int = 96
    seed: int = 20260602
    cohorts: tuple[Cohort, ...] = (
        Cohort("civic_learners", "Civic Learner", 0.25),
        Cohort("high_arousal", "High-Arousal Scroll", 0.35),
        Cohort("casual_scrollers", "Casual Scroller", 0.25),
        Cohort("deep_researchers", "Deep Research", 0.15),
    )


@dataclass(frozen=True)
class Campaign:
    name: str
    description: str
    adversary_budget: float
    defender_budget: float
    detection_pressure: float
    apply: Callable


def _idx(labels: list[str], names: tuple[str, ...] | list[str]) -> list[int]:
    return [labels.index(name) for name in names]


def _feature_index(name: str) -> int:
    return SOCIAL_MEDIA.feature_labels.index(name)


def _normalized_intent(intent: str) -> np.ndarray:
    telemetry = np.array(SOCIAL_MEDIA.intent_presets[intent], dtype=np.float64)
    norm = np.linalg.norm(telemetry)
    return telemetry / norm if norm > 0 else telemetry


def _edge_cost(edge_names: list[tuple[str, str]], magnitude: float) -> float:
    return float(len(edge_names) * abs(magnitude))


def _effective_budget(budget: float, detection_pressure: float) -> float:
    """Diminishing adversarial effect under detection and throttling pressure."""
    return float(budget / (1.0 + detection_pressure * max(budget, 0.0)))


def _apply_beta(kernel, labels: list[str], edges: list[tuple[str, str]], boost: float) -> None:
    for source, target in edges:
        i, j = labels.index(source), labels.index(target)
        if kernel.topo.adjacency_mask[i, j]:
            kernel.sponsor_edge(i, j, boost)


def _apply_friction(kernel, labels: list[str], edges: list[tuple[str, str]], reduction: float) -> None:
    for source, target in edges:
        i, j = labels.index(source), labels.index(target)
        if kernel.topo.adjacency_mask[i, j]:
            kernel.sponsor_edge_friction(i, j, reduction)


def _foreign_wedge(kernel, labels: list[str], budget: float, detection_pressure: float) -> float:
    effective = _effective_budget(budget, detection_pressure)
    beta_boost = 0.55 * effective
    friction_reduction = 0.35 * effective
    escape_friction_increase = 0.30 * effective
    _apply_beta(kernel, labels, ADVERSARY_BETA_EDGES, beta_boost)
    _apply_friction(kernel, labels, ADVERSARY_FRICTION_EDGES, friction_reduction)
    _apply_friction(kernel, labels, ADVERSARY_SUPPRESSION_EDGES, -escape_friction_increase)
    target = labels.index(FOREIGN_INFLUENCE_TARGET)
    neutral = np.zeros(kernel.topo.F, dtype=np.float64)
    current_bias = kernel.get_diagnostic(neutral)["node_bias"][target]
    kernel.set_node_bias(target, current_bias + 0.08 * effective)
    return (
        _edge_cost(ADVERSARY_BETA_EDGES, beta_boost)
        + _edge_cost(ADVERSARY_FRICTION_EDGES, friction_reduction)
        + _edge_cost(ADVERSARY_SUPPRESSION_EDGES, escape_friction_increase)
        + 0.08 * effective
    )


def _off_ramp_defense(kernel, labels: list[str], budget: float) -> float:
    friction_reduction = 0.50 * budget
    beta_boost = 0.25 * budget
    _apply_friction(kernel, labels, OFF_RAMP_EDGES, friction_reduction)
    _apply_beta(kernel, labels, OFF_RAMP_EDGES, beta_boost)
    return _edge_cost(OFF_RAMP_EDGES, friction_reduction + beta_boost)


def _prebunk_defense(kernel, labels: list[str], budget: float) -> float:
    beta_boost = 0.45 * budget
    _apply_beta(kernel, labels, PREBUNK_EDGES, beta_boost)
    return _edge_cost(PREBUNK_EDGES, beta_boost)


def _throttle_defense(kernel, labels: list[str], budget: float) -> float:
    friction_increase = 0.40 * budget
    _apply_friction(kernel, labels, THROTTLE_EDGES, -friction_increase)
    return _edge_cost(THROTTLE_EDGES, friction_increase)


def campaign_catalog(adversary_budget: float = 1.0, defender_budget: float = 1.0) -> list[Campaign]:
    def baseline(kernel, labels):
        return {"adversary_cost": 0.0, "defender_cost": 0.0}

    def foreign_only(kernel, labels):
        return {
            "adversary_cost": _foreign_wedge(kernel, labels, adversary_budget, 0.0),
            "defender_cost": 0.0,
        }

    def off_ramp(kernel, labels):
        return {
            "adversary_cost": _foreign_wedge(kernel, labels, adversary_budget, 0.15),
            "defender_cost": _off_ramp_defense(kernel, labels, defender_budget),
        }

    def prebunk(kernel, labels):
        return {
            "adversary_cost": _foreign_wedge(kernel, labels, adversary_budget, 0.20),
            "defender_cost": _prebunk_defense(kernel, labels, defender_budget),
        }

    def throttle(kernel, labels):
        return {
            "adversary_cost": _foreign_wedge(kernel, labels, adversary_budget, 0.35),
            "defender_cost": _throttle_defense(kernel, labels, defender_budget),
        }

    def combined(kernel, labels):
        return {
            "adversary_cost": _foreign_wedge(kernel, labels, adversary_budget, 0.45),
            "defender_cost": (
                _off_ramp_defense(kernel, labels, 0.45 * defender_budget)
                + _prebunk_defense(kernel, labels, 0.35 * defender_budget)
                + _throttle_defense(kernel, labels, 0.20 * defender_budget)
            ),
        }

    return [
        Campaign("baseline", "Uncontested platform dynamics.", 0.0, 0.0, 0.0, baseline),
        Campaign("foreign_wedge", "Budgeted foreign influence pressure on risk-basin edges.", adversary_budget, 0.0, 0.0, foreign_only),
        Campaign("off_ramp_defense", "Foreign pressure plus topology repair through credible off-ramps.", adversary_budget, defender_budget, 0.15, off_ramp),
        Campaign("prebunk_defense", "Foreign pressure plus credibility-first prebunk routing.", adversary_budget, defender_budget, 0.20, prebunk),
        Campaign("throttle_defense", "Foreign pressure plus throttling of high-risk narrative loops.", adversary_budget, defender_budget, 0.35, throttle),
        Campaign("combined_defense", "Foreign pressure plus mixed off-ramp, prebunk, and throttle policy.", adversary_budget, defender_budget, 0.45, combined),
    ]


def build_kernel(config: InfluenceConfig):
    return SOCIAL_MEDIA.build_kernel(
        feedback_rate=config.feedback_rate,
        feedback_noise=0.0,
        temperature=config.temperature,
    )


def _cohort_assignments(config: InfluenceConfig) -> tuple[np.ndarray, list[str], np.ndarray]:
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
    telemetry = np.vstack([
        np.tile(_normalized_intent(config.cohorts[idx].intent), (count, 1))
        for idx, count in enumerate(counts)
        if count > 0
    ])
    return cohort_ids, cohort_names, telemetry


def simulate_campaign(config: InfluenceConfig, campaign: Campaign) -> dict:
    kernel = build_kernel(config)
    labels = kernel.topo.labels
    costs = campaign.apply(kernel, labels)

    rng = np.random.default_rng(config.seed)
    n = kernel.topo.N
    risk_idx = np.array(_idx(labels, RISK_NODES), dtype=int)
    protective_idx = np.array(_idx(labels, PROTECTIVE_NODES), dtype=int)
    target_idx = labels.index(FOREIGN_INFLUENCE_TARGET)
    start = labels.index("Onboarding")
    credibility_i = _feature_index("Credibility")
    conflict_i = _feature_index("Conflict")

    cohort_ids, cohort_names, telemetries = _cohort_assignments(config)
    positions = np.full(config.agents, start, dtype=int)

    risk_series = []
    protective_series = []
    target_series = []
    credibility_series = []
    drift_series = []
    escape_series = []
    cohort_risk_series = {name: [] for name in cohort_names}
    edge_counts = np.zeros((n, n), dtype=np.float64)

    for _step in range(config.steps):
        P_all = kernel.transition_matrix_batch(telemetries, step=_step)
        rows = P_all[np.arange(config.agents), positions, :]

        in_risk = np.isin(positions, risk_idx)
        if np.any(in_risk):
            escape_series.append(float(np.mean(1.0 - rows[in_risk][:, risk_idx].sum(axis=1))))
        else:
            escape_series.append(1.0)

        cdf = np.cumsum(rows, axis=1)
        draws = rng.random((config.agents, 1))
        next_positions = np.argmax(cdf >= draws, axis=1)
        row_sums = rows.sum(axis=1)
        next_positions[row_sums < 1e-12] = positions[row_sums < 1e-12]
        np.add.at(edge_counts, (positions, next_positions), 1.0)

        visited = kernel.topo.node_features[next_positions]
        lam = config.feedback_rate
        telemetries = (1.0 - lam) * telemetries + lam * visited
        if config.noise_sigma > 0:
            telemetries += rng.normal(scale=config.noise_sigma, size=telemetries.shape)
        norms = np.linalg.norm(telemetries, axis=1, keepdims=True)
        telemetries = np.where(norms > 0, telemetries / norms, telemetries)
        positions = next_positions

        risk_series.append(float(np.mean(np.isin(positions, risk_idx))))
        protective_series.append(float(np.mean(np.isin(positions, protective_idx))))
        target_series.append(float(np.mean(positions == target_idx)))
        credibility_series.append(float(np.mean(visited[:, credibility_i])))
        drift_series.append(float(np.mean(telemetries[:, conflict_i] - telemetries[:, credibility_i])))

        for idx, name in enumerate(cohort_names):
            mask = cohort_ids == idx
            cohort_risk_series[name].append(float(np.mean(np.isin(positions[mask], risk_idx))))

    final_window = max(8, config.steps // 4)
    cohort_final_risk = {
        name: float(np.mean(series[-final_window:])) for name, series in cohort_risk_series.items()
    }
    node_counts = np.bincount(positions, minlength=n).astype(np.float64)
    occupancy = node_counts / max(float(node_counts.sum()), 1.0)
    occupancy_entropy = -float(np.sum(occupancy[occupancy > 0] * np.log(occupancy[occupancy > 0])))
    normalized_entropy = occupancy_entropy / np.log(n)

    edge_flow = edge_counts / max(float(edge_counts.sum()), 1.0)
    edge_current = edge_flow - edge_flow.T
    eps = 1e-12
    rev = edge_flow.T
    mask = (edge_flow > eps) & (rev > eps)
    entropy_production = 0.0
    if np.any(mask):
        entropy_production = 0.5 * float(np.sum(edge_flow[mask] * np.log(edge_flow[mask] / rev[mask])))

    final_risk = float(np.mean(risk_series[-final_window:]))
    final_escape = float(np.mean(escape_series[-final_window:]))
    final_drift = float(np.mean(drift_series[-final_window:]))
    lock_in = final_risk >= 0.18 and final_escape <= 0.55 and final_drift > -0.05

    return {
        "campaign": campaign.name,
        "description": campaign.description,
        "lambda": config.feedback_rate,
        "tau": config.temperature,
        "sigma": config.noise_sigma,
        "agents": config.agents,
        "steps": config.steps,
        "adversary_budget": campaign.adversary_budget,
        "defender_budget": campaign.defender_budget,
        "adversary_cost": float(costs["adversary_cost"]),
        "defender_cost": float(costs["defender_cost"]),
        "risk_share": final_risk,
        "protective_share": float(np.mean(protective_series[-final_window:])),
        "target_share": float(np.mean(target_series[-final_window:])),
        "credibility_exposure": float(np.mean(credibility_series[-final_window:])),
        "conflict_credibility_drift": final_drift,
        "escape_probability": final_escape,
        "population_entropy": normalized_entropy,
        "polarization_gap": float(max(cohort_final_risk.values()) - min(cohort_final_risk.values())),
        "cohort_risk": cohort_final_risk,
        "edge_current_norm": float(np.linalg.norm(edge_current, ord="fro")),
        "entropy_production": entropy_production,
        "lock_in": bool(lock_in),
        "risk_series": risk_series,
    }


def run_campaign_set(
    config: InfluenceConfig | None = None,
    adversary_budget: float = 1.0,
    defender_budget: float = 1.0,
) -> list[dict]:
    config = config or InfluenceConfig()
    campaigns = campaign_catalog(adversary_budget, defender_budget)
    rows = [simulate_campaign(config, campaign) for campaign in campaigns]
    baseline = next(row for row in rows if row["campaign"] == "baseline")
    foreign = next(row for row in rows if row["campaign"] == "foreign_wedge")
    for row in rows:
        row["risk_delta_vs_baseline"] = row["risk_share"] - baseline["risk_share"]
        row["risk_delta_vs_foreign_wedge"] = row["risk_share"] - foreign["risk_share"]
        row["adversary_roi"] = (
            row["risk_delta_vs_baseline"] / row["adversary_cost"]
            if row["adversary_cost"] > 0
            else 0.0
        )
        row["defender_roi"] = (
            (foreign["risk_share"] - row["risk_share"]) / row["defender_cost"]
            if row["defender_cost"] > 0
            else 0.0
        )
    return rows


def run_budget_sweep(
    config: InfluenceConfig | None = None,
    budgets: list[float] | None = None,
    risk_threshold: float = 0.18,
) -> list[dict]:
    config = config or InfluenceConfig()
    budgets = budgets or [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0]
    rows = []
    for budget in budgets:
        campaign = campaign_catalog(adversary_budget=budget, defender_budget=0.0)[1]
        row = simulate_campaign(config, campaign)
        row["crosses_threshold"] = bool(row["risk_share"] >= risk_threshold)
        rows.append(row)
    return rows


def summarize(campaigns: list[dict], budget_sweep: list[dict], risk_threshold: float = 0.18) -> dict:
    compact_campaigns = [
        {key: value for key, value in row.items() if key != "risk_series"}
        for row in campaigns
    ]
    defenses = [row for row in compact_campaigns if row["defender_cost"] > 0]
    best_defense = max(defenses, key=lambda row: row["defender_roi"]) if defenses else None
    foreign = next(row for row in compact_campaigns if row["campaign"] == "foreign_wedge")
    threshold_rows = [row for row in budget_sweep if row["risk_share"] >= risk_threshold]
    critical_budget = min((row["adversary_budget"] for row in threshold_rows), default=None)
    return {
        "campaigns": compact_campaigns,
        "budget_sweep": [
            {key: value for key, value in row.items() if key != "risk_series"}
            for row in budget_sweep
        ],
        "foreign_wedge_risk": foreign["risk_share"],
        "foreign_wedge_adversary_roi": foreign["adversary_roi"],
        "best_defense": best_defense,
        "risk_threshold": risk_threshold,
        "critical_budget": critical_budget,
        "institutional_readout": (
            "threshold_crossed"
            if critical_budget is not None
            else "no_threshold_crossing"
        ),
    }


def render_report(summary: dict) -> str:
    lines = [
        "# Geopolitical Influence DTE Report",
        "",
        "## Scope",
        "",
        (
            "This is a defensive, abstract influence-wargame simulation. Actors are "
            "budgeted perturbations to recommendation topology, not operational bot tactics."
        ),
        "",
        "## Campaign Outcomes",
        "",
        "| Campaign | Risk | Target | Protective | Escape | Polarization | Adv ROI | Def ROI | Lock-In |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["campaigns"]:
        lines.append(
            f"| {row['campaign']} | {row['risk_share']:.3f} | {row['target_share']:.3f} | "
            f"{row['protective_share']:.3f} | {row['escape_probability']:.3f} | "
            f"{row['polarization_gap']:.3f} | {row['adversary_roi']:.4f} | "
            f"{row['defender_roi']:.4f} | {row['lock_in']} |"
        )

    lines.extend([
        "",
        "## Budget Threshold",
        "",
        "| Adversary Budget | Risk | Target | Escape | Crosses Threshold |",
        "|---:|---:|---:|---:|---|",
    ])
    for row in summary["budget_sweep"]:
        lines.append(
            f"| {row['adversary_budget']:.2f} | {row['risk_share']:.3f} | "
            f"{row['target_share']:.3f} | {row['escape_probability']:.3f} | "
            f"{row['crosses_threshold']} |"
        )

    best = summary.get("best_defense")
    if best:
        lines.extend([
            "",
            "## Readout",
            "",
            (
                f"Best defender ROI in this run is `{best['campaign']}` at "
                f"{best['defender_roi']:.4f}. Critical adversary budget for the "
                f"{summary['risk_threshold']:.2f} risk threshold is "
                f"`{summary['critical_budget']}`."
            ),
        ])
    return "\n".join(lines) + "\n"


def write_outputs(
    summary: dict,
    output_json: Path = Path("geopolitical_influence_output.json"),
    output_md: Path = Path("GEOPOLITICAL_INFLUENCE_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_md.write_text(render_report(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run defensive geopolitical influence DTE simulation.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--adversary-budget", type=float, default=1.0)
    parser.add_argument("--defender-budget", type=float, default=1.0)
    parser.add_argument("--risk-threshold", type=float, default=0.18)
    parser.add_argument("--output-json", type=Path, default=Path("geopolitical_influence_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("GEOPOLITICAL_INFLUENCE_REPORT.md"))
    args = parser.parse_args()

    config = InfluenceConfig(
        agents=args.agents if args.agents is not None else (96 if args.quick else 320),
        steps=args.steps if args.steps is not None else (32 if args.quick else 96),
    )
    campaigns = run_campaign_set(config, args.adversary_budget, args.defender_budget)
    budget_sweep = run_budget_sweep(config, risk_threshold=args.risk_threshold)
    summary = summarize(campaigns, budget_sweep, args.risk_threshold)
    write_outputs(summary, args.output_json, args.output_md)
    print(json.dumps({
        "foreign_wedge_risk": summary["foreign_wedge_risk"],
        "best_defense": summary["best_defense"]["campaign"] if summary["best_defense"] else None,
        "critical_budget": summary["critical_budget"],
        "institutional_readout": summary["institutional_readout"],
    }, indent=2))


if __name__ == "__main__":
    main()

"""
Legitimacy ontology for DTE influence-resilience experiments.

This module grounds narrative simulations in explicit civic-legitimacy
dimensions and basin labels. It is intentionally defensive: it supports
aggregate topology-risk analysis, not attribution or individual profiling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_LABELS = [
    "legitimacy_hunger",
    "institutional_trust",
    "recognition_grievance",
    "agency_sensitivity",
    "corruption_sensitivity",
    "identity_salience",
    "epistemic_certainty",
    "threat_arousal",
    "anti_elite_resentment",
    "procedural_fairness",
    "civic_repair_orientation",
    "pluralism_tolerance",
]

BASIN_CLASSES = [
    "legitimacy_preserving",
    "grievance_open",
    "delegitimation",
    "epistemic_closure",
    "mobilization_pressure",
]

RISK_BASINS = {"delegitimation", "epistemic_closure", "mobilization_pressure"}
EXIT_BASINS = {"legitimacy_preserving", "grievance_open"}


@dataclass(frozen=True)
class FeatureDimension:
    key: str
    description: str
    protected_meaning: str
    risk_when_exploited: str


@dataclass(frozen=True)
class NarrativeNode:
    key: str
    label: str
    basin_class: str
    features: tuple[float, ...]
    description: str

    def vector(self) -> np.ndarray:
        return np.array(self.features, dtype=np.float64)


@dataclass(frozen=True)
class LegitimacyMetricWeights:
    exit_conductance: float = 0.40
    irreversibility: float = 0.25
    entropy_production: float = 0.15
    closure_pressure: float = 0.20


FEATURE_DIMENSIONS = [
    FeatureDimension(
        "legitimacy_hunger",
        "Need for institutions to justify authority and deserve consent.",
        "Demand for accountable authority.",
        "Authority is framed as impossible to legitimate.",
    ),
    FeatureDimension(
        "institutional_trust",
        "Prior trust in institutional competence and good faith.",
        "Trust calibrated by evidence and performance.",
        "Trust is collapsed into naive obedience or total cynicism.",
    ),
    FeatureDimension(
        "recognition_grievance",
        "Need to be seen, respected, and not humiliated.",
        "Legitimate demand for dignity and recognition.",
        "Humiliation becomes a route into revenge identity.",
    ),
    FeatureDimension(
        "agency_sensitivity",
        "Reactivity to coercion, censorship, surveillance, or elite control.",
        "Concern for autonomy and consent.",
        "Every constraint is framed as enemy domination.",
    ),
    FeatureDimension(
        "corruption_sensitivity",
        "Salience of betrayal, graft, hypocrisy, or rigged systems.",
        "Detection of real institutional failure.",
        "All institutions are framed as inherently fraudulent.",
    ),
    FeatureDimension(
        "identity_salience",
        "Degree to which group identity organizes interpretation.",
        "Belonging and collective meaning.",
        "Narratives route all evidence through friend/enemy identity.",
    ),
    FeatureDimension(
        "epistemic_certainty",
        "Preference for closure, certainty, and totalizing explanation.",
        "Need for coherence under uncertainty.",
        "Contrary evidence is absorbed as proof of conspiracy.",
    ),
    FeatureDimension(
        "threat_arousal",
        "Fear, urgency, anger, danger, and emergency framing.",
        "Attention to actual danger.",
        "Emergency framing suppresses deliberation and exits.",
    ),
    FeatureDimension(
        "anti_elite_resentment",
        "Hostility toward perceived detached or contemptuous elites.",
        "Critique of elite failure and status contempt.",
        "Anti-elite affect becomes generalized delegitimation.",
    ),
    FeatureDimension(
        "procedural_fairness",
        "Orientation toward due process and rule-governed legitimacy.",
        "Institutions remain contestable and corrigible.",
        "Procedural routes are framed as fake or useless.",
    ),
    FeatureDimension(
        "civic_repair_orientation",
        "Availability of constructive institutional or local repair pathways.",
        "Grievance can route into reform, accountability, and repair.",
        "Repair exits are blocked or mocked as complicity.",
    ),
    FeatureDimension(
        "pluralism_tolerance",
        "Tolerance for ambiguity, dissent, and multiple legitimate perspectives.",
        "Disagreement remains survivable.",
        "Dissent is treated as betrayal or enemy contamination.",
    ),
]


NARRATIVE_NODES = [
    NarrativeNode(
        "procedural_repair",
        "Procedural Repair",
        "legitimacy_preserving",
        (0.70, 0.75, 0.35, 0.35, 0.45, 0.20, 0.20, 0.15, 0.20, 0.95, 0.95, 0.85),
        "Critique routed through due process, reform, auditability, and repair.",
    ),
    NarrativeNode(
        "local_civic_agency",
        "Local Civic Agency",
        "legitimacy_preserving",
        (0.55, 0.65, 0.45, 0.65, 0.35, 0.35, 0.20, 0.20, 0.25, 0.80, 0.95, 0.75),
        "Local action and community repair remain reachable.",
    ),
    NarrativeNode(
        "credible_investigation",
        "Credible Investigation",
        "legitimacy_preserving",
        (0.80, 0.70, 0.25, 0.35, 0.75, 0.15, 0.25, 0.20, 0.25, 0.90, 0.75, 0.85),
        "Institutional failure is examined through evidence and accountability.",
    ),
    NarrativeNode(
        "unheard_grievance",
        "Unheard Grievance",
        "grievance_open",
        (0.80, 0.35, 0.90, 0.75, 0.65, 0.65, 0.45, 0.50, 0.55, 0.45, 0.45, 0.40),
        "Recognition grievance is intense but not yet closed to repair.",
    ),
    NarrativeNode(
        "elite_hypocrisy",
        "Elite Hypocrisy",
        "grievance_open",
        (0.75, 0.30, 0.70, 0.65, 0.90, 0.45, 0.55, 0.55, 0.75, 0.35, 0.35, 0.35),
        "Elite failure and double standards are foregrounded.",
    ),
    NarrativeNode(
        "rigged_system",
        "Rigged System",
        "delegitimation",
        (0.90, 0.05, 0.85, 0.85, 0.95, 0.75, 0.85, 0.80, 0.90, 0.10, 0.05, 0.10),
        "Institutions are framed as structurally fraudulent and irredeemable.",
    ),
    NarrativeNode(
        "state_is_alien",
        "State Is Alien",
        "delegitimation",
        (0.85, 0.05, 0.80, 0.90, 0.85, 0.90, 0.80, 0.85, 0.95, 0.05, 0.05, 0.05),
        "The state is framed as an occupying enemy rather than a corrigible institution.",
    ),
    NarrativeNode(
        "truth_is_suppressed",
        "Truth Is Suppressed",
        "epistemic_closure",
        (0.70, 0.05, 0.65, 0.95, 0.80, 0.65, 0.98, 0.85, 0.85, 0.05, 0.05, 0.05),
        "Contrary evidence is treated as proof that the hidden truth is being suppressed.",
    ),
    NarrativeNode(
        "all_exits_are_controlled",
        "All Exits Are Controlled",
        "epistemic_closure",
        (0.75, 0.02, 0.70, 0.95, 0.85, 0.70, 1.00, 0.90, 0.90, 0.02, 0.02, 0.02),
        "Credible off-ramps are preemptively framed as enemy manipulation.",
    ),
    NarrativeNode(
        "defensive_escalation",
        "Defensive Escalation",
        "mobilization_pressure",
        (0.70, 0.05, 0.80, 0.95, 0.85, 0.85, 0.90, 1.00, 0.90, 0.05, 0.02, 0.02),
        "Urgency and enemy construction make escalation appear defensive.",
    ),
]


def feature_index(name: str) -> int:
    return FEATURE_LABELS.index(name)


def node_by_key(key: str) -> NarrativeNode:
    for node in NARRATIVE_NODES:
        if node.key == key:
            return node
    raise KeyError(key)


def validate_ontology() -> None:
    if len(FEATURE_LABELS) != len(FEATURE_DIMENSIONS):
        raise ValueError("Feature labels and dimensions are out of sync")
    known_basins = set(BASIN_CLASSES)
    for node in NARRATIVE_NODES:
        if node.basin_class not in known_basins:
            raise ValueError(f"Unknown basin class for {node.key}: {node.basin_class}")
        if len(node.features) != len(FEATURE_LABELS):
            raise ValueError(f"Feature length mismatch for {node.key}")
        if not all(0.0 <= value <= 1.0 for value in node.features):
            raise ValueError(f"Feature out of [0,1] range for {node.key}")


def legitimacy_drift(telemetry: np.ndarray) -> float:
    """Positive means repair/pluralism dominates closure/arousal pressure."""
    t = np.asarray(telemetry, dtype=np.float64)
    positive = (
        t[feature_index("procedural_fairness")]
        + t[feature_index("civic_repair_orientation")]
        + t[feature_index("pluralism_tolerance")]
    )
    negative = (
        t[feature_index("threat_arousal")]
        + t[feature_index("epistemic_certainty")]
        + t[feature_index("anti_elite_resentment")]
    )
    return float((positive - negative) / 3.0)


def closure_pressure(telemetry: np.ndarray) -> float:
    """High values indicate epistemic closure pressure rather than grievance alone."""
    t = np.asarray(telemetry, dtype=np.float64)
    raw = (
        t[feature_index("epistemic_certainty")]
        + t[feature_index("threat_arousal")]
        + t[feature_index("anti_elite_resentment")]
        - t[feature_index("pluralism_tolerance")]
    ) / 3.0
    return float(np.clip(raw, 0.0, 1.0))


def basin_indices(nodes: list[NarrativeNode], basin_classes: set[str]) -> list[int]:
    return [idx for idx, node in enumerate(nodes) if node.basin_class in basin_classes]


def legitimacy_warning_score(
    exit_conductance: float,
    irreversibility: float,
    entropy_production: float,
    closure: float,
    weights: LegitimacyMetricWeights = LegitimacyMetricWeights(),
) -> float:
    score = (
        weights.exit_conductance * (1.0 - np.clip(exit_conductance, 0.0, 1.0))
        + weights.irreversibility * np.clip(irreversibility, 0.0, 1.0)
        + weights.entropy_production * np.clip(entropy_production, 0.0, 1.0)
        + weights.closure_pressure * np.clip(closure, 0.0, 1.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def project_social_telemetry_to_legitimacy(
    telemetry: np.ndarray,
    social_feature_labels: list[str],
) -> np.ndarray:
    """
    Project the Social Media adapter telemetry into legitimacy-ontology space.

    This is an explicit bridge for the current prototype. It keeps the game
    running on the social-media topology while letting legitimacy metrics use
    political-theoretic coordinates such as institutional_trust and
    procedural_fairness.
    """
    t = np.asarray(telemetry, dtype=np.float64)
    if t.ndim == 1:
        t = t[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    lookup = {label: idx for idx, label in enumerate(social_feature_labels)}

    def s(label: str) -> np.ndarray:
        return t[:, lookup[label]]

    legitimacy = np.zeros((t.shape[0], len(FEATURE_LABELS)), dtype=np.float64)
    values = {
        "legitimacy_hunger": 0.55 * s("News") + 0.35 * s("Identity") + 0.10 * s("Depth"),
        "institutional_trust": 0.70 * s("Credibility") + 0.20 * s("Depth") + 0.10 * (1.0 - s("Conflict")),
        "recognition_grievance": 0.55 * s("Identity") + 0.30 * s("Social Proof") + 0.15 * s("Conflict"),
        "agency_sensitivity": 0.45 * s("Identity") + 0.35 * s("Novelty") + 0.20 * s("Conflict"),
        "corruption_sensitivity": 0.50 * s("News") + 0.35 * s("Conflict") + 0.15 * (1.0 - s("Credibility")),
        "identity_salience": 0.70 * s("Identity") + 0.20 * s("Social Proof") + 0.10 * s("Conflict"),
        "epistemic_certainty": 0.55 * (1.0 - s("Credibility")) + 0.25 * s("Conflict") + 0.20 * (1.0 - s("Depth")),
        "threat_arousal": 0.75 * s("Conflict") + 0.25 * s("Novelty"),
        "anti_elite_resentment": 0.55 * s("Conflict") + 0.25 * s("Identity") + 0.20 * (1.0 - s("Credibility")),
        "procedural_fairness": 0.55 * s("Credibility") + 0.25 * s("News") + 0.20 * s("Depth"),
        "civic_repair_orientation": 0.45 * s("Depth") + 0.35 * s("Practicality") + 0.20 * s("Credibility"),
        "pluralism_tolerance": 0.45 * s("Depth") + 0.35 * s("Credibility") + 0.20 * (1.0 - s("Conflict")),
    }
    for key, value in values.items():
        legitimacy[:, feature_index(key)] = np.clip(value, 0.0, 1.0)
    return legitimacy[0] if squeeze else legitimacy


def ontology_summary() -> dict:
    validate_ontology()
    return {
        "feature_count": len(FEATURE_LABELS),
        "node_count": len(NARRATIVE_NODES),
        "basin_classes": BASIN_CLASSES,
        "risk_node_keys": [
            node.key for node in NARRATIVE_NODES if node.basin_class in RISK_BASINS
        ],
        "exit_node_keys": [
            node.key for node in NARRATIVE_NODES if node.basin_class in EXIT_BASINS
        ],
    }


validate_ontology()

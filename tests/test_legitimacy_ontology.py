import numpy as np

from legitimacy_ontology import (
    EXIT_BASINS,
    FEATURE_LABELS,
    NARRATIVE_NODES,
    RISK_BASINS,
    basin_indices,
    closure_pressure,
    legitimacy_drift,
    legitimacy_warning_score,
    node_by_key,
    ontology_summary,
    project_social_telemetry_to_legitimacy,
    validate_ontology,
)


def test_ontology_validates_and_summarizes():
    validate_ontology()
    summary = ontology_summary()

    assert summary["feature_count"] == len(FEATURE_LABELS)
    assert summary["node_count"] == len(NARRATIVE_NODES)
    assert "rigged_system" in summary["risk_node_keys"]
    assert "procedural_repair" in summary["exit_node_keys"]


def test_risk_and_exit_basin_indices_are_disjoint():
    risk = set(basin_indices(NARRATIVE_NODES, RISK_BASINS))
    exits = set(basin_indices(NARRATIVE_NODES, EXIT_BASINS))

    assert risk
    assert exits
    assert risk.isdisjoint(exits)


def test_legitimacy_metrics_distinguish_repair_from_closure():
    repair = node_by_key("procedural_repair").vector()
    closure = node_by_key("truth_is_suppressed").vector()

    assert legitimacy_drift(repair) > legitimacy_drift(closure)
    assert closure_pressure(closure) > closure_pressure(repair)


def test_warning_score_is_bounded_and_increases_with_closure():
    calm = legitimacy_warning_score(
        exit_conductance=0.9,
        irreversibility=0.1,
        entropy_production=0.1,
        closure=0.1,
    )
    stressed = legitimacy_warning_score(
        exit_conductance=0.2,
        irreversibility=0.7,
        entropy_production=0.5,
        closure=0.8,
    )

    assert 0.0 <= calm <= 1.0
    assert 0.0 <= stressed <= 1.0
    assert stressed > calm


def test_node_vectors_are_numeric_unit_interval():
    matrix = np.vstack([node.vector() for node in NARRATIVE_NODES])

    assert matrix.shape == (len(NARRATIVE_NODES), len(FEATURE_LABELS))
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)


def test_social_projection_populates_institutional_trust():
    social_labels = [
        "News", "Entertainment", "Practicality", "Identity", "Conflict",
        "Novelty", "Social Proof", "Credibility", "Commercial", "Depth",
    ]
    telemetry = np.array([0.7, 0.1, 0.5, 0.2, 0.1, 0.2, 0.2, 0.9, 0.1, 0.8])
    projected = project_social_telemetry_to_legitimacy(telemetry, social_labels)

    assert projected.shape == (len(FEATURE_LABELS),)
    assert projected[FEATURE_LABELS.index("institutional_trust")] > 0.75
    assert projected[FEATURE_LABELS.index("procedural_fairness")] > 0.75

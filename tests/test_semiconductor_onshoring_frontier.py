from pathlib import Path

from semiconductor_onshoring_frontier import (
    FrontierConfig,
    StrategyPoint,
    evaluate_point,
    control_for,
    render_report,
    run_frontier,
    scenario_for,
    write_outputs,
)
from semiconductor_onshoring import OnshoringConfig, simulate


def test_strategy_point_evaluation_has_frontier_metrics():
    config = FrontierConfig(agents=40, steps=20)
    baseline_point = StrategyPoint(0.0, 1.0, 1.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0)
    baseline = simulate(
        OnshoringConfig(agents=config.agents, steps=config.steps, seed=config.seed),
        scenario_for(baseline_point),
        None,
    )
    row = evaluate_point(
        config,
        StrategyPoint(1.0, 1.2, 1.5, 0.8, 1.0, 1.0, 0.9, 0.8, 1.5),
        baseline,
    )

    assert 0.0 <= row["onshore_share"] <= 1.0
    assert row["packaging_capacity_multiplier"] == 1.5
    assert row["finished_flow_ratio"] >= 0.0
    assert row["classification"]


def test_control_preserves_independent_edge_magnitudes():
    control = control_for(
        StrategyPoint(0.0, 1.0, 1.0, 0.8, 0.4, 1.1, 1.3, 0.9, 2.5),
    )

    assert control.beta_edge_boosts[("CHIPS Subsidy Credit", "Intel US Fabs")] == 0.8
    assert control.beta_edge_boosts[("NVIDIA AI Accelerator Demand", "TSMC Arizona Fabs")] == 2.5
    assert control.friction_edge_deltas[("Section 232 Tariff Offset", "TSMC Arizona Fabs")] == 0.4
    assert control.friction_edge_deltas[("Japan Chemicals Materials", "US Wafer Fabrication")] == 0.9


def test_quick_frontier_and_outputs(tmp_path: Path):
    payload = run_frontier(FrontierConfig(agents=32, steps=16), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["best_score"]
    assert "Semiconductor Onshoring Frontier Report" in report
    write_outputs(payload, tmp_path / "frontier.json", tmp_path / "frontier.md")
    assert (tmp_path / "frontier.json").exists()
    assert (tmp_path / "frontier.md").exists()

from pathlib import Path

from social_media_phase import (
    SimulationConfig,
    render_report,
    run_interventions,
    run_phase_grid,
    simulate,
    summarize,
    write_outputs,
)


def test_social_phase_simulation_metrics_are_bounded():
    result = simulate(
        SimulationConfig(
            intent="High-Arousal Scroll",
            feedback_rate=0.2,
            temperature=1.0,
            noise_sigma=0.0,
            agents=32,
            steps=16,
        )
    )
    for key in [
        "risk_share",
        "protective_share",
        "escape_probability",
        "edge_current_norm",
        "entropy_production",
        "lock_in",
    ]:
        assert key in result
    assert 0.0 <= result["risk_share"] <= 1.0
    assert 0.0 <= result["protective_share"] <= 1.0
    assert 0.0 <= result["escape_probability"] <= 1.0
    assert len(result["risk_series"]) == 16


def test_phase_grid_respects_requested_dimensions():
    grid = run_phase_grid(
        lambdas=[0.0, 0.2],
        taus=[0.8],
        sigmas=[0.0],
        intents=["High-Arousal Scroll", "Deep Research"],
        agents=24,
        steps=12,
    )
    assert len(grid) == 4
    assert {row["intent"] for row in grid} == {"High-Arousal Scroll", "Deep Research"}


def test_intervention_rows_have_roi_fields():
    rows = run_interventions(
        SimulationConfig(
            intent="High-Arousal Scroll",
            feedback_rate=0.25,
            temperature=0.8,
            noise_sigma=0.0,
            agents=32,
            steps=16,
        )
    )
    assert any(row["intervention"] == "baseline" for row in rows)
    assert any(row["intervention"] == "adversarial_rumor_beta" for row in rows)
    for row in rows:
        assert "cost" in row
        assert "risk_delta_vs_baseline" in row
        assert "risk_reduction_per_cost" in row


def test_report_renderer_and_writer(tmp_path: Path):
    grid = run_phase_grid(
        lambdas=[0.2],
        taus=[1.0],
        sigmas=[0.0],
        intents=["High-Arousal Scroll"],
        agents=24,
        steps=12,
    )
    interventions = run_interventions(
        SimulationConfig(
            intent="High-Arousal Scroll",
            feedback_rate=0.2,
            temperature=1.0,
            noise_sigma=0.0,
            agents=24,
            steps=12,
        )
    )
    summary = summarize(grid, interventions)
    report = render_report(summary, interventions)
    assert "Social Media Phase Diagram Report" in report
    payload = write_outputs(
        grid,
        interventions,
        output_json=tmp_path / "phase.json",
        output_md=tmp_path / "phase.md",
    )
    assert payload["summary"]["grid_cells"] == 1
    assert (tmp_path / "phase.json").exists()
    assert (tmp_path / "phase.md").exists()

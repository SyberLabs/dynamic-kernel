from pathlib import Path

from geopolitical_influence import (
    InfluenceConfig,
    campaign_catalog,
    render_report,
    run_budget_sweep,
    run_campaign_set,
    simulate_campaign,
    summarize,
    write_outputs,
)


def test_campaign_simulation_metrics_are_bounded():
    config = InfluenceConfig(agents=32, steps=12, noise_sigma=0.0)
    campaign = campaign_catalog(adversary_budget=0.75, defender_budget=0.5)[1]
    row = simulate_campaign(config, campaign)

    assert row["campaign"] == "foreign_wedge"
    for key in [
        "risk_share",
        "protective_share",
        "target_share",
        "credibility_exposure",
        "escape_probability",
        "population_entropy",
        "polarization_gap",
    ]:
        assert 0.0 <= row[key] <= 1.0
    assert row["adversary_cost"] > 0
    assert len(row["risk_series"]) == 12


def test_campaign_set_adds_roi_fields():
    rows = run_campaign_set(
        InfluenceConfig(agents=32, steps=12, noise_sigma=0.0),
        adversary_budget=0.75,
        defender_budget=0.5,
    )

    assert {row["campaign"] for row in rows} >= {"baseline", "foreign_wedge", "combined_defense"}
    for row in rows:
        assert "risk_delta_vs_baseline" in row
        assert "risk_delta_vs_foreign_wedge" in row
        assert "adversary_roi" in row
        assert "defender_roi" in row


def test_budget_sweep_and_report_outputs(tmp_path: Path):
    config = InfluenceConfig(agents=24, steps=10, noise_sigma=0.0)
    campaigns = run_campaign_set(config, adversary_budget=0.5, defender_budget=0.5)
    sweep = run_budget_sweep(config, budgets=[0.0, 0.5], risk_threshold=0.10)
    summary = summarize(campaigns, sweep, risk_threshold=0.10)
    report = render_report(summary)

    assert "Geopolitical Influence DTE Report" in report
    assert "best_defense" in summary
    assert len(summary["budget_sweep"]) == 2
    write_outputs(summary, tmp_path / "geo.json", tmp_path / "geo.md")
    assert (tmp_path / "geo.json").exists()
    assert (tmp_path / "geo.md").exists()

from pathlib import Path

from influence_budget_sweep import render_report, run_budget_sweep, summarize_budget_sweep, write_outputs


def test_budget_sweep_summary_identifies_schema():
    summary = summarize_budget_sweep([
        {
            "defender_budget": 0.0,
            "q_eval_peak_risk": 0.20,
            "q_threshold_crossed": True,
        },
        {
            "defender_budget": 5.0,
            "q_eval_peak_risk": 0.15,
            "q_threshold_crossed": False,
        },
    ])

    assert summary["min_safe_budget"] == 5.0
    assert summary["robust_safe_budget"] == 5.0
    assert summary["threshold_crossing_budgets"] == [0.0]


def test_budget_sweep_tiny_run_and_outputs(tmp_path: Path):
    payload = run_budget_sweep(
        budgets=[0.0, 5.0],
        episodes=1,
        agents=24,
        rounds=2,
        steps_per_round=4,
    )
    report = render_report(payload)

    assert "Minimum Defender Budget Report" in report
    assert len(payload["rows"]) == 2
    write_outputs(payload, tmp_path / "budget.json", tmp_path / "budget.md")
    assert (tmp_path / "budget.json").exists()
    assert (tmp_path / "budget.md").exists()

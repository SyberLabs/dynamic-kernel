from pathlib import Path

from influence_wargame import render_report, run_wargame, write_outputs


def test_wargame_tiny_sweep_has_summary():
    payload = run_wargame(
        lambdas=[0.35],
        taus=[0.65],
        adversary_budgets=[6.0],
        defender_budgets=[0.0, 3.0],
        defender_policies=["none", "risk_threshold", "structural_warning"],
        seeds=1,
        agents=24,
        rounds=2,
        steps_per_round=4,
    )

    assert payload["summary"]["total_runs"] == 6
    assert payload["summary"]["cells"] == 2
    assert "structural_warning" in payload["summary"]["policy_summary"]
    assert "best_policy_counts" in payload["summary"]


def test_wargame_report_and_outputs(tmp_path: Path):
    payload = run_wargame(
        lambdas=[0.35],
        taus=[0.65],
        adversary_budgets=[6.0],
        defender_budgets=[3.0],
        defender_policies=["none", "structural_warning"],
        seeds=1,
        agents=24,
        rounds=2,
        steps_per_round=4,
    )
    report = render_report(payload)

    assert "Influence War Game Report" in report
    assert "Policy Summary" in report
    write_outputs(payload, tmp_path / "wargame.json", tmp_path / "wargame.md")
    assert (tmp_path / "wargame.json").exists()
    assert (tmp_path / "wargame.md").exists()

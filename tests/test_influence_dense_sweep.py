from pathlib import Path

from influence_dense_sweep import render_report, run_dense_sweep, summarize_dense_sweep, write_outputs


def test_dense_summary_reports_first_safe_and_robust_safe_budget():
    rows = [
        {
            "seed": 1,
            "defender_budget": 0.0,
            "q_eval_peak_risk": 0.19,
            "q_eval_final_risk": 0.18,
            "q_threshold_crossed": True,
            "q_pause_rate": 0.0,
        },
        {
            "seed": 2,
            "defender_budget": 0.0,
            "q_eval_peak_risk": 0.16,
            "q_eval_final_risk": 0.14,
            "q_threshold_crossed": False,
            "q_pause_rate": 0.0,
        },
        {
            "seed": 1,
            "defender_budget": 0.25,
            "q_eval_peak_risk": 0.15,
            "q_eval_final_risk": 0.12,
            "q_threshold_crossed": False,
            "q_pause_rate": 0.1,
        },
        {
            "seed": 2,
            "defender_budget": 0.25,
            "q_eval_peak_risk": 0.14,
            "q_eval_final_risk": 0.11,
            "q_threshold_crossed": False,
            "q_pause_rate": 0.1,
        },
    ]

    summary = summarize_dense_sweep(rows)

    assert summary["all_seed_safe_budget"] == 0.25
    assert summary["empirical_robust_safe_budget"] == 0.25
    assert summary["monotonicity_violation_count"] == 0


def test_dense_sweep_tiny_run_and_outputs(tmp_path: Path):
    payload = run_dense_sweep(
        budgets=[0.0, 0.25],
        seeds=1,
        episodes=1,
        agents=24,
        rounds=2,
        steps_per_round=4,
    )
    report = render_report(payload)

    assert "Dense Defender Budget Frontier Report" in report
    assert "Interpretation Guardrail" in report
    assert payload["summary"]["runs"] == 2
    write_outputs(payload, tmp_path / "dense.json", tmp_path / "dense.md")
    assert (tmp_path / "dense.json").exists()
    assert (tmp_path / "dense.md").exists()

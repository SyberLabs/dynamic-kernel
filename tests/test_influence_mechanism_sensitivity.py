from pathlib import Path

from influence_mechanism_sensitivity import (
    infer_mechanism,
    render_report,
    run_mechanism_sensitivity,
    summarize_axis,
    write_outputs,
)


def test_summarize_axis_finds_group_thresholds():
    rows = [
        {
            "action_unit": 1.0,
            "defender_budget": 0.0,
            "seed": 1,
            "q_eval_peak_risk": 0.19,
            "q_eval_final_risk": 0.18,
            "q_threshold_crossed": True,
            "q_pause_rate": 0.0,
        },
        {
            "action_unit": 1.0,
            "defender_budget": 0.25,
            "seed": 1,
            "q_eval_peak_risk": 0.15,
            "q_eval_final_risk": 0.12,
            "q_threshold_crossed": False,
            "q_pause_rate": 0.0,
        },
        {
            "action_unit": 2.0,
            "defender_budget": 0.0,
            "seed": 1,
            "q_eval_peak_risk": 0.16,
            "q_eval_final_risk": 0.14,
            "q_threshold_crossed": False,
            "q_pause_rate": 0.0,
        },
        {
            "action_unit": 2.0,
            "defender_budget": 0.25,
            "seed": 1,
            "q_eval_peak_risk": 0.14,
            "q_eval_final_risk": 0.11,
            "q_threshold_crossed": False,
            "q_pause_rate": 0.0,
        },
    ]

    summary = summarize_axis(rows, group_key="action_unit")

    assert summary["groups"]["1.0"]["empirical_robust_safe_budget"] == 0.25
    assert summary["groups"]["2.0"]["empirical_robust_safe_budget"] == 0.0


def test_infer_mechanism_flags_immune_bootstrap():
    payload = {
        "action_unit": {
            "summary": {
                "groups": {
                    "0.5": {"empirical_robust_safe_budget": 0.25},
                    "1.0": {"empirical_robust_safe_budget": 0.25},
                }
            }
        },
        "immune": {
            "summary": {
                "groups": {
                    "none": {"empirical_robust_safe_budget": 1.0},
                    "default": {"empirical_robust_safe_budget": 0.25},
                }
            }
        },
    }

    inference = infer_mechanism(payload)

    assert inference["immune_bootstrap_supported"] is True
    assert inference["action_unit_artifact_supported"] is False


def test_mechanism_sensitivity_tiny_run_and_outputs(tmp_path: Path):
    payload = run_mechanism_sensitivity(
        budgets=[0.0, 0.25],
        action_units=[0.5],
        seeds=1,
        episodes=1,
        agents=16,
        rounds=2,
        steps_per_round=3,
    )
    report = render_report(payload)

    assert "Mechanism Sensitivity Report" in report
    assert "Immune-Response Sensitivity" in report
    write_outputs(payload, tmp_path / "mechanism.json", tmp_path / "mechanism.md")
    assert (tmp_path / "mechanism.json").exists()
    assert (tmp_path / "mechanism.md").exists()

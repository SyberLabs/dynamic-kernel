from pathlib import Path

from influence_seed_sweep import (
    render_report,
    run_seed_robust_sweeps,
    summarize_policy_seed_sweep,
    summarize_q_budget_seed_sweep,
    write_outputs,
)


def test_q_budget_seed_summary_identifies_robust_safe_budget():
    rows = [
        {
            "seed": 1,
            "defender_budget": 0.0,
            "q_eval_peak_risk": 0.20,
            "q_eval_final_risk": 0.16,
            "baseline_peak_risk": 0.18,
            "q_threshold_crossed": True,
            "q_pause_rate": 0.0,
            "q_action_counts": {"amplify": 1},
        },
        {
            "seed": 2,
            "defender_budget": 0.0,
            "q_eval_peak_risk": 0.16,
            "q_eval_final_risk": 0.12,
            "baseline_peak_risk": 0.18,
            "q_threshold_crossed": False,
            "q_pause_rate": 0.0,
            "q_action_counts": {"polarize": 1},
        },
        {
            "seed": 1,
            "defender_budget": 5.0,
            "q_eval_peak_risk": 0.14,
            "q_eval_final_risk": 0.10,
            "baseline_peak_risk": 0.19,
            "q_threshold_crossed": False,
            "q_pause_rate": 0.25,
            "q_action_counts": {"pause": 1},
        },
        {
            "seed": 2,
            "defender_budget": 5.0,
            "q_eval_peak_risk": 0.15,
            "q_eval_final_risk": 0.11,
            "baseline_peak_risk": 0.19,
            "q_threshold_crossed": False,
            "q_pause_rate": 0.25,
            "q_action_counts": {"stealth_amplify": 1},
        },
    ]

    summary = summarize_q_budget_seed_sweep(rows)

    assert summary["all_seed_safe_budget"] == 5.0
    assert summary["empirical_robust_safe_budget"] == 5.0
    assert summary["mean_safe_budget"] == 5.0
    assert summary["aggregate_q_action_counts"]["amplify"] == 1


def test_policy_seed_summary_compares_exp3_to_fixed():
    rows = [
        {
            "seed": 1,
            "policy": "structural_warning",
            "threshold_crossed": False,
            "peak_risk": 0.15,
            "final_risk": 0.10,
            "mean_risk": 0.08,
            "final_escape_probability": 0.60,
            "total_defender_cost": 3.0,
        },
        {
            "seed": 1,
            "policy": "exp3_defender",
            "threshold_crossed": False,
            "peak_risk": 0.16,
            "final_risk": 0.12,
            "mean_risk": 0.09,
            "final_escape_probability": 0.58,
            "total_defender_cost": 4.0,
        },
    ]

    summary = summarize_policy_seed_sweep(rows)

    assert summary["best_policy_counts"]["structural_warning"] == 1
    assert summary["exp3_minus_best_fixed_final_risk"] > 0


def test_seed_robust_sweep_tiny_run_and_outputs(tmp_path: Path):
    payload = run_seed_robust_sweeps(
        budgets=[0.0, 5.0],
        policies=["none", "structural_warning", "exp3_defender"],
        seeds=1,
        episodes=1,
        agents=24,
        rounds=2,
        steps_per_round=4,
    )
    report = render_report(payload)

    assert "Seed-Robust Influence Sweep Report" in report
    assert payload["q_budget"]["summary"]["runs"] == 2
    assert payload["policy"]["summary"]["runs"] == 3
    write_outputs(payload, tmp_path / "seed.json", tmp_path / "seed.md")
    assert (tmp_path / "seed.json").exists()
    assert (tmp_path / "seed.md").exists()

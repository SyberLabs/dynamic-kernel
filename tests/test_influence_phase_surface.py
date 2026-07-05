from pathlib import Path

from influence_phase_surface import render_report, run_phase_surface, summarize_phase_surface, write_outputs


def test_phase_surface_summary_marks_safe_gain_by_action_unit():
    rows = [
        {
            "immune_gain": 0.0,
            "action_unit": 1.0,
            "seed": 1,
            "q_eval_peak_risk": 0.19,
            "q_eval_final_risk": 0.18,
            "q_threshold_crossed": True,
            "q_pause_rate": 0.0,
        },
        {
            "immune_gain": 0.75,
            "action_unit": 1.0,
            "seed": 1,
            "q_eval_peak_risk": 0.15,
            "q_eval_final_risk": 0.12,
            "q_threshold_crossed": False,
            "q_pause_rate": 0.0,
        },
        {
            "immune_gain": 1.25,
            "action_unit": 1.0,
            "seed": 1,
            "q_eval_peak_risk": 0.14,
            "q_eval_final_risk": 0.10,
            "q_threshold_crossed": False,
            "q_pause_rate": 0.0,
        },
    ]

    summary = summarize_phase_surface(rows)

    assert summary["robust_safe_gain_by_action_unit"]["1.00"] == 0.75
    assert summary["cells"]["0.75|1.00"]["classification"] == "safe"


def test_phase_surface_tiny_run_and_outputs(tmp_path: Path):
    payload = run_phase_surface(
        gains=[0.0, 0.75],
        action_units=[1.0],
        seeds=1,
        episodes=1,
        agents=16,
        rounds=2,
        steps_per_round=3,
    )
    report = render_report(payload)

    assert "Influence Phase Surface Report" in report
    assert "Crossing Rate Surface" in report
    write_outputs(payload, tmp_path / "surface.json", tmp_path / "surface.md")
    assert (tmp_path / "surface.json").exists()
    assert (tmp_path / "surface.md").exists()

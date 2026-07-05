from pathlib import Path

from social_media_scaling import (
    ScalingCase,
    estimate_runtime_seconds,
    render_report,
    run_scaling,
    write_outputs,
)


def test_runtime_estimator_scales_linearly():
    base = estimate_runtime_seconds(
        seconds_per_agent_step=1e-6,
        agents=10,
        steps=20,
        cells=5,
        seeds=2,
        interventions=3,
    )
    doubled = estimate_runtime_seconds(
        seconds_per_agent_step=1e-6,
        agents=20,
        steps=20,
        cells=5,
        seeds=2,
        interventions=3,
    )
    assert doubled == 2 * base


def test_run_scaling_small_case_has_summary():
    payload = run_scaling([
        ScalingCase("tiny", agents=16, steps=8, cells=2, seeds=1, interventions=1)
    ])
    assert payload["summary"]["recommendation"] in {"prepare_slurm", "stay_local_for_tier2"}
    assert payload["results"][0]["simulated_runs"] == 2
    assert payload["results"][0]["agent_steps"] == 16 * 8 * 2


def test_scaling_report_and_outputs(tmp_path: Path):
    payload = run_scaling([
        ScalingCase("tiny", agents=16, steps=8, cells=1, seeds=1, interventions=1)
    ])
    report = render_report(payload["results"], payload["summary"])
    assert "Social Media Scaling Report" in report
    write_outputs(payload, tmp_path / "scaling.json", tmp_path / "scaling.md")
    assert (tmp_path / "scaling.json").exists()
    assert (tmp_path / "scaling.md").exists()

from pathlib import Path

from social_media_robustness import (
    RobustnessCell,
    _wilson_interval,
    render_report,
    run_robustness,
    write_outputs,
)


def test_wilson_interval_is_bounded():
    low, high = _wilson_interval(successes=3, n=5)
    assert 0.0 <= low <= high <= 1.0


def test_run_robustness_tiny_case_has_summary():
    payload = run_robustness(
        cells=[RobustnessCell("High-Arousal Scroll", 0.35, 0.50, 0.00)],
        seeds=1,
        agents=16,
        steps=8,
    )
    summary = payload["summary"]
    assert summary["replicates"] == 1
    assert summary["primary_intervention"] == "off_ramp_friction"
    assert summary["comparator_intervention"] == "generic_diversity_beta"
    assert summary["institutional_result"] in {"pass", "inconclusive"}
    assert len(payload["replicates"][0]["interventions"]) == 6


def test_robustness_report_and_outputs(tmp_path: Path):
    payload = run_robustness(
        cells=[RobustnessCell("High-Arousal Scroll", 0.35, 0.50, 0.00)],
        seeds=1,
        agents=16,
        steps=8,
    )
    report = render_report(payload)
    assert "Social Media Robustness Report" in report
    assert "Primary claim" in report
    write_outputs(payload, tmp_path / "robustness.json", tmp_path / "robustness.md")
    assert (tmp_path / "robustness.json").exists()
    assert (tmp_path / "robustness.md").exists()

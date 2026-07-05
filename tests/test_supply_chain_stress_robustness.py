from pathlib import Path

from supply_chain_stress_robustness import (
    RobustnessConfig,
    _classify,
    _mean_ci,
    render_report,
    run_robustness_suite,
    write_outputs,
)


def _metric(mean: float, low: float | None = None, high: float | None = None) -> dict:
    return {
        "mean": mean,
        "ci_low": mean if low is None else low,
        "ci_high": mean if high is None else high,
        "n": 3,
    }


def test_mean_ci_singleton_is_exact():
    row = _mean_ci([0.25])
    assert row["mean"] == 0.25
    assert row["ci_low"] == 0.25
    assert row["ci_high"] == 0.25


def test_classification_treats_capacity_overflow_as_backfire():
    summary = {
        "control": "port_reroute",
        "fulfillment_delta": _metric(0.0, -0.01, 0.01),
        "critical_delta": _metric(0.0, -0.01, 0.01),
        "overflow_delta": _metric(0.05, 0.03, 0.07),
        "net_benefit_delta": _metric(-0.02, -0.03, -0.01),
        "capacity_adjusted_roi": _metric(-0.01, -0.02, -0.001),
        "positive_seed_rate": 0.0,
    }

    assert _classify(summary) == "backfire"


def test_quick_robustness_suite_and_outputs(tmp_path: Path):
    payload = run_robustness_suite(
        RobustnessConfig(seeds=2, agents=24, steps=8, severities=(2.0,)),
        quick=True,
    )
    report = render_report(payload)

    assert payload["rows"]
    assert payload["summary"]
    assert "Supply Chain Stress Robustness Report" in report
    write_outputs(payload, tmp_path / "robustness.json", tmp_path / "robustness.md")
    assert (tmp_path / "robustness.json").exists()
    assert (tmp_path / "robustness.md").exists()

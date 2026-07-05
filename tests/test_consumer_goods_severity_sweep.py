from pathlib import Path

from consumer_goods_severity_sweep import (
    render_report,
    run_severity_sweep,
    severity_scenario,
    write_outputs,
)


def test_severity_scenario_tightens_cold_chain_capacity():
    mild = severity_scenario("cold_chain", 0.0)
    harsh = severity_scenario("cold_chain", 2.0)

    assert mild.gate_capacity_caps["cold_chain_gate"] > harsh.gate_capacity_caps["cold_chain_gate"]
    assert mild.edge_capacity_caps[("Cold Chain Carrier", "Regional DC")] > harsh.edge_capacity_caps[("Cold Chain Carrier", "Regional DC")]


def test_quick_consumer_goods_severity_sweep_outputs(tmp_path: Path):
    payload = run_severity_sweep(
        families=("cold_chain",),
        severities=(0.0, 1.0),
        seeds=(20260611,),
        agents=24,
        steps=12,
    )
    report = render_report(payload)

    assert payload["rows"]
    assert payload["grouped"]
    assert payload["best_by_family_severity"]
    assert payload["boundary_summary"]["cold_chain"]["max_useful_non_backfire_severity"] is not None
    assert "Consumer Goods Severity Robustness Report" in report
    assert "Boundary Summary" in report

    write_outputs(payload, tmp_path / "severity.json", tmp_path / "severity.md")
    assert (tmp_path / "severity.json").exists()
    assert (tmp_path / "severity.md").exists()

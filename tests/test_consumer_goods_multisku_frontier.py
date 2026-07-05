from pathlib import Path

from consumer_goods_multisku_frontier import (
    frontier_scenario,
    render_report,
    run_frontier,
    write_outputs,
)


def test_frontier_scenario_tightens_reserved_slots():
    mild = frontier_scenario("reserved_slots", 0.0)
    harsh = frontier_scenario("reserved_slots", 3.0)

    assert mild.gate_capacity_caps["core_reserved_gate"] > harsh.gate_capacity_caps["core_reserved_gate"]
    assert mild.edge_capacity_caps[("Reserved Cold Carrier", "North DC")] > harsh.edge_capacity_caps[
        ("Reserved Cold Carrier", "North DC")
    ]


def test_quick_multisku_frontier_outputs(tmp_path: Path):
    payload = run_frontier(
        families=("promotion_intensity",),
        severities=(0.0, 2.0),
        seeds=(20260617,),
        agents=32,
        steps=10,
    )
    report = render_report(payload)

    assert payload["rows"]
    assert payload["grouped"]
    assert payload["boundary_summary"]
    assert "Consumer Goods Multi-SKU Calibrated Frontier" in report

    write_outputs(payload, tmp_path / "frontier.json", tmp_path / "frontier.md")
    assert (tmp_path / "frontier.json").exists()
    assert (tmp_path / "frontier.md").exists()

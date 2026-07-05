from pathlib import Path

from supply_chain_resilience import (
    SimulationConfig,
    build_kernel,
    render_report,
    run_scenarios,
    summarize,
    write_outputs,
)


def test_supply_chain_topology_is_rich_and_directed():
    kernel = build_kernel(SimulationConfig(agents=16, steps=4))

    assert kernel.topo.N == 22
    assert kernel.topo.F == 7
    assert "Planning Desk" in kernel.topo.labels
    assert not kernel.topo.adjacency_mask.T.tolist() == kernel.topo.adjacency_mask.tolist()


def test_supply_chain_scenarios_have_resilience_metrics():
    rows = run_scenarios(SimulationConfig(agents=24, steps=8))
    summary = summarize(rows)

    assert len(rows) == 24
    assert summary["scenario_count"] == 24
    assert "port_congestion" in summary["shocks"]
    assert all("fulfillment_share" in row for row in rows)
    assert all("resilience_roi" in row for row in rows)


def test_supply_chain_report_and_outputs(tmp_path: Path):
    rows = run_scenarios(SimulationConfig(agents=24, steps=8))
    payload = {"summary": summarize(rows), "rows": rows}
    report = render_report(payload)

    assert "Supply Chain Resilience Report" in report
    assert "Control Matrix" in report
    write_outputs(payload, tmp_path / "supply.json", tmp_path / "supply.md")
    assert (tmp_path / "supply.json").exists()
    assert (tmp_path / "supply.md").exists()

from pathlib import Path

from consumer_goods_circulation import (
    SimulationConfig,
    _choice_point_inventory,
    controls,
    render_report,
    run_pilot,
    scenarios,
    simulate,
    write_outputs,
)


def test_consumer_goods_simulation_reports_feasibility_metrics():
    row = simulate(
        SimulationConfig(agents=32, steps=16),
        scenarios()[1],
        controls()[1],
    )

    assert 0.0 <= row["service_completion_rate"] <= 1.0
    assert 0.0 <= row["priority_service_rate"] <= 1.0
    assert 0.0 <= row["lost_demand_rate"] <= 1.0
    assert "cold_chain_gate" in row["gate_attempts"]
    assert "cold_chain_gate" in row["gate_inventory_blocked"]
    assert "cold_chain_gate" in row["gate_capacity_blocked"]
    assert "production_gate" in row["gate_inventory_end"]
    assert row["limiting_part"]
    assert 0.0 <= row["gate_starvation_rate"] <= 1.0
    assert 0.0 <= row["gate_service_capacity_block_rate"] <= 1.0
    assert 0.0 <= row["gate_contention_rate"] <= 1.0
    assert row["gate_primary_pressure"] in {
        "inventory_starvation",
        "service_capacity",
        "ordinary_contention",
        "none",
    }


def test_gate_pressure_decomposes_blocked_attempts():
    row = simulate(
        SimulationConfig(agents=48, steps=18),
        scenarios()[1],
        controls()[0],
    )

    for gate_name, blocked in row["gate_blocked"].items():
        decomposed = row["gate_inventory_blocked"][gate_name] + row["gate_capacity_blocked"][gate_name]
        assert blocked == decomposed


def test_choice_point_inventory_marks_serial_corridors():
    inventory = _choice_point_inventory()

    assert any(item["choice_type"] == "choice_point" for item in inventory)
    assert any(item["choice_type"] == "serial_corridor" for item in inventory)


def test_quick_consumer_goods_pilot_outputs(tmp_path: Path):
    payload = run_pilot(SimulationConfig(agents=24, steps=12), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["summary"]
    assert "Consumer Goods Cold-Chain Circulation Pilot" in report

    write_outputs(payload, tmp_path / "consumer.json", tmp_path / "consumer.md")
    assert (tmp_path / "consumer.json").exists()
    assert (tmp_path / "consumer.md").exists()

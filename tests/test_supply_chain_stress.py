from pathlib import Path

from supply_chain_resilience import SimulationConfig
from supply_chain_stress import (
    StressShock,
    render_report,
    run_hard_capacity_cases,
    run_severity_sweep,
    simulate_stress,
    write_outputs,
)


def test_hard_failure_records_blocked_mass():
    row = simulate_stress(
        SimulationConfig(agents=24, steps=8),
        StressShock(
            name="test_chip_failure",
            family="hard_failure",
            failed_edges=(("Planning Desk", "Chip Fab A"),),
        ),
    )

    assert row["hard_blocked_mass"] >= 0.0
    assert "fulfillment_share" in row


def test_capacity_case_records_overflow_rate():
    row = simulate_stress(
        SimulationConfig(agents=48, steps=12),
        StressShock(
            name="test_capacity",
            family="capacity",
            capacity_caps={("Planning Desk", "Chip Fab A"): 1},
        ),
    )

    assert row["capacity_attempts"] > 0
    assert row["capacity_overflow_rate"] >= 0.0


def test_node_capacity_case_records_node_overflow_rate():
    row = simulate_stress(
        SimulationConfig(agents=48, steps=12),
        StressShock(
            name="test_node_capacity",
            family="node_capacity",
            node_capacity_caps={"Chip Fab A": 1},
        ),
    )

    assert row["node_capacity_attempts"] > 0
    assert row["node_capacity_overflow_rate"] >= 0.0
    assert row["capacity_attempts"] >= row["node_capacity_attempts"]


def test_bom_gate_records_production_blocking():
    row = simulate_stress(
        SimulationConfig(agents=96, steps=32),
        StressShock(name="test_bom", family="bom"),
        enforce_bom=True,
    )

    assert row["bom_enforced"] is True
    assert row["bom_attempts"] > 0
    assert row["bom_blocked_events"] > 0
    assert row["bom_block_rate"] > 0.0
    assert set(row["bom_inventory_end"]) == {"battery", "electronics", "chassis", "packaging"}


def test_stress_suites_and_outputs(tmp_path: Path):
    config = SimulationConfig(agents=24, steps=8)
    severity = run_severity_sweep(severities=[0.0, 2.0], config=config)
    hard = run_hard_capacity_cases(config=config)
    payload = {"severity": severity, "hard_capacity": hard}
    report = render_report(payload)

    assert "Supply Chain Stress Report" in report
    assert "Hard Failure And Capacity Cases" in report
    write_outputs(payload, tmp_path / "stress.json", tmp_path / "stress.md")
    assert (tmp_path / "stress.json").exists()
    assert (tmp_path / "stress.md").exists()

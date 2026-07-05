from pathlib import Path

from semiconductor_onshoring import (
    OnshoringConfig,
    Scenario,
    controls,
    render_report,
    run_suite,
    scenarios,
    simulate,
    write_outputs,
)


def test_onshoring_simulation_reports_core_metrics():
    row = simulate(
        OnshoringConfig(agents=48, steps=24),
        scenarios()[0],
        controls()[0],
    )

    assert row["gates_enforced"] is True
    assert 0.0 <= row["onshore_share"] <= 1.0
    assert row["lot_total_us_finished"] >= 0
    assert row["lot_onshore_finished"] <= row["lot_total_us_finished"]
    assert 0.0 <= row["flow_onshore_share"] <= 1.0
    assert "us_fab_gate" in row["gate_attempts"]
    assert "us_advanced_packaging_gate" in row["gate_attempts"]
    assert "taiwan_export_gate" in row["gate_attempts"]
    assert "us_fab_gate" in row["gate_replenished"]
    assert "us_fab_gate" in row["gate_capacity_blocked"]
    assert 0.0 <= row["gate_backlog_pressure"] <= 1.0
    assert 0.0 <= row["gate_starvation_index"] <= 1.0
    assert 0.0 <= row["gate_pressure_rate"] <= 1.0
    assert row["limiting_part"] in {
        "domestic_wafers",
        "materials",
        "eda_ip",
        "tooling",
        "power_labor",
        "packaged_us_wafers",
        "us_packaging_inputs",
        "taiwan_wafers",
        "taiwan_packaging_inputs",
    }


def test_gate_service_capacity_limits_completions():
    config = OnshoringConfig(agents=48, steps=24)
    control = Scenario(
        "gate_cap_test",
        "test",
        gate_replenishment={
            ("us_fab_gate", "materials"): 8,
            ("us_fab_gate", "power_labor"): 8,
        },
        gate_capacity_caps={"us_fab_gate": 1},
    )
    row = simulate(config, scenarios()[0], control)

    assert row["gate_completions"]["us_fab_gate"] <= config.steps
    assert row["gate_capacity_caps"]["us_fab_gate"] == 1


def test_randomization_key_makes_mechanically_identical_controls_comparable():
    config = OnshoringConfig(
        agents=48,
        steps=24,
        randomization_key="paired_experiment",
    )
    first = Scenario("first_label", "test", friction_delta=0.2)
    second = Scenario("second_label", "test", friction_delta=0.2)

    first_row = simulate(config, scenarios()[0], first)
    second_row = simulate(config, scenarios()[0], second)

    assert first_row["lot_total_us_finished"] == second_row["lot_total_us_finished"]
    assert first_row["lot_onshore_finished"] == second_row["lot_onshore_finished"]
    assert first_row["gate_attempts"] == second_row["gate_attempts"]


def test_topology_variant_edges_are_applied():
    base = simulate(OnshoringConfig(agents=32, steps=16), scenarios()[0], controls()[0])
    variant = simulate(
        OnshoringConfig(
            agents=32,
            steps=16,
            additional_edges=(("Export Control Review", "Market Allocation Desk", 1.0),),
        ),
        scenarios()[0],
        controls()[0],
    )

    assert base["total_us_finished_flow"] != variant["total_us_finished_flow"]


def test_quick_suite_and_outputs(tmp_path: Path):
    payload = run_suite(OnshoringConfig(agents=32, steps=16), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["best_by_scenario"]
    assert "Semiconductor Onshoring Report" in report
    write_outputs(payload, tmp_path / "onshoring.json", tmp_path / "onshoring.md")
    assert (tmp_path / "onshoring.json").exists()
    assert (tmp_path / "onshoring.md").exists()

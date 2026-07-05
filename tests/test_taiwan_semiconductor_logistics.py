from pathlib import Path

from taiwan_semiconductor_logistics import (
    TaiwanConfig,
    controls,
    paired_run,
    render_report,
    run_suite,
    shocks,
    simulate,
    write_outputs,
)


def test_gated_simulation_records_dependency_gates():
    row = simulate(
        TaiwanConfig(agents=48, steps=24),
        shocks()[0],
        controls()[0],
        enforce_gates=True,
    )

    assert row["gates_enforced"] is True
    assert "fab_input_gate" in row["gate_attempts"]
    assert "exportable_chip_gate" in row["gate_attempts"]
    assert row["gate_attempts"]["fab_input_gate"] >= 0
    assert row["limiting_part"] in {
        "materials",
        "chemicals",
        "lithography",
        "design_ip",
        "energy",
        "wafer_output",
        "packaging",
    }


def test_paired_run_computes_feasibility_gap():
    row = paired_run(
        TaiwanConfig(agents=48, steps=24),
        shocks()[0],
        controls()[0],
    )

    assert row["feasibility_gap"] >= 0.0
    assert row["classification"]


def test_quick_suite_and_outputs(tmp_path: Path):
    payload = run_suite(TaiwanConfig(agents=32, steps=16), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["best_by_shock"]
    assert "Taiwan Semiconductor Logistics Prototype Report" in report
    write_outputs(payload, tmp_path / "taiwan.json", tmp_path / "taiwan.md")
    assert (tmp_path / "taiwan.json").exists()
    assert (tmp_path / "taiwan.md").exists()

from pathlib import Path

from supply_chain_bom import (
    BOMConfig,
    StressShock,
    render_report,
    run_bom_suite,
    simulate_bom_pair,
    write_outputs,
)


def test_bom_pair_computes_feasibility_gap():
    row = simulate_bom_pair(
        BOMConfig(agents=64, steps=32),
        StressShock(name="test_nominal", family="test"),
        "no_control",
    )

    assert row["bom_attempts"] > 0
    assert row["feasibility_gap"] >= 0.0
    assert row["limiting_component"] in {"battery", "electronics", "chassis", "packaging"}


def test_quick_bom_suite_and_outputs(tmp_path: Path):
    payload = run_bom_suite(BOMConfig(agents=32, steps=16), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["best_by_shock"]
    assert "Supply Chain BOM Feasibility Report" in report
    write_outputs(payload, tmp_path / "bom.json", tmp_path / "bom.md")
    assert (tmp_path / "bom.json").exists()
    assert (tmp_path / "bom.md").exists()

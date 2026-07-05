from pathlib import Path

from consumer_goods_multisku import (
    MultiSKUConfig,
    controls,
    render_report,
    run_pilot,
    scenarios,
    simulate,
    write_outputs,
)


def test_multisku_simulation_reports_sku_and_gate_metrics():
    row = simulate(
        MultiSKUConfig(agents=48, steps=18),
        scenarios()[2],
        controls()[1],
    )

    assert 0.0 <= row["service_completion_rate"] <= 1.0
    assert 0.0 <= row["substitution_rate"] <= 1.0
    assert "Promo Yogurt" in row["sku_service_rates"]
    assert "Frozen Dessert" in row["sku_lost_rates"]
    assert "core_reserved_gate" in row["gate_attempts"]
    assert "core_reserved_gate" in row["gate_capacity_blocked"]
    assert row["gate_primary_pressure"] in {
        "inventory_starvation",
        "service_capacity",
        "ordinary_contention",
        "none",
    }


def test_multisku_quick_pilot_outputs(tmp_path: Path):
    payload = run_pilot(MultiSKUConfig(agents=36, steps=12), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["summary"]
    assert "Consumer Goods Multi-SKU Controlled Topology" in report
    assert "SKU Service Rates" in report

    write_outputs(payload, tmp_path / "multisku.json", tmp_path / "multisku.md")
    assert (tmp_path / "multisku.json").exists()
    assert (tmp_path / "multisku.md").exists()

from pathlib import Path

from semiconductor_onshoring_frontier import FrontierConfig
from semiconductor_onshoring_inventory_phase_diagram import (
    render_report,
    run_inventory_phase_diagram,
    write_outputs,
)


def test_quick_inventory_phase_diagram_outputs(tmp_path: Path):
    payload = run_inventory_phase_diagram(FrontierConfig(agents=32, steps=16), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["inventory_renewals"]
    assert payload["classification_counts"]
    assert "Inventory-Renewal Phase Diagram" in report
    assert "Renewal Slices" in report

    write_outputs(payload, tmp_path / "inventory_phase.json", tmp_path / "inventory_phase.md")
    assert (tmp_path / "inventory_phase.json").exists()
    assert (tmp_path / "inventory_phase.md").exists()

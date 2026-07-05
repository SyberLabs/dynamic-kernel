from pathlib import Path

from semiconductor_onshoring_frontier import FrontierConfig
from semiconductor_onshoring_phase_diagram import render_report, run_phase_diagram, write_outputs


def test_quick_phase_diagram_outputs(tmp_path: Path):
    payload = run_phase_diagram(FrontierConfig(agents=32, steps=16), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["classification_counts"]
    assert "Semiconductor Onshoring Phase Diagram" in report
    assert "FabCap \\ PkgCap" in report

    write_outputs(payload, tmp_path / "phase.json", tmp_path / "phase.md")
    assert (tmp_path / "phase.json").exists()
    assert (tmp_path / "phase.md").exists()

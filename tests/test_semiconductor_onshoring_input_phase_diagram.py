from pathlib import Path

from semiconductor_onshoring_frontier import FrontierConfig
from semiconductor_onshoring_input_phase_diagram import (
    render_report,
    run_input_phase_diagram,
    write_outputs,
)


def test_quick_input_phase_diagram_outputs(tmp_path: Path):
    payload = run_input_phase_diagram(FrontierConfig(agents=32, steps=16), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["materials_levels"]
    assert payload["classification_counts"]
    assert "Input-Replenishment Phase Diagram" in report
    assert "Materials Slices" in report

    write_outputs(payload, tmp_path / "input_phase.json", tmp_path / "input_phase.md")
    assert (tmp_path / "input_phase.json").exists()
    assert (tmp_path / "input_phase.md").exists()

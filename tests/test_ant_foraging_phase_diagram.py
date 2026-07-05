from pathlib import Path

from ant_foraging_dte import SimulationConfig
from ant_foraging_phase_diagram import render_report, run_phase_diagram, write_outputs


def test_quick_ant_phase_diagram_outputs(tmp_path: Path):
    payload = run_phase_diagram(SimulationConfig(agents=28, steps=48), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["summary"]["classification_counts"]
    assert "Ant Foraging Phase Diagram" in report
    assert "Memory Ratio" in report or "memory ratio" in report
    assert "Phase Matrices" in report

    write_outputs(payload, tmp_path / "ant_phase.json", tmp_path / "ant_phase.md")
    assert (tmp_path / "ant_phase.json").exists()
    assert (tmp_path / "ant_phase.md").exists()


def test_delayed_depletion_phase_diagram_exposes_familiarity_boundary():
    payload = run_phase_diagram(SimulationConfig(agents=48, steps=90), quick=True)
    counts = payload["summary"]["classification_counts"]
    delayed = payload["summary"]["scenario_summary"]["rich_patch_delayed_depletion"]

    assert counts.get("adaptive_memory", 0) > 0
    assert counts.get("deadly_familiarity", 0) > 0
    assert delayed["deadly_min_memory_ratio"] is not None
    assert delayed["adaptive_count"] > 0

from pathlib import Path

from microbiome_ecology_dte import (
    MicrobiomeConfig,
    render_report,
    run_pilot,
    scenarios,
    simulate,
    write_outputs,
)


def test_microbiome_simulation_reports_phase_and_memory_metrics():
    row = simulate(MicrobiomeConfig(agents=48, steps=56), scenarios()[0])

    assert 0.0 <= row["probiotic_survival_rate"] <= 1.0
    assert 0.0 <= row["antibiotic_overlap_fraction"] <= 1.0
    assert 0.0 <= row["intervention_phase_error"] <= 1.0
    assert 0.0 <= row["final_beneficial_occupancy"] <= 1.0
    assert 0.0 <= row["final_pathobiont_occupancy"] <= 1.0
    assert 0.0 <= row["final_diversity_entropy"] <= 1.0
    assert row["dominant_stale_memory_layer"] in {
        "none",
        "structural_memory",
        "preference_memory",
        "state_memory",
    }


def test_quick_microbiome_pilot_exposes_timing_regimes():
    payload = run_pilot(quick=True)
    classes = payload["summary"]["classification_counts"]

    assert classes.get("early_washout", 0) >= 1
    assert classes.get("on_phase_recovery", 0) >= 1
    assert classes.get("late_lockin", 0) >= 1


def test_microbiome_report_and_outputs(tmp_path: Path):
    payload = run_pilot(MicrobiomeConfig(agents=48, steps=64), quick=True)
    report = render_report(payload)

    assert "Microbiome Ecology DTE Prototype" in report
    assert "non-medical" in report
    assert "Phase error" in report

    write_outputs(payload, tmp_path / "microbiome.json", tmp_path / "microbiome.md")
    assert (tmp_path / "microbiome.json").exists()
    assert (tmp_path / "microbiome.md").exists()

from pathlib import Path

from microbiome_ecology_dte import MicrobiomeConfig
from microbiome_timing_sweep import render_report, run_timing_sweep, scenario_for_offset, write_outputs


def test_timing_scenario_offsets_move_probiotic_start():
    early = scenario_for_offset(-14)
    on_time = scenario_for_offset(0)
    late = scenario_for_offset(20)

    assert early.probiotic_start < early.antibiotic_end
    assert on_time.probiotic_start == on_time.antibiotic_end
    assert late.probiotic_start > late.antibiotic_end


def test_quick_microbiome_timing_sweep_identifies_window():
    payload = run_timing_sweep(MicrobiomeConfig(agents=96, steps=126), quick=True, seeds=2)
    summary = payload["summary"]

    assert summary["offset_summary"]
    assert summary["viable_window"] is not None
    assert summary["best_offset"]["mean_beneficial_occupancy"] >= 0.55
    assert any(row["washout_rate"] > 0 for row in summary["offset_summary"])
    assert any(row["lockin_rate"] > 0 for row in summary["offset_summary"])


def test_microbiome_timing_report_outputs(tmp_path: Path):
    payload = run_timing_sweep(MicrobiomeConfig(agents=48, steps=72), quick=True, seeds=1)
    report = render_report(payload)

    assert "Microbiome Timing Sweep" in report
    assert "non-medical" in report
    assert "Offset Frontier" in report

    write_outputs(payload, tmp_path / "timing.json", tmp_path / "timing.md")
    assert (tmp_path / "timing.json").exists()
    assert (tmp_path / "timing.md").exists()

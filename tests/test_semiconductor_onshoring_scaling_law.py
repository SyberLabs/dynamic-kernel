from pathlib import Path

from semiconductor_onshoring_scaling_law import render_report, run_scaling_law, write_outputs


def test_small_scaling_law_outputs(tmp_path: Path):
    payload = run_scaling_law(
        agent_levels=(40, 80),
        seeds=(20260606,),
        steps=16,
    )
    report = render_report(payload)

    assert payload["rows"]
    assert payload["grouped"]
    assert payload["policy_summary"]
    assert "Resource-Scaling Law Report" in report
    assert "Robust Phase Summary" in report

    write_outputs(payload, tmp_path / "scaling.json", tmp_path / "scaling.md")
    assert (tmp_path / "scaling.json").exists()
    assert (tmp_path / "scaling.md").exists()

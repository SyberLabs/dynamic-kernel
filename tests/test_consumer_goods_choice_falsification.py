from pathlib import Path

from consumer_goods_choice_falsification import (
    falsification_controls,
    render_report,
    run_choice_falsification,
    write_outputs,
)


def test_falsification_controls_include_decomposition_variants():
    roles = {control.name.split("__", 1)[1] for control in falsification_controls("carrier_priority")}

    assert "canonical" in roles
    assert "edges_only" in roles
    assert "gate_only" in roles
    assert "wrong_dry_lane" in roles


def test_quick_choice_falsification_outputs(tmp_path: Path):
    payload = run_choice_falsification(seeds=(20260611,), agents=24, steps=12)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["grouped"]
    assert payload["verdicts"]
    assert "Choice-Point Falsification Report" in report
    assert "Verdicts" in report

    write_outputs(payload, tmp_path / "choice.json", tmp_path / "choice.md")
    assert (tmp_path / "choice.json").exists()
    assert (tmp_path / "choice.md").exists()

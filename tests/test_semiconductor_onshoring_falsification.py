from collections import Counter
from pathlib import Path

from semiconductor_onshoring import EDGES
from semiconductor_onshoring_falsification import (
    _rewired_choice_topology,
    render_report,
    run_falsification_suite,
    write_outputs,
)


def test_rewired_choice_topology_preserves_directed_degrees():
    removed, added = _rewired_choice_topology(7001, swaps=8)
    original = {(source, target) for source, target, _ in EDGES}
    rewired = (original - set(removed)) | {(source, target) for source, target, _ in added}

    assert Counter(source for source, _ in original) == Counter(source for source, _ in rewired)
    assert Counter(target for _, target in original) == Counter(target for _, target in rewired)


def test_small_falsification_suite_outputs(tmp_path: Path):
    payload = run_falsification_suite(
        agents=32,
        steps=12,
        seeds=(20260606,),
    )
    report = render_report(payload)

    assert payload["choice_point_relocation"]["grouped"]
    assert payload["topology_surgery"]["grouped"]
    assert payload["feedback_continuum"]["grouped"]
    assert payload["randomized_topology_nulls"]["grouped"]
    assert "Falsification Report" in report

    write_outputs(payload, tmp_path / "falsification.json", tmp_path / "falsification.md")
    assert (tmp_path / "falsification.json").exists()
    assert (tmp_path / "falsification.md").exists()

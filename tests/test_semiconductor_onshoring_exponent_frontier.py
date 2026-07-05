from pathlib import Path

from semiconductor_onshoring_exponent_frontier import (
    render_report,
    run_exponent_frontier,
    write_outputs,
)
from semiconductor_onshoring_scaling_law import run_scaling_law


def test_small_exponent_frontier_outputs(tmp_path: Path):
    payload = run_exponent_frontier(
        agent_levels=(80,),
        exponents=(1.0, 2.0),
        seeds=(20260606,),
        steps=16,
    )
    report = render_report(payload)

    assert payload["rows"]
    assert payload["grouped"]
    assert "robust_cells" in payload
    assert "All-Resource Exponent Frontier" in report
    assert "phase islands" in report

    write_outputs(payload, tmp_path / "frontier.json", tmp_path / "frontier.md")
    assert (tmp_path / "frontier.json").exists()
    assert (tmp_path / "frontier.md").exists()


def test_frontier_matches_scaling_law_for_same_resource_policy():
    scaling = run_scaling_law(
        agent_levels=(80,),
        seeds=(20260606,),
        steps=16,
    )
    frontier = run_exponent_frontier(
        agent_levels=(80,),
        exponents=(1.0,),
        seeds=(20260606,),
        steps=16,
    )

    scaling_row = next(row for row in scaling["rows"] if row["policy"] == "linear_all")
    frontier_row = frontier["rows"][0]
    assert scaling_row["lot_total_us_finished"] == frontier_row["lot_total_us_finished"]
    assert scaling_row["lot_onshore_finished"] == frontier_row["lot_onshore_finished"]
    assert scaling_row["gate_attempts"] == frontier_row["gate_attempts"]

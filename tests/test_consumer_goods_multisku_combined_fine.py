from pathlib import Path

from consumer_goods_multisku_combined_fine import (
    render_report,
    run_combined_fine_frontier,
    write_outputs,
)


def test_quick_combined_fine_frontier_outputs(tmp_path: Path):
    payload = run_combined_fine_frontier(
        severities=(1.5, 2.0),
        seeds=(20260617,),
        agents=32,
        steps=10,
    )
    report = render_report(payload)

    assert payload["rows"]
    assert payload["fine_summary"]["best_rows"]
    assert "Stability Bands" in report

    write_outputs(payload, tmp_path / "combined.json", tmp_path / "combined.md")
    assert (tmp_path / "combined.json").exists()
    assert (tmp_path / "combined.md").exists()

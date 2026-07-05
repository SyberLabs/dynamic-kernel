import json
from pathlib import Path

from social_media_reduce import reduce_payloads
from social_media_robustness import RobustnessCell, run_robustness


def test_reduce_payloads_combines_replicates(tmp_path: Path):
    first = run_robustness(
        cells=[RobustnessCell("High-Arousal Scroll", 0.35, 0.50, 0.00)],
        seeds=1,
        agents=16,
        steps=8,
    )
    second = run_robustness(
        cells=[RobustnessCell("High-Arousal Scroll", 0.50, 0.80, 0.03)],
        seeds=1,
        agents=16,
        steps=8,
        seed_base=20270000,
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    reduced = reduce_payloads([first_path, second_path])

    assert reduced["summary"]["cells"] == 2
    assert reduced["summary"]["replicates"] == 2
    assert len(reduced["summary"]["source_files"]) == 2

import json

from neural_v2_seed_validation import SeedValidationConfig, run_seed_validation
from neural_v2_seed_validation_figure import (
    generate_seed_validation_figure,
)


def test_seed_validation_figure_generates_svg_and_report(tmp_path):
    payload = run_seed_validation(
        SeedValidationConfig(
            ticks=30,
            shift_tick=10,
            batch_size=30,
            seeds=(0,),
            hard_reward_delay=2,
            adversarial_reward_delay=2,
        ),
        write_outputs=False,
    )
    input_path = tmp_path / "seed_validation.json"
    svg_path = tmp_path / "figure.svg"
    report_path = tmp_path / "report.md"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    result = generate_seed_validation_figure(input_path, svg_path, report_path)

    assert result["svg"] == str(svg_path)
    assert svg_path.exists()
    assert "<svg" in svg_path.read_text(encoding="utf-8")
    assert "Neural V2 policy-lane boundary" in svg_path.read_text(encoding="utf-8")
    assert "regime-boundary" in report_path.read_text(encoding="utf-8")

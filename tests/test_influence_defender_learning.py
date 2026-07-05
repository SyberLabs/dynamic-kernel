from pathlib import Path

from influence_defender_learning import render_report, run_exp3_defender_game, write_outputs
from influence_game import GameConfig


def test_exp3_defender_game_runs():
    payload = run_exp3_defender_game(
        GameConfig(
            rounds=3,
            agents=24,
            steps_per_round=4,
            adversary_budget=3.0,
            defender_budget=3.0,
            dynamic_defense=True,
            trust_hysteresis=True,
            noise_sigma=0.0,
        )
    )

    assert payload["summary"]["rounds"] == 3
    assert payload["summary"]["final_action_probabilities"]
    assert len(payload["rounds"]) == 3


def test_exp3_report_and_outputs(tmp_path: Path):
    payload = run_exp3_defender_game(
        GameConfig(rounds=2, agents=24, steps_per_round=4, noise_sigma=0.0)
    )
    report = render_report(payload)

    assert "No-Regret Defender Report" in report
    write_outputs(payload, tmp_path / "exp3.json", tmp_path / "exp3.md")
    assert (tmp_path / "exp3.json").exists()
    assert (tmp_path / "exp3.md").exists()

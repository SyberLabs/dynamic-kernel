from pathlib import Path

from influence_game import GameConfig
from influence_qlearn import _adversary_reward, q_state_key, render_report, train_q_adversary, write_outputs


def test_q_state_key_is_stable():
    key = q_state_key(
        {"risk_share": 0.12, "escape_probability": 0.52, "warning_score": 0.48},
        adversary_budget_left=3.0,
        detection_pressure=1.0,
    )

    assert key.startswith("r")
    assert ":e" in key
    assert ":b" in key


def test_q_training_tiny_run_has_evaluation():
    payload = train_q_adversary(
        GameConfig(
            rounds=2,
            agents=24,
            steps_per_round=4,
            adversary_budget=2.0,
            defender_budget=2.0,
            dynamic_defense=True,
            trust_hysteresis=True,
            noise_sigma=0.0,
        ),
        episodes=2,
    )

    assert payload["summary"]["episodes"] == 2
    assert payload["summary"]["visited_states"] > 0
    assert "evaluation" in payload
    assert "baseline" in payload
    assert "q_pause_rate" in payload["summary"]


def test_qlearn_report_and_outputs(tmp_path: Path):
    payload = train_q_adversary(
        GameConfig(
            rounds=2,
            agents=24,
            steps_per_round=4,
            adversary_budget=2.0,
            defender_budget=2.0,
            dynamic_defense=True,
            trust_hysteresis=True,
            noise_sigma=0.0,
        ),
        episodes=1,
    )
    report = render_report(payload)

    assert "Q-Learning Influence Adversary Report" in report
    write_outputs(payload, tmp_path / "q.json", tmp_path / "q.md")
    assert (tmp_path / "q.json").exists()
    assert (tmp_path / "q.md").exists()


def test_terminal_reward_and_delta_tax_change_reward():
    metrics = {
        "risk_share": 0.20,
        "target_share": 0.05,
        "polarization_gap": 0.10,
    }
    terminal = _adversary_reward(
        metrics,
        adv_cost=1.0,
        detection_pressure=2.0,
        detection_delta=0.1,
        risk_threshold=0.18,
        is_terminal=True,
        terminal_bonus=10.0,
        detection_tax_mode="delta",
    )
    non_terminal = _adversary_reward(
        metrics,
        adv_cost=1.0,
        detection_pressure=2.0,
        detection_delta=0.1,
        risk_threshold=0.18,
        is_terminal=False,
        terminal_bonus=10.0,
        detection_tax_mode="delta",
    )

    assert terminal > non_terminal

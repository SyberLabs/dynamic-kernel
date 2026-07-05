from pathlib import Path

from influence_game import (
    ADVERSARY_ACTIONS,
    GameConfig,
    adversary_budget_spend,
    adversary_detection_increment,
    apply_immune_response,
    defender_policy,
    initialize_state,
    _apply_trust_hysteresis,
    compare_defender_policies,
    render_report,
    run_game,
    write_outputs,
)
import numpy as np


def test_influence_game_runs_stateful_rounds():
    payload = run_game(
        GameConfig(rounds=3, agents=32, steps_per_round=8, noise_sigma=0.0),
        adversary_policy_name="escalating",
        defender_policy_name="structural_warning",
    )

    assert payload["summary"]["rounds"] == 3
    assert len(payload["rounds"]) == 3
    for row in payload["rounds"]:
        assert 0.0 <= row["risk_share"] <= 1.0
        assert 0.0 <= row["escape_probability"] <= 1.0
        assert 0.0 <= row["warning_score"] <= 1.0
        assert row["adversary_action"] in ADVERSARY_ACTIONS
        assert row["defender_action"] in {"observe", "off_ramp", "prebunk", "throttle", "combined"}


def test_defender_policy_comparison_has_best_policy():
    comparison = compare_defender_policies(
        GameConfig(rounds=3, agents=32, steps_per_round=8, noise_sigma=0.0)
    )

    assert comparison["best_policy"] in comparison["policies"]
    assert set(comparison["policies"]) == {"none", "risk_threshold", "structural_warning"}


def test_fixed_defender_doctrines_choose_expected_actions():
    config = GameConfig(rounds=1, agents=16, steps_per_round=4, defender_budget=3)
    state = initialize_state(config)

    assert defender_policy(state, config, "off_ramp_first") == "off_ramp"
    assert defender_policy(state, config, "prebunk_first") == "prebunk"
    assert defender_policy(state, config, "throttle_first") == "throttle"
    assert defender_policy(state, config, "combined_first") == "combined"


def test_stealth_actions_have_lower_budget_and_detection_cost():
    assert adversary_budget_spend("stealth_amplify", 1.0) < adversary_budget_spend("amplify", 1.0)
    assert adversary_detection_increment("stealth_amplify") < adversary_detection_increment("amplify")


def test_dynamic_immune_response_adds_defender_budget():
    config = GameConfig(
        rounds=1,
        agents=16,
        steps_per_round=4,
        defender_budget=1.0,
        dynamic_defense=True,
        max_defender_budget=3.0,
    )
    state = initialize_state(config)
    state.previous_metrics = {"edge_current_norm": 0.18, "entropy_production": 0.10}

    added = apply_immune_response(state, config)

    assert added > 0.0
    assert state.defender_budget_left > 1.0


def test_trust_hysteresis_slows_low_trust_recovery():
    config = GameConfig(trust_hysteresis=True, trust_floor=0.5, trust_recovery_slowdown=0.25)
    previous = np.zeros((1, 10))
    updated = np.zeros((1, 10))
    previous[0, 7] = 0.2
    updated[0, 7] = 0.6

    slowed = _apply_trust_hysteresis(previous, updated, config)

    assert slowed[0, 7] == 0.3


def test_influence_game_report_outputs(tmp_path: Path):
    config = GameConfig(rounds=2, agents=24, steps_per_round=6, noise_sigma=0.0)
    payload = run_game(config)
    comparison = compare_defender_policies(config)
    report = render_report(payload, comparison)

    assert "Influence Game DTE Report" in report
    assert "Defender Policy Comparison" in report
    write_outputs(payload, comparison, tmp_path / "game.json", tmp_path / "game.md")
    assert (tmp_path / "game.json").exists()
    assert (tmp_path / "game.md").exists()

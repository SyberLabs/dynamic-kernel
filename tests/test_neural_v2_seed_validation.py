from neural_v2_seed_validation import (
    SeedValidationConfig,
    render_seed_validation_report,
    run_seed_validation,
)


def test_seed_validation_runs_and_reports_regimes():
    payload = run_seed_validation(
        SeedValidationConfig(
            ticks=32,
            shift_tick=12,
            batch_size=40,
            seeds=(0,),
            hard_reward_delay=2,
            adversarial_reward_delay=2,
            adversarial_switch_period=4,
        ),
        write_outputs=False,
    )
    regimes = {row["regime"] for row in payload["summary"]}
    assert {"clean", "hard", "adversarial_switch"}.issubset(regimes)
    routers = {row["router"] for row in payload["summary"]}
    assert "dte_local_regret" in routers
    assert "hard_dte_reliability_arbitrated_exp3" in routers
    assert all("paired_delta_vs_local" in row for row in payload["summary"])

    report = render_seed_validation_report(payload)
    assert "Neural V2 Seed Validation" in report
    assert "Delta vs Local" in report

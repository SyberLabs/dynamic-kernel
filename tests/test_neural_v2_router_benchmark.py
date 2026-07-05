from neural_v2_router_benchmark import (
    BenchmarkConfig,
    AdversarialSwitchSweepConfig,
    FrontierConfig,
    HardBenchmarkConfig,
    make_hard_task_stream,
    render_report,
    render_adversarial_switch_report,
    render_frontier_report,
    render_hard_report,
    run_adversarial_switch_benchmark,
    run_benchmark,
    run_dte_router,
    run_frontier,
    run_hard_benchmark,
    run_hard_dte_router,
    run_hard_oracle_router,
    run_oracle_router,
    make_task_stream,
)


def test_dte_local_regret_beats_dte_surprise_on_shared_stream():
    config = BenchmarkConfig(ticks=80, shift_tick=35, batch_size=120, seeds=(0,))
    stream = make_task_stream(0, config)
    surprise = run_dte_router("surprise_only", stream, 0, config)
    local = run_dte_router("local_regret", stream, 0, config)
    assert local["post_shift_mean_regret"] < surprise["post_shift_mean_regret"]


def test_oracle_is_upper_bound_on_reward_for_dte_router():
    config = BenchmarkConfig(ticks=50, shift_tick=25, batch_size=80, seeds=(0,))
    stream = make_task_stream(0, config)
    local = run_dte_router("local_regret", stream, 0, config)
    oracle = run_oracle_router(stream, 0, config)
    assert oracle["post_shift_mean_reward"] >= local["post_shift_mean_reward"]
    assert oracle["post_shift_mean_regret"] == 0.0


def test_router_benchmark_runs_and_reports_all_baselines():
    payload = run_benchmark(
        BenchmarkConfig(ticks=40, shift_tick=20, batch_size=60, seeds=(0,)),
        write_outputs=False,
    )
    summary = payload["summary"]
    for label in [
        "dte_surprise_only",
        "dte_local_regret",
        "dte_ucb",
        "dte_contextual_ucb",
        "dte_arbitrated_ucb",
        "dte_reliability_arbitrated_ucb",
        "dte_exp3",
        "static_contextual",
        "epsilon_bandit",
        "ucb",
        "exp3",
        "oracle",
    ]:
        assert label in summary
    report = render_report(payload)
    assert "strong" in report
    assert "dte_local_regret" in report


def test_hard_dte_local_regret_beats_hard_dte_surprise_on_shared_stream():
    config = HardBenchmarkConfig(ticks=80, shift_tick=35, batch_size=120, seeds=(0,))
    stream = make_hard_task_stream(0, config)
    surprise = run_hard_dte_router("surprise_only", stream, 0, config)
    local = run_hard_dte_router("local_regret", stream, 0, config)
    assert local["post_shift_mean_regret"] < surprise["post_shift_mean_regret"]


def test_hard_oracle_is_upper_bound_on_reward_for_dte_router():
    config = HardBenchmarkConfig(ticks=50, shift_tick=25, batch_size=80, seeds=(0,))
    stream = make_hard_task_stream(0, config)
    local = run_hard_dte_router("local_regret", stream, 0, config)
    oracle = run_hard_oracle_router(stream, 0, config)
    assert oracle["post_shift_mean_reward"] >= local["post_shift_mean_reward"]
    assert oracle["post_shift_mean_regret"] == 0.0


def test_hard_router_benchmark_runs_and_reports_boundary():
    payload = run_hard_benchmark(
        HardBenchmarkConfig(ticks=40, shift_tick=20, batch_size=60, seeds=(0,)),
        write_outputs=False,
    )
    summary = payload["summary"]
    for label in [
        "hard_dte_surprise_only",
        "hard_dte_local_regret",
        "hard_dte_ucb",
        "hard_dte_contextual_ucb",
        "hard_dte_arbitrated_ucb",
        "hard_dte_reliability_arbitrated_ucb",
        "hard_dte_exp3",
        "hard_dte_reliability_arbitrated_exp3",
        "hard_static_contextual",
        "hard_epsilon_bandit",
        "hard_ucb",
        "hard_exp3",
        "hard_oracle",
    ]:
        assert label in summary
    report = render_hard_report(payload)
    assert "Institutional Boundary" in report
    assert "hard_dte_local_regret" in report


def test_frontier_runs_and_reports_axis_gaps():
    payload = run_frontier(
        FrontierConfig(
            ticks=35,
            shift_tick=16,
            batch_size=40,
            seeds=(0,),
            context_noise_values=(0.0, 0.22),
            label_noise_values=(0.28,),
            reward_delay_values=(0,),
            language_degradation_values=(0.62,),
            verifier_bonus_values=(0.12,),
        ),
        write_outputs=False,
    )
    assert "context_noise" in payload["summary"]
    assert len(payload["rows"]) == 6
    context_rows = payload["summary"]["context_noise"]["rows"]
    assert len(context_rows) == 2
    assert "dte_minus_contextual_regret" in context_rows[0]
    assert "dte_minus_ucb_regret" in context_rows[0]
    assert "dte_minus_exp3_regret" in context_rows[0]
    assert "dte_ucb_vs_dte_regret_gain" in context_rows[0]
    assert "dte_contextual_ucb_vs_dte_regret_gain" in context_rows[0]
    assert "dte_arbitrated_ucb_vs_dte_regret_gain" in context_rows[0]
    assert "dte_arbitrated_ucb_vs_contextual_ucb_regret_gain" in context_rows[0]
    assert "dte_reliability_arbitrated_ucb_vs_dte_regret_gain" in context_rows[0]
    assert (
        "dte_reliability_arbitrated_ucb_vs_arbitrated_ucb_regret_gain"
        in context_rows[0]
    )
    assert "dte_exp3_vs_dte_regret_gain" in context_rows[0]
    assert "dte_exp3_vs_reliability_arbitrated_ucb_regret_gain" in context_rows[0]
    assert "dte_exp3_minus_external_exp3_regret" in context_rows[0]
    report = render_frontier_report(payload)
    assert "Parameter Frontier" in report
    assert "DTE - Contextual" in report
    assert "DTE - UCB" in report
    assert "DTE - EXP3" in report
    assert "DTE-native UCB" in report
    assert "DTE-contextual UCB" in report
    assert "DTE-arbitrated UCB" in report
    assert "DTE-reliability-arbitrated UCB" in report
    assert "DTE-EXP3" in report


def test_adversarial_switch_benchmark_separates_switching_from_attribution():
    payload = run_adversarial_switch_benchmark(
        AdversarialSwitchSweepConfig(
            ticks=35,
            shift_tick=12,
            batch_size=40,
            seeds=(0,),
            switch_period_values=(2, 8),
            label_noise_values=(0.0, 0.28),
            reward_delay=2,
        ),
        write_outputs=False,
    )
    assert len(payload["rows"]) == 4
    row = payload["rows"][0]
    assert "dte_exp3_vs_reliability_ucb_gain" in row
    assert "exp3_reliability_gate_gain" in row
    assert "dte_exp3_minus_external_exp3" in row
    assert "winner" in row
    report = render_adversarial_switch_report(payload)
    assert "Adversarial-Switching" in report
    assert "attribution" in report

from kernel_local_regret_witness import run_condition, summarize


def test_local_regret_releases_stale_memory_more_than_surprise_only():
    surprise = run_condition(0.0, seed=0, agents=120, ticks=40)
    regret = run_condition(1.0, seed=0, agents=120, ticks=40)

    assert regret["final_stale_delta"] < 0.1 * surprise["final_stale_delta"]
    assert regret["final_p_entry_stale"] < 0.65
    assert surprise["final_p_entry_stale"] > 0.85
    assert regret["last10_entry_stale_share"] < surprise["last10_entry_stale_share"]


def test_witness_summary_groups_conditions():
    runs = [
        run_condition(0.0, seed=0, agents=80, ticks=20),
        run_condition(1.0, seed=0, agents=80, ticks=20),
    ]
    summary = summarize(runs)
    assert summary["0.0"]["runs"] == 1
    assert summary["1.0"]["runs"] == 1
    assert (
        summary["1.0"]["mean_final_stale_delta"]
        < summary["0.0"]["mean_final_stale_delta"]
    )

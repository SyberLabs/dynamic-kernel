from pathlib import Path

from two_route_memory_theorem import (
    SweepConfig,
    TwoRouteConfig,
    decay_only_recovery_bound,
    recovery_lower_bound,
    render_report,
    run_sweep,
    simulate,
    write_outputs,
)


def test_two_route_model_separates_no_memory_recovery_and_unrecovered_lockin():
    no_memory = simulate(
        TwoRouteConfig(
            cycles=60,
            deplete_cycle=20,
            rho=0.0,
            eta=0.12,
            epsilon=0.20,
        )
    )
    recovered = simulate(
        TwoRouteConfig(
            cycles=90,
            deplete_cycle=30,
            rho=0.12,
            eta=0.12,
            epsilon=0.02,
        )
    )
    unrecovered = simulate(
        TwoRouteConfig(
            cycles=90,
            deplete_cycle=30,
            rho=0.90,
            eta=0.01,
            epsilon=0.02,
        )
    )

    assert no_memory["classification"] == "no_lockin"
    assert recovered["classification"] == "recovered_stale_lockin"
    assert recovered["observed_recovery_time"] is not None
    assert unrecovered["classification"] == "unrecovered_stale_lockin"
    assert unrecovered["observed_recovery_time"] is None
    assert unrecovered["post_depletion_empty_rate"] > recovered["post_depletion_empty_rate"]
    assert unrecovered["decay_only_recovery_bound"] > unrecovered["stale_lockin_duration"]


def test_recovery_lower_bound_matches_memory_decay_scale():
    assert decay_only_recovery_bound(delta_memory_at_depletion=0.5, eta=0.10, theta=0.6) == 0
    assert decay_only_recovery_bound(delta_memory_at_depletion=1.2, eta=0.10, theta=0.6) > 0
    assert decay_only_recovery_bound(delta_memory_at_depletion=1.2, eta=0.0, theta=0.6) is None
    assert recovery_lower_bound(delta_memory_at_depletion=1.2, eta=0.10, theta=0.6) == decay_only_recovery_bound(
        delta_memory_at_depletion=1.2,
        eta=0.10,
        theta=0.6,
    )


def test_quick_two_route_sweep_outputs(tmp_path: Path):
    payload = run_sweep(SweepConfig(cycles=50, deplete_cycle=18), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["summary"]["classification_counts"]
    assert "Two-Route Memory-Ecology Theorem Simulator" in report
    assert "Maximum Lock-In" in report

    write_outputs(payload, tmp_path / "two_route.json", tmp_path / "two_route.md")
    assert (tmp_path / "two_route.json").exists()
    assert (tmp_path / "two_route.md").exists()

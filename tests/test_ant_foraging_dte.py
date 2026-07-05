from pathlib import Path

from ant_foraging_dte import (
    ForagingScenario,
    SimulationConfig,
    policies,
    render_report,
    run_pilot,
    simulate,
    write_outputs,
)


def test_ant_foraging_simulation_reports_biological_metrics():
    row = simulate(
        SimulationConfig(agents=48, steps=44),
        ForagingScenario(
            name="depletion_test",
            food_inventory={"Rich Food Patch": 18, "Sparse Food Patch": 900},
        ),
        policies()[1],
    )

    assert 0.0 <= row["food_completion_rate"] <= 1.0
    assert 0.0 <= row["returns_per_100_ant_steps"]
    assert 0.0 <= row["empty_food_visit_rate"] <= 1.0
    assert 0.0 <= row["fork_entropy"] <= 1.0
    assert 0.0 <= row["lock_in_index"] <= 1.0
    assert 0.0 <= row["hazard_rate"] <= 1.0
    assert 0.0 <= row["memory_staleness"]["structural_stale_flow"] <= 1.0
    assert 0.0 <= row["memory_staleness"]["preference_stale_concentration"] <= 1.0
    assert 0.0 <= row["memory_staleness"]["state_stale_alignment"] <= 1.0
    assert row["dominant_stale_memory_layer"] in {
        "none",
        "structural_memory",
        "preference_memory",
        "state_memory",
    }
    assert row["dominant_branch"] in {
        "Short Trail",
        "Long Trail",
        "Risky Ridge",
        "Shaded Detour",
        "none",
    }


def test_depletion_trap_is_detectable_against_static_short_path():
    config = SimulationConfig(agents=56, steps=58)
    scenario = ForagingScenario(
        name="rich_patch_depletion",
        food_inventory={"Rich Food Patch": 22, "Sparse Food Patch": 1000},
    )
    no_pheromone = simulate(config, scenario, policies()[0])
    rapid_evaporation = simulate(config, scenario, policies()[4])

    assert no_pheromone["empty_food_visit_rate"] >= 0.30
    assert "Rich Food Patch" in no_pheromone["memory_staleness"]["stale_nodes"]
    assert no_pheromone["dominant_stale_memory_layer"] != "none"
    assert rapid_evaporation["food_returned"] >= no_pheromone["food_returned"]


def test_quick_ant_pilot_outputs_regime_report(tmp_path: Path):
    payload = run_pilot(SimulationConfig(agents=32, steps=32), quick=True)
    report = render_report(payload)

    assert payload["rows"]
    assert payload["summary"]["class_counts"]
    assert "Ant Foraging DTE Prototype" in report
    assert "Empty visits" in report
    assert "Stale layer" in report

    write_outputs(payload, tmp_path / "ants.json", tmp_path / "ants.md")
    assert (tmp_path / "ants.json").exists()
    assert (tmp_path / "ants.md").exists()

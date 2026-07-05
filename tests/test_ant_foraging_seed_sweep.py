from pathlib import Path

from ant_foraging_dte import SimulationConfig
from ant_foraging_seed_sweep import (
    render_report,
    run_seed_robust_phase_sweep,
    summarize_seed_robust_phase,
    write_outputs,
)


def test_seed_summary_detects_majority_deadly_boundary():
    rows = [
        {
            "seed": 1,
            "scenario": "test",
            "pheromone_deposit": 0.0,
            "evaporation": 0.01,
            "scout_share": 0.1,
            "memory_ratio": 0.0,
            "phase": "baseline",
            "food_completion_rate": 0.20,
            "empty_food_visit_rate": 0.20,
            "score": 0.10,
        },
        {
            "seed": 1,
            "scenario": "test",
            "pheromone_deposit": 0.5,
            "evaporation": 0.01,
            "scout_share": 0.1,
            "memory_ratio": 50.0,
            "phase": "deadly_familiarity",
            "food_completion_rate": 0.18,
            "empty_food_visit_rate": 0.50,
            "score": 0.00,
        },
        {
            "seed": 2,
            "scenario": "test",
            "pheromone_deposit": 0.5,
            "evaporation": 0.01,
            "scout_share": 0.1,
            "memory_ratio": 50.0,
            "phase": "neutral_memory",
            "food_completion_rate": 0.22,
            "empty_food_visit_rate": 0.42,
            "score": 0.05,
        },
    ]

    summary = summarize_seed_robust_phase(rows)
    scenario = summary["scenario_summary"]["test"]

    assert scenario["any_deadly_min_memory_ratio"] == 50.0
    assert scenario["majority_deadly_min_memory_ratio"] == 50.0
    assert scenario["all_seed_deadly_min_memory_ratio"] is None


def test_quick_seed_robust_ant_sweep_outputs(tmp_path: Path):
    payload = run_seed_robust_phase_sweep(
        SimulationConfig(agents=28, steps=48),
        quick=True,
        seeds=1,
    )
    report = render_report(payload)

    assert payload["rows"]
    assert payload["summary"]["phase_counts"]
    assert "Seed-Robust Ant Memory Sweep" in report
    assert "Robust Thresholds" in report

    write_outputs(payload, tmp_path / "seed.json", tmp_path / "seed.md")
    assert (tmp_path / "seed.json").exists()
    assert (tmp_path / "seed.md").exists()


def test_seed_robust_ant_sweep_reproduces_familiarity_signal():
    payload = run_seed_robust_phase_sweep(
        SimulationConfig(agents=48, steps=90),
        quick=True,
        seeds=3,
    )
    delayed = payload["summary"]["scenario_summary"]["rich_patch_delayed_depletion"]

    assert delayed["any_deadly_min_memory_ratio"] is not None
    assert delayed["majority_deadly_min_memory_ratio"] is not None

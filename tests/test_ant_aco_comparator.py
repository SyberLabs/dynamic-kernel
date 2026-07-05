from pathlib import Path

from ant_aco_comparator import (
    ACOConfig,
    ComparisonPoint,
    render_report,
    run_comparison,
    simulate_aco,
    summarize_comparison,
    write_outputs,
)
from ant_foraging_dte import ForagingScenario, SimulationConfig


def test_aco_simulation_reports_same_core_metrics():
    row = simulate_aco(
        ACOConfig(agents=32, steps=36),
        ForagingScenario(
            name="aco_depletion_test",
            food_inventory={"Rich Food Patch": 24, "Sparse Food Patch": 900},
        ),
        ComparisonPoint("test", pheromone_deposit=0.25, evaporation=0.04, scout_share=0.25),
    )

    assert row["framework"] == "ACO"
    assert 0.0 <= row["food_completion_rate"] <= 1.0
    assert 0.0 <= row["empty_food_visit_rate"] <= 1.0
    assert 0.0 <= row["hazard_rate"] <= 1.0
    assert 0.0 <= row["fork_entropy"] <= 1.0
    assert row["dominant_branch"] in {
        "Short Trail",
        "Long Trail",
        "Risky Ridge",
        "Shaded Detour",
        "none",
    }


def test_comparison_summary_counts_framework_wins():
    rows = [
        {
            "framework": "DTE",
            "scenario": "s",
            "point": "p",
            "food_returned": 10,
            "returns_per_100_ant_steps": 1.0,
            "food_completion_rate": 0.30,
            "empty_food_visit_rate": 0.10,
            "rich_visit_share": 0.40,
            "hazard_rate": 0.01,
            "fork_entropy": 0.90,
            "lock_in_index": 0.30,
            "mean_return_path_length": 4.0,
            "pheromone_mass": 1.0,
            "pheromone_max": 1.2,
            "score": 0.20,
            "shock_recovery_steps": None,
        },
        {
            "framework": "ACO",
            "scenario": "s",
            "point": "p",
            "food_returned": 9,
            "returns_per_100_ant_steps": 0.9,
            "food_completion_rate": 0.20,
            "empty_food_visit_rate": 0.20,
            "rich_visit_share": 0.50,
            "hazard_rate": 0.01,
            "fork_entropy": 0.80,
            "lock_in_index": 0.40,
            "mean_return_path_length": 5.0,
            "pheromone_mass": 1.0,
            "pheromone_max": 1.2,
            "score": 0.10,
            "shock_recovery_steps": None,
        },
    ]

    summary = summarize_comparison(rows)

    assert summary["win_counts"]["DTE"] == 1
    assert summary["best_dte_advantage"]["dte_score_minus_aco"] > 0


def test_quick_aco_comparison_outputs(tmp_path: Path):
    payload = run_comparison(
        dte_config=SimulationConfig(agents=24, steps=32),
        aco_config=ACOConfig(agents=24, steps=32),
        quick=True,
        seeds=1,
    )
    report = render_report(payload)

    assert payload["rows"]
    assert payload["summary"]["comparisons"]
    assert "Ant DTE vs Classical ACO Comparator" in report
    assert "Framework Wins" in report

    write_outputs(payload, tmp_path / "aco.json", tmp_path / "aco.md")
    assert (tmp_path / "aco.json").exists()
    assert (tmp_path / "aco.md").exists()

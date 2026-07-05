from pathlib import Path

from semiconductor_onshoring_classification_robustness import (
    render_report as render_robustness_report,
    run_classification_robustness,
)
from semiconductor_onshoring_feasibility_preference import (
    render_report as render_surface_report,
    run_feasibility_preference_surface,
    write_outputs,
)
from semiconductor_onshoring_model_benchmark import (
    render_report as render_benchmark_report,
    run_model_benchmark,
)


def _small_surface():
    return run_feasibility_preference_surface(
        agent_levels=(40,),
        domestic_pulls=(0.0, 1.0),
        tariffs=(0.0,),
        seeds=(20260606,),
        steps=16,
    )


def test_small_feasibility_preference_surface_outputs(tmp_path: Path):
    payload = _small_surface()
    report = render_surface_report(payload)

    assert payload["rows"]
    assert payload["grouped"]
    assert "Feasibility-Preference Surface" in report
    write_outputs(payload, tmp_path / "surface.json", tmp_path / "surface.md")
    assert (tmp_path / "surface.json").exists()
    assert (tmp_path / "surface.md").exists()


def test_classification_robustness_reuses_surface_rows():
    payload = run_classification_robustness(_small_surface())
    report = render_robustness_report(payload)

    assert payload["threshold_configs"] == 54
    assert payload["results"]
    assert "Classification-Robustness Report" in report


def test_model_benchmark_compares_three_model_classes():
    payload = run_model_benchmark(_small_surface())
    report = render_benchmark_report(payload)

    assert payload["comparisons"]
    assert payload["frozen_rows"]
    assert payload["static_cells"]
    assert "Model Benchmark" in report

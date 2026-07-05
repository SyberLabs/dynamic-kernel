from optimizer import SymmetryMode
from neural_reassessment import (
    build_current_neural_kernel,
    render_report,
    run_mode_probe,
    run_reassessment,
)


def test_current_complete_neural_default_is_symmetric_and_stalls():
    row = run_mode_probe(
        mode=SymmetryMode.ENTROPY_PI,
        n=8,
        density=1.0,
        directed=False,
        steps=20,
    )
    assert abs(row["delta_sigma"]) < 1e-8
    assert row["beta_std"] == 0.0
    assert row["health"] == "STALLED"


def test_sparse_directed_weight_symmetry_has_signal():
    row = run_mode_probe(
        mode=SymmetryMode.WEIGHT_SYMMETRY,
        n=8,
        density=0.35,
        directed=True,
        steps=20,
    )
    assert row["delta_sigma"] > 0.0


def test_reassessment_report_states_verdict(tmp_path, monkeypatch):
    payload = {
        "mode_rows": [
            {
                "case": "demo",
                "mode": "ENTROPY_PI",
                "delta_sigma": 0.0,
                "beta_std": 0.0,
                "pi_std": 0.0,
                "mixing_time": 1.0,
                "health": "STALLED",
            }
        ],
        "composite_probe": {
            "target_node_1": 0.4,
            "pi_node_1": 0.12,
            "steps": 1,
            "target_feasibility": {"status": "PARTIAL", "l1_error": 0.5},
        },
    }
    report = render_report(payload)
    assert "advanced optimizer" in report
    assert "instrument" in report
    assert "What Would Make It A Serious Neural Application" in report


def test_current_neural_kernel_has_identical_features():
    kernel = build_current_neural_kernel(n=5)
    assert kernel.topo.node_features.shape == (5, 4)
    assert (kernel.topo.node_features == kernel.topo.node_features[0]).all()

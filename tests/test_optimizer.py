import numpy as np

from kernel import DynamicTopologyKernel, topology_from_edges
from optimizer import SymmetryMode, SymmetryOptimizer


def dense_kernel(n=6, beta_init=3.0):
    nodes = {f"N{i}": np.full(4, 0.25) for i in range(n)}
    edges = [(f"N{i}", f"N{j}", 1.0) for i in range(n) for j in range(n) if i != j]
    topo = topology_from_edges(nodes, edges, undirected=False)
    beta = np.zeros((n, n))
    beta[topo.adjacency_mask] = beta_init
    return DynamicTopologyKernel(
        topology=topo,
        beta=beta,
        node_bias=np.zeros(n),
        feedback_noise=0.0,
        feedback_rate=0.0,
    )


def test_optimizer_step_preserves_constraints():
    kern = dense_kernel()
    opt = SymmetryOptimizer(kern, mode=SymmetryMode.ENTROPY_PI, eta=0.02, noise_sigma=0.02)
    frame = opt.step()
    beta = np.array(frame["beta"])
    assert np.all(beta[kern.topo.adjacency_mask] >= 0)
    assert np.all(beta[kern.topo.adjacency_mask] <= opt.beta_max)
    assert np.all(beta[~kern.topo.adjacency_mask] == 0)


def test_gradient_matches_central_difference_entry():
    kern = dense_kernel(n=4)
    opt = SymmetryOptimizer(kern, mode=SymmetryMode.SPECTRAL_GAP, eps=1e-3, noise_sigma=0.0)
    grad = opt._gradient()
    i, j = np.argwhere(kern.topo.adjacency_mask)[0]
    beta = kern._beta.copy()
    plus = beta.copy()
    minus = beta.copy()
    plus[i, j] += opt.eps
    minus[i, j] -= opt.eps
    expected = (opt._eval_sigma(plus) - opt._eval_sigma(minus)) / (2 * opt.eps)
    assert abs(grad[i, j] - expected) < 1e-4


def test_entropy_pi_stationary_is_probability_vector():
    kern = dense_kernel()
    opt = SymmetryOptimizer(kern, mode=SymmetryMode.ENTROPY_PI)
    frame = opt.snapshot()
    pi = np.array(frame["pi"])
    assert abs(float(pi.sum()) - 1.0) < 1e-6
    assert np.all(pi >= 0.0)


def test_snapshot_includes_flow_diagnostics():
    kern = dense_kernel()
    opt = SymmetryOptimizer(kern, mode=SymmetryMode.ENTROPY_PI)
    frame = opt.snapshot()
    assert "entropy_production" in frame
    assert "irreversible_flux" in frame
    assert "edge_current" in frame
    current = np.array(frame["edge_current"])
    assert current.shape == kern._beta.shape
    np.testing.assert_allclose(current, -current.T, atol=1e-10)


def test_composite_snapshot_includes_target_feasibility():
    kern = dense_kernel()
    target = np.full(kern.topo.N, (1.0 - 0.4) / (kern.topo.N - 1))
    target[0] = 0.4
    opt = SymmetryOptimizer(
        kern,
        mode=SymmetryMode.COMPOSITE,
        target_pi=target,
        composite_lambda=0.0,
        noise_sigma=0.0,
    )
    frame = opt.snapshot()
    feasibility = frame["target_feasibility"]
    assert feasibility is not None
    assert feasibility["status"] in {"REACHABLE", "PARTIAL", "CONSTRAINED"}
    assert feasibility["l1_error"] >= 0.0
    assert len(feasibility["achieved_pi"]) == kern.topo.N


def test_entropy_pi_analytic_gradient_matches_central_difference():
    nodes = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([0.2, 0.8]),
        "C": np.array([0.7, 0.3]),
        "D": np.array([0.1, 1.0]),
    }
    edges = [
        ("A", "B", 1.0),
        ("A", "C", 1.4),
        ("B", "C", 1.1),
        ("B", "D", 1.2),
        ("C", "A", 1.3),
        ("C", "D", 0.9),
        ("D", "A", 1.5),
        ("D", "B", 1.0),
    ]
    topo = topology_from_edges(nodes, edges, undirected=False)
    beta = np.zeros((topo.N, topo.N))
    beta[topo.adjacency_mask] = 2.0
    kern = DynamicTopologyKernel(
        topology=topo,
        beta=beta,
        node_bias=np.array([0.1, 0.0, 0.2, 0.0]),
        feedback_noise=0.0,
        feedback_rate=0.0,
    )
    telemetry = np.array([0.6, 0.4])
    telemetry = telemetry / np.linalg.norm(telemetry)
    opt = SymmetryOptimizer(
        kern,
        mode=SymmetryMode.ENTROPY_PI,
        telemetry=telemetry,
        eps=1e-5,
        noise_sigma=0.0,
    )
    grad = opt._gradient()
    i, j = np.argwhere(topo.adjacency_mask)[0]
    beta0 = kern._beta.copy()
    plus = beta0.copy()
    minus = beta0.copy()
    plus[i, j] += opt.eps
    minus[i, j] -= opt.eps
    expected = (opt._eval_sigma(plus) - opt._eval_sigma(minus)) / (2 * opt.eps)
    assert abs(grad[i, j] - expected) < 1e-7

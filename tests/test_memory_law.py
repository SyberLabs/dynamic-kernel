"""
Tests for the opt-in Axis-5 preference-memory update law
(kernel.configure_memory_law / memory_law_step), promoted from the
design-space program. The contract under test:

  - DEFAULT (static / unconfigured) leaves all kernel behavior unchanged
  - traffic mode reinforces visited edges and evaporates the delta
  - reward_gated mode requires node_reward and deposits nothing when the
    reward is zero
  - adaptive_eta raises evaporation under reward-expectation surprise and
    decays stale memory faster than the base eta
  - the law never breaks choice-point invariance
"""

import numpy as np
import pytest

from kernel import DynamicTopologyKernel, topology_from_edges


def make_topo():
    return topology_from_edges(
        nodes={
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
            "C": np.array([0.5, 0.5]),
        },
        edges=[("A", "B", 2.0), ("B", "C", 1.0), ("C", "A", 1.5)],
        undirected=True,
    )


def make_kernel(**kwargs):
    return DynamicTopologyKernel(
        topology=make_topo(), beta=1.0, feedback_noise=0.0, **kwargs
    )


def uniform_traffic(n=3, mass=1.0):
    return np.full((n, n), mass)


def edge_mask(kernel):
    return kernel.topo.adjacency_mask.astype(float)


def test_unconfigured_kernel_is_static():
    kernel = make_kernel()
    before = kernel._sponsor_friction.copy()
    kernel.memory_law_step(uniform_traffic())  # no-op without configuration
    np.testing.assert_array_equal(kernel._sponsor_friction, before)
    assert kernel.memory_law_state() == {"mode": "static"}


def test_static_mode_resets_law():
    kernel = make_kernel()
    kernel.configure_memory_law(mode="traffic", rho=0.5, eta=0.1)
    kernel.configure_memory_law(mode="static")
    before = kernel._sponsor_friction.copy()
    kernel.memory_law_step(uniform_traffic())
    np.testing.assert_array_equal(kernel._sponsor_friction, before)


def test_traffic_mode_accumulates_and_evaporates():
    kernel = make_kernel()
    kernel.configure_memory_law(mode="traffic", rho=0.5, eta=0.1)
    traffic = uniform_traffic(mass=1.0)
    kernel.memory_law_step(traffic)
    np.testing.assert_allclose(
        kernel._sponsor_friction - kernel._friction_baseline,
        0.5 * traffic * edge_mask(kernel),
    )
    # Second step with zero traffic: pure evaporation of the delta
    kernel.memory_law_step(np.zeros((3, 3)))
    np.testing.assert_allclose(
        kernel._sponsor_friction - kernel._friction_baseline,
        0.5 * traffic * edge_mask(kernel) * 0.9,
    )


def test_reward_gated_requires_reward_and_gates_on_it():
    kernel = make_kernel()
    kernel.configure_memory_law(mode="reward_gated", rho=0.5, eta=0.1)
    with pytest.raises(ValueError, match="node_reward"):
        kernel.memory_law_step(uniform_traffic())
    kernel.memory_law_step(uniform_traffic(), node_reward=np.zeros(3))
    np.testing.assert_allclose(
        kernel._sponsor_friction, kernel._friction_baseline
    )
    kernel.memory_law_step(
        uniform_traffic(), node_reward=np.array([1.0, 0.0, 0.0])
    )
    delta = kernel._sponsor_friction - kernel._friction_baseline
    inbound = kernel.topo.adjacency_mask[:, 0]
    assert np.all(delta[inbound, 0] > 0)  # rewarded destination reinforced
    assert delta[0, 0] == 0.0             # non-edges do not accumulate memory
    np.testing.assert_allclose(delta[:, 1:], 0.0)


def test_beta_channel():
    kernel = make_kernel()
    kernel.configure_memory_law(mode="traffic", channel="beta", rho=0.2, eta=0.0)
    friction_before = kernel._sponsor_friction.copy()
    kernel.memory_law_step(uniform_traffic())
    np.testing.assert_array_equal(kernel._sponsor_friction, friction_before)
    np.testing.assert_allclose(
        kernel._beta - kernel._beta_baseline, 0.2 * edge_mask(kernel)
    )


def test_adaptive_eta_rises_on_surprise_and_clears_stale_memory_faster():
    def run(mode):
        kernel = make_kernel()
        kernel.configure_memory_law(
            mode=mode, rho=0.5, eta=0.02, eta_max=0.5,
            surprise_gain=4.0, initial_expectation=1.0,
        )
        traffic = uniform_traffic(mass=1.0)
        # Build memory while reward flows
        for _ in range(10):
            kernel.memory_law_step(traffic, node_reward=np.ones(3))
        # Reward collapses; traffic keeps hitting the stale destinations
        for _ in range(5):
            kernel.memory_law_step(traffic, node_reward=np.zeros(3))
        return kernel

    adaptive = run("adaptive_eta")
    fixed = run("reward_gated")
    assert np.max(adaptive._last_eta_effective) > 0.02
    stale_adaptive = np.sum(adaptive._sponsor_friction - adaptive._friction_baseline)
    stale_fixed = np.sum(fixed._sponsor_friction - fixed._friction_baseline)
    assert stale_adaptive < 0.5 * stale_fixed


def test_adaptive_eta_no_surprise_keeps_base_eta():
    kernel = make_kernel()
    kernel.configure_memory_law(
        mode="adaptive_eta", rho=0.5, eta=0.02, initial_expectation=1.0
    )
    kernel.memory_law_step(uniform_traffic(), node_reward=np.ones(3))
    np.testing.assert_allclose(kernel._last_eta_effective, 0.02)


def test_opportunity_cost_diagnostic_identifies_stale_edges():
    kernel = make_kernel()
    traffic = np.zeros((3, 3))
    traffic[0, 1] = 2.0
    rewards = np.array([0.0, 0.2, 1.0])
    diag = kernel.opportunity_cost_diagnostic(traffic, rewards)
    assert diag["mean_opportunity_cost"] == pytest.approx(0.8)
    assert diag["stale_flow_share"] == pytest.approx(1.0)
    assert diag["destination_regret"][1] == pytest.approx(0.8)


def test_adaptive_eta_can_use_opportunity_cost_without_reward_collapse():
    kernel = make_kernel()
    kernel.configure_memory_law(
        mode="adaptive_eta",
        rho=0.0,
        eta=0.02,
        eta_max=0.5,
        surprise_gain=0.0,
        opportunity_gain=1.0,
        initial_expectation=0.2,
    )
    traffic = np.zeros((3, 3))
    traffic[0, 1] = 1.0
    kernel.memory_law_step(traffic, node_reward=np.array([0.0, 0.2, 1.0]))
    assert kernel._last_eta_effective[1] > 0.02
    state = kernel.memory_law_state()
    assert state["last_opportunity_diagnostic"]["stale_flow_share"] == pytest.approx(1.0)


def test_choice_point_invariance_unaffected():
    topo = topology_from_edges(
        nodes={"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0]),
               "C": np.array([0.7, 0.7])},
        edges=[("A", "B", 5.0), ("B", "C", 1.0), ("C", "A", 2.0)],
        undirected=False,
    )
    kernel = DynamicTopologyKernel(topology=topo, feedback_noise=0.0)
    kernel.configure_memory_law(mode="traffic", rho=2.0, eta=0.05)
    rng = np.random.default_rng(3)
    for _ in range(5):
        kernel.memory_law_step(rng.random((3, 3)))
        P = kernel.transition_matrix(rng.normal(size=2))
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            assert P[i, j] == pytest.approx(1.0)


def test_invalid_arguments_rejected():
    kernel = make_kernel()
    with pytest.raises(ValueError, match="mode"):
        kernel.configure_memory_law(mode="bogus")
    with pytest.raises(ValueError, match="channel"):
        kernel.configure_memory_law(mode="traffic", channel="bogus")
    kernel.configure_memory_law(mode="traffic", rho=0.1, eta=0.1)
    with pytest.raises(ValueError, match="shape"):
        kernel.memory_law_step(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        kernel.memory_law_step(-np.ones((3, 3)))

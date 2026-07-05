"""
Regression tests for the external-review fixes (stages 1-4).

Covers:
  1. EXP3-IX estimator — the two-route switching scenario recovers
     (the pre-fix gain r * p_hat locked in forever).
  2. EXP3 weight renormalization — no cap saturation.
  3. Leverage field — analytic dP/dS and dpi/dS match finite differences,
     including the softplus floor gate; singleton rows have zero leverage.
  4. Injectable RNG — reproducible, isolated from the global stream.
  5. Sponsorship decay tick semantics — per time-step, not per agent-step.
  6. simulate_batch honors a custom feedback_fn.
  7. Vectorized transition_entropy matches the row-wise definition.
  8. TV mixing-time estimator — periodic honesty and known chains.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel import AgentState, DynamicTopologyKernel, topology_from_edges
from simulator import PopulationSimulator


def make_two_route_kernel(**exp3_kwargs):
    nodes = {
        "S": np.array([1.0, 0.0]),
        "H": np.array([1.0, 0.0]),
        "L": np.array([1.0, 0.0]),
    }
    edges = [
        ("S", "H", 1.0),
        ("S", "L", 1.0),
        ("H", "S", 1.0),
        ("L", "S", 1.0),
    ]
    topo = topology_from_edges(nodes=nodes, edges=edges, undirected=False)
    kernel = DynamicTopologyKernel(topology=topo, alpha=1.0, temperature=1.0)
    kernel.configure_edge_learning(
        mode="exp3",
        policy="arbitrated",
        policy_mix_min=1.0,
        policy_mix_max=1.0,
        policy_temperature=0.25,
        exp3_gamma=0.10,
        exp3_eta=0.20,
        **exp3_kwargs,
    )
    return topo, kernel


class TestExp3Estimator:
    def test_two_route_switching_recovers(self):
        """Pre-fix (gain = r * p_hat) this NEVER recovered; the IX estimator
        must re-prefer the sparse arm quickly after the popular arm dies."""
        topo, kernel = make_two_route_kernel()
        tel = np.array([1.0, 0.0])
        iS, iH, iL = (topo.labels.index(x) for x in ("S", "H", "L"))
        switch, horizon, mass = 200, 400, 1000.0
        recovery = None
        for t in range(1, horizon + 1):
            P = kernel.transition_matrix(tel)
            pH, pL = P[iS, iH], P[iS, iL]
            if t > switch and pL > 0.5 and recovery is None:
                recovery = t - switch
                break
            rH = 0.9 if t <= switch else 0.0
            traffic = np.zeros((topo.N, topo.N))
            traffic[iS, iH] = mass * pH
            traffic[iS, iL] = mass * pL
            reward = np.zeros((topo.N, topo.N))
            reward[iS, iH] = rH
            reward[iS, iL] = 0.5
            kernel.edge_learning_step(traffic, edge_reward=reward)
        assert recovery is not None, "EXP3 lane failed to recover after switch"
        assert recovery <= 100

    def test_gain_not_throttled_by_unpopularity(self):
        """A high-reward sparse edge must gain at least as much log-weight as
        a low-reward popular edge (the pre-fix rule inverted this)."""
        topo, kernel = make_two_route_kernel()
        iS, iH, iL = (topo.labels.index(x) for x in ("S", "H", "L"))
        traffic = np.zeros((topo.N, topo.N))
        traffic[iS, iH] = 0.95   # popular, low reward
        traffic[iS, iL] = 0.05   # sparse, high reward
        reward = np.zeros((topo.N, topo.N))
        reward[iS, iH] = 0.5
        reward[iS, iL] = 0.9
        kernel.edge_learning_step(traffic, edge_reward=reward)
        w = kernel.edge_learning_state()["edge_policy_weight"]
        assert w[iS, iL] > w[iS, iH]

    def test_weights_renormalized_no_cap_saturation(self):
        topo, kernel = make_two_route_kernel()
        iS, iH, iL = (topo.labels.index(x) for x in ("S", "H", "L"))
        traffic = np.zeros((topo.N, topo.N))
        traffic[iS, iH] = 0.5
        traffic[iS, iL] = 0.5
        reward = np.zeros((topo.N, topo.N))
        reward[iS, iH] = 1.0
        reward[iS, iL] = 0.8
        for _ in range(500):
            kernel.edge_learning_step(traffic, edge_reward=reward)
        w = kernel.edge_learning_state()["edge_policy_weight"]
        assert w[iS].max() == pytest.approx(1.0)
        # the reward gap must survive 500 updates (pre-fix cap collapse
        # would drive both to the cap and erase it)
        assert w[iS, iL] < w[iS, iH]

    def test_exp3_ix_validation(self):
        topo, kernel = make_two_route_kernel()
        with pytest.raises(ValueError, match="exp3_ix"):
            kernel.configure_edge_learning(
                mode="exp3", policy="arbitrated", exp3_ix=-0.1
            )


def make_random_kernel(seed=7, N=6, F=3, temperature=0.7):
    rng = np.random.default_rng(seed)
    labels = [f"n{i}" for i in range(N)]
    nodes = {lab: rng.normal(size=F) for lab in labels}
    edges = []
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < 0.6:
                edges.append((labels[i], labels[j], float(1.0 + 2.0 * rng.random())))
    topo = topology_from_edges(nodes=nodes, edges=edges, undirected=False)
    kernel = DynamicTopologyKernel(topology=topo, alpha=1.0, temperature=temperature)
    tel = rng.normal(size=F)
    return topo, kernel, tel / np.linalg.norm(tel)


class TestLeverageField:
    def test_edge_leverage_matches_finite_difference(self):
        topo, kernel, tel = make_random_kernel()
        L = kernel.edge_leverage(tel)
        P0 = kernel.transition_matrix(tel)
        eps = 1e-6
        for (i, j) in np.argwhere(topo.adjacency_mask):
            kernel._sponsor_friction[i, j] += eps
            P1 = kernel.transition_matrix(tel)
            kernel._sponsor_friction[i, j] -= eps
            fd = (P1[i, j] - P0[i, j]) / eps
            assert fd == pytest.approx(L[i, j], abs=5e-5)

    def test_leverage_includes_floor_gate(self):
        """Below the softplus floor, naive P(1-P)/tau overestimates by orders
        of magnitude; the gated formula must track the true sensitivity."""
        topo, kernel, tel = make_random_kernel()
        i, j = np.argwhere(topo.adjacency_mask)[0]
        # push the edge far below the floor via sponsorship
        kernel._sponsor_friction[i, j] = 10.0
        L = kernel.edge_leverage(tel)
        P = kernel.transition_matrix(tel)
        naive = P[i, j] * (1 - P[i, j]) / kernel.temperature
        assert L[i, j] < naive / 50.0

    def test_singleton_row_has_zero_leverage(self):
        """Choice-point invariance is the P -> 1 limit of the leverage field."""
        nodes = {"a": np.array([1.0]), "b": np.array([1.0]), "c": np.array([1.0])}
        edges = [("a", "b", 1.0), ("b", "c", 1.0), ("b", "a", 1.0), ("c", "a", 1.0)]
        topo = topology_from_edges(nodes=nodes, edges=edges, undirected=False)
        kernel = DynamicTopologyKernel(topology=topo, alpha=1.0, temperature=1.0)
        L = kernel.edge_leverage(np.array([1.0]))
        ia, ib = topo.labels.index("a"), topo.labels.index("b")
        assert L[ia, ib] == pytest.approx(0.0, abs=1e-12)

    def test_stationary_leverage_matches_finite_difference(self):
        topo, kernel, tel = make_random_kernel(seed=11, N=7)
        G = kernel.stationary_leverage(tel)
        pi0 = kernel.stationary_distribution(tel)
        eps = 1e-6
        for (i, j) in np.argwhere(topo.adjacency_mask)[:8]:
            kernel._sponsor_friction[i, j] += eps
            pi1 = kernel.stationary_distribution(tel)
            kernel._sponsor_friction[i, j] -= eps
            fd = (pi1 - pi0) / eps
            assert np.allclose(fd, G[i, j], atol=5e-4)

    def test_stationary_leverage_preserves_normalization(self):
        topo, kernel, tel = make_random_kernel(seed=11, N=7)
        G = kernel.stationary_leverage(tel)
        assert np.max(np.abs(G.sum(axis=2))) < 1e-12


class TestInjectableRng:
    def test_same_seed_same_paths(self):
        _, k1, tel = make_random_kernel(seed=3)
        _, k2, _ = make_random_kernel(seed=3)
        k1._rng = np.random.default_rng(99)
        k2._rng = np.random.default_rng(99)
        batch = np.tile(tel, (8, 1))
        starts = np.zeros(8, dtype=int)
        p1 = k1.simulate_batch(batch.copy(), starts.copy(), steps=15)
        p2 = k2.simulate_batch(batch.copy(), starts.copy(), steps=15)
        assert np.array_equal(p1, p2)

    def test_isolated_from_global_stream(self):
        _, k1, tel = make_random_kernel(seed=3)
        _, k2, _ = make_random_kernel(seed=3)
        k1._rng = np.random.default_rng(123)
        k2._rng = np.random.default_rng(123)
        batch = np.tile(tel, (6, 1))
        starts = np.zeros(6, dtype=int)
        np.random.seed(1)
        p1 = k1.simulate_batch(batch.copy(), starts.copy(), steps=10)
        np.random.seed(2)
        _ = np.random.random(1000)  # perturb the global stream
        p2 = k2.simulate_batch(batch.copy(), starts.copy(), steps=10)
        assert np.array_equal(p1, p2)

    def test_constructor_accepts_rng(self):
        nodes = {"a": np.array([1.0]), "b": np.array([1.0])}
        edges = [("a", "b", 1.0), ("b", "a", 1.0)]
        topo = topology_from_edges(nodes=nodes, edges=edges, undirected=False)
        kernel = DynamicTopologyKernel(
            topology=topo, rng=np.random.default_rng(0)
        )
        agent = AgentState(telemetry=np.array([1.0]), position=0)
        node, _ = kernel.step(agent)
        assert node in (0, 1)


class TestDecayTickSemantics:
    def make_sponsored_kernel(self, decay=0.5):
        nodes = {"a": np.array([1.0]), "b": np.array([1.0]), "c": np.array([1.0])}
        edges = [
            ("a", "b", 1.0), ("a", "c", 1.0),
            ("b", "a", 1.0), ("c", "a", 1.0),
        ]
        topo = topology_from_edges(nodes=nodes, edges=edges, undirected=False)
        kernel = DynamicTopologyKernel(
            topology=topo, sponsor_decay=decay, feedback_noise=0.0
        )
        kernel.sponsor_edge(0, 1, 5.0)
        return kernel

    def delta(self, kernel):
        return float((kernel._beta - kernel._beta_baseline)[0, 1])

    def test_simulate_batch_decays_per_step_not_per_agent(self):
        for K in (1, 10):
            kernel = self.make_sponsored_kernel(decay=0.5)
            batch = np.tile(np.array([1.0]), (K, 1))
            kernel.simulate_batch(batch, np.zeros(K, dtype=int), steps=3)
            # 3 time-steps -> delta * 0.5^3, regardless of population size
            assert self.delta(kernel) == pytest.approx(5.0 * 0.5**3)

    def test_step_decay_optout(self):
        kernel = self.make_sponsored_kernel(decay=0.5)
        agent = AgentState(telemetry=np.array([1.0]), position=0)
        kernel.step(agent, decay=False)
        assert self.delta(kernel) == pytest.approx(5.0)
        kernel.step(agent)
        assert self.delta(kernel) == pytest.approx(2.5)

    def test_population_simulator_applies_decay(self):
        kernel = self.make_sponsored_kernel(decay=0.1)
        sim = PopulationSimulator(kernel, K=5, rng=np.random.default_rng(0))
        for _ in range(4):
            sim.tick()
        assert self.delta(kernel) == pytest.approx(5.0 * 0.9**4)


class TestBatchFeedbackFn:
    def test_custom_feedback_fn_is_honored(self):
        nodes = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
        edges = [("a", "b", 1.0), ("b", "a", 1.0)]
        topo = topology_from_edges(nodes=nodes, edges=edges, undirected=False)
        calls = []

        def spy_feedback(telemetry, features):
            calls.append(1)
            return np.array([0.5, 0.5])

        kernel = DynamicTopologyKernel(topology=topo, feedback_fn=spy_feedback)
        K, steps = 4, 3
        batch = np.tile(np.array([1.0, 0.0]), (K, 1))
        kernel.simulate_batch(batch, np.zeros(K, dtype=int), steps=steps)
        assert len(calls) == K * steps


class TestDiagnostics:
    def test_transition_entropy_matches_rowwise(self):
        _, kernel, tel = make_random_kernel(seed=5)
        P = kernel.transition_matrix(tel)
        expected = np.array([kernel.row_entropy(P[i]) for i in range(P.shape[0])])
        assert np.allclose(kernel.transition_entropy(tel), expected)

    def test_tv_mixing_finite_for_aperiodic_chain(self):
        """Triangle plus a chord has cycle lengths 2 and 3 -> aperiodic,
        so TV decay must reach the threshold at a finite time."""
        nodes = {x: np.array([0.0]) for x in "abc"}
        edges = [("a", "b", 1.0), ("b", "c", 1.0), ("c", "a", 1.0), ("a", "c", 1.0)]
        topo = topology_from_edges(nodes=nodes, edges=edges, undirected=False)
        kernel = DynamicTopologyKernel(
            topology=topo, beta=0.0, temperature=1.0, feedback_noise=0.0
        )
        t = kernel.mixing_time_estimate(np.array([0.0]), method="tv")
        assert np.isfinite(t)
        assert t >= 1

    def test_tv_mixing_inf_for_periodic_chain(self):
        """A pure 2-cycle never converges in TV — inf is the honest answer."""
        nodes = {"a": np.array([0.0]), "b": np.array([0.0])}
        edges = [("a", "b", 1.0), ("b", "a", 1.0)]
        topo = topology_from_edges(nodes=nodes, edges=edges, undirected=False)
        kernel = DynamicTopologyKernel(
            topology=topo, beta=0.0, temperature=1.0, feedback_noise=0.0
        )
        t = kernel.mixing_time_estimate(np.array([0.0]), method="tv", horizon=300)
        assert np.isinf(t)

    def test_spectral_method_still_available(self):
        _, kernel, tel = make_random_kernel(seed=5)
        t = kernel.mixing_time_estimate(tel, method="spectral")
        assert t > 0

    def test_invalid_method_raises(self):
        _, kernel, tel = make_random_kernel(seed=5)
        with pytest.raises(ValueError, match="method"):
            kernel.mixing_time_estimate(tel, method="bogus")

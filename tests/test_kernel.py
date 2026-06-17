"""
Unit tests for DynamicTopologyKernel and topology utilities.
"""
import numpy as np
import pytest

from kernel import DynamicTopologyKernel, Topology, topology_from_edges, AgentState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_topo():
    """3-node undirected triangle with uniform features."""
    return topology_from_edges(
        nodes={
            "A": np.array([1.0, 0.0, 0.0]),
            "B": np.array([0.0, 1.0, 0.0]),
            "C": np.array([0.0, 0.0, 1.0]),
        },
        edges=[("A", "B", 2.0), ("B", "C", 2.0), ("A", "C", 3.0)],
        undirected=True,
    )


@pytest.fixture
def directed_topo():
    """3-node directed chain A→B→C with no return edges."""
    return topology_from_edges(
        nodes={
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.5, 0.5]),
            "C": np.array([0.0, 1.0]),
        },
        edges=[("A", "B", 1.0), ("B", "C", 1.0)],
        undirected=False,
    )


@pytest.fixture
def kernel(simple_topo):
    return DynamicTopologyKernel(
        topology=simple_topo,
        alpha=1.0,
        feedback_rate=0.1,
        temperature=1.0,
        feedback_noise=0.0,
    )


# ---------------------------------------------------------------------------
# topology_from_edges
# ---------------------------------------------------------------------------

class TestTopologyFromEdges:
    def test_undirected_mirrors_edges(self, simple_topo):
        D = simple_topo.distance_matrix
        assert D[0, 1] == D[1, 0] == 2.0
        assert D[0, 2] == D[2, 0] == 3.0

    def test_directed_no_mirror(self, directed_topo):
        D = directed_topo.distance_matrix
        assert D[0, 1] == 1.0
        assert np.isinf(D[1, 0]), "directed edge should NOT be mirrored"

    def test_adjacency_mask_excludes_self_loops(self, simple_topo):
        mask = simple_topo.adjacency_mask
        for i in range(3):
            assert not mask[i, i], "self-loops should be False in adjacency mask"

    def test_adjacency_mask_correct_edges(self, simple_topo):
        mask = simple_topo.adjacency_mask
        # All 3 edges present in both directions (undirected)
        assert mask[0, 1] and mask[1, 0]
        assert mask[1, 2] and mask[2, 1]
        assert mask[0, 2] and mask[2, 0]

    def test_N_and_F_properties(self, simple_topo):
        assert simple_topo.N == 3
        assert simple_topo.F == 3


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

class TestTransitionMatrix:
    def test_row_stochastic(self, kernel):
        tel = np.array([1.0, 0.0, 0.0])
        P = kernel.transition_matrix(tel)
        row_sums = P.sum(axis=1)
        # Rows with neighbors must sum to 1; isolated rows to 0
        connected = kernel.topo.adjacency_mask.any(axis=1)
        np.testing.assert_allclose(row_sums[connected], 1.0, atol=1e-10)

    def test_non_edges_are_zero(self, kernel):
        tel = np.array([1.0, 0.0, 0.0])
        P = kernel.transition_matrix(tel)
        mask = kernel.topo.adjacency_mask
        # Non-edge cells must be exactly 0
        non_edges = ~mask & ~np.eye(3, dtype=bool)
        assert np.all(P[non_edges] == 0.0)

    def test_aligned_node_gets_higher_prob(self, kernel):
        """Node A (feature=[1,0,0]) should dominate when telemetry=[1,0,0]."""
        tel = np.array([1.0, 0.0, 0.0])
        P = kernel.transition_matrix(tel)
        # From node B (idx=1), neighbor A (idx=0) should be preferred over C (idx=2)
        assert P[1, 0] > P[1, 2], "A aligns better with fashion telemetry than C"

    def test_temperature_effect(self, simple_topo):
        """High temperature should make transitions more uniform."""
        tel = np.array([1.0, 0.0, 0.0])
        low_tau = DynamicTopologyKernel(simple_topo, temperature=0.05, feedback_noise=0.0)
        high_tau = DynamicTopologyKernel(simple_topo, temperature=10.0, feedback_noise=0.0)
        P_low = low_tau.transition_matrix(tel)
        P_high = high_tau.transition_matrix(tel)
        # High temperature → smaller max probability per row
        for i in range(3):
            row_low = P_low[i]
            row_high = P_high[i]
            if row_low.sum() > 0:
                assert row_high.max() <= row_low.max() + 1e-6


# ---------------------------------------------------------------------------
# Batch transition matrix
# ---------------------------------------------------------------------------

class TestBatchTransitionMatrix:
    def test_shape(self, kernel):
        K = 10
        tels = np.random.dirichlet(np.ones(3), size=K)
        P = kernel.transition_matrix_batch(tels)
        assert P.shape == (K, 3, 3)

    def test_each_row_stochastic(self, kernel):
        K = 5
        tels = np.random.dirichlet(np.ones(3), size=K)
        P = kernel.transition_matrix_batch(tels)
        connected = kernel.topo.adjacency_mask.any(axis=1)
        for k in range(K):
            row_sums = P[k].sum(axis=1)
            np.testing.assert_allclose(row_sums[connected], 1.0, atol=1e-10)

    def test_matches_single(self, kernel):
        tel = np.array([0.5, 0.3, 0.2])
        P_single = kernel.transition_matrix(tel)
        P_batch = kernel.transition_matrix_batch(tel[np.newaxis, :])
        np.testing.assert_allclose(P_single, P_batch[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Sponsorship API
# ---------------------------------------------------------------------------

class TestSponsorAPI:
    def test_sponsor_edge_increases_prob(self, kernel):
        tel = np.array([0.5, 0.3, 0.2])
        P_before = kernel.transition_matrix(tel)
        kernel.sponsor_edge(0, 1, boost=20.0)  # A → B
        P_after = kernel.transition_matrix(tel)
        assert P_after[0, 1] > P_before[0, 1], "sponsoring edge should increase probability"

    def test_sponsor_node_boosts_all_inbound(self, kernel):
        tel = np.array([0.5, 0.3, 0.2])
        P_before = kernel.transition_matrix(tel)
        kernel.sponsor_node(2, boost=20.0)  # Node C
        P_after = kernel.transition_matrix(tel)
        # From A (idx=0) to C (idx=2) should increase
        assert P_after[0, 2] > P_before[0, 2]

    def test_sponsor_decay_erodes_delta(self, simple_topo):
        kern = DynamicTopologyKernel(
            simple_topo, temperature=1.0, feedback_noise=0.0, sponsor_decay=0.5
        )
        kern.sponsor_edge(0, 1, boost=10.0)
        beta_before = kern._beta[0, 1]
        agent = AgentState(telemetry=np.array([1.0, 0.0, 0.0]), position=0)
        kern.step(agent)
        beta_after = kern._beta[0, 1]
        assert beta_after < beta_before, "decay should erode sponsored delta"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_row_entropy_uniform(self, kernel):
        """Perfectly uniform row should have entropy = log2(num_neighbors)."""
        # Uniform row over 2 neighbors
        row = np.array([0.5, 0.5, 0.0])
        H = kernel.row_entropy(row)
        assert abs(H - 1.0) < 1e-10, f"expected 1 bit, got {H}"

    def test_row_entropy_deterministic(self, kernel):
        row = np.array([1.0, 0.0, 0.0])
        assert kernel.row_entropy(row) == 0.0

    def test_get_diagnostic_keys(self, kernel):
        tel = np.array([1.0, 0.0, 0.0])
        diag = kernel.get_diagnostic(tel)
        for key in ["alignment", "transition_matrix", "row_entropy",
                    "effective_rank", "mixing_time", "temperature",
                    "edge_flux", "edge_current", "entropy_production",
                    "irreversible_flux"]:
            assert key in diag

    def test_flow_diagnostic_current_is_antisymmetric(self, kernel):
        tel = np.array([0.4, 0.4, 0.2])
        flow = kernel.flow_diagnostic(tel)
        current = flow["edge_current"]
        np.testing.assert_allclose(current, -current.T, atol=1e-12)
        assert flow["entropy_production"] >= -1e-12
        assert flow["irreversible_flux"] >= 0.0

    def test_direct_stationary_matches_power_iteration(self, kernel):
        tel = np.array([0.4, 0.4, 0.2])
        power_pi = kernel.stationary_distribution(tel)
        direct_pi = kernel.stationary_distribution_direct(tel)
        np.testing.assert_allclose(direct_pi, power_pi, atol=1e-8)

    def test_effective_rank_gte_one(self, kernel):
        tel = np.array([1.0, 0.0, 0.0])
        ranks = kernel.effective_rank(tel)
        connected = kernel.topo.adjacency_mask.any(axis=1)
        assert np.all(ranks[connected] >= 1.0 - 1e-10)


# ---------------------------------------------------------------------------
# Agent step
# ---------------------------------------------------------------------------

class TestAgentStep:
    def test_step_moves_agent(self, kernel):
        np.random.seed(0)
        agent = AgentState(telemetry=np.array([1.0, 0.0, 0.0]), position=0)
        for _ in range(10):
            kernel.step(agent)
        assert len(agent.history) == 11  # initial + 10 steps

    def test_step_updates_telemetry(self, kernel):
        agent = AgentState(telemetry=np.array([1.0, 0.0, 0.0]), position=0)
        tel_before = agent.telemetry.copy()
        kernel.step(agent)
        # Telemetry should change due to feedback
        assert not np.allclose(agent.telemetry, tel_before)

    def test_directed_terminal_node_stays(self, directed_topo):
        """Node C in directed chain has no outbound edges — agent should stay."""
        kern = DynamicTopologyKernel(directed_topo, temperature=1.0, feedback_noise=0.0)
        agent = AgentState(telemetry=np.array([0.5, 0.5]), position=2)  # C
        node, _ = kern.step(agent)
        assert node == 2, "terminal node should not move"

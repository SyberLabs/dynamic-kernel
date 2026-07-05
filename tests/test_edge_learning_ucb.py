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
        edges=[("A", "B", 1.0), ("A", "C", 1.0), ("B", "C", 1.0)],
        undirected=False,
    )


def make_kernel():
    return DynamicTopologyKernel(
        topology=make_topo(),
        beta=0.0,
        temperature=0.4,
        feedback_noise=0.0,
        weight_floor=0.0,
    )


def test_edge_learning_static_by_default():
    kernel = make_kernel()
    telemetry = np.array([1.0, 0.0])
    before = kernel.transition_matrix(telemetry)
    traffic = np.zeros((3, 3))
    traffic[0, 1] = 1.0
    assert kernel.edge_learning_step(traffic, node_reward=np.ones(3))["mode"] == "static"
    after = kernel.transition_matrix(telemetry)
    np.testing.assert_allclose(after, before)
    assert kernel.edge_learning_state() == {"mode": "static"}


def test_edge_learning_reward_potential_raises_rewarded_edge_probability():
    kernel = make_kernel()
    telemetry = np.array([1.0, 0.0])
    before = kernel.transition_matrix(telemetry)
    kernel.configure_edge_learning(
        mode="ucb",
        reward_gain=2.0,
        uncertainty_gain=0.0,
        initial_reward=0.0,
    )
    traffic = np.zeros((3, 3))
    traffic[0, 1] = 10.0
    kernel.edge_learning_step(traffic, node_reward=np.array([0.0, 1.0, 0.0]))
    after = kernel.transition_matrix(telemetry)
    assert after[0, 1] > before[0, 1]
    assert after[0, 1] > after[0, 2]


def test_edge_learning_ucb_bonus_prefers_under_sampled_edge():
    kernel = make_kernel()
    kernel.configure_edge_learning(
        mode="ucb",
        reward_gain=0.0,
        uncertainty_gain=1.0,
        ucb_c=1.0,
    )
    traffic = np.zeros((3, 3))
    traffic[0, 1] = 20.0
    kernel.edge_learning_step(traffic, node_reward=np.zeros(3))
    state = kernel.edge_learning_state()
    # A->C is admissible and unvisited, so it should carry a larger optimism
    # bonus than the heavily sampled A->B edge.
    assert state["potential"][0, 2] > state["potential"][0, 1]


def test_edge_learning_rejects_invalid_inputs():
    kernel = make_kernel()
    with pytest.raises(ValueError, match="mode"):
        kernel.configure_edge_learning(mode="bogus")
    with pytest.raises(ValueError, match="policy"):
        kernel.configure_edge_learning(mode="ucb", policy="bogus")
    with pytest.raises(ValueError, match="policy_reliability"):
        kernel.configure_edge_learning(
            mode="ucb",
            policy="arbitrated",
            policy_reliability="bogus",
        )
    with pytest.raises(ValueError, match="policy_mix_min"):
        kernel.configure_edge_learning(
            mode="ucb",
            policy="arbitrated",
            policy_mix_min=0.5,
            policy_mix_max=0.1,
        )
    with pytest.raises(ValueError, match="exp3"):
        kernel.configure_edge_learning(mode="exp3")
    kernel.configure_edge_learning(mode="ucb")
    with pytest.raises(ValueError, match="exactly one"):
        kernel.edge_learning_step(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        kernel.edge_learning_step(-np.ones((3, 3)), node_reward=np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        kernel.edge_learning_step(np.zeros((3, 3)), node_reward=np.zeros(2))


def test_contextual_ucb_keeps_separate_context_reward_estimates():
    kernel = make_kernel()
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
    kernel.configure_edge_learning(
        mode="ucb",
        reward_gain=2.0,
        uncertainty_gain=0.0,
        context_centroids=centroids,
    )
    traffic = np.zeros((3, 3))
    traffic[0, 1] = 1.0
    kernel.edge_learning_step(
        traffic,
        node_reward=np.array([0.0, 1.0, 0.0]),
        context_index=0,
    )
    kernel.edge_learning_step(
        traffic,
        node_reward=np.array([0.0, 0.1, 0.0]),
        context_index=1,
    )
    state = kernel.edge_learning_state()
    assert state["edge_reward_estimate"][0, 0, 1] == pytest.approx(1.0)
    assert state["edge_reward_estimate"][1, 0, 1] == pytest.approx(0.1)


def test_contextual_ucb_transition_uses_active_telemetry_context():
    kernel = make_kernel()
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
    kernel.configure_edge_learning(
        mode="ucb",
        reward_gain=2.0,
        uncertainty_gain=0.0,
        context_centroids=centroids,
    )
    traffic_b = np.zeros((3, 3))
    traffic_c = np.zeros((3, 3))
    traffic_b[0, 1] = 1.0
    traffic_c[0, 2] = 1.0
    kernel.edge_learning_step(
        traffic_b,
        node_reward=np.array([0.0, 1.0, 0.0]),
        context_index=0,
    )
    kernel.edge_learning_step(
        traffic_c,
        node_reward=np.array([0.0, 0.0, 1.0]),
        context_index=1,
    )
    p_a = kernel.transition_matrix(np.array([1.0, 0.0]))
    p_b = kernel.transition_matrix(np.array([0.0, 1.0]))
    assert p_a[0, 1] > p_a[0, 2]
    assert p_b[0, 2] > p_b[0, 1]


def test_contextual_ucb_requires_context_index_on_update():
    kernel = make_kernel()
    kernel.configure_edge_learning(
        mode="ucb",
        context_centroids=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="context_index"):
        kernel.edge_learning_step(np.zeros((3, 3)), node_reward=np.zeros(3))


def test_arbitrated_ucb_can_be_disabled_by_zero_mix():
    kernel = make_kernel()
    telemetry = np.array([1.0, 0.0])
    before = kernel.transition_matrix(telemetry)
    kernel.configure_edge_learning(
        mode="ucb",
        policy="arbitrated",
        reward_gain=2.0,
        uncertainty_gain=2.0,
        policy_mix_min=0.0,
        policy_mix_max=0.0,
    )
    traffic = np.zeros((3, 3))
    traffic[0, 1] = 10.0
    kernel.edge_learning_step(traffic, node_reward=np.array([0.0, 1.0, 0.0]))
    after = kernel.transition_matrix(telemetry)
    np.testing.assert_allclose(after, before)
    np.testing.assert_allclose(kernel.edge_learning_state()["last_policy_mix"], 0.0)


def test_arbitrated_ucb_mixes_toward_under_sampled_edge_policy():
    kernel = make_kernel()
    telemetry = np.array([1.0, 0.0])
    before = kernel.transition_matrix(telemetry)
    kernel.configure_edge_learning(
        mode="ucb",
        policy="arbitrated",
        reward_gain=0.0,
        uncertainty_gain=1.0,
        ucb_c=1.0,
        policy_mix_max=0.75,
        policy_uncertainty_scale=0.01,
        policy_temperature=0.05,
    )
    traffic = np.zeros((3, 3))
    traffic[0, 1] = 20.0
    kernel.edge_learning_step(traffic, node_reward=np.zeros(3))
    after = kernel.transition_matrix(telemetry)
    state = kernel.edge_learning_state()
    assert after[0, 2] > before[0, 2]
    assert after[0, 2] > after[0, 1]
    assert state["last_policy_mix"][0] > 0.0


def test_reliability_gated_arbitration_reduces_mix_for_ambiguous_context():
    kernel = make_kernel()
    kernel.configure_edge_learning(
        mode="ucb",
        policy="arbitrated",
        reward_gain=0.0,
        uncertainty_gain=1.0,
        ucb_c=1.0,
        context_centroids=np.array([[1.0, 0.0], [0.0, 1.0]]),
        policy_mix_max=0.75,
        policy_uncertainty_scale=0.01,
        policy_temperature=0.05,
        policy_reliability="centroid_margin",
        policy_reliability_floor=0.0,
        policy_reliability_scale=0.10,
    )
    traffic = np.zeros((3, 3))
    traffic[0, 1] = 20.0
    kernel.edge_learning_step(traffic, node_reward=np.zeros(3), context_index=0)

    kernel.transition_matrix(np.array([1.0, 0.0]))
    crisp_state = kernel.edge_learning_state()
    crisp_mix = crisp_state["last_policy_mix"][0]
    crisp_reliability = crisp_state["last_policy_reliability"][0]

    ambiguous = np.array([1.0, 1.0])
    ambiguous = ambiguous / np.linalg.norm(ambiguous)
    kernel.transition_matrix(ambiguous)
    ambiguous_state = kernel.edge_learning_state()
    ambiguous_mix = ambiguous_state["last_policy_mix"][0]
    ambiguous_reliability = ambiguous_state["last_policy_reliability"][0]

    assert crisp_reliability > ambiguous_reliability
    assert crisp_mix > ambiguous_mix
    assert ambiguous_mix == pytest.approx(0.0)


def test_exp3_edge_learning_increases_rewarded_edge_weight():
    kernel = make_kernel()
    kernel.configure_edge_learning(
        mode="exp3",
        policy="arbitrated",
        exp3_eta=1.0,
        exp3_gamma=0.0,
        policy_mix_max=0.5,
        policy_uncertainty_scale=0.01,
        policy_temperature=1.0,
    )
    traffic = np.zeros((3, 3))
    traffic[0, 1] = 1.0
    kernel.edge_learning_step(traffic, node_reward=np.array([0.0, 1.0, 0.0]))
    state = kernel.edge_learning_state()
    assert state["mode"] == "exp3"
    assert state["edge_policy_weight"][0, 1] > state["edge_policy_weight"][0, 2]
    assert state["potential"][0, 1] > state["potential"][0, 2]


def test_exp3_arbitration_shifts_toward_rewarded_policy_lane():
    kernel = make_kernel()
    telemetry = np.array([1.0, 0.0])
    before = kernel.transition_matrix(telemetry)
    kernel.configure_edge_learning(
        mode="exp3",
        policy="arbitrated",
        exp3_eta=2.0,
        exp3_gamma=0.0,
        policy_mix_max=0.75,
        policy_uncertainty_scale=0.01,
        policy_temperature=1.0,
    )
    traffic = np.zeros((3, 3))
    traffic[0, 1] = 1.0
    kernel.edge_learning_step(traffic, node_reward=np.array([0.0, 1.0, 0.0]))
    after = kernel.transition_matrix(telemetry)
    state = kernel.edge_learning_state()
    assert after[0, 1] > before[0, 1]
    assert after[0, 1] > after[0, 2]
    assert state["last_policy_mix"][0] > 0.0

import numpy as np
import pytest

from kernel import DynamicTopologyKernel, topology_from_edges
from simulator import PopulationSimulator


def make_two_node_kernel():
    topo = topology_from_edges(
        nodes={
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
        },
        edges=[("A", "B", 1.0), ("B", "A", 1.0)],
        undirected=False,
    )
    return DynamicTopologyKernel(topology=topo, beta=1.0, feedback_noise=0.0)


def test_population_simulator_applies_memory_law_from_completed_traffic():
    kernel = make_two_node_kernel()
    kernel.configure_memory_law(mode="traffic", rho=0.5, eta=0.0)
    sim = PopulationSimulator(kernel, K=4, time_multiplier=1.0)

    first = sim.tick()
    assert first["memory_update"]["traffic_mass"] == 0.0
    assert int(np.sum(sim.state == 1)) == 4

    second = sim.tick()
    assert second["memory_update"]["traffic_mass"] == pytest.approx(4.0)
    delta = kernel._sponsor_friction - kernel._friction_baseline
    assert delta[0, 1] == pytest.approx(2.0)
    assert delta[0, 0] == 0.0


def test_population_simulator_reward_gated_memory_requires_rewards():
    kernel = make_two_node_kernel()
    kernel.configure_memory_law(mode="reward_gated", rho=0.5, eta=0.0)
    sim = PopulationSimulator(kernel, K=2, time_multiplier=1.0)

    with pytest.raises(ValueError, match="node_rewards"):
        sim.tick()


def test_population_simulator_reports_jsonable_memory_state():
    kernel = make_two_node_kernel()
    kernel.configure_memory_law(
        mode="adaptive_eta",
        rho=0.0,
        eta=0.02,
        opportunity_gain=1.0,
        initial_expectation=0.0,
    )
    sim = PopulationSimulator(
        kernel,
        K=2,
        time_multiplier=1.0,
        node_rewards=np.array([1.0, 0.1]),
    )

    sim.tick()
    sim.tick()
    snapshot = sim.get_report_snapshot()
    state = snapshot["memory_law"]["state"]
    assert isinstance(state["reward_expectation"], list)
    assert isinstance(state["last_eta_effective"], list)
    assert snapshot["memory_law"]["node_rewards"] == [1.0, 0.1]

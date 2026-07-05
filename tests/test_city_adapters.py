import numpy as np

from adapters import RHIZOME_CITY, WHEEL_CITY


def test_wheel_city_peripheral_routes_prefer_cbd():
    kern = WHEEL_CITY.build_kernel(feedback_noise=0.0)
    resident = np.array([0.7, 0.1, 0.2, 0.3])
    p = kern.transition_matrix(resident)
    cbd = kern.topo.labels.index("CBD")
    inbound = [
        p[i, cbd]
        for i in range(kern.topo.N)
        if i != cbd and kern.topo.adjacency_mask[i, cbd]
    ]
    assert min(inbound) > 0.7


def test_rhizome_city_has_high_mean_entropy():
    kern = RHIZOME_CITY.build_kernel(feedback_noise=0.0)
    resident = np.array([0.7, 0.1, 0.2, 0.3])
    mean_entropy = float(np.mean(kern.transition_entropy(resident)))
    assert mean_entropy > 1.5


def test_city_stationary_distribution_is_probability_vector():
    kern = WHEEL_CITY.build_kernel(feedback_noise=0.0)
    commuter = np.array([0.1, 0.8, 0.0, 0.8])
    pi = kern.stationary_distribution(commuter)
    assert pi.shape == (12,)
    assert abs(float(pi.sum()) - 1.0) < 1e-6
    assert np.all(pi >= 0.0)

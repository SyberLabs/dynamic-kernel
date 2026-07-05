import numpy as np

from adapters import SOCIAL_MEDIA


def _intent(name: str) -> np.ndarray:
    telemetry = np.array(SOCIAL_MEDIA.intent_presets[name], dtype=np.float64)
    return telemetry / np.linalg.norm(telemetry)


def test_social_media_adapter_is_sink_free_directed_graph():
    kern = SOCIAL_MEDIA.build_kernel(feedback_noise=0.0)
    assert kern.topo.N == 16
    assert kern.topo.F == 10
    assert not SOCIAL_MEDIA.undirected
    assert kern.topo.adjacency_mask.sum() >= 50
    assert np.all(kern.topo.adjacency_mask.any(axis=1))


def test_high_arousal_intent_prefers_conflict_loop():
    kern = SOCIAL_MEDIA.build_kernel(feedback_noise=0.0)
    labels = kern.topo.labels
    p = kern.transition_matrix(_intent("High-Arousal Scroll"))
    civic = labels.index("Civic Debate")
    conflict = labels.index("Conflict Commentary")
    science = labels.index("Science Explainers")
    conspiracy = labels.index("Conspiracy/Rumor")

    assert p[civic, conflict] > p[civic, science]
    assert p[conflict, conspiracy] > p[conflict, labels.index("Longform Off-Ramp")]


def test_deep_research_off_ramp_prefers_credible_content():
    kern = SOCIAL_MEDIA.build_kernel(feedback_noise=0.0)
    labels = kern.topo.labels
    p = kern.transition_matrix(_intent("Deep Research"))
    off_ramp = labels.index("Longform Off-Ramp")
    science = labels.index("Science Explainers")
    local_news = labels.index("Local News")
    friend_updates = labels.index("Friend Updates")

    assert p[off_ramp, science] > p[off_ramp, friend_updates]
    assert p[off_ramp, local_news] > p[off_ramp, friend_updates]


def test_social_media_flow_diagnostics_are_finite():
    kern = SOCIAL_MEDIA.build_kernel(feedback_noise=0.0)
    diag = kern.get_diagnostic(_intent("High-Arousal Scroll"))
    assert diag["entropy_production"] >= 0.0
    assert diag["irreversible_flux"] >= 0.0
    assert np.all(np.isfinite(diag["edge_current"]))

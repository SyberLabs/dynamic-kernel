import numpy as np

from neural_v2_adaptive_routing import (
    DEFAULT_CONDITIONS,
    LANGUAGE,
    SYMBOLIC,
    build_neural_v2_kernel,
    module_feature_table,
    run_condition,
    summarize,
)


def condition(label):
    return next(c for c in DEFAULT_CONDITIONS if c.label == label)


def test_neural_v2_modules_have_distinct_semantic_features():
    features = module_feature_table()
    language = features["Language Expert"]
    symbolic = features["Symbolic Expert"]
    assert not np.allclose(language, symbolic)
    assert language[1] > symbolic[1]
    assert symbolic[2] > language[2]


def test_neural_v2_kernel_uses_memory_law_with_local_regret():
    kernel = build_neural_v2_kernel(condition("local_regret"))
    state = kernel.memory_law_state()
    assert state["mode"] == "adaptive_eta"
    assert state["opportunity_gain"] > 0.0
    assert kernel.topo.adjacency_mask[0, LANGUAGE]
    assert kernel.topo.adjacency_mask[0, SYMBOLIC]


def test_local_regret_improves_post_shift_routing_against_surprise_only():
    surprise = run_condition(condition("surprise_only"), seed=0, ticks=100, batch_size=160)
    regret = run_condition(condition("local_regret"), seed=0, ticks=100, batch_size=160)

    assert regret["post_shift_mean_regret"] < surprise["post_shift_mean_regret"]
    assert regret["post_shift_symbolic_share"] > surprise["post_shift_symbolic_share"]
    assert regret["post_shift_language_share"] < surprise["post_shift_language_share"]


def test_summary_groups_conditions():
    rows = [
        run_condition(condition("base_forgetting"), seed=0, ticks=40, shift_tick=20, batch_size=80),
        run_condition(condition("local_regret"), seed=0, ticks=40, shift_tick=20, batch_size=80),
    ]
    summary = summarize(rows)
    assert summary["base_forgetting"]["runs"] == 1
    assert summary["local_regret"]["runs"] == 1

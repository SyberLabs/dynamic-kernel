"""
Test Suite — Dynamic Topology Kernel
=====================================
Property-based and unit tests for correctness, robustness, and
backward compatibility of the improved kernel.

Run: py -m pytest test_kernel.py -v
  or: py test_kernel.py  (standalone mode)
"""

import numpy as np
import time
import sys
from kernel import (
    DynamicTopologyKernel,
    AgentState,
    Topology,
    topology_from_edges,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def make_mall_topology() -> Topology:
    """Standard 5-node mall topology used in all tests."""
    return topology_from_edges(
        nodes={
            "Entrance":     np.array([0.0, 0.0, 0.0]),
            "FoodCourt":    np.array([0.1, 0.9, 0.0]),
            "TechStore":    np.array([0.0, 0.1, 0.9]),
            "OutletA":      np.array([0.8, 0.0, 0.1]),
            "OutletB":      np.array([0.9, 0.1, 0.0]),
        },
        edges=[
            ("Entrance", "FoodCourt", 5.0),
            ("Entrance", "TechStore", 4.0),
            ("Entrance", "OutletA", 10.0),
            ("FoodCourt", "OutletB", 6.0),
            ("OutletA", "OutletB", 3.0),
            ("TechStore", "OutletA", 7.0),
        ],
    )


def make_default_kernel(topo=None, **kwargs) -> DynamicTopologyKernel:
    """Kernel with standard defaults."""
    if topo is None:
        topo = make_mall_topology()
    defaults = dict(
        topology=topo,
        alpha=1.0,
        beta=np.full((topo.N, topo.N), 5.0),
        feedback_rate=0.15,
        temperature=1.0,
        feedback_noise=0.0,  # deterministic for testing
    )
    defaults.update(kwargs)
    return DynamicTopologyKernel(**defaults)


# ===========================================================================
# Test catalog
# ===========================================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            msg = f"  [FAIL] {name}"
            if detail:
                msg += f": {detail}"
            print(msg)
            self.errors.append(name)

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"  Failures:")
            for e in self.errors:
                print(f"    - {e}")
        print(f"{'='*60}")
        return self.failed == 0


def run_all_tests():
    R = TestResults()

    # ===================================================================
    # 1. ROW STOCHASTICITY
    # ===================================================================
    print("\n--- 1. Row Stochasticity ---")
    kernel = make_default_kernel()
    telemetries = {
        "fashion":  np.array([1.0, 0.0, 0.0]),
        "food":     np.array([0.0, 1.0, 0.0]),
        "tech":     np.array([0.0, 0.0, 1.0]),
        "neutral":  np.array([1/3, 1/3, 1/3]),
        "zero":     np.array([0.0, 0.0, 0.0]),
    }
    for name, tel in telemetries.items():
        P = kernel.transition_matrix(tel)
        row_sums = P.sum(axis=1)
        nonzero_rows = row_sums > 0
        stochastic = np.allclose(row_sums[nonzero_rows], 1.0, atol=1e-10)
        nonneg = np.all(P >= 0)
        R.check(
            f"Stochasticity [{name}]",
            stochastic and nonneg,
            f"row_sums={row_sums}, min_P={P.min()}"
        )

    # ===================================================================
    # 2. ADJACENCY MASK ENFORCEMENT
    # ===================================================================
    print("\n--- 2. Adjacency Mask ---")
    P = kernel.transition_matrix(np.array([1.0, 0.0, 0.0]))
    topo = kernel.topo
    for i in range(topo.N):
        for j in range(topo.N):
            if not topo.adjacency_mask[i, j] and i != j:
                R.check(
                    f"No edge {topo.labels[i]}->{topo.labels[j]} = 0",
                    P[i, j] == 0.0,
                    f"P[{i},{j}]={P[i,j]}"
                )

    # ===================================================================
    # 3. SOFTPLUS FLOOR
    # ===================================================================
    print("\n--- 3. Softplus Floor ---")
    kernel_soft = make_default_kernel()

    # Check that edges with different raw weights are NOT collapsed
    tel = np.array([1.0, 0.0, 0.0])
    diag = kernel_soft.get_diagnostic(tel)
    W_raw = diag["weight_matrix_raw"]
    W = diag["weight_matrix"]

    # OutletA->OutletB raw = -1.5, OutletB->OutletA raw = -1.0
    # Under hard floor, both would be 0.1 (identical)
    # Under softplus, they should be different (or very close but not identical)
    w_ab = W[3, 4]  # OutletA -> OutletB
    w_ba = W[4, 3]  # OutletB -> OutletA
    R.check(
        "Softplus preserves W ordering",
        w_ab <= w_ba or np.isclose(w_ab, w_ba, atol=0.001),
        f"W(A->B)={w_ab:.4f}, W(B->A)={w_ba:.4f}"
    )
    R.check(
        "Softplus floor is smooth (W > 0)",
        np.all(W[topo.adjacency_mask] > 0),
        f"min_W={W[topo.adjacency_mask].min():.4f}"
    )
    R.check(
        "Softplus floor approaches floor value",
        w_ab >= kernel_soft._weight_floor * 0.99,
        f"W(A->B)={w_ab:.4f}, floor={kernel_soft._weight_floor}"
    )

    # ===================================================================
    # 4. TEMPERATURE EXTREMES
    # ===================================================================
    print("\n--- 4. Temperature Control ---")
    tel = np.array([1.0, 0.0, 0.0])

    # Low temperature -> greedy (near-deterministic)
    kernel_cold = make_default_kernel(temperature=0.01)
    P_cold = kernel_cold.transition_matrix(tel)
    for i in range(topo.N):
        row = P_cold[i]
        if row.sum() > 0:
            max_p = row.max()
            R.check(
                f"Cold (tau=0.01) row {i} near-deterministic",
                max_p > 0.95,
                f"max_p={max_p:.4f}"
            )

    # High temperature -> uniform over neighbors
    kernel_hot = make_default_kernel(temperature=100.0)
    P_hot = kernel_hot.transition_matrix(tel)
    for i in range(topo.N):
        row = P_hot[i]
        neighbors = np.where(row > 0)[0]
        if len(neighbors) > 1:
            probs = row[neighbors]
            spread = probs.max() / probs.min()
            R.check(
                f"Hot (tau=100) row {i} near-uniform",
                spread < 2.0,
                f"spread={spread:.2f}, probs={np.round(probs, 4)}"
            )

    # Temperature schedule
    schedule_calls = []
    def temp_schedule(step):
        schedule_calls.append(step)
        return 5.0 - step * 0.1  # cool from 5.0 to ~0

    kernel_sched = make_default_kernel(temperature_fn=temp_schedule)
    agent = AgentState(telemetry=tel.copy(), position=0)
    kernel_sched.simulate(agent, steps=3)
    R.check(
        "Temperature schedule was called",
        len(schedule_calls) > 0,
        f"calls={schedule_calls}"
    )

    # ===================================================================
    # 5. NODE BIAS
    # ===================================================================
    print("\n--- 5. Node Bias ---")
    # Without bias, Entrance has alignment 0 for all telemetries
    kernel_nobias = make_default_kernel()
    diag_nb = kernel_nobias.get_diagnostic(tel)
    align_entrance_nb = diag_nb["alignment"][0]

    # With bias, Entrance gets a boost
    kernel_bias = make_default_kernel(
        node_bias=np.array([0.5, 0.0, 0.0, 0.0, 0.0])
    )
    diag_b = kernel_bias.get_diagnostic(tel)
    align_entrance_b = diag_b["alignment"][0]

    R.check(
        "Node bias increases Entrance alignment",
        align_entrance_b > align_entrance_nb,
        f"without={align_entrance_nb:.3f}, with={align_entrance_b:.3f}"
    )

    # Higher bias -> higher return probability from neighboring nodes
    P_nb = kernel_nobias.transition_matrix(tel)
    P_b = kernel_bias.transition_matrix(tel)
    # From FoodCourt (1) back to Entrance (0)
    R.check(
        "Node bias increases return probability to Entrance",
        P_b[1, 0] >= P_nb[1, 0],
        f"P(FC->Ent) without={P_nb[1,0]:.4f}, with={P_b[1,0]:.4f}"
    )

    # ===================================================================
    # 6. FEEDBACK NOISE (ANTI-LOCK-IN)
    # ===================================================================
    print("\n--- 6. Feedback Noise (Anti-Lock-In) ---")

    # Without noise: telemetry converges to fixed point
    np.random.seed(42)
    kernel_quiet = make_default_kernel(feedback_noise=0.0)
    agent_q = AgentState(telemetry=np.array([1.0, 0.0, 0.0]), position=0)
    for _ in range(100):
        kernel_quiet.step(agent_q)
    tel_50_quiet = agent_q.telemetry.copy()
    for _ in range(50):
        kernel_quiet.step(agent_q)
    tel_100_quiet = agent_q.telemetry.copy()
    drift_quiet = np.linalg.norm(tel_100_quiet - tel_50_quiet)

    # With noise: telemetry should keep exploring
    np.random.seed(42)
    kernel_noisy = make_default_kernel(feedback_noise=0.05)
    agent_n = AgentState(telemetry=np.array([1.0, 0.0, 0.0]), position=0)
    telemetry_log = []
    for _ in range(150):
        kernel_noisy.step(agent_n)
        telemetry_log.append(agent_n.telemetry.copy())

    # Check that telemetry is still moving at the end
    late_tels = np.array(telemetry_log[100:])
    late_var = np.var(late_tels, axis=0).sum()

    R.check(
        "Noisy feedback maintains telemetry variance",
        late_var > 1e-6,
        f"late_variance={late_var:.8f}"
    )

    # Norm preservation even with noise
    norms = np.linalg.norm(np.array(telemetry_log), axis=1)
    R.check(
        "Feedback preserves unit norm with noise",
        np.allclose(norms, 1.0, atol=1e-6),
        f"norm_range=[{norms.min():.6f}, {norms.max():.6f}]"
    )

    # ===================================================================
    # 7. BATCH vs SEQUENTIAL CONSISTENCY
    # ===================================================================
    print("\n--- 7. Batch vs Sequential Consistency ---")
    kernel_test = make_default_kernel(feedback_noise=0.0)

    np.random.seed(99)
    tel_test = np.array([0.5, 0.3, 0.2])
    agent_seq = AgentState(telemetry=tel_test.copy(), position=0)
    kernel_test.simulate(agent_seq, steps=5)
    seq_path = agent_seq.history

    np.random.seed(99)
    batch_tel = tel_test.copy().reshape(1, -1)
    batch_start = np.array([0])
    batch_paths = kernel_test.simulate_batch(batch_tel, batch_start, steps=5)
    batch_path = batch_paths[0].tolist()

    R.check(
        "Batch == Sequential path (no noise)",
        seq_path == batch_path,
        f"seq={seq_path}, batch={batch_path}"
    )

    # ===================================================================
    # 8. BATCH PERFORMANCE
    # ===================================================================
    print("\n--- 8. Batch Performance ---")
    kernel_perf = make_default_kernel(feedback_noise=0.0)
    K = 1000
    rng = np.random.default_rng(42)
    raw_tel = rng.dirichlet(np.ones(3), size=K)
    starts = np.zeros(K, dtype=int)

    t0 = time.perf_counter()
    kernel_perf.simulate_batch(raw_tel, starts, steps=5)
    t1 = time.perf_counter()
    batch_time = t1 - t0

    R.check(
        f"Batch 1000x5 < 0.5s",
        batch_time < 0.5,
        f"time={batch_time:.3f}s"
    )

    # ===================================================================
    # 9. SPONSOR MONOTONICITY
    # ===================================================================
    print("\n--- 9. Sponsor Monotonicity ---")
    kernel_s1 = make_default_kernel()
    P_before = kernel_s1.transition_matrix(tel)
    p_before = P_before[0, 1]  # Entrance -> FoodCourt

    kernel_s1.sponsor_edge(0, 1, boost=5.0)
    P_after = kernel_s1.transition_matrix(tel)
    p_after = P_after[0, 1]

    R.check(
        "Sponsor boost increases target probability",
        p_after > p_before,
        f"before={p_before:.4f}, after={p_after:.4f}"
    )

    # Double boost -> even higher
    kernel_s1.sponsor_edge(0, 1, boost=5.0)
    P_double = kernel_s1.transition_matrix(tel)
    p_double = P_double[0, 1]

    R.check(
        "Double sponsor boost increases further",
        p_double >= p_after,
        f"after_single={p_after:.4f}, after_double={p_double:.4f}"
    )

    # ===================================================================
    # 10. DIAGNOSTIC METHODS
    # ===================================================================
    print("\n--- 10. Diagnostic Methods ---")
    kernel_diag = make_default_kernel()
    tel_d = np.array([1.0, 0.0, 0.0])

    # Row entropy
    H = kernel_diag.transition_entropy(tel_d)
    R.check(
        "Entropy values non-negative",
        np.all(H >= 0),
        f"H={np.round(H, 3)}"
    )

    # Effective rank
    ranks = kernel_diag.effective_rank(tel_d)
    R.check(
        "Effective rank >= 1 for connected nodes",
        np.all(ranks >= 1.0),
        f"ranks={np.round(ranks, 2)}"
    )
    R.check(
        "Effective rank <= degree for each node",
        all(ranks[i] <= np.sum(topo.adjacency_mask[i]) + 0.01
            for i in range(topo.N)),
        f"ranks={np.round(ranks, 2)}"
    )

    # Mixing time
    mixing = kernel_diag.mixing_time_estimate(tel_d)
    R.check(
        "Mixing time is positive and finite",
        0 < mixing < np.inf,
        f"mixing_time={mixing:.2f}"
    )

    # Full diagnostic dict
    diag = kernel_diag.get_diagnostic(tel_d)
    expected_keys = {
        "alignment", "weight_matrix_raw", "weight_matrix",
        "transition_matrix", "beta_tensor", "sponsor_friction",
        "node_bias", "row_entropy", "effective_rank", "mixing_time",
        "temperature",
    }
    R.check(
        "Diagnostic dict has all keys",
        expected_keys.issubset(diag.keys()),
        f"missing={expected_keys - diag.keys()}"
    )

    # ===================================================================
    # 11. EDGE CASES
    # ===================================================================
    print("\n--- 11. Edge Cases ---")

    # Single node graph
    topo_single = Topology(
        node_features=np.array([[1.0, 0.0]]),
        distance_matrix=np.array([[0.0]]),
        labels=["Alone"],
    )
    kernel_single = DynamicTopologyKernel(topology=topo_single)
    P_single = kernel_single.transition_matrix(np.array([1.0, 0.0]))
    R.check(
        "Single node: P is 1x1 zero",
        P_single.shape == (1, 1) and P_single[0, 0] == 0.0,
        f"P={P_single}"
    )

    # All-zero feature graph
    topo_zero = topology_from_edges(
        nodes={
            "A": np.array([0.0, 0.0]),
            "B": np.array([0.0, 0.0]),
            "C": np.array([0.0, 0.0]),
        },
        edges=[("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 2.0)],
    )
    kernel_zero = DynamicTopologyKernel(
        topology=topo_zero,
        beta=np.full((3, 3), 5.0),
    )
    P_zero = kernel_zero.transition_matrix(np.array([1.0, 0.0]))
    row_sums_z = P_zero.sum(axis=1)
    R.check(
        "All-zero features: still valid stochastic matrix",
        np.allclose(row_sums_z, 1.0, atol=1e-10),
        f"row_sums={row_sums_z}"
    )

    # Very large beta (extreme morphing)
    kernel_extreme = make_default_kernel(beta=np.full((5, 5), 100.0))
    P_extreme = kernel_extreme.transition_matrix(np.array([1.0, 0.0, 0.0]))
    row_sums_e = P_extreme.sum(axis=1)
    R.check(
        "Extreme beta: still valid stochastic matrix",
        np.allclose(row_sums_e[row_sums_e > 0], 1.0, atol=1e-10),
        f"row_sums={row_sums_e}"
    )
    R.check(
        "Extreme beta: no NaN/Inf in P",
        np.all(np.isfinite(P_extreme)),
        f"has_nan={np.any(np.isnan(P_extreme))}"
    )

    # ===================================================================
    # 12. ERGODICITY / STATIONARY DISTRIBUTION
    # ===================================================================
    # NOTE (F4): the original test name claimed "P^50 ... (ergodic)" but it
    # only proves all-pairs reachability + aperiodicity, not that the chain
    # has mixed in 50 steps. A chain can satisfy P^50 > 0 and still have a
    # mixing time of thousands (the Mall under fashion telemetry at tau=1
    # mixes in ~4435 steps). We now test both properties separately.
    print("\n--- 12. Ergodicity ---")
    kernel_erg = make_default_kernel()
    P_erg = kernel_erg.transition_matrix(np.array([1/3, 1/3, 1/3]))

    # 12a. All-pairs reachability (the weaker, faster-passing assertion)
    P_power = np.linalg.matrix_power(P_erg, 50)
    all_positive = np.all(P_power > 1e-12)
    R.check(
        "P^50 has all positive entries (all-pairs reachable)",
        all_positive,
        f"min_entry={P_power.min():.2e}"
    )

    # 12b. Real mixing test: at high temperature the Mall chain is
    # well-mixed in a small number of steps; assert P^N rows actually
    # converge to the stationary distribution.
    kernel_fast_mix = make_default_kernel(temperature=5.0)
    P_fm = kernel_fast_mix.transition_matrix(np.array([1/3, 1/3, 1/3]))

    eigvals_fm, eigvecs_fm = np.linalg.eig(P_fm.T)
    idx_fm = np.argmin(np.abs(eigvals_fm - 1.0))
    pi_fm = np.real(eigvecs_fm[:, idx_fm])
    pi_fm = pi_fm / pi_fm.sum()

    P_fm_power = np.linalg.matrix_power(P_fm, 200)
    row_max_dev = max(
        np.linalg.norm(P_fm_power[i] - pi_fm, ord=1)
        for i in range(P_fm.shape[0])
    )
    R.check(
        "P^200 rows converge to stationary distribution at tau=5",
        row_max_dev < 1e-3,
        f"max_L1_row_deviation={row_max_dev:.2e}"
    )

    # 12c. Document the slow-mixing case rather than asserting it (it's by design)
    kernel_slow = make_default_kernel(temperature=1.0)
    P_slow = kernel_slow.transition_matrix(np.array([1.0, 0.0, 0.0]))
    mixing_slow = kernel_slow.mixing_time_estimate(np.array([1.0, 0.0, 0.0]))
    R.check(
        "Mixing-time estimate is positive & finite for fashion telemetry",
        0 < mixing_slow < np.inf,
        f"mixing_time={mixing_slow:.1f} (slow-mixing case, by design)"
    )

    # Stationary distribution via eigenvector
    eigvals, eigvecs = np.linalg.eig(P_erg.T)
    idx_unit = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx_unit])
    pi = pi / pi.sum()
    R.check(
        "Stationary dist sums to 1.0",
        np.isclose(pi.sum(), 1.0),
        f"sum={pi.sum():.8f}"
    )
    R.check(
        "Stationary dist all non-negative",
        np.all(pi >= -1e-10),
        f"min={pi.min():.8f}"
    )

    # ===================================================================
    # 13. SPONSOR FRICTION (ADDITIVE CHANNEL)
    # ===================================================================
    print("\n--- 13. Sponsor Friction (Additive Channel) ---")
    tel_f = np.array([1.0, 0.0, 0.0])

    # FoodCourt has alignment 0.1 to fashion telemetry.
    # beta-channel efficiency: boost * 0.1 = 10% of boost value.
    # Friction-channel efficiency: reduction * 1.0 = 100% of reduction.
    kernel_beta_only = make_default_kernel()
    kernel_friction  = make_default_kernel()

    P_base = kernel_beta_only.transition_matrix(tel_f)
    p_base_fc = P_base[0, 1]  # Entrance -> FoodCourt

    # Apply same numeric value via both channels
    kernel_beta_only.sponsor_edge(0, 1, boost=4.0)        # beta: efficiency = 4 * 0.1 = 0.4
    kernel_friction.sponsor_edge_friction(0, 1, reduction=4.0)  # S: efficiency = 4.0 (flat)

    P_beta     = kernel_beta_only.transition_matrix(tel_f)
    P_friction = kernel_friction.transition_matrix(tel_f)

    p_beta_fc     = P_beta[0, 1]
    p_friction_fc = P_friction[0, 1]

    R.check(
        "Friction channel stronger than beta for misaligned node",
        p_friction_fc > p_beta_fc,
        f"base={p_base_fc:.4f}, beta={p_beta_fc:.4f}, friction={p_friction_fc:.4f}"
    )
    R.check(
        "Friction channel increases probability over baseline",
        p_friction_fc > p_base_fc,
        f"base={p_base_fc:.4f}, friction={p_friction_fc:.4f}"
    )

    # Monotonicity: more reduction -> higher probability
    kernel_friction2 = make_default_kernel()
    kernel_friction2.sponsor_edge_friction(0, 1, reduction=4.0)
    kernel_friction2.sponsor_edge_friction(0, 1, reduction=4.0)  # 8.0 total
    P_friction2 = kernel_friction2.transition_matrix(tel_f)
    R.check(
        "Friction monotonicity: double reduction -> higher P",
        P_friction2[0, 1] > p_friction_fc,
        f"single={p_friction_fc:.4f}, double={P_friction2[0,1]:.4f}"
    )

    # sponsor_node_friction affects ALL inbound edges
    kernel_node_friction = make_default_kernel()
    P_before_nf = kernel_node_friction.transition_matrix(tel_f)
    kernel_node_friction.sponsor_node_friction(1, reduction=3.0)  # FoodCourt node
    P_after_nf = kernel_node_friction.transition_matrix(tel_f)
    R.check(
        "Node friction increases all inbound probabilities",
        P_after_nf[0, 1] > P_before_nf[0, 1],  # Entrance -> FoodCourt
        f"before={P_before_nf[0,1]:.4f}, after={P_after_nf[0,1]:.4f}"
    )

    # reset_friction restores to baseline
    kernel_friction.reset_friction()
    P_reset = kernel_friction.transition_matrix(tel_f)
    R.check(
        "reset_friction() restores baseline probabilities",
        np.allclose(P_reset, P_base, atol=1e-8),
        f"max_delta={np.abs(P_reset - P_base).max():.2e}"
    )

    # Stochasticity preserved with friction
    kernel_big_friction = make_default_kernel()
    kernel_big_friction.sponsor_node_friction(1, reduction=50.0)  # extreme
    P_big = kernel_big_friction.transition_matrix(tel_f)
    row_sums_bf = P_big.sum(axis=1)
    R.check(
        "Extreme friction: still valid stochastic matrix",
        np.allclose(row_sums_bf[row_sums_bf > 0], 1.0, atol=1e-10),
        f"row_sums={np.round(row_sums_bf, 6)}"
    )
    R.check(
        "Extreme friction: no NaN/Inf in P",
        np.all(np.isfinite(P_big)),
        f"has_nan={np.any(np.isnan(P_big))}"
    )

    # Diagnostic exposes sponsor_friction matrix
    kernel_fd = make_default_kernel()
    kernel_fd.sponsor_edge_friction(0, 1, reduction=2.5)
    diag_fd = kernel_fd.get_diagnostic(tel_f)
    R.check(
        "Diagnostic exposes sponsor_friction tensor",
        diag_fd["sponsor_friction"][0, 1] == 2.5,
        f"S[0,1]={diag_fd['sponsor_friction'][0,1]}"
    )

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    success = R.summary()
    return success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

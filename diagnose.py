"""
Comprehensive diagnostic analysis of the Dynamic Topology Kernel.
Writes structured output to diagnose_output.txt in compact format.
"""
import numpy as np
import sys, io, contextlib
from kernel import DynamicTopologyKernel, AgentState, Topology, topology_from_edges

def run_diagnostics():
    out = []
    def p(s=""):
        out.append(str(s))

    # ---------------------------------------------------------------
    # 1. Build topology
    # ---------------------------------------------------------------
    topo = topology_from_edges(
        nodes={
            "Entrance":     np.array([0.1, 0.1, 0.1]),  # weakly attracts all intents
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
    labels = topo.labels
    N = topo.N

    p("=" * 60)
    p("SECTION 1: TOPOLOGY STRUCTURE")
    p("=" * 60)
    p(f"Nodes: {labels}")
    p(f"N={N}, F={topo.F}")
    p(f"\nAdjacency mask:")
    p(topo.adjacency_mask.astype(int))
    p(f"\nDistance matrix:")
    for i in range(N):
        row = " ".join(f"{topo.distance_matrix[i,j]:6.1f}" for j in range(N))
        p(f"  {labels[i]:12s} | {row}")
    p(f"\nNode features:")
    for i in range(N):
        p(f"  {labels[i]:12s} | {topo.node_features[i]}")

    # ---------------------------------------------------------------
    # 2. Kernel instantiation
    # ---------------------------------------------------------------
    kernel = DynamicTopologyKernel(
        topology=topo, alpha=1.0,
        beta=np.full((5, 5), 5.0),
        feedback_rate=0.15,
    )

    p(f"\n{'='*60}")
    p("SECTION 2: TRANSITION MATRIX ANALYSIS")
    p("=" * 60)

    test_telemetries = {
        "fashion":  np.array([1.0, 0.0, 0.0]),
        "food":     np.array([0.0, 1.0, 0.0]),
        "tech":     np.array([0.0, 0.0, 1.0]),
        "neutral":  np.array([1/3, 1/3, 1/3]),
        "zero":     np.array([0.0, 0.0, 0.0]),
    }

    for name, tel in test_telemetries.items():
        diag = kernel.get_diagnostic(tel)
        P = diag["transition_matrix"]
        alignment = diag["alignment"]
        W = diag["weight_matrix"]

        p(f"\n--- Telemetry: {name} = {tel} ---")
        p(f"  Alignment: {np.round(alignment, 4)}")

        p(f"  W (dynamic weights):")
        for i in range(N):
            row = " ".join(f"{W[i,j]:7.2f}" for j in range(N))
            p(f"    {labels[i]:12s} | {row}")

        p(f"  P (transition probs):")
        for i in range(N):
            row = " ".join(f"{P[i,j]:7.4f}" for j in range(N))
            p(f"    {labels[i]:12s} | {row}")

        row_sums = P.sum(axis=1)
        p(f"  Row sums: {np.round(row_sums, 8)}")

        # Check irreducibility
        P_power = np.linalg.matrix_power(P, 50)
        p(f"  P^50 row 0: {np.round(P_power[0], 6)}")

    # ---------------------------------------------------------------
    # 3. CRITICAL BUG CHECKS
    # ---------------------------------------------------------------
    p(f"\n{'='*60}")
    p("SECTION 3: BUG AND EDGE-CASE ANALYSIS")
    p("=" * 60)

    # 3a. Entrance row — the Entrance has feature [0,0,0].
    # alignment = N_j . a_t — but Entrance features are zero,
    # so Entrance alignment is always 0.
    # This means the cost to reach Entrance is purely alpha*D 
    # (never reduced by morphing). Is that intended?
    p("\n[RESOLVED-1] Entrance node features fixed to [0.1, 0.1, 0.1].")
    p("  Entrance now weakly attracts all agent archetypes via beta.")
    p("  Alignment is no longer always-zero; return routing is meaningful.")
    p(f"  alignment_Entrance(fashion) = {np.dot(np.array([0.1,0.1,0.1]), np.array([1,0,0])):.3f}")

    # 3b. Check if P is a valid stochastic matrix for all telemetries
    p("\n[CHECK-2] Row-stochasticity validation:")
    all_ok = True
    for name, tel in test_telemetries.items():
        P = kernel.transition_matrix(tel)
        rs = P.sum(axis=1)
        for i in range(N):
            if abs(rs[i] - 1.0) > 1e-10 and rs[i] > 0:
                p(f"  FAIL: {name} row {i} sums to {rs[i]}")
                all_ok = False
    if all_ok:
        p("  PASS: All rows sum to ~1.0 or 0.0 (isolated).")

    # 3c. Negative weight floor — does the 0.1 floor create
    # probability concentration artifacts?
    p("\n[CHECK-3] Weight floor analysis:")
    tel_fashion = np.array([1.0, 0.0, 0.0])
    diag = kernel.get_diagnostic(tel_fashion)
    W = diag["weight_matrix"]
    raw_W = (kernel.alpha * topo.distance_matrix) - (kernel._beta * (topo.node_features @ tel_fashion)[np.newaxis, :])
    floored_count = np.sum((raw_W < 0.1) & topo.adjacency_mask)
    p(f"  Edges where W was clipped to 0.1 floor: {floored_count}")
    floored_edges = np.argwhere((raw_W < 0.1) & topo.adjacency_mask)
    for idx in floored_edges:
        i, j = idx
        p(f"    {labels[i]} -> {labels[j]}: raw={raw_W[i,j]:.3f}, floored to 0.1")

    # 3d. Ergodicity check — can every node reach every other node?
    p("\n[CHECK-4] Graph connectivity (reachability):")
    adj = topo.adjacency_mask.astype(float)
    reachable = np.linalg.matrix_power(adj + np.eye(N), N) > 0
    if reachable.all():
        p("  Graph is strongly connected (ergodic Markov chain).")
    else:
        p("  WARNING: Graph is NOT strongly connected!")
        for i in range(N):
            for j in range(N):
                if not reachable[i, j]:
                    p(f"    {labels[i]} cannot reach {labels[j]}")

    # 3e. Feedback drift analysis
    p("\n[CHECK-5] Feedback drift analysis:")
    p("  Running 100-step simulation, tracking telemetry norm and direction.")
    np.random.seed(42)
    agent = AgentState(telemetry=np.array([1.0, 0.0, 0.0]), position=0)
    telemetry_log = [agent.telemetry.copy()]
    for _ in range(100):
        kernel.step(agent)
        telemetry_log.append(agent.telemetry.copy())
    tlog = np.array(telemetry_log)
    p(f"  Telemetry at t=0:   {tlog[0]}")
    p(f"  Telemetry at t=10:  {np.round(tlog[10], 4)}")
    p(f"  Telemetry at t=50:  {np.round(tlog[50], 4)}")
    p(f"  Telemetry at t=100: {np.round(tlog[100], 4)}")
    norms = np.linalg.norm(tlog, axis=1)
    p(f"  Norm range: [{norms.min():.6f}, {norms.max():.6f}]")
    p(f"  All norms ~1.0: {np.allclose(norms, 1.0, atol=1e-6)}")

    # 3f. Stationary distribution
    p("\n[CHECK-6] Stationary distribution (fashion telemetry, fixed):")
    P_fashion = kernel.transition_matrix(np.array([1.0, 0.0, 0.0]))
    eigenvalues, eigvecs = np.linalg.eig(P_fashion.T)
    idx_stat = np.argmin(np.abs(eigenvalues - 1.0))
    stat = np.real(eigvecs[:, idx_stat])
    stat = stat / stat.sum()
    for i in range(N):
        p(f"  {labels[i]:12s}: {stat[i]:.6f}")

    # ---------------------------------------------------------------
    # 4. BATCH PERFORMANCE
    # ---------------------------------------------------------------
    p(f"\n{'='*60}")
    p("SECTION 4: BATCH SIMULATION (1000 agents)")
    p("=" * 60)

    K = 1000
    rng = np.random.default_rng(42)
    raw_tel = rng.dirichlet(np.ones(3), size=K)
    starts = np.zeros(K, dtype=int)

    import time
    t0 = time.perf_counter()
    all_paths = kernel.simulate_batch(raw_tel, starts, steps=5)
    t1 = time.perf_counter()

    final_nodes = all_paths[:, -1]
    p(f"\n  Time for 1000 agents x 5 steps: {t1-t0:.3f}s")
    p(f"  Final destination distribution:")
    for i, label in enumerate(labels):
        count = np.sum(final_nodes == i)
        p(f"    {label:12s}: {count:4d} ({count/K*100:5.1f}%)")

    # ---------------------------------------------------------------
    # 5. SPONSOR EFFECT ANALYSIS
    # ---------------------------------------------------------------
    p(f"\n{'='*60}")
    p("SECTION 5: SPONSOR EFFECT ANALYSIS")
    p("=" * 60)

    # Reset kernel
    kernel2 = DynamicTopologyKernel(
        topology=topo, alpha=1.0,
        beta=np.full((5, 5), 5.0),
        feedback_rate=0.15,
    )

    P_before = kernel2.transition_matrix(np.array([1.0, 0.0, 0.0]))
    p(f"\n  Before sponsorship:")
    p(f"    Entrance -> FoodCourt: {P_before[0,1]:.4f}")
    p(f"    Entrance -> TechStore: {P_before[0,2]:.4f}")
    p(f"    Entrance -> OutletA:   {P_before[0,3]:.4f}")

    kernel2.sponsor_edge(0, 1, boost=8.0)
    P_after = kernel2.transition_matrix(np.array([1.0, 0.0, 0.0]))
    p(f"  After sponsoring Entrance->FoodCourt (+8.0):")
    p(f"    Entrance -> FoodCourt: {P_after[0,1]:.4f}")
    p(f"    Entrance -> TechStore: {P_after[0,2]:.4f}")
    p(f"    Entrance -> OutletA:   {P_after[0,3]:.4f}")

    # Note: FoodCourt features = [0.1, 0.9, 0.0]
    # Fashion telemetry = [1.0, 0.0, 0.0]
    # alignment_FoodCourt = 0.1
    # The boost makes beta for edge (0,1) = 13.0
    # So W(0,1) = 1*5 - 13*0.1 = 5 - 1.3 = 3.7
    # Without boost: W(0,1) = 5 - 5*0.1 = 4.5
    p(f"\n  Sponsor mechanism analysis:")
    p(f"    FoodCourt alignment to fashion: 0.1")
    p(f"    Without sponsor: W = 1*5 - 5*0.1 = 4.5")
    p(f"    With sponsor (+8): W = 1*5 - 13*0.1 = 3.7")
    p(f"    Sponsor effect is WEAK for misaligned nodes!")
    p(f"    sponsor_boost * alignment = {8.0 * 0.1:.1f} effective W reduction")

    # ---------------------------------------------------------------
    # 6. simulate_batch LOOP BUG CHECK
    # ---------------------------------------------------------------
    p(f"\n{'='*60}")
    p("SECTION 6: BATCH vs SEQUENTIAL CONSISTENCY")
    p("=" * 60)

    # The batch method has a Python loop - does it produce same 
    # results as sequential simulation?
    kernel3 = DynamicTopologyKernel(
        topology=topo, alpha=1.0,
        beta=np.full((5, 5), 5.0),
        feedback_rate=0.15,
    )
    np.random.seed(99)
    tel = np.array([0.5, 0.3, 0.2])
    agent_seq = AgentState(telemetry=tel.copy(), position=0)
    kernel3.simulate(agent_seq, steps=5)
    seq_path = agent_seq.history

    np.random.seed(99)
    batch_tel = tel.copy().reshape(1, -1)
    batch_start = np.array([0])
    batch_paths = kernel3.simulate_batch(batch_tel, batch_start, steps=5)
    batch_path = batch_paths[0].tolist()

    p(f"  Sequential path: {seq_path}")
    p(f"  Batch path:      {batch_path}")
    p(f"  Match: {seq_path == batch_path}")

    # Write output
    return "\n".join(out)


if __name__ == "__main__":
    result = run_diagnostics()
    with open("diagnose_output.txt", "w", encoding="ascii", errors="replace") as f:
        f.write(result)
    print("Diagnostics written to diagnose_output.txt")

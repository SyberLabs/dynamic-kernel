"""
Comparative diagnostic: old kernel behavior vs. improved kernel.
Shows how each fix addresses the issues identified in the analysis.
"""
import numpy as np
import time
from kernel import DynamicTopologyKernel, AgentState, topology_from_edges

def run_comparison():
    out = []
    def p(s=""):
        out.append(str(s))

    topo = topology_from_edges(
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
    labels = topo.labels
    N = topo.N
    tel_fashion = np.array([1.0, 0.0, 0.0])

    # ===================================================================
    p("=" * 60)
    p("COMPARISON 1: SOFTPLUS FLOOR vs HARD CLIP")
    p("=" * 60)

    # "Old" behavior: hard clip (simulated by setting sharpness very high)
    kernel_hard = DynamicTopologyKernel(
        topology=topo, alpha=1.0,
        beta=np.full((N, N), 5.0),
        feedback_rate=0.15, feedback_noise=0.0,
        floor_sharpness=1000.0,  # effectively hard clip
    )
    kernel_soft = DynamicTopologyKernel(
        topology=topo, alpha=1.0,
        beta=np.full((N, N), 5.0),
        feedback_rate=0.15, feedback_noise=0.0,
        floor_sharpness=5.0,  # smooth
    )

    diag_hard = kernel_hard.get_diagnostic(tel_fashion)
    diag_soft = kernel_soft.get_diagnostic(tel_fashion)

    p("\n  Weight matrix (hard floor, sharpness=1000):")
    for i in range(N):
        row = " ".join(f"{diag_hard['weight_matrix'][i,j]:7.3f}" for j in range(N))
        p(f"    {labels[i]:12s} | {row}")

    p("\n  Weight matrix (soft floor, sharpness=5):")
    for i in range(N):
        row = " ".join(f"{diag_soft['weight_matrix'][i,j]:7.3f}" for j in range(N))
        p(f"    {labels[i]:12s} | {row}")

    p(f"\n  OutletA->OutletB (raw=-1.500):")
    p(f"    Hard: W={diag_hard['weight_matrix'][3,4]:.4f}")
    p(f"    Soft: W={diag_soft['weight_matrix'][3,4]:.4f}")
    p(f"  OutletB->OutletA (raw=-1.000):")
    p(f"    Hard: W={diag_hard['weight_matrix'][4,3]:.4f}")
    p(f"    Soft: W={diag_soft['weight_matrix'][4,3]:.4f}")
    p(f"\n  Hard floor collapses both to 0.1000 (IDENTICAL)")
    p(f"  Soft floor preserves discrimination")

    # Row entropy comparison
    p(f"\n  Row entropy (bits):")
    p(f"    {'Node':12s} | {'Hard':>8s} | {'Soft':>8s} | {'Delta':>8s}")
    for i in range(N):
        h_hard = diag_hard['row_entropy'][i]
        h_soft = diag_soft['row_entropy'][i]
        p(f"    {labels[i]:12s} | {h_hard:8.4f} | {h_soft:8.4f} | {h_soft-h_hard:+8.4f}")

    # ===================================================================
    p(f"\n{'='*60}")
    p("COMPARISON 2: TELEMETRY LOCK-IN vs EXPLORATION")
    p("=" * 60)

    # No noise
    np.random.seed(42)
    kernel_locked = DynamicTopologyKernel(
        topology=topo, alpha=1.0,
        beta=np.full((N, N), 5.0),
        feedback_rate=0.15, feedback_noise=0.0,
    )
    agent_l = AgentState(telemetry=tel_fashion.copy(), position=0)
    tel_log_locked = [agent_l.telemetry.copy()]
    for _ in range(100):
        kernel_locked.step(agent_l)
        tel_log_locked.append(agent_l.telemetry.copy())

    # With noise
    np.random.seed(42)
    kernel_explore = DynamicTopologyKernel(
        topology=topo, alpha=1.0,
        beta=np.full((N, N), 5.0),
        feedback_rate=0.15, feedback_noise=0.05,
    )
    agent_e = AgentState(telemetry=tel_fashion.copy(), position=0)
    tel_log_explore = [agent_e.telemetry.copy()]
    for _ in range(100):
        kernel_explore.step(agent_e)
        tel_log_explore.append(agent_e.telemetry.copy())

    tl = np.array(tel_log_locked)
    te = np.array(tel_log_explore)

    p(f"\n  Telemetry variance over last 50 steps:")
    late_var_l = np.var(tl[50:], axis=0).sum()
    late_var_e = np.var(te[50:], axis=0).sum()
    p(f"    No noise:   {late_var_l:.8f}")
    p(f"    With noise:  {late_var_e:.8f}")
    p(f"    Ratio:       {late_var_e/max(late_var_l, 1e-20):.1f}x more exploration")

    p(f"\n  Unique nodes visited (100 steps):")
    path_l = agent_l.history
    path_e = agent_e.history
    p(f"    No noise:   {len(set(path_l))} nodes: {set(labels[i] for i in path_l)}")
    p(f"    With noise:  {len(set(path_e))} nodes: {set(labels[i] for i in path_e)}")

    # ===================================================================
    p(f"\n{'='*60}")
    p("COMPARISON 3: NODE BIAS (ZERO-FEATURE FIX)")
    p("=" * 60)

    kernel_nobias = DynamicTopologyKernel(
        topology=topo, alpha=1.0,
        beta=np.full((N, N), 5.0),
        feedback_rate=0.15, feedback_noise=0.0,
    )
    kernel_bias = DynamicTopologyKernel(
        topology=topo, alpha=1.0,
        beta=np.full((N, N), 5.0),
        feedback_rate=0.15, feedback_noise=0.0,
        node_bias=np.array([0.3, 0.0, 0.0, 0.0, 0.0]),
    )

    # Stationary distribution comparison
    P_nb = kernel_nobias.transition_matrix(tel_fashion)
    P_b = kernel_bias.transition_matrix(tel_fashion)

    eigvals_nb, eigvecs_nb = np.linalg.eig(P_nb.T)
    idx_nb = np.argmin(np.abs(eigvals_nb - 1.0))
    pi_nb = np.real(eigvecs_nb[:, idx_nb])
    pi_nb = pi_nb / pi_nb.sum()

    eigvals_b, eigvecs_b = np.linalg.eig(P_b.T)
    idx_b = np.argmin(np.abs(eigvals_b - 1.0))
    pi_b = np.real(eigvecs_b[:, idx_b])
    pi_b = pi_b / pi_b.sum()

    p(f"\n  Stationary distribution (fashion telemetry):")
    p(f"    {'Node':12s} | {'No bias':>10s} | {'With bias':>10s} | {'Delta':>10s}")
    for i in range(N):
        p(f"    {labels[i]:12s} | {pi_nb[i]:10.6f} | {pi_b[i]:10.6f} | {pi_b[i]-pi_nb[i]:+10.6f}")

    # NOTE: previously printed pi_b[i] here, which used the leaked loop variable
    # from the for-loop just above (i == N-1), reporting OutletB's share instead
    # of Entrance's. Indexing both sides with [0] explicitly.
    p(f"\n  Entrance share: {pi_nb[0]*100:.3f}% -> {pi_b[0]*100:.3f}% ({pi_b[0]/max(pi_nb[0],1e-20):.1f}x)")

    # ===================================================================
    p(f"\n{'='*60}")
    p("COMPARISON 4: BATCH PERFORMANCE")
    p("=" * 60)

    kernel_perf = DynamicTopologyKernel(
        topology=topo, alpha=1.0,
        beta=np.full((N, N), 5.0),
        feedback_rate=0.15, feedback_noise=0.0,
    )

    K = 5000
    rng = np.random.default_rng(42)
    raw_tel = rng.dirichlet(np.ones(3), size=K)
    starts = np.zeros(K, dtype=int)

    t0 = time.perf_counter()
    kernel_perf.simulate_batch(raw_tel, starts, steps=5)
    t1 = time.perf_counter()

    p(f"\n  {K} agents x 5 steps: {t1-t0:.3f}s")
    p(f"  Per agent-step: {(t1-t0)/(K*5)*1e6:.0f} us")

    # ===================================================================
    p(f"\n{'='*60}")
    p("COMPARISON 5: TEMPERATURE EFFECT")
    p("=" * 60)

    for tau in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        k = DynamicTopologyKernel(
            topology=topo, alpha=1.0,
            beta=np.full((N, N), 5.0),
            feedback_rate=0.15, feedback_noise=0.0,
            temperature=tau,
        )
        H = k.transition_entropy(tel_fashion)
        eff_r = k.effective_rank(tel_fashion)
        mix = k.mixing_time_estimate(tel_fashion)
        p(f"  tau={tau:5.1f} | mean_entropy={H.mean():.3f} bits | "
          f"mean_eff_rank={eff_r.mean():.2f} | mixing_time={mix:.1f}")

    return "\n".join(out)


if __name__ == "__main__":
    result = run_comparison()
    with open("comparison_output.txt", "w", encoding="ascii", errors="replace") as f:
        f.write(result)
    print(result)

"""
Run the kernel and capture output to a plain ASCII file.
"""
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Redirect print to a file
import contextlib

with open("test_output.txt", "w", encoding="utf-8") as f:
    with contextlib.redirect_stdout(f):
        import numpy as np
        from kernel import (
            DynamicTopologyKernel, AgentState, topology_from_edges
        )

        # --- Build the mall topology ---
        topo = topology_from_edges(
            nodes={
                "Entrance":          np.array([0.1, 0.1, 0.1]),

                "Food Court":        np.array([0.1, 0.9, 0.0]),
                "Tech Store":        np.array([0.0, 0.1, 0.9]),
                "Premium Outlet A":  np.array([0.8, 0.0, 0.1]),
                "Premium Outlet B":  np.array([0.9, 0.1, 0.0]),
            },
            edges=[
                ("Entrance", "Food Court", 5.0),
                ("Entrance", "Tech Store", 4.0),
                ("Entrance", "Premium Outlet A", 10.0),
                ("Food Court", "Premium Outlet B", 6.0),
                ("Premium Outlet A", "Premium Outlet B", 3.0),
                ("Tech Store", "Premium Outlet A", 7.0),
            ],
        )

        kernel = DynamicTopologyKernel(
            topology=topo,
            alpha=1.0,
            beta=np.full((5, 5), 5.0),
            feedback_rate=0.15,
        )

        np.random.seed(42)

        # --- Scenario 1 ---
        print("=" * 60)
        print("  SCENARIO 1: Fashion-biased agent (no sponsorship)")
        print("=" * 60)

        agent = AgentState(
            telemetry=np.array([1.0, 0.0, 0.0]),
            position=0,
        )
        path = kernel.simulate(agent, steps=4, verbose=True)
        labels = topo.labels
        print(f"  Path: {' -> '.join(labels[i] for i in path)}\n")

        # --- Scenario 2 ---
        print("=" * 60)
        print("  SCENARIO 2: Food Court sponsors Entrance->Food Court edge")
        print("=" * 60)

        kernel.sponsor_edge(0, 1, boost=8.0)

        agent2 = AgentState(
            telemetry=np.array([1.0, 0.0, 0.0]),
            position=0,
        )
        path2 = kernel.simulate(agent2, steps=4, verbose=True)
        print(f"  Path: {' -> '.join(labels[i] for i in path2)}\n")

        # --- Scenario 3: Batch ---
        print("=" * 60)
        print("  SCENARIO 3: Batch -- 1000 agents, distribution analysis")
        print("=" * 60)

        K = 1000
        rng = np.random.default_rng(42)
        raw = rng.dirichlet(np.ones(3), size=K)
        starts = np.zeros(K, dtype=int)

        all_paths = kernel.simulate_batch(raw, starts, steps=5)

        final_nodes = all_paths[:, -1]
        print(f"\n  Final destination distribution ({K} agents, 5 steps):")
        for i, label in enumerate(labels):
            count = np.sum(final_nodes == i)
            print(f"    {label:25s}: {count:4d}  ({count/K*100:5.1f}%)")

        # --- Diagnostic ---
        print(f"\n{'='*60}")
        print("  DIAGNOSTIC: Transition matrix for pure-fashion telemetry")
        print("=" * 60)
        diag = kernel.get_diagnostic(np.array([1.0, 0.0, 0.0]))
        P = diag["transition_matrix"]
        print(f"\n  Alignment vector: {np.round(diag['alignment'], 3)}")
        print(f"\n  Transition matrix P:")
        header = "  " + " " * 22 + "".join(f"{l:>14s}" for l in labels)
        print(header)
        for i, label in enumerate(labels):
            row_str = "".join(f"{P[i,j]:14.4f}" for j in range(len(labels)))
            print(f"  {label:20s} {row_str}")

        # --- Additional diagnostics ---
        print(f"\n{'='*60}")
        print("  STOCHASTICITY CHECK: Row sums of P")
        print("=" * 60)
        row_sums = P.sum(axis=1)
        for i, label in enumerate(labels):
            print(f"    {label:25s}: {row_sums[i]:.8f}")

        print(f"\n  Beta tensor (current):")
        print(diag["beta_tensor"])

        # --- Eigenvalue analysis of transition matrix (stationary dist) ---
        print(f"\n{'='*60}")
        print("  STATIONARY DISTRIBUTION (ergodic check)")
        print("=" * 60)
        # Check if P is irreducible by looking at P^100
        P_clean = diag["transition_matrix"].copy()
        P_100 = np.linalg.matrix_power(P_clean, 100)
        print(f"\n  P^100 (rows should converge if ergodic):")
        for i, label in enumerate(labels):
            row_str = "".join(f"{P_100[i,j]:14.6f}" for j in range(len(labels)))
            print(f"  {label:20s} {row_str}")

        # Left eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(P_clean.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        print(f"\n  Stationary distribution (left eigenvector):")
        for i, label in enumerate(labels):
            print(f"    {label:25s}: {stationary[i]:.6f}")

print("Output written to test_output.txt")

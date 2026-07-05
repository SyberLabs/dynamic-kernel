"""
Neural Optimizer — Empirical Investigation
==========================================
Runs each symmetry mode for N_STEPS optimizer steps on a fresh 12-node
fully-connected network and reports the key theoretical metrics.

No HTTP server needed — drives kernel + optimizer directly.

Run:   py investigate_neural.py
"""

import sys
import numpy as np
from optimizer import SymmetryOptimizer, SymmetryMode
from kernel import DynamicTopologyKernel, topology_from_edges

# ── Configuration ─────────────────────────────────────────────────────────────
N_NEURONS  = 12
F_FEATURES = 4
BETA_INIT  = 3.0
N_STEPS    = 200
ETA        = 0.05
EPS        = 1e-3
NOISE      = 0.0           # clean gradient, no noise, so results are deterministic
TELEMETRY  = np.full(F_FEATURES, 1.0 / F_FEATURES)

MODES = [
    SymmetryMode.ENTROPY_PI,
    SymmetryMode.ROW_ENTROPY,
    SymmetryMode.DETAILED_BALANCE,
    SymmetryMode.SPECTRAL_GAP,
    SymmetryMode.WEIGHT_SYMMETRY,
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_kernel(n: int = N_NEURONS, f: int = F_FEATURES):
    """Fully-connected undirected N-node network with random asymmetric weights."""
    labels = [f"N{i+1}" for i in range(n)]
    nodes  = {label: np.full(f, 1.0 / f) for label in labels}
    edges  = [(labels[i], labels[j], 5.0)
              for i in range(n) for j in range(i+1, n)]
    topo   = topology_from_edges(nodes=nodes, edges=edges, undirected=True)
    
    # Initialize with random, highly asymmetric weights to force optimization
    rng = np.random.default_rng(42) # Fixed seed for reproducibility
    beta = rng.uniform(0.1, 8.0, size=(n, n))
    beta[~topo.adjacency_mask] = 0.0
    
    return DynamicTopologyKernel(
        topology=topo, alpha=1.0, beta=beta,
        feedback_rate=0.0, temperature=1.0, feedback_noise=0.0,
    )


def kl_from_uniform(pi: np.ndarray) -> float:
    """KL(π ‖ uniform).  0 = perfectly uniform.  Higher = more concentrated."""
    n = len(pi)
    uniform = np.full(n, 1.0 / n)
    pi_safe = np.clip(pi, 1e-12, None)
    return float(np.sum(pi_safe * np.log(pi_safe / uniform)))


def detailed_balance_residual(kernel: DynamicTopologyKernel, tel: np.ndarray) -> float:
    """max |π_i P_ij - π_j P_ji| across all edges.  0 = perfectly reversible."""
    p  = kernel.transition_matrix(tel)
    pi = kernel.stationary_distribution(tel)
    flow = pi[:, None] * p
    diff = np.abs(flow - flow.T)
    return float(diff[kernel.topo.adjacency_mask].max())


def weight_asymmetry(beta: np.ndarray) -> float:
    """‖β - βᵀ‖_F.  0 = perfectly symmetric weight matrix."""
    return float(np.linalg.norm(beta - beta.T))


def row_entropy_variance(kernel: DynamicTopologyKernel, tel: np.ndarray) -> float:
    """Variance of per-node row entropy.  0 = all nodes equally uncertain."""
    diag = kernel.get_diagnostic(tel)
    return float(np.var(diag["row_entropy"]))


def spectral_gap(kernel: DynamicTopologyKernel, tel: np.ndarray) -> float:
    """λ₁ - λ₂ of transition matrix.  Larger = faster mixing."""
    p    = kernel.transition_matrix(tel)
    eigs = np.sort(np.abs(np.linalg.eigvals(p)))[::-1]
    return float(eigs[0] - eigs[1]) if len(eigs) >= 2 else 0.0


def sigma_label(mode: SymmetryMode) -> str:
    return {
        SymmetryMode.ENTROPY_PI:       "H(π)        [nats]",
        SymmetryMode.ROW_ENTROPY:      "-Var(H_row) [nats²]",
        SymmetryMode.DETAILED_BALANCE: "-Σflow_gap² ",
        SymmetryMode.SPECTRAL_GAP:     "λ₁-λ₂       ",
        SymmetryMode.WEIGHT_SYMMETRY:  "-‖β-βᵀ‖_F  ",
    }[mode]


def print_section(title: str):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")


def print_metrics(label: str, kernel: DynamicTopologyKernel,
                  tel: np.ndarray, sigma: float, grad_norm: float,
                  tick: int, mode: SymmetryMode):
    pi   = kernel.stationary_distribution(tel)
    kl   = kl_from_uniform(pi)
    db   = detailed_balance_residual(kernel, tel)
    wa   = weight_asymmetry(kernel._beta)
    rev  = row_entropy_variance(kernel, tel)
    sg   = spectral_gap(kernel, tel)
    diag = kernel.get_diagnostic(tel)
    h    = diag["row_entropy"]
    mix  = diag["mixing_time"]
    mix_str = f"{mix:.2f}" if not np.isinf(mix) else "inf"

    print(f"\n  [{label}] tick={tick:4d}  Σ={sigma:+.5f}  ‖∇Σ‖={grad_norm:.2e}")
    print(f"  {'─'*64}")
    print(f"  KL(π ‖ uniform)        = {kl:.6f}   "
          f"{'← perfectly uniform' if kl < 0.001 else '← concentrated' if kl > 0.05 else ''}")
    print(f"  Detailed balance max   = {db:.6f}   "
          f"{'← reversible!' if db < 0.002 else '← significant flow gap'}")
    print(f"  Weight asymmetry ‖β-βᵀ‖= {wa:.4f}    "
          f"{'← symmetric' if wa < 0.1 else ''}")
    print(f"  Var(row_entropy)       = {rev:.6f}   "
          f"{'← equivariant' if rev < 0.0001 else ''}")
    print(f"  Spectral gap λ₁-λ₂    = {sg:.6f}   "
          f"{'← fast mixing' if sg > 0.5 else '← slow mixing'}")
    print(f"  Mixing time (est.)     = {mix_str}")
    print(f"  π min/max              = {pi.min():.4f} / {pi.max():.4f}  "
          f"(range {pi.max()-pi.min():.4f})")
    print(f"  H(row) min/max         = {h.min():.4f} / {h.max():.4f}  "
          f"(var {np.var(h):.6f})")


def run_mode(mode: SymmetryMode) -> dict:
    """Run one mode for N_STEPS and return summary metrics."""
    kernel = build_kernel()
    opt    = SymmetryOptimizer(
        kernel, mode=mode, eta=ETA, eps=EPS,
        noise_sigma=NOISE, telemetry=TELEMETRY,
        converge_eps=1e-5, converge_ticks=30,
    )

    # Baseline snapshot
    snap0 = opt.snapshot(grad_norm=0.0)
    pi0   = np.array(snap0["pi"])

    print_section(f"MODE: {mode.value}  —  Σ = {sigma_label(mode)}")
    print_metrics("INIT  ", kernel, TELEMETRY,
                  snap0["sigma_value"], 0.0, 0, mode)

    sigma_traj = [snap0["sigma_value"]]
    grad_traj  = []
    last_snap  = snap0

    for step in range(N_STEPS):
        snap = opt.step()
        sigma_traj.append(snap["sigma_value"])
        grad_traj.append(snap["grad_norm"])
        last_snap = snap
        if snap["converged"]:
            print(f"\n  ✓ Converged at tick {snap['tick']} "
                  f"(grad_norm < 1e-5 for 30 consecutive steps)")
            break

    print_metrics("FINAL ", kernel, TELEMETRY,
                  last_snap["sigma_value"], last_snap["grad_norm"],
                  last_snap["tick"], mode)

    # Sigma monotonicity check
    diffs = np.diff(sigma_traj)
    pct_increasing = 100.0 * np.sum(diffs > 0) / len(diffs) if len(diffs) else 0
    print(f"\n  Sigma trajectory:  start={sigma_traj[0]:.5f}  "
          f"end={sigma_traj[-1]:.5f}  "
          f"delta={sigma_traj[-1]-sigma_traj[0]:+.5f}")
    print(f"  Monotone ascent:   {pct_increasing:.1f}% of steps "
          f"({'✓ clean gradient' if pct_increasing > 90 else '~ noisy but ascending' if pct_increasing > 60 else '✗ non-monotone'})")

    return {
        "mode":          mode.value,
        "ticks":         last_snap["tick"],
        "converged":     last_snap["converged"],
        "sigma_init":    sigma_traj[0],
        "sigma_final":   sigma_traj[-1],
        "sigma_delta":   sigma_traj[-1] - sigma_traj[0],
        "kl_final":      kl_from_uniform(np.array(last_snap["pi"])),
        "db_final":      detailed_balance_residual(kernel, TELEMETRY),
        "wa_final":      weight_asymmetry(kernel._beta),
        "rev_final":     row_entropy_variance(kernel, TELEMETRY),
        "sg_final":      spectral_gap(kernel, TELEMETRY),
        "mix_final":     last_snap["mixing_time"],
        "pct_mono":      pct_increasing,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "▓"*72)
    print("  DYNAMIC TOPOLOGY ENGINE — Neural Optimizer Empirical Investigation")
    print(f"  Network: {N_NEURONS} neurons, fully connected, uniform features")
    print(f"  Steps per mode: {N_STEPS}  |  η={ETA}  |  ε={EPS}  |  noise={NOISE}")
    print("▓"*72)

    results = []
    for mode in MODES:
        r = run_mode(mode)
        results.append(r)

    # ── Comparative Summary Table ─────────────────────────────────────────────
    print_section("COMPARATIVE SUMMARY")
    header = (
        f"  {'Mode':<20} {'ΔΣ':>9} {'KL(π,u)':>9} {'DetBal':>9} "
        f"{'WtAsym':>9} {'RowVar':>9} {'SpGap':>9} {'Mix':>7} {'Conv':>5}"
    )
    print(header)
    print("  " + "─"*74)
    for r in results:
        mix_s = f"{r['mix_final']:.2f}" if r['mix_final'] >= 0 else "inf"
        conv  = "✓" if r["converged"] else f"~{r['ticks']}"
        print(
            f"  {r['mode']:<20} "
            f"{r['sigma_delta']:>+9.4f} "
            f"{r['kl_final']:>9.5f} "
            f"{r['db_final']:>9.5f} "
            f"{r['wa_final']:>9.4f} "
            f"{r['rev_final']:>9.5f} "
            f"{r['sg_final']:>9.5f} "
            f"{mix_s:>7} "
            f"{conv:>5}"
        )

    # ── Theoretical Interpretation ────────────────────────────────────────────
    print_section("THEORETICAL INTERPRETATION")

    best_kl   = min(results, key=lambda r: r["kl_final"])
    best_db   = min(results, key=lambda r: r["db_final"])
    best_wa   = min(results, key=lambda r: r["wa_final"])
    best_rev  = min(results, key=lambda r: r["rev_final"])
    best_sg   = max(results, key=lambda r: r["sg_final"])

    print(f"""
  Q: Which mode drives π closest to uniform?
  A: {best_kl['mode']}  (KL={best_kl['kl_final']:.5f})
     {'→ ENTROPY_PI is the direct optimizer of this property.' 
      if best_kl['mode'] == 'ENTROPY_PI' 
      else '→ Unexpected — another mode incidentally equalises π.'}

  Q: Which mode achieves thermodynamic reversibility (detailed balance)?
  A: {best_db['mode']}  (residual={best_db['db_final']:.5f})
     {'→ DETAILED_BALANCE correctly minimises flow asymmetry.' 
      if best_db['mode'] == 'DETAILED_BALANCE'
      else '→ Interesting — a different mode incidentally achieves reversibility.'}

  Q: Which mode equalises per-node routing uncertainty?
  A: {best_rev['mode']}  (Var(H_row)={best_rev['rev_final']:.5f})
     {'→ ROW_ENTROPY is the correct optimizer.' 
      if best_rev['mode'] == 'ROW_ENTROPY'
      else '→ Unexpected — check if modes are cross-entangled.'}

  Q: Which mode achieves fastest algebraic mixing?
  A: {best_sg['mode']}  (spectral gap={best_sg['sg_final']:.5f})
     {'→ SPECTRAL_GAP correctly widens the gap.' 
      if best_sg['mode'] == 'SPECTRAL_GAP'
      else '→ Another mode incidentally achieves faster mixing.'}

  Q: Which mode makes the weight matrix most symmetric?
  A: {best_wa['mode']}  (‖β-βᵀ‖={best_wa['wa_final']:.4f})
     {'→ WEIGHT_SYMMETRY correctly collapses asymmetry.' 
      if best_wa['mode'] == 'WEIGHT_SYMMETRY'
      else '→ Note: another mode is incidentally more symmetric.'}

  CROSS-COUPLING OBSERVATION:
  On a fully-connected undirected symmetric network, several of these modes
  are expected to converge to the same attractor (uniform π + detailed balance).
  The interesting divergences are:
   • SPECTRAL_GAP may concentrate weights on specific paths (fast-mixing != uniform)
   • WEIGHT_SYMMETRY operates on β directly, not on π — may diverge from DB
   • ROW_ENTROPY equalises H_i but not necessarily π (a node can have high entropy
     but low stationary mass — the two are distinct properties)
    """)

    print("  Investigation complete. Run with different N_NEURONS or NOISE to")
    print("  observe phase transitions and attractor geometry.\n")


if __name__ == "__main__":
    main()

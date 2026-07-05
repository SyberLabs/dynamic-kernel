"""
Neural Port — Deep Behavioral Investigation
============================================
Covers 6 behavioral dimensions:

  1. Convergence speed per mode (how many ticks to plateau?)
  2. What the beta matrix actually looks like at convergence (structural signature)
  3. Sparse network behavior (density < 1.0)
  4. COMPOSITE mode — lambda sweep (symmetry vs utility tradeoff)
  5. Noise sensitivity — does noise help escape local minima?
  6. Mode chaining — does the ORDER of modes matter?

Run:  py deep_investigate.py
"""

import sys, io
import numpy as np
from optimizer import SymmetryOptimizer, SymmetryMode
from kernel import DynamicTopologyKernel, topology_from_edges

# ── stdout to UTF-8 so Unicode box chars work on Windows ─────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Shared config ─────────────────────────────────────────────────────────────
N           = 12
F           = 4
ETA         = 0.05
EPS         = 1e-3
N_STEPS     = 300
TEL         = np.full(F, 1.0 / F)
MODES_MAIN  = [
    SymmetryMode.ENTROPY_PI,
    SymmetryMode.ROW_ENTROPY,
    SymmetryMode.DETAILED_BALANCE,
    SymmetryMode.SPECTRAL_GAP,
    SymmetryMode.WEIGHT_SYMMETRY,
]

# ── Kernel factory ─────────────────────────────────────────────────────────────
def make_kernel(n=N, f=F, density=1.0, seed=42, beta_lo=0.5, beta_hi=8.0):
    """Optionally sparse, randomly-weighted kernel. Density 1.0 = fully connected."""
    rng    = np.random.default_rng(seed)
    labels = [f"N{i+1}" for i in range(n)]
    nodes  = {lbl: np.full(f, 1.0 / f) for lbl in labels}
    edges  = []
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < density:
                edges.append((labels[i], labels[j], 5.0))
    # Ensure connectivity: add a ring backbone so no sinks exist
    for i in range(n):
        j = (i+1) % n
        pair = (labels[i], labels[j], 5.0)
        rev  = (labels[j], labels[i], 5.0)
        if pair not in edges and rev not in edges:
            edges.append(pair)
    topo = topology_from_edges(nodes=nodes, edges=edges, undirected=True)
    beta = np.zeros((n, n))
    beta[topo.adjacency_mask] = rng.uniform(beta_lo, beta_hi,
                                            size=topo.adjacency_mask.sum())
    return DynamicTopologyKernel(
        topology=topo, alpha=1.0, beta=beta,
        feedback_rate=0.0, temperature=1.0, feedback_noise=0.0,
    )

# ── Metric helpers ─────────────────────────────────────────────────────────────
def kl_uniform(pi):
    n  = len(pi)
    p  = np.clip(pi, 1e-12, None)
    return float(np.sum(p * np.log(p * n)))

def db_residual(kernel, tel):
    P  = kernel.transition_matrix(tel)
    pi = kernel.stationary_distribution(tel)
    flow = pi[:, None] * P
    return float(np.abs(flow - flow.T)[kernel.topo.adjacency_mask].max())

def wt_asym(beta):
    return float(np.linalg.norm(beta - beta.T))

def row_var(kernel, tel):
    return float(np.var(kernel.get_diagnostic(tel)["row_entropy"]))

def spec_gap(kernel, tel):
    eigs = np.sort(np.abs(np.linalg.eigvals(kernel.transition_matrix(tel))))[::-1]
    return float(eigs[0] - eigs[1]) if len(eigs) >= 2 else 0.0

def snapshot(kernel, tel):
    pi  = kernel.stationary_distribution(tel)
    mix = kernel.get_diagnostic(tel)["mixing_time"]
    return dict(
        kl  = kl_uniform(pi),
        db  = db_residual(kernel, tel),
        wa  = wt_asym(kernel._beta),
        rv  = row_var(kernel, tel),
        sg  = spec_gap(kernel, tel),
        mix = float(mix) if not np.isinf(mix) else 999.0,
        pi  = pi,
    )

def run(kernel, mode, n_steps=N_STEPS, noise=0.0, target_pi=None,
        composite_lambda=0.5, composite_utility_type="kl"):
    opt = SymmetryOptimizer(
        kernel, mode=mode, eta=None, eps=EPS,
        noise_sigma=noise, telemetry=TEL,
        target_pi=target_pi, composite_lambda=composite_lambda,
        composite_utility_type=composite_utility_type,
        converge_eps=1e-5, converge_ticks=40,
    )
    sigma_curve = []
    grad_curve  = []
    converged_at = None
    for t in range(n_steps):
        s = opt.step()
        sigma_curve.append(s["sigma_value"])
        grad_curve.append(s["grad_norm"])
        if s["converged"] and converged_at is None:
            converged_at = t + 1
            break
    final = snapshot(kernel, TEL)
    final["sigma_curve"]   = sigma_curve
    final["grad_curve"]    = grad_curve
    final["converged_at"]  = converged_at
    final["ticks"]         = len(sigma_curve)
    return final

def hdr(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")

def row_fmt(*cols, widths=None):
    widths = widths or [22] + [9]*10
    parts  = [str(c).ljust(w) if i==0 else str(c).rjust(w)
               for i,(c,w) in enumerate(zip(cols, widths))]
    print("  " + "".join(parts))

# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 1 — Convergence speed per mode
# ═══════════════════════════════════════════════════════════════════════════════
hdr("DIMENSION 1: Convergence Speed & Sigma Gain per Mode")
print("  Asymmetric 12-node fully-connected network, 300 max steps, no noise.")
print()

headers = ["Mode","ticks","conv?","ΔΣ","KL(π,u)","DetBal","WtAsym","RowVar","SpGap","Mix"]
row_fmt(*headers, widths=[22,7,6,10,9,9,9,9,9,6])
print("  " + "-"*87)

conv_results = {}
for mode in MODES_MAIN:
    k = make_kernel()
    s0 = snapshot(k, TEL)
    r  = run(k, mode)
    conv = str(r["converged_at"]) if r["converged_at"] else f">{N_STEPS}"
    dsig = r["sigma_curve"][-1] - r["sigma_curve"][0] if r["sigma_curve"] else 0
    row_fmt(
        mode.value, r["ticks"], conv,
        f"{dsig:+.4f}",
        f"{r['kl']:.5f}",
        f"{r['db']:.5f}",
        f"{r['wa']:.3f}",
        f"{r['rv']:.5f}",
        f"{r['sg']:.5f}",
        f"{r['mix']:.2f}",
        widths=[22,7,6,10,9,9,9,9,9,6]
    )
    conv_results[mode] = {"s0": s0, "final": r}

# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 2 — Beta matrix structure at convergence
# ═══════════════════════════════════════════════════════════════════════════════
hdr("DIMENSION 2: Beta Matrix Structure at Convergence")
print("  Shows the distribution of edge weights after 300 steps.")
print("  Stats: mean, std, min, max, skew, % edges at max (=20)\n")

for mode in MODES_MAIN:
    k = make_kernel()
    run(k, mode)
    beta = k._beta
    mask = k.topo.adjacency_mask
    vals = beta[mask]
    skew = float(np.mean(((vals - vals.mean()) / (vals.std()+1e-12))**3))
    pct_max = 100.0 * np.sum(vals >= 19.5) / len(vals)
    pct_min = 100.0 * np.sum(vals <= 0.1)  / len(vals)
    print(f"  {mode.value:<22}  "
          f"mean={vals.mean():5.2f}  std={vals.std():4.2f}  "
          f"[{vals.min():.2f}, {vals.max():.2f}]  "
          f"skew={skew:+.2f}  "
          f"@max={pct_max:.0f}%  @min={pct_min:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 3 — Sparse network behavior
# ═══════════════════════════════════════════════════════════════════════════════
hdr("DIMENSION 3: Sparse Network Behavior (density sweep)")
print("  WEIGHT_SYMMETRY mode.  How does reducing connectivity affect convergence?\n")

densities = [1.0, 0.75, 0.5, 0.35]
row_fmt("Density","edges","ticks","ΔΣ","KL(π,u)","DetBal","WtAsym","SpGap",
        widths=[10,7,7,10,9,9,9,9])
print("  " + "-"*70)

for d in densities:
    k = make_kernel(density=d)
    n_edges = int(k.topo.adjacency_mask.sum() // 2)  # undirected
    r = run(k, SymmetryMode.WEIGHT_SYMMETRY)
    dsig = r["sigma_curve"][-1] - r["sigma_curve"][0] if r["sigma_curve"] else 0
    row_fmt(f"{d:.2f}", n_edges, r["ticks"],
            f"{dsig:+.4f}", f"{r['kl']:.5f}",
            f"{r['db']:.5f}", f"{r['wa']:.3f}", f"{r['sg']:.5f}",
            widths=[10,7,7,10,9,9,9,9])

# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 4 — COMPOSITE mode lambda sweep
# ═══════════════════════════════════════════════════════════════════════════════
hdr("DIMENSION 4: COMPOSITE Mode — Lambda Sweep (KL Utility)")
print("  lambda=1.0 => pure ENTROPY_PI symmetry")
print("  lambda=0.0 => pure utility (match target_pi: node 0 gets 40%, KL mode)")
print()

target = np.full(N, (1.0 - 0.40) / (N - 1))
target[0] = 0.40

lambdas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
row_fmt("Lambda","ticks","KL(π,u)","pi[N1]","pi_max","DetBal","SpGap",
        widths=[9,7,9,9,9,9,9])
print("  " + "-"*61)

for lam in lambdas:
    k = make_kernel()
    r = run(k, SymmetryMode.COMPOSITE,
            composite_lambda=lam, target_pi=target, composite_utility_type="kl")
    pi = r["pi"]
    row_fmt(f"{lam:.1f}", r["ticks"],
            f"{r['kl']:.5f}", f"{pi[0]:.4f}", f"{max(pi):.4f}",
            f"{r['db']:.5f}", f"{r['sg']:.5f}",
            widths=[9,7,9,9,9,9,9])

# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 5 — Noise sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
hdr("DIMENSION 5: Noise Sensitivity — Does Noise Help?")
print("  ENTROPY_PI mode on a mild asymmetric init (beta_lo=3, beta_hi=6).")
print("  Testing whether gradient noise escapes local plateaus.\n")

noises = [0.0, 0.01, 0.05, 0.1, 0.2]
row_fmt("Noise","ticks","ΔΣ","KL(π,u)","Monotone%",
        widths=[9,7,10,9,11])
print("  " + "-"*46)

for noise in noises:
    k = make_kernel(beta_lo=3.0, beta_hi=6.0)
    r = run(k, SymmetryMode.ENTROPY_PI, noise=noise)
    sc = r["sigma_curve"]
    dsig = sc[-1] - sc[0] if sc else 0
    diffs = np.diff(sc)
    pct_mono = 100.0 * np.sum(diffs > 0) / len(diffs) if len(diffs) else 0
    row_fmt(f"{noise:.2f}", r["ticks"],
            f"{dsig:+.5f}", f"{r['kl']:.5f}", f"{pct_mono:.1f}%",
            widths=[9,7,10,9,11])

# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION 6 — Mode chaining (order effects)
# ═══════════════════════════════════════════════════════════════════════════════
hdr("DIMENSION 6: Mode Chaining — Does Order Matter?")
print("  Run 150 steps of Mode A, then 150 steps of Mode B on the SAME kernel.")
print("  Compare to running Mode B alone for 300 steps.\n")

chains = [
    (SymmetryMode.WEIGHT_SYMMETRY, SymmetryMode.ENTROPY_PI),
    (SymmetryMode.ENTROPY_PI,      SymmetryMode.WEIGHT_SYMMETRY),
    (SymmetryMode.SPECTRAL_GAP,    SymmetryMode.DETAILED_BALANCE),
    (SymmetryMode.DETAILED_BALANCE,SymmetryMode.SPECTRAL_GAP),
]

row_fmt("Chain (A->B)","KL(final)","DetBal","WtAsym","SpGap",
        widths=[30,10,9,9,9])
print("  " + "-"*67)

for mA, mB in chains:
    k = make_kernel()
    # Phase A
    optA = SymmetryOptimizer(k, mode=mA, eta=ETA, eps=EPS,
                              telemetry=TEL, converge_eps=1e-5, converge_ticks=40)
    for _ in range(150): optA.step()
    # Phase B (same kernel — weights carry over)
    optB = SymmetryOptimizer(k, mode=mB, eta=ETA, eps=EPS,
                              telemetry=TEL, converge_eps=1e-5, converge_ticks=40)
    for _ in range(150): optB.step()
    s = snapshot(k, TEL)
    label = f"{mA.value[:8]}->{mB.value[:8]}"
    row_fmt(label, f"{s['kl']:.5f}", f"{s['db']:.5f}",
            f"{s['wa']:.3f}", f"{s['sg']:.5f}",
            widths=[30,10,9,9,9])

print()
print("  -- Baseline: B alone for 300 steps --")
baselines = set()
for mA, mB in chains:
    if mB not in baselines:
        baselines.add(mB)
        k  = make_kernel()
        r  = run(k, mB)
        s  = snapshot(k, TEL)
        row_fmt(f"BASELINE {mB.value[:14]}", f"{s['kl']:.5f}",
                f"{s['db']:.5f}", f"{s['wa']:.3f}", f"{s['sg']:.5f}",
                widths=[30,10,9,9,9])

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY VERDICT
# ═══════════════════════════════════════════════════════════════════════════════
hdr("BEHAVIORAL SUMMARY & DIAGNOSES")
print("""
  KEY FINDINGS:

  [1] CONVERGENCE
      Each mode is characterized by its sigma gain (ΔΣ) and convergence speed.
      Small ΔΣ means the optimizer found the attractor is nearby — easy problem.
      Large ΔΣ means significant structural reorganization happened.

  [2] BETA STRUCTURE
      Check mean/std/skew of edge weights at convergence.
      - WEIGHT_SYMMETRY should collapse skew toward 0 (equal β_ij, β_ji pairs).
      - SPECTRAL_GAP may INCREASE skew (directional highways boost one edge).
      - ENTROPY_PI / ROW_ENTROPY should show moderate, uniform weights.

  [3] SPARSE NETWORKS
      Fewer edges = fewer degrees of freedom for the optimizer.
      Expect WtAsym to remain higher as density decreases (fewer edges to equalize).

  [4] COMPOSITE
      lambda=1 → pure symmetry, pi[N1] ≈ 1/N (uniform)
      lambda=0 → pure utility, pi[N1] should approach 0.40 (the target)
      The crossover lambda where pi[N1] starts deviating from uniform reveals
      how "expensive" (in symmetry cost) the utility objective is.

  [5] NOISE
      If ΔΣ increases with noise => noise helps escape plateaus.
      If ΔΣ decreases with noise => noise disrupts clean gradient.
      Monotone% < 70% with high noise = optimizer becomes a random walk.

  [6] MODE CHAINING
      If WS->EP gives lower KL than EP alone → WS "pre-conditions" the landscape.
      If SG->DB gives lower DB than DB alone → gap-widening helps find reversibility.
      Path-dependence is evidence that the loss landscape is non-convex.
""")

"""
DTE Design-Space Exploration — Settings library.

Implements the "Settings" model-family parameterization described in
DTE_DESIGN_SPACE_RESEARCH_PROMPT.md. The existing kernel stack becomes
Settings() == DEFAULT, one labeled point in an explicit grid.

Six axes (DEFAULT marked):

    Axis 1  alignment    : bilinear* | cosine | rbf | poly | threshold
    Axis 2  composition  : additive* | multiplicative | bottleneck | two_stage
    Axis 3  link         : softmax* | powerlaw | eps_greedy | local_tau
    Axis 4  telemetry    : ema* | frozen | momentum | two_timescale
    Axis 5  memory       : static* (kernel) / reward_gated* (two-route theorem)
                           | traffic | adaptive_eta
    Axis 6  ecology      : step* | logistic | shock

Two witnesses carry the diagnostic battery:

    - the two-route memory-ecology model (axes 3, 5, 6 are live; axes 1, 2, 4
      have no analog — there is no telemetry or feature geometry to vary);
    - the ant foraging terrain on the full kernel (axes 1-4 are live; axis 5
      preference memory lives in the pheromone deposit loop, axis 6 in the
      scenario inventory scripts).

Nothing here mutates kernel.py or the existing experiment scripts: the
DEFAULT code path stays DEFAULT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from kernel import DynamicTopologyKernel, Topology, topology_from_edges
from ant_foraging_dte import (
    BRANCH_NODES,
    FOOD_NODES,
    INTENTS,
    ForagingPolicy,
    ForagingScenario,
    SimulationConfig,
    _entropy,
    _friction_matrix,
    _normalize,
    _normalize_rows,
    _sample_next,
    topology as ant_topology,
)

BIG = 1.0e6

ALIGNMENT_CELLS = ("bilinear", "cosine", "rbf", "poly", "threshold")
COMPOSITION_CELLS = ("additive", "multiplicative", "bottleneck", "two_stage")
LINK_CELLS = ("softmax", "powerlaw", "eps_greedy", "local_tau")
TELEMETRY_CELLS = ("ema", "frozen", "momentum", "two_timescale")
MEMORY_CELLS = ("static", "reward_gated", "traffic", "adaptive_eta")
ECOLOGY_CELLS = ("step", "logistic", "shock")


@dataclass(frozen=True)
class Settings:
    """One cell of the model-family grid. Defaults reproduce kernel.py."""

    alignment: str = "bilinear"
    composition: str = "additive"
    link: str = "softmax"
    telemetry: str = "ema"

    # Axis 1 parameters
    rbf_sigma: float = 0.75
    poly_c: float = 1.0
    poly_d: int = 2
    threshold_theta: float = 0.5

    # Axis 2 parameters
    two_stage_quantile: float = 0.75   # feasibility gate on alpha*D - S

    # Axis 3 parameters
    powerlaw_gamma: float = 2.2        # matches ACO heuristic_beta in the comparator
    eps_greedy_eps: float = 0.10
    local_tau_floor: float = 0.25      # min degree multiplier for tau_i

    # Axis 4 parameters
    momentum_mu: float = 0.6
    fast_weight: float = 0.7           # two_timescale visible mix
    slow_rate: float = 0.02            # two_timescale slow lambda

    def label(self) -> str:
        parts = []
        if self.alignment != "bilinear":
            parts.append(f"a1={self.alignment}")
        if self.composition != "additive":
            parts.append(f"a2={self.composition}")
        if self.link != "softmax":
            parts.append(f"a3={self.link}")
        if self.telemetry != "ema":
            parts.append(f"a4={self.telemetry}")
        return "DEFAULT" if not parts else ",".join(parts)


# ===========================================================================
# SettingsKernel — axes 1-3 inside the transition computation
# ===========================================================================

class SettingsKernel(DynamicTopologyKernel):
    """
    Drop-in kernel whose transition computation is assembled from the
    Settings cell. With Settings() it is numerically identical to
    DynamicTopologyKernel. Degenerate events (empty admissible rows,
    nonpositive power-law weights) are counted, not hidden.
    """

    def __init__(self, topology: Topology, settings: Settings | None = None, **kwargs):
        super().__init__(topology, **kwargs)
        self.settings = settings or Settings()
        self.degeneracy = {
            "empty_admissible_rows": 0,
            "nonpositive_powerlaw_weights": 0,
        }
        deg = self.topo.adjacency_mask.sum(axis=1).astype(float)
        mean_deg = max(1.0, float(deg[deg > 0].mean())) if np.any(deg > 0) else 1.0
        self._degree_tau_scale = np.maximum(
            self.settings.local_tau_floor, deg / mean_deg
        )
        # Two-stage feasibility threshold from the at-init cost distribution.
        base_cost = self.alpha * self.topo.distance_matrix - self._sponsor_friction
        finite = base_cost[self.topo.adjacency_mask]
        self._two_stage_threshold = (
            float(np.quantile(finite, self.settings.two_stage_quantile))
            if finite.size
            else 0.0
        )

    # --- Axis 1: alignment ------------------------------------------------

    def _alignment_batch(self, T: np.ndarray) -> np.ndarray:
        """(K, F) telemetry -> (K, N) alignment, bias included."""
        s = self.settings
        Fm = self.topo.node_features
        if s.alignment == "bilinear":
            raw = T @ Fm.T
        elif s.alignment == "cosine":
            node_norms = np.linalg.norm(Fm, axis=1)
            t_norms = np.linalg.norm(T, axis=1, keepdims=True)
            denom = np.maximum(t_norms * node_norms[np.newaxis, :], 1e-12)
            raw = (T @ Fm.T) / denom
        elif s.alignment == "rbf":
            d2 = np.sum((T[:, np.newaxis, :] - Fm[np.newaxis, :, :]) ** 2, axis=2)
            raw = np.exp(-d2 / max(s.rbf_sigma ** 2, 1e-12))
        elif s.alignment == "poly":
            raw = (T @ Fm.T + s.poly_c) ** s.poly_d
        elif s.alignment == "threshold":
            dot = T @ Fm.T
            raw = np.where(dot > s.threshold_theta, dot, -BIG)
        else:
            raise ValueError(f"unknown alignment cell: {s.alignment}")
        return raw + self._node_bias[np.newaxis, :]

    # --- Axis 2: composition ----------------------------------------------

    def _compose_W(self, A_b: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """
        A_b: (K, 1, N) destination alignment. Returns (W, extra_mask).
        extra_mask is an additional admissibility mask (two_stage) or None.
        """
        s = self.settings
        D = self.topo.distance_matrix[np.newaxis, :, :]
        beta = self._beta[np.newaxis, :, :]
        S = self._sponsor_friction[np.newaxis, :, :]
        base = self.alpha * D

        if s.composition == "additive":
            W = base - beta * A_b - S
            return self._soft_floor(W), None
        if s.composition == "multiplicative":
            # All channels squashed to bounded factors so they multiply.
            A_n = 1.0 / (1.0 + np.exp(-np.clip(A_b, -60.0, 60.0)))
            beta_n = beta / (1.0 + beta)
            S_n = S / (1.0 + np.abs(S))
            with np.errstate(invalid="ignore"):
                W = base * (1.0 - beta_n * A_n) * (1.0 - S_n)
            return self._soft_floor(W), None
        if s.composition == "bottleneck":
            W = np.maximum(base - S, -beta * A_b)
            return self._soft_floor(W), None
        if s.composition == "two_stage":
            # Feasibility gate on physical cost, preference-only weights after.
            # NOTE: the softplus floor is deliberately skipped — it is
            # calibrated to additive cost-scale W and would collapse all
            # strongly-aligned (very negative) preference weights to the
            # floor. This is itself a finding: the floor encodes an implicit
            # additive-composition assumption.
            cost = base - S
            extra_mask = cost <= self._two_stage_threshold  # (1, N, N)
            W = -beta * A_b  # broadcasts to (K, N, N)
            return W, extra_mask
        raise ValueError(f"unknown composition cell: {s.composition}")

    # --- Axis 3: link -----------------------------------------------------

    def _apply_link(self, W: np.ndarray, mask: np.ndarray, step) -> np.ndarray:
        """W, mask: (K, N, N). Returns row-stochastic P (rows may sum to 0)."""
        s = self.settings
        tau = self._resolve_temperature(step)

        if s.link in ("softmax", "local_tau"):
            if s.link == "local_tau":
                tau_row = tau * self._degree_tau_scale  # (N,)
                tau_eff = np.maximum(tau_row, 1e-8)[np.newaxis, :, np.newaxis]
            else:
                tau_eff = tau
            neg_W = np.where(mask, -W / tau_eff, -np.inf)
            row_max = np.max(neg_W, axis=2, keepdims=True)
            row_max = np.where(np.isfinite(row_max), row_max, 0.0)
            exp_W = np.exp(neg_W - row_max)
            exp_W = np.where(mask, exp_W, 0.0)
            row_sums = np.sum(exp_W, axis=2, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            return exp_W / row_sums

        if s.link == "powerlaw":
            # Classical ACO transition rule: P ~ W^-gamma over admissible
            # edges. Requires W > 0; the softplus floor normally guarantees
            # this. Nonpositive admissible weights are clamped and counted.
            bad = mask & (W <= 0)
            n_bad = int(np.sum(bad))
            if n_bad:
                self.degeneracy["nonpositive_powerlaw_weights"] += n_bad
            W_pos = np.maximum(W, 1e-9)
            with np.errstate(over="ignore"):
                weights = np.where(mask, W_pos ** (-s.powerlaw_gamma), 0.0)
            weights = np.where(np.isfinite(weights), weights, 0.0)
            row_sums = np.sum(weights, axis=2, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            return weights / row_sums

        if s.link == "eps_greedy":
            eps = s.eps_greedy_eps
            W_masked = np.where(mask, W, np.inf)
            best = np.argmin(W_masked, axis=2)  # (K, N)
            K, N, _ = W.shape
            onehot = np.zeros_like(W)
            k_idx, i_idx = np.meshgrid(np.arange(K), np.arange(N), indexing="ij")
            onehot[k_idx, i_idx, best] = 1.0
            counts = np.sum(mask, axis=2, keepdims=True)
            uniform = np.where(mask, 1.0 / np.maximum(counts, 1), 0.0)
            has_row = counts[..., 0] > 0
            P = (1.0 - eps) * onehot + eps * uniform
            P = np.where(has_row[..., np.newaxis], P, 0.0)
            return P

        raise ValueError(f"unknown link cell: {s.link}")

    # --- Assembled transition computation -----------------------------------

    def _P_from_telemetry(self, T: np.ndarray, step) -> np.ndarray:
        A = self._alignment_batch(T)          # (K, N)
        A_b = A[:, np.newaxis, :]             # (K, 1, N)
        W, extra_mask = self._compose_W(A_b)  # (K, N, N)
        K = T.shape[0]
        adjacency = np.broadcast_to(
            self.topo.adjacency_mask[np.newaxis, :, :], W.shape
        )
        if extra_mask is None:
            mask = adjacency
        else:
            mask = adjacency & extra_mask
            # Rows with outgoing edges but an empty admissible set are
            # degenerate cells of the two-stage gate. Recorded, then fall
            # back to the full adjacency so simulation can proceed.
            row_has = mask.any(axis=2)
            adj_has = adjacency.any(axis=2)
            empty = adj_has & ~row_has
            n_empty = int(np.sum(empty))
            if n_empty:
                self.degeneracy["empty_admissible_rows"] += n_empty
                mask = np.where(empty[:, :, np.newaxis], adjacency, mask)
        return self._apply_link(W, mask, step)

    def transition_matrix(self, telemetry: np.ndarray, step=None) -> np.ndarray:
        T = np.asarray(telemetry, dtype=np.float64)[np.newaxis, :]
        return self._P_from_telemetry(T, step)[0]

    def transition_matrix_batch(self, telemetry_batch: np.ndarray, step=None) -> np.ndarray:
        return self._P_from_telemetry(
            np.asarray(telemetry_batch, dtype=np.float64), step
        )


# ===========================================================================
# Axis 4 — telemetry feedback state for batched ant simulation
# ===========================================================================

class TelemetryState:
    """
    Holds whatever extra state the Axis-4 telemetry rule needs (velocity,
    fast/slow split) and exposes the same operations the ant simulation
    performs: bulk visit update, per-ant target blend, per-ant reset.
    `visible()` is what the kernel sees.
    """

    def __init__(self, settings: Settings, initial: np.ndarray):
        self.s = settings
        self.t = initial.copy()
        if settings.telemetry == "momentum":
            self.v = np.zeros_like(initial)
        elif settings.telemetry == "two_timescale":
            self.fast = initial.copy()
            self.slow = initial.copy()

    def visible(self) -> np.ndarray:
        if self.s.telemetry == "two_timescale":
            w = self.s.fast_weight
            return _normalize_rows(w * self.fast + (1.0 - w) * self.slow)
        return self.t

    def visit_update(self, visited_features: np.ndarray, rate: float) -> None:
        mode = self.s.telemetry
        if mode == "ema":
            self.t = (1.0 - rate) * self.t + rate * visited_features
        elif mode == "frozen":
            pass
        elif mode == "momentum":
            mu = self.s.momentum_mu
            self.v = mu * self.v + (1.0 - mu) * (visited_features - self.t)
            self.t = self.t + self.v
        elif mode == "two_timescale":
            ls = self.s.slow_rate
            self.fast = (1.0 - rate) * self.fast + rate * visited_features
            self.slow = (1.0 - ls) * self.slow + ls * visited_features
        else:
            raise ValueError(f"unknown telemetry cell: {mode}")

    def blend(self, ant: int, target: np.ndarray, rate: float) -> None:
        """Scenario-semantic overwrite (e.g. return-carrier intent)."""
        if self.s.telemetry == "two_timescale":
            self.fast[ant] = (1.0 - rate) * self.fast[ant] + rate * target
        else:
            self.t[ant] = (1.0 - rate) * self.t[ant] + rate * target

    def reset(self, ant: int, base: np.ndarray) -> None:
        if self.s.telemetry == "momentum":
            self.v[ant] = 0.0
            self.t[ant] = base
        elif self.s.telemetry == "two_timescale":
            self.fast[ant] = base
            self.slow[ant] = base
        else:
            self.t[ant] = base

    def normalize(self) -> None:
        if self.s.telemetry == "two_timescale":
            self.fast = _normalize_rows(self.fast)
            self.slow = _normalize_rows(self.slow)
        else:
            self.t = _normalize_rows(self.t)


# ===========================================================================
# Ant foraging with Settings — faithful port of ant_foraging_dte.simulate
# with the kernel and telemetry rule swapped per cell. With Settings() this
# reproduces the original run exactly (same rng draw order).
# ===========================================================================

def _build_settings_kernel(
    config: SimulationConfig, policy: ForagingPolicy, settings: Settings
) -> SettingsKernel:
    topo = ant_topology()
    beta = np.full((topo.N, topo.N), config.beta_strength, dtype=float)
    node_bias = np.zeros(topo.N, dtype=float)
    node_bias[topo.labels.index("Nest")] = 0.18
    node_bias[topo.labels.index("Trail Fork")] = 0.06
    return SettingsKernel(
        topology=topo,
        settings=settings,
        alpha=config.alpha,
        beta=beta,
        feedback_rate=config.feedback_rate,
        temperature=policy.temperature,
        feedback_noise=0.0,
        node_bias=node_bias,
    )


def _assign_intents_settings(
    rng: np.random.Generator, config: SimulationConfig, policy: ForagingPolicy
) -> tuple[np.ndarray, list[str]]:
    exploiter_share = max(0.0, 1.0 - policy.scout_share - policy.risk_averse_share)
    names = ["Scout", "Exploiter", "Risk Averse"]
    probs = np.array(
        [policy.scout_share, exploiter_share, policy.risk_averse_share], dtype=float
    )
    probs = probs / probs.sum()
    classes = rng.choice(names, size=config.agents, p=probs)
    telemetry = np.array(
        [_normalize(np.array(INTENTS[name], dtype=float)) for name in classes]
    )
    return telemetry, [str(name) for name in classes]


@dataclass(frozen=True)
class AdaptiveEvaporation:
    """
    Axis-5 adaptive evaporation, colony-level analog of the two-route
    adaptive_eta rule. The colony tracks its expected pickup success rate
    (pickups / food-node visits) with an EMA; when realized success falls
    below expectation (ants arriving at empty patches), evaporation rises
    with the surprise:

        evap_t = min(eta_max, evap_base + gain * max(0, rhat - success_t))

    The signal only fires when ants actually visit depleted patches — i.e.
    exactly under stale lock-in — and fades once expectations adjust, by
    which time the stale pheromone should be gone.
    """

    gain: float = 2.0
    eta_max: float = 0.35
    track_rate: float = 0.10
    initial_expectation: float = 1.0


def simulate_ant_settings(
    config: SimulationConfig,
    scenario: ForagingScenario,
    policy: ForagingPolicy,
    settings: Settings,
    seed_offset: int = 0,
    adaptive_evap: AdaptiveEvaporation | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed + seed_offset)
    kernel = _build_settings_kernel(config, policy, settings)
    topo = kernel.topo
    idx = {label: i for i, label in enumerate(topo.labels)}

    nest = idx["Nest"]
    hazard = idx["Hazard Zone"]
    food_idx = {idx[name]: name for name in FOOD_NODES}
    branch_idx = {idx[name]: name for name in BRANCH_NODES}
    fork = idx["Trail Fork"]

    positions = np.full(config.agents, nest, dtype=int)
    telemetries, classes = _assign_intents_settings(rng, config, policy)
    base_intents = np.array(
        [_normalize(np.array(INTENTS[name], dtype=float)) for name in classes]
    )
    return_intent = _normalize(np.array(INTENTS["Return Carrier"], dtype=float))
    tstate = TelemetryState(settings, telemetries)

    carrying = np.zeros(config.agents, dtype=bool)
    carried_source: list[str | None] = [None for _ in range(config.agents)]
    paths: list[list[tuple[int, int]]] = [[] for _ in range(config.agents)]

    food_remaining = dict(scenario.food_inventory)
    initial_food_total = sum(food_remaining.values())
    static_friction = _friction_matrix(topo, scenario.initial_friction)
    shock_friction = _friction_matrix(topo, scenario.shock_friction)
    pheromone = np.zeros((topo.N, topo.N), dtype=float)

    edge_counts = np.zeros((topo.N, topo.N), dtype=int)
    fork_counts = {name: 0 for name in BRANCH_NODES}
    returns_by_source = {name: 0 for name in FOOD_NODES}
    visits_by_source = {name: 0 for name in FOOD_NODES}
    empty_food_visits = 0
    hazard_hits = 0
    food_returns = 0
    path_lengths_to_return: list[int] = []
    returns_by_step: list[int] = []
    shock_active = False

    rhat = adaptive_evap.initial_expectation if adaptive_evap else 0.0
    surprise = 0.0
    evap_trace: list[float] = []

    for step in range(config.steps):
        if scenario.shock_step is not None and step >= scenario.shock_step:
            shock_active = True

        if adaptive_evap is not None:
            evap_t = min(
                adaptive_evap.eta_max,
                policy.evaporation + adaptive_evap.gain * surprise,
            )
        else:
            evap_t = policy.evaporation
        evap_trace.append(evap_t)
        pheromone *= max(0.0, 1.0 - evap_t)
        friction = static_friction + pheromone
        if shock_active:
            friction = friction + shock_friction
        kernel._sponsor_friction = friction

        p_all = kernel.transition_matrix_batch(tstate.visible(), step=step)
        transition_rows = p_all[np.arange(config.agents), positions]
        next_positions = _sample_next(rng, transition_rows)

        step_returns = 0
        step_pickups = 0
        step_food_visits = 0
        previous_positions = positions.copy()
        visited_features = topo.node_features[next_positions]
        tstate.visit_update(visited_features, config.feedback_rate)

        for ant in range(config.agents):
            src = int(previous_positions[ant])
            dst = int(next_positions[ant])
            if src != dst and np.isfinite(topo.distance_matrix[src, dst]):
                edge_counts[src, dst] += 1
                paths[ant].append((src, dst))
                if len(paths[ant]) > config.max_path_memory:
                    paths[ant] = paths[ant][-config.max_path_memory:]
                if src == fork and dst in branch_idx:
                    fork_counts[branch_idx[dst]] += 1

            if dst == hazard:
                hazard_hits += 1
                carrying[ant] = False
                carried_source[ant] = None
                paths[ant] = []
                next_positions[ant] = nest
                tstate.reset(ant, base_intents[ant])
                continue

            if carrying[ant] and dst == nest:
                food_returns += 1
                step_returns += 1
                source = carried_source[ant]
                if source is not None:
                    returns_by_source[source] += 1
                path_lengths_to_return.append(len(paths[ant]))
                if policy.pheromone_deposit > 0.0 and paths[ant]:
                    deposit = policy.pheromone_deposit / math.sqrt(max(1, len(paths[ant])))
                    for i, j in paths[ant]:
                        pheromone[i, j] += deposit
                    np.clip(pheromone, 0.0, policy.pheromone_cap, out=pheromone)
                carrying[ant] = False
                carried_source[ant] = None
                paths[ant] = []
                tstate.reset(ant, base_intents[ant])
                continue

            if not carrying[ant] and dst in food_idx:
                food_name = food_idx[dst]
                visits_by_source[food_name] += 1
                step_food_visits += 1
                if food_remaining.get(food_name, 0) > 0:
                    food_remaining[food_name] -= 1
                    step_pickups += 1
                    carrying[ant] = True
                    carried_source[ant] = food_name
                    tstate.blend(ant, return_intent, config.return_feedback_rate)
                else:
                    empty_food_visits += 1

            if carrying[ant]:
                tstate.blend(ant, return_intent, config.return_feedback_rate)
            elif dst == nest:
                paths[ant] = []

        positions = next_positions
        tstate.normalize()
        returns_by_step.append(step_returns)

        if adaptive_evap is not None and step_food_visits > 0:
            success = step_pickups / step_food_visits
            surprise = max(0.0, rhat - success)
            rhat += adaptive_evap.track_rate * (success - rhat)

    fork_total = sum(fork_counts.values())
    lock_in_index = max(fork_counts.values()) / fork_total if fork_total else 0.0
    branch_winner = max(fork_counts, key=fork_counts.get) if fork_total else "none"
    rich_returns = returns_by_source["Rich Food Patch"]
    sparse_returns = returns_by_source["Sparse Food Patch"]
    return_total = max(1, rich_returns + sparse_returns)
    food_visit_total = max(1, sum(visits_by_source.values()))

    shock_recovery_steps: int | None = None
    if (
        scenario.shock_step is not None
        and scenario.shock_step >= 12
        and scenario.shock_step < len(returns_by_step)
    ):
        pre = np.mean(returns_by_step[max(0, scenario.shock_step - 12): scenario.shock_step])
        target = 0.8 * pre
        if target > 0:
            for t in range(scenario.shock_step + 6, config.steps):
                window = returns_by_step[max(0, t - 8): t + 1]
                if np.mean(window) >= target:
                    shock_recovery_steps = int(t - scenario.shock_step)
                    break

    return {
        "settings": settings.label(),
        "scenario": scenario.name,
        "policy": policy.name,
        "agents": config.agents,
        "steps": config.steps,
        "food_returned": int(food_returns),
        "returns_per_100_ant_steps": round(
            100.0 * food_returns / (config.agents * config.steps), 4
        ),
        "food_completion_rate": round(food_returns / max(1, initial_food_total), 4),
        "empty_food_visits": int(empty_food_visits),
        "empty_food_visit_rate": round(empty_food_visits / food_visit_total, 4),
        "rich_visit_share": round(
            visits_by_source["Rich Food Patch"] / food_visit_total, 4
        ),
        "sparse_visit_share": round(
            visits_by_source["Sparse Food Patch"] / food_visit_total, 4
        ),
        "hazard_hits": int(hazard_hits),
        "hazard_rate": round(hazard_hits / (config.agents * config.steps), 4),
        "fork_entropy": round(_entropy(fork_counts), 4),
        "lock_in_index": round(lock_in_index, 4),
        "dominant_branch": branch_winner,
        "shock_recovery_steps": shock_recovery_steps,
        "pheromone_mass": round(float(np.sum(pheromone)), 4),
        "pheromone_max": round(float(np.max(pheromone)), 4),
        "mean_realized_evaporation": round(float(np.mean(evap_trace)), 4),
        "max_realized_evaporation": round(float(np.max(evap_trace)), 4),
        "degeneracy": dict(kernel.degeneracy),
    }


# ===========================================================================
# Two-route model with Settings — axes 3, 5, 6
# ===========================================================================

@dataclass(frozen=True)
class TwoRouteCell:
    """Axis selections (and their parameters) for the two-route witness."""

    link: str = "softmax"           # softmax | powerlaw | eps_greedy | local_tau
    memory: str = "reward_gated"    # static | reward_gated | traffic | adaptive_eta
    ecology: str = "step"           # step | logistic | shock

    powerlaw_gamma: float = 5.0     # sharpness analog of kappa=5
    powerlaw_s0: float = 0.05       # zero-offset so 0-score routes stay defined

    adaptive_eta_max: float = 0.5
    adaptive_gain: float = 4.0
    reward_track_rate: float = 0.25

    logistic_growth: float = 0.0    # r in logistic regeneration
    logistic_capacity: float = 1.0  # K
    harvest: float = 0.08           # consumption pressure per unit p_rich

    shock_rate: float = 0.0         # Poisson arrival probability per cycle
    shock_log_mean: float = math.log(0.2)
    shock_log_sigma: float = 0.5
    seed: int = 0

    def label(self) -> str:
        parts = []
        if self.link != "softmax":
            parts.append(f"link={self.link}")
        if self.memory != "reward_gated":
            parts.append(f"mem={self.memory}")
        if self.ecology != "step":
            parts.append(f"eco={self.ecology}")
        return "DEFAULT" if not parts else ",".join(parts)


@dataclass(frozen=True)
class TwoRouteParams:
    """Mirror of two_route_memory_theorem.TwoRouteConfig (kept comparable)."""

    cycles: int = 120
    deplete_cycle: int = 36
    rich_initial_reward: float = 1.0
    sparse_reward: float = 0.62
    rho: float = 0.45
    eta: float = 0.02
    epsilon: float = 0.04
    inverse_temperature: float = 5.0


def two_route_link_probs(
    s_h: float, s_l: float, params: TwoRouteParams, cell: TwoRouteCell
) -> tuple[float, float]:
    """
    Axis-3 link on the single two-route decision node. local_tau collapses to
    softmax here (the decision node's degree is constant), which is recorded
    by the caller as an axis degeneracy on this topology.
    """
    if cell.link in ("softmax", "local_tau"):
        scaled = np.array([s_h, s_l], dtype=float) * params.inverse_temperature
        scaled -= float(np.max(scaled))
        exp = np.exp(scaled)
        q_h = float(exp[0] / np.sum(exp))
    elif cell.link == "powerlaw":
        a = (max(s_h, 0.0) + cell.powerlaw_s0) ** cell.powerlaw_gamma
        b = (max(s_l, 0.0) + cell.powerlaw_s0) ** cell.powerlaw_gamma
        q_h = a / (a + b) if (a + b) > 0 else 0.5
    elif cell.link == "eps_greedy":
        # Hard argmax; the epsilon mixing below is its ONLY exploration
        # mechanism (the kappa->inf limit of the softmax cell).
        q_h = 1.0 if s_h > s_l else (0.5 if s_h == s_l else 0.0)
    else:
        raise ValueError(f"unknown two-route link cell: {cell.link}")
    q_l = 1.0 - q_h
    eps = params.epsilon
    p_h = (1.0 - eps) * q_h + 0.5 * eps
    p_l = (1.0 - eps) * q_l + 0.5 * eps
    return p_h, p_l


def two_route_choice_point(cell: TwoRouteCell, params: TwoRouteParams) -> bool:
    """
    Outdegree-1 precondition: with only one admissible route, every link
    function must put probability 1 on it (uniform fallback included).
    All links here normalize over the admissible set, so this checks the
    normalization contract rather than the weight value.
    """
    for s in (0.0, 0.3, 5.0, -2.0):
        if cell.link in ("softmax", "local_tau"):
            p = 1.0  # softmax over a single finite score
        elif cell.link == "powerlaw":
            a = (max(s, 0.0) + cell.powerlaw_s0) ** cell.powerlaw_gamma
            p = a / a if a > 0 else None
            if p is None:
                return False
        elif cell.link == "eps_greedy":
            p = (1.0 - params.epsilon) * 1.0 + params.epsilon * 1.0
        if abs(p - 1.0) > 1e-12:
            return False
    return True


def simulate_two_route_cell(
    params: TwoRouteParams, cell: TwoRouteCell
) -> dict[str, Any]:
    """
    Generalized two-route run. Stale lock-in is generalized to: cycle counts
    as stale iff p_rich > p_sparse while the rich route's CURRENT reward is
    below the sparse reward. Under the step ecology this reduces exactly to
    the original post-depletion definition.
    """
    rng = np.random.default_rng(cell.seed)
    m_h = 0.0
    m_l = 0.0
    eta_h = params.eta
    eta_l = params.eta
    rhat_h = params.rich_initial_reward
    rhat_l = params.sparse_reward
    stock = cell.logistic_capacity

    stale_duration = 0
    stale_eligible_cycles = 0
    empty_rich_mass = 0.0
    throughput = 0.0
    recovery_cycle: int | None = None
    onset_cycle: int | None = None
    delta_at_onset: float | None = None
    locked_at_end = False
    episodes = 0
    in_episode = False

    for cycle in range(params.cycles):
        # --- Axis 6: ecology -> current rich reward -----------------------
        if cell.ecology == "step":
            r_h = params.rich_initial_reward if cycle < params.deplete_cycle else 0.0
        elif cell.ecology in ("logistic", "shock"):
            r_h = params.rich_initial_reward * stock / cell.logistic_capacity
        else:
            raise ValueError(f"unknown ecology cell: {cell.ecology}")
        r_l = params.sparse_reward

        p_h, p_l = two_route_link_probs(r_h + m_h, r_l + m_l, params, cell)
        throughput += p_h * r_h + p_l * r_l

        stale_eligible = r_h < r_l
        if stale_eligible:
            stale_eligible_cycles += 1
            empty_rich_mass += p_h
            if p_h > p_l:
                stale_duration += 1
                locked_at_end = True
                if onset_cycle is None:
                    onset_cycle = cycle
                    delta_at_onset = m_h - m_l
                if not in_episode:
                    episodes += 1
                    in_episode = True
            else:
                locked_at_end = False
                in_episode = False
                if onset_cycle is not None and recovery_cycle is None:
                    recovery_cycle = cycle
        else:
            in_episode = False

        # --- Axis 5: preference-memory update law -------------------------
        if cell.memory == "static":
            pass
        elif cell.memory == "reward_gated":
            m_h = m_h * (1.0 - params.eta) + params.rho * p_h * r_h
            m_l = m_l * (1.0 - params.eta) + params.rho * p_l * r_l
        elif cell.memory == "traffic":
            m_h = m_h * (1.0 - params.eta) + params.rho * p_h
            m_l = m_l * (1.0 - params.eta) + params.rho * p_l
        elif cell.memory == "adaptive_eta":
            surprise_h = p_h * abs(r_h - rhat_h)
            surprise_l = p_l * abs(r_l - rhat_l)
            eta_h = min(params.eta + cell.adaptive_gain * surprise_h, cell.adaptive_eta_max)
            eta_l = min(params.eta + cell.adaptive_gain * surprise_l, cell.adaptive_eta_max)
            rhat_h += cell.reward_track_rate * p_h * (r_h - rhat_h)
            rhat_l += cell.reward_track_rate * p_l * (r_l - rhat_l)
            m_h = m_h * (1.0 - eta_h) + params.rho * p_h * r_h
            m_l = m_l * (1.0 - eta_l) + params.rho * p_l * r_l
        else:
            raise ValueError(f"unknown memory cell: {cell.memory}")

        # --- Axis 6: stock dynamics ----------------------------------------
        if cell.ecology in ("logistic", "shock"):
            growth = cell.logistic_growth * stock * (1.0 - stock / cell.logistic_capacity)
            consumption = cell.harvest * p_h * stock
            stock = max(0.0, stock + growth - consumption)
            if cell.ecology == "shock" and rng.random() < cell.shock_rate:
                factor = float(
                    np.exp(rng.normal(cell.shock_log_mean, cell.shock_log_sigma))
                )
                stock = min(cell.logistic_capacity, stock * min(1.0, factor))

    observed_recovery: int | None = None
    if onset_cycle is not None and recovery_cycle is not None:
        observed_recovery = recovery_cycle - onset_cycle

    classification = "no_lockin"
    if stale_duration > 0 and locked_at_end:
        classification = "unrecovered_stale_lockin"
    elif stale_duration > 0:
        classification = "recovered_stale_lockin"
    elif stale_eligible_cycles > 0 and (
        empty_rich_mass / stale_eligible_cycles >= 0.25
    ):
        classification = "diffuse_empty_drag"

    return {
        "cell": cell.label(),
        "rho": params.rho,
        "eta": params.eta,
        "epsilon": params.epsilon,
        "memory_ratio": float(params.rho / max(params.eta, 1e-12)),
        "stale_lockin_duration": int(stale_duration),
        "lockin_episodes": int(episodes),
        "observed_recovery_time": observed_recovery,
        "delta_memory_at_onset": delta_at_onset,
        "post_depletion_empty_rate": (
            float(empty_rich_mass / stale_eligible_cycles)
            if stale_eligible_cycles
            else 0.0
        ),
        "stale_eligible_cycles": int(stale_eligible_cycles),
        "throughput": float(throughput),
        "final_memory_gap": float(m_h - m_l),
        "classification": classification,
    }


def two_route_grid(quick: bool = False) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Same rho/eta/epsilon grids as two_route_memory_theorem.sweep_grid."""
    if quick:
        return (0.12, 0.45, 0.90), (0.01, 0.04, 0.12), (0.02, 0.10)
    return (
        (0.08, 0.12, 0.24, 0.45, 0.70, 0.90),
        (0.005, 0.01, 0.02, 0.04, 0.08, 0.12),
        (0.0, 0.02, 0.06, 0.10, 0.18),
    )


def sweep_two_route_cell(
    cell: TwoRouteCell, quick: bool = False, cycles: int = 120, deplete_cycle: int = 36
) -> dict[str, Any]:
    """Full diagnostic battery item 2 for one Settings cell."""
    rhos, etas, epsilons = two_route_grid(quick)
    rows = []
    for rho in rhos:
        for eta in etas:
            for epsilon in epsilons:
                params = TwoRouteParams(
                    cycles=cycles,
                    deplete_cycle=deplete_cycle,
                    rho=rho,
                    eta=eta,
                    epsilon=epsilon,
                )
                rows.append(simulate_two_route_cell(params, cell))

    lockin_rows = [r for r in rows if r["stale_lockin_duration"] > 0]
    unrecovered = [r for r in rows if r["classification"] == "unrecovered_stale_lockin"]
    class_counts: dict[str, int] = {}
    for r in rows:
        class_counts[r["classification"]] = class_counts.get(r["classification"], 0) + 1
    max_lock = max(
        lockin_rows, key=lambda r: r["stale_lockin_duration"], default=None
    )
    return {
        "cell": cell.label(),
        "choice_point_pass": two_route_choice_point(cell, TwoRouteParams()),
        "runs": len(rows),
        "classification_counts": class_counts,
        "lockin_cells": len(lockin_rows),
        "first_lockin_memory_ratio": min(
            (r["memory_ratio"] for r in lockin_rows), default=None
        ),
        "first_unrecovered_memory_ratio": min(
            (r["memory_ratio"] for r in unrecovered), default=None
        ),
        "max_lockin_duration": (
            None
            if max_lock is None
            else {
                "rho": max_lock["rho"],
                "eta": max_lock["eta"],
                "epsilon": max_lock["epsilon"],
                "duration": max_lock["stale_lockin_duration"],
                "recovery": max_lock["observed_recovery_time"],
                "empty_rate": max_lock["post_depletion_empty_rate"],
            }
        ),
        "mean_throughput": float(np.mean([r["throughput"] for r in rows])),
        "rows": rows,
    }


# ===========================================================================
# Diagnostic battery items 1 and 4 on the kernel — choice-point invariance
# and entropy / mixing-time profiles
# ===========================================================================

def _synthetic_outdegree1_topology(stretch: float = 1.0) -> Topology:
    """3-node directed cycle: every node has outdegree exactly 1."""
    rng = np.random.default_rng(7)
    nodes = {
        "A": rng.random(3),
        "B": rng.random(3),
        "C": rng.random(3),
    }
    return topology_from_edges(
        nodes=nodes,
        edges=[("A", "B", 5.0 * stretch), ("B", "C", 1.0), ("C", "A", 2.0)],
        undirected=False,
    )


def kernel_choice_point_check(
    settings: Settings, stretch: float = 1.0, draws: int = 25
) -> dict[str, Any]:
    """
    Battery item 1: at every outdegree-1 node, P on the single edge must be
    exactly 1 regardless of W. Run over random telemetry/bias draws.
    """
    topo = _synthetic_outdegree1_topology(stretch=stretch)
    rng = np.random.default_rng(11)
    kernel = SettingsKernel(
        topology=topo,
        settings=settings,
        alpha=1.0,
        beta=np.full((3, 3), 2.0),
        node_bias=rng.normal(0.0, 0.5, size=3),
    )
    edges = [(0, 1), (1, 2), (2, 0)]
    worst = 1.0
    for _ in range(draws):
        telemetry = rng.normal(0.0, 1.0, size=3)
        P = kernel.transition_matrix(telemetry)
        for i, j in edges:
            worst = min(worst, float(P[i, j]))
    return {
        "settings": settings.label(),
        "stretch": stretch,
        "min_edge_probability": worst,
        "passes": bool(worst > 1.0 - 1e-9),
        "degeneracy": dict(kernel.degeneracy),
    }


def kernel_entropy_profile(settings: Settings) -> dict[str, Any]:
    """
    Battery item 4: mean row entropy and mixing-time estimate on the ant
    topology for the normalized mean intent telemetry. Cheap triage statistic.
    """
    topo = ant_topology()
    kernel = SettingsKernel(
        topology=topo,
        settings=settings,
        alpha=1.25,
        beta=np.full((topo.N, topo.N), 0.85),
        temperature=0.95,
        feedback_noise=0.0,
    )
    mean_intent = _normalize(
        np.mean([_normalize(np.array(v, dtype=float)) for v in INTENTS.values()], axis=0)
    )
    H = kernel.transition_entropy(mean_intent)
    mixing = kernel.mixing_time_estimate(mean_intent)
    out_rows = topo.adjacency_mask.any(axis=1)
    return {
        "settings": settings.label(),
        "mean_row_entropy_bits": float(np.mean(H[out_rows])),
        "max_row_entropy_bits": float(np.max(H[out_rows])),
        "mixing_time_estimate": float(mixing) if np.isfinite(mixing) else None,
        "degeneracy": dict(kernel.degeneracy),
    }


def sanitize(obj: Any) -> Any:
    """Make numpy types JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

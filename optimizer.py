from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Any, Literal

import numpy as np

from kernel import DynamicTopologyKernel

# ── Per-mode recommended learning rates (empirically calibrated) ───────────────
# Gradient magnitudes span ~4 orders of magnitude across modes.
# These defaults give each mode an equally aggressive effective step.
MODE_DEFAULT_ETA: dict[str, float] = {
    "ENTROPY_PI":       0.50,
    "ROW_ENTROPY":      0.30,
    "DETAILED_BALANCE": 0.80,   # KL flux formulation, still needs high η
    "SPECTRAL_GAP":     0.02,   # noisy landscape — keep step small
    "WEIGHT_SYMMETRY":  0.05,   # original default is correct
    "COMPOSITE":        0.20,
}


class SymmetryMode(str, Enum):
    ENTROPY_PI = "ENTROPY_PI"
    ROW_ENTROPY = "ROW_ENTROPY"
    DETAILED_BALANCE = "DETAILED_BALANCE"
    SPECTRAL_GAP = "SPECTRAL_GAP"
    WEIGHT_SYMMETRY = "WEIGHT_SYMMETRY"
    COMPOSITE = "COMPOSITE"


class SymmetryOptimizer:
    """
    Finite-difference optimizer over the kernel beta tensor.

    The optimizer intentionally treats DynamicTopologyKernel as a black box:
    it perturbs beta, asks the kernel for diagnostics, and restores state.
    """

    def __init__(
        self,
        kernel: DynamicTopologyKernel,
        mode: str | SymmetryMode = SymmetryMode.ENTROPY_PI,
        eta: float | None = None,          # None → use MODE_DEFAULT_ETA
        eps: float = 1e-3,
        beta_max: float = 20.0,
        noise_sigma: float = 0.0,
        telemetry: np.ndarray | None = None,
        target_pi: np.ndarray | None = None,
        composite_lambda: float = 0.5,
        composite_utility_type: Literal["l2", "kl"] = "kl",
        normalize_gradient: bool = False,
        converge_eps: float = 1e-4,
        converge_ticks: int = 50,
        rng: np.random.Generator | None = None,
    ):
        self.kernel = kernel
        self.mode = SymmetryMode(mode)
        self.eta = float(eta) if eta is not None else MODE_DEFAULT_ETA[self.mode.value]
        self.eps = max(float(eps), 1e-8)
        self.beta_max = max(float(beta_max), 0.0)
        self.noise_sigma = max(float(noise_sigma), 0.0)
        self.telemetry = self._normalize_telemetry(telemetry)
        self.target_pi = self._normalize_distribution(target_pi)
        self.composite_lambda = min(max(float(composite_lambda), 0.0), 1.0)
        self.composite_utility_type: Literal["l2", "kl"] = composite_utility_type
        self.normalize_gradient = bool(normalize_gradient)
        self.converge_eps = max(float(converge_eps), 0.0)
        self.converge_ticks = max(int(converge_ticks), 1)
        self.tick = 0
        self.paused = False
        self.converged = False
        self._stable_ticks = 0
        self._history = {
            "tick": deque(maxlen=500),
            "sigma": deque(maxlen=500),
            "grad_norm": deque(maxlen=500),
        }
        # Rolling window for health metric (last 20 sigma values)
        self._sigma_window: deque[float] = deque(maxlen=20)
        self._rng = rng or np.random.default_rng()
        self._feasibility_cache: tuple[tuple, dict[str, Any] | None] | None = None

    @property
    def edge_mask(self) -> np.ndarray:
        return self.kernel.topo.adjacency_mask

    def configure(
        self,
        mode: str | None = None,
        eta: float | None = None,
        eps: float | None = None,
        beta_max: float | None = None,
        noise_sigma: float | None = None,
        telemetry: list[float] | np.ndarray | None = None,
        target_pi: list[float] | np.ndarray | None = None,
        composite_lambda: float | None = None,
        composite_utility_type: str | None = None,
        normalize_gradient: bool | None = None,
        paused: bool | None = None,
    ) -> None:
        if mode is not None:
            next_mode = SymmetryMode(mode)
            if next_mode != self.mode:
                self._reset_convergence()
                self._clear_history()
                self._sigma_window.clear()
                # When mode changes without explicit η, adopt mode default
                if eta is None:
                    self.eta = MODE_DEFAULT_ETA[next_mode.value]
            self.mode = next_mode
        if eta is not None:
            self.eta = float(eta)
        if eps is not None:
            self.eps = max(float(eps), 1e-8)
        if beta_max is not None:
            self.beta_max = max(float(beta_max), 0.0)
            self._feasibility_cache = None
        if noise_sigma is not None:
            self.noise_sigma = max(float(noise_sigma), 0.0)
        if telemetry is not None:
            self.telemetry = self._normalize_telemetry(np.asarray(telemetry, dtype=np.float64))
            self._feasibility_cache = None
        if target_pi is not None:
            self.target_pi = self._normalize_distribution(np.asarray(target_pi, dtype=np.float64))
            self._feasibility_cache = None
        if composite_lambda is not None:
            self.composite_lambda = min(max(float(composite_lambda), 0.0), 1.0)
        if composite_utility_type is not None:
            if composite_utility_type in ("l2", "kl"):
                self.composite_utility_type = composite_utility_type  # type: ignore[assignment]
                self._feasibility_cache = None
        if normalize_gradient is not None:
            self.normalize_gradient = bool(normalize_gradient)
        if paused is not None:
            self.paused = bool(paused)
            if not self.paused:
                self.converged = False

    def reset_beta(self, beta_init: float = 3.0) -> None:
        beta = np.zeros_like(self.kernel._beta)
        beta[self.edge_mask] = float(beta_init)
        self.kernel._beta = beta
        self.kernel._beta_baseline = beta.copy()
        self.tick = 0
        self.paused = False
        self._reset_convergence()
        self._clear_history()
        self._feasibility_cache = None

    def step(self) -> dict[str, Any]:
        if self.paused or self.converged:
            return self.snapshot(grad_norm=0.0)

        gradient = self._gradient()
        grad_norm = float(np.linalg.norm(gradient[self.edge_mask]))

        # Optional: normalize gradient so η is a pure step-fraction
        if self.normalize_gradient and grad_norm > 1e-12:
            gradient = gradient / grad_norm

        update = self.eta * gradient
        if self.noise_sigma > 0:
            noise = self.noise_sigma * self._rng.normal(size=update.shape)
            noise[~self.edge_mask] = 0.0
            update += noise

        beta = self.kernel._beta.copy()
        beta[self.edge_mask] = beta[self.edge_mask] + update[self.edge_mask]
        beta = np.clip(beta, 0.0, self.beta_max)
        beta[~self.edge_mask] = 0.0
        self.kernel._beta = beta

        self.tick += 1
        if grad_norm < self.converge_eps:
            self._stable_ticks += 1
            if self._stable_ticks >= self.converge_ticks:
                self.converged = True
                self.paused = True
        else:
            self._stable_ticks = 0

        return self.snapshot(grad_norm=grad_norm)

    def snapshot(self, grad_norm: float | None = None) -> dict[str, Any]:
        sigma = self._eval_sigma(self.kernel._beta)
        diag = self.kernel.get_diagnostic(self.telemetry)
        pi = self._stationary_distribution()
        mixing = diag["mixing_time"]
        grad = 0.0 if grad_norm is None else float(grad_norm)

        if self.tick == 0 or not self._history["tick"] or self._history["tick"][-1] != self.tick:
            self._history["tick"].append(int(self.tick))
            self._history["sigma"].append(float(sigma))
            self._history["grad_norm"].append(float(grad))
            self._sigma_window.append(float(sigma))

        health, health_msg = self._compute_health()
        target_feasibility = self._composite_feasibility()

        return {
            "tick": int(self.tick),
            "beta": self.kernel._beta.tolist(),
            "sigma_value": float(sigma),
            "grad_norm": float(grad),
            "pi": pi.tolist(),
            "row_entropy": diag["row_entropy"].tolist(),
            "mixing_time": float(mixing) if not np.isinf(mixing) else -1.0,
            "mode": self.mode.value,
            "converged": bool(self.converged),
            "paused": bool(self.paused),
            "history": self.history(),
            "eigenvalues": self._eigenvalues().tolist(),
            "entropy_production": float(diag["entropy_production"]),
            "irreversible_flux": float(diag["irreversible_flux"]),
            "edge_current": diag["edge_current"].tolist(),
            # Enhancement additions
            "health": health,
            "health_msg": health_msg,
            "recommended_eta": MODE_DEFAULT_ETA[self.mode.value],
            "normalize_gradient": self.normalize_gradient,
            "composite_utility_type": self.composite_utility_type,
            "target_feasibility": target_feasibility,
        }

    def history(self) -> dict[str, list[float]]:
        return {key: list(values) for key, values in self._history.items()}

    def _eval_sigma(self, beta_perturbed: np.ndarray) -> float:
        old_beta = self.kernel._beta
        self.kernel._beta = beta_perturbed
        try:
            if self.mode == SymmetryMode.ENTROPY_PI:
                return self._sigma_entropy_pi()
            if self.mode == SymmetryMode.ROW_ENTROPY:
                return self._sigma_row_entropy()
            if self.mode == SymmetryMode.DETAILED_BALANCE:
                return self._sigma_detailed_balance()
            if self.mode == SymmetryMode.SPECTRAL_GAP:
                return self._sigma_spectral_gap()
            if self.mode == SymmetryMode.WEIGHT_SYMMETRY:
                return self._sigma_weight_symmetry(beta_perturbed)
            if self.mode == SymmetryMode.COMPOSITE:
                symmetry = self._sigma_entropy_pi()
                utility = self._sigma_target_pi()
                return self.composite_lambda * symmetry + (1.0 - self.composite_lambda) * utility
            raise ValueError(f"Unsupported mode: {self.mode}")
        finally:
            self.kernel._beta = old_beta

    def _gradient(self) -> np.ndarray:
        if self.mode == SymmetryMode.ENTROPY_PI:
            return self._gradient_entropy_pi()
        if self.mode == SymmetryMode.COMPOSITE:
            sym_grad = self._normalize_edge_gradient(self._gradient_entropy_pi())
            utility_grad = self._normalize_edge_gradient(self._gradient_target_pi())
            grad = (
                self.composite_lambda * sym_grad
                + (1.0 - self.composite_lambda) * utility_grad
            )
            grad[~self.edge_mask] = 0.0
            return grad

        beta = self.kernel._beta.copy()
        grad = np.zeros_like(beta)
        for i, j in np.argwhere(self.edge_mask):
            plus = beta.copy()
            minus = beta.copy()
            plus[i, j] = min(self.beta_max, plus[i, j] + self.eps)
            minus[i, j] = max(0.0, minus[i, j] - self.eps)
            actual_eps = plus[i, j] - minus[i, j]
            if actual_eps <= 0:
                continue
            grad[i, j] = (self._eval_sigma(plus) - self._eval_sigma(minus)) / actual_eps
        grad[~self.edge_mask] = 0.0
        return grad

    def _sigma_entropy_pi(self) -> float:
        pi = self._stationary_distribution()
        p = pi[pi > 0]
        return float(-np.sum(p * np.log(p)))

    def _sigma_row_entropy(self) -> float:
        entropy = self.kernel.get_diagnostic(self.telemetry)["row_entropy"]
        connected = self.edge_mask.any(axis=1)
        values = entropy[connected] if np.any(connected) else entropy
        return float(-np.var(values))

    def _sigma_detailed_balance(self) -> float:
        """KL flux formulation: -Σ_{i,j} π_i P_ij log(π_i P_ij / π_j P_ji).

        This replaces the old L2 squared-difference which produced O(1e-4)
        values that vanished below the finite-difference epsilon.
        KL flux scales as O(1) and gives the gradient estimator a real signal.
        """
        p  = self.kernel.transition_matrix(self.telemetry)
        pi = self._stationary_distribution()
        eps = 1e-12
        fwd = pi[:, np.newaxis] * p          # forward flux  π_i P_ij
        rev = (pi[:, np.newaxis] * p).T      # reverse flux  π_j P_ji
        # Only sum over edges that have non-trivial flux in both directions
        mask = (fwd > eps) & (rev > eps)
        kl = np.sum(fwd[mask] * np.log(fwd[mask] / rev[mask]))
        return float(-kl)                    # negate: maximise reversibility

    def _sigma_spectral_gap(self) -> float:
        p = self.kernel.transition_matrix(self.telemetry)
        pi = self._stationary_distribution()
        centered = p - np.ones((self.kernel.topo.N, 1)) @ pi[np.newaxis, :]
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if len(singular_values) == 0:
            return 0.0
        smooth = 0.05
        max_s = float(np.max(singular_values))
        soft_radius = max_s + smooth * float(
            np.log(np.sum(np.exp((singular_values - max_s) / smooth)))
        )
        return float(1.0 - soft_radius)

    def _sigma_weight_symmetry(self, beta: np.ndarray) -> float:
        return float(-np.linalg.norm(beta - beta.T))

    def _sigma_target_pi(self) -> float:
        """Utility: how close is π to the target distribution?

        l2  mode: -N² Σ(π_i - target_i)²     (scaled L2, O(log N) magnitude)
        kl  mode: -KL(target ‖ π)             (information-theoretic, natural scale)
        """
        pi = self._stationary_distribution()
        target = self.target_pi
        if target is None:
            target = np.full(self.kernel.topo.N, 1.0 / self.kernel.topo.N)
        N = self.kernel.topo.N
        if self.composite_utility_type == "kl":
            eps = 1e-12
            t = np.clip(target, eps, None)
            p = np.clip(pi, eps, None)
            return float(-np.sum(t * np.log(t / p)))
        else:  # l2
            return float(-N ** 2 * np.sum((pi - target) ** 2))

    def _gradient_entropy_pi(self) -> np.ndarray:
        pi = self._stationary_distribution()
        eps = 1e-12
        value_grad = -np.log(np.clip(pi, eps, None)) - 1.0
        return self._stationary_value_gradient(value_grad)

    def _gradient_target_pi(self) -> np.ndarray:
        pi = self._stationary_distribution()
        target = self.target_pi
        if target is None:
            target = np.full(self.kernel.topo.N, 1.0 / self.kernel.topo.N)

        eps = 1e-12
        if self.composite_utility_type == "kl":
            value_grad = np.clip(target, eps, None) / np.clip(pi, eps, None)
        else:
            N = self.kernel.topo.N
            value_grad = -2.0 * (N ** 2) * (pi - target)
        return self._stationary_value_gradient(value_grad)

    def _stationary_value_gradient(self, value_grad: np.ndarray) -> np.ndarray:
        """Analytic beta gradient for frozen-telemetry stationary objectives."""
        P = self.kernel.transition_matrix(self.telemetry).copy()
        row_sums = P.sum(axis=1)
        dangling = row_sums < 1e-12
        if np.any(dangling):
            P[dangling, :] = 0.0
            P[dangling, dangling] = 1.0

        pi = self._stationary_distribution()
        n = self.kernel.topo.N
        fundamental = np.eye(n) - P + np.ones((n, 1)) @ pi[np.newaxis, :]
        try:
            z = np.linalg.solve(fundamental, value_grad)
        except np.linalg.LinAlgError:
            z = np.linalg.pinv(fundamental) @ value_grad

        expected_z = P @ z
        coeff = self._beta_logit_derivative()
        grad = np.zeros_like(self.kernel._beta)
        grad[self.edge_mask] = (
            pi[:, np.newaxis]
            * P
            * coeff
            * (z[np.newaxis, :] - expected_z[:, np.newaxis])
        )[self.edge_mask]
        grad[~self.edge_mask] = 0.0
        return grad

    def _beta_logit_derivative(self) -> np.ndarray:
        """Return d(-W/tau)/d beta_ij for the active telemetry vector."""
        features = self.kernel.topo.node_features
        alignment = features @ self.telemetry + self.kernel._node_bias
        W_raw = (
            (self.kernel.alpha * self.kernel.topo.distance_matrix)
            - (self.kernel._beta * alignment[np.newaxis, :])
            - self.kernel._sponsor_friction
        )

        floor = self.kernel._weight_floor
        sharpness = self.kernel._floor_sharpness
        delta = sharpness * (W_raw - floor)
        floor_derivative = np.ones_like(W_raw)
        finite = np.isfinite(delta)
        stable = finite & (delta <= 20.0)
        with np.errstate(over="ignore", invalid="ignore"):
            floor_derivative[stable] = 1.0 / (1.0 + np.exp(-delta[stable]))

        tau = self.kernel._resolve_temperature(None)
        coeff = alignment[np.newaxis, :] * floor_derivative / tau
        coeff[~self.edge_mask] = 0.0
        return coeff

    def _normalize_edge_gradient(self, gradient: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(gradient[self.edge_mask]))
        if norm <= 1e-12:
            return gradient
        out = gradient / norm
        out[~self.edge_mask] = 0.0
        return out

    def _eigenvalues(self) -> np.ndarray:
        vals = np.linalg.eigvals(self.kernel.transition_matrix(self.telemetry))
        return np.sort(np.abs(vals))[::-1]

    def _stationary_distribution(self) -> np.ndarray:
        try:
            return self.kernel.stationary_distribution_direct(self.telemetry)
        except AttributeError:
            return self.kernel.stationary_distribution(self.telemetry)

    def _composite_feasibility(self) -> dict[str, Any] | None:
        if self.mode != SymmetryMode.COMPOSITE or self.target_pi is None:
            return None

        key = (
            tuple(np.round(self.target_pi, 8)),
            tuple(np.round(self.telemetry, 8)),
            round(self.beta_max, 8),
            self.composite_utility_type,
            tuple(map(tuple, self.edge_mask.astype(int))),
        )
        if self._feasibility_cache is not None and self._feasibility_cache[0] == key:
            return self._feasibility_cache[1]

        old_beta = self.kernel._beta.copy()
        starts = [np.clip(old_beta.copy(), 0.0, self.beta_max)]
        target_start = np.zeros_like(old_beta)
        target_columns = np.tile(self.beta_max * self.target_pi, (self.kernel.topo.N, 1))
        target_start[self.edge_mask] = target_columns[self.edge_mask]
        starts.append(target_start)

        target = self.target_pi
        initial_pi = self._stationary_distribution()
        initial_l1 = float(np.sum(np.abs(initial_pi - target)))
        best_pi = initial_pi
        best_utility = self._sigma_target_pi()

        try:
            for start in starts:
                beta = start.copy()
                beta[~self.edge_mask] = 0.0
                self.kernel._beta = beta
                for _ in range(80):
                    grad = self._normalize_edge_gradient(self._gradient_target_pi())
                    beta = self.kernel._beta.copy()
                    beta[self.edge_mask] = beta[self.edge_mask] + 0.35 * grad[self.edge_mask]
                    beta = np.clip(beta, 0.0, self.beta_max)
                    beta[~self.edge_mask] = 0.0
                    self.kernel._beta = beta
                utility = self._sigma_target_pi()
                if utility > best_utility:
                    best_utility = utility
                    best_pi = self._stationary_distribution()
        finally:
            self.kernel._beta = old_beta

        l1_error = float(np.sum(np.abs(best_pi - target)))
        max_abs_error = float(np.max(np.abs(best_pi - target)))
        improvement = float(initial_l1 - l1_error)
        target_mass = float(np.dot(best_pi, target > (1.0 / max(self.kernel.topo.N, 1))))

        if l1_error <= 0.10 and max_abs_error <= 0.05:
            status = "REACHABLE"
            message = "Target appears reachable under current topology and beta cap."
        elif improvement > 0.05 or l1_error <= 0.35:
            status = "PARTIAL"
            message = "Target is partially reachable; topology or beta cap still constrains it."
        else:
            status = "CONSTRAINED"
            message = "Target appears constrained by topology, features, or beta cap."

        result: dict[str, Any] = {
            "status": status,
            "message": message,
            "estimated": True,
            "l1_error": l1_error,
            "max_abs_error": max_abs_error,
            "initial_l1_error": initial_l1,
            "improvement": improvement,
            "achieved_pi": best_pi.tolist(),
            "target_focus_mass": target_mass,
            "probe_steps": 80,
        }
        self._feasibility_cache = (key, result)
        return result

    def _normalize_telemetry(self, telemetry: np.ndarray | None) -> np.ndarray:
        if telemetry is None:
            telemetry = np.full(self.kernel.topo.F, 1.0 / self.kernel.topo.F)
        telemetry = np.asarray(telemetry, dtype=np.float64)
        if telemetry.shape != (self.kernel.topo.F,):
            telemetry = np.full(self.kernel.topo.F, 1.0 / self.kernel.topo.F)
        norm = np.linalg.norm(telemetry)
        return telemetry / norm if norm > 0 else telemetry

    def _normalize_distribution(self, values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.kernel.topo.N,):
            return None
        values = np.clip(values, 0.0, None)
        total = values.sum()
        if total <= 0:
            return np.full(self.kernel.topo.N, 1.0 / self.kernel.topo.N)
        return values / total

    def _compute_health(self) -> tuple[str, str]:
        """Derive a health badge from the rolling sigma window.

        Returns (health_level, message) where health_level is one of:
          HEALTHY  — optimizer making meaningful progress
          SLOW     — optimizer moving but sluggishly
          STALLED  — near-zero sigma gain; likely at attractor or gradient dead-zone
        """
        w = list(self._sigma_window)
        if len(w) < 4:
            return "HEALTHY", "Warming up..."

        diffs = np.diff(w)
        rate = float(np.mean(np.abs(diffs)))   # mean |ΔΣ| per tick

        if rate > 5e-4:
            return "HEALTHY", ""
        if rate > 5e-5:
            return "SLOW", "Optimizer is making slow progress."

        # Stalled — emit mode-specific guidance
        suggestions = {
            SymmetryMode.DETAILED_BALANCE: (
                "STALLED",
                "Try WEIGHT_SYMMETRY first (150 steps), then switch back to DETAILED_BALANCE."
            ),
            SymmetryMode.ENTROPY_PI: (
                "STALLED",
                "Near entropy maximum. Increase \u03b7 or pre-condition with WEIGHT_SYMMETRY."
            ),
            SymmetryMode.ROW_ENTROPY: (
                "STALLED",
                "Row entropy variance is already low. Try ENTROPY_PI for global equalization."
            ),
            SymmetryMode.SPECTRAL_GAP: (
                "STALLED",
                "Spectral gap near ceiling. Decrease \u03b7 to 0.01 to avoid oscillation."
            ),
            SymmetryMode.WEIGHT_SYMMETRY: (
                "STALLED",
                "Weight matrix is symmetric. Switch to ENTROPY_PI to continue optimization."
            ),
            SymmetryMode.COMPOSITE: (
                "STALLED",
                "Composite objective is balanced. Try adjusting \u03bb or the utility type."
            ),
        }
        return suggestions.get(self.mode, ("STALLED", "Optimizer stalled."))

    def compute_gradient_matrix(self) -> list[list[float]]:
        """Return the full N\u00d7N gradient matrix for the current mode.
        Expensive (2\u00d7|E| kernel calls) — call on-demand, not every tick.
        """
        return self._gradient().tolist()

    def _reset_convergence(self) -> None:
        self.converged = False
        self._stable_ticks = 0

    def _clear_history(self) -> None:
        for values in self._history.values():
            values.clear()

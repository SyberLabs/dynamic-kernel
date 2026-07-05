from collections import deque

import numpy as np
from typing import List, Tuple, Dict, Optional
from kernel import DynamicTopologyKernel


class PopulationSimulator:
    """
    A continuous stateful wrapper around the DynamicTopologyKernel.
    Tracks K agents with realistic transit delays and dwell times.

    Phase 3 additions:
    - Cumulative visit tracking (total_node_visits, total_edge_crossings)
    - tick_count for ROI and rate calculations
    - get_report_snapshot() for the export endpoint
    - reset_history() to clear accumulated stats without rebuilding

    BUG-5 FIX (directed topology deadlocks):
    - Agents that reach a sink node (no outbound edges) are teleported back to
      node 0 (the canonical entry point) after a brief dwell. This prevents
      100% stationary inflation and keeps all topologies alive.
    """

    def __init__(
        self,
        kernel: DynamicTopologyKernel,
        K: int,
        time_multiplier: float = 5.0,
        node_rewards: Optional[np.ndarray] = None,
        dwell_range: Tuple[int, int] = (20, 60),
        sink_dwell_range: Tuple[int, int] = (10, 25),
        rng: Optional[np.random.Generator] = None,
    ):
        self.kernel = kernel
        self.K = K
        self.time_multiplier = time_multiplier
        self.dwell_range = dwell_range
        self.sink_dwell_range = sink_dwell_range
        # None -> legacy global np.random (seed-compatible); Generator -> isolated
        self._rng = rng

        # 0 = At Node, 1 = In Transit
        self.state = np.zeros(K, dtype=int)

        # Initialize everyone at the Entrance (node 0)
        self.current_node    = np.zeros(K, dtype=int)
        self.target_node     = np.zeros(K, dtype=int)
        self.ticks_remaining = np.zeros(K, dtype=int)

        self.telemetries = np.zeros((K, self.kernel.topo.F), dtype=np.float64)
        self.telemetries[:, 0] = 1.0  # default to feature 0
        self.node_rewards: Optional[np.ndarray] = None
        self.last_memory_update: Optional[dict] = None
        self.last_opportunity_cost: Optional[dict] = None
        if node_rewards is not None:
            self.set_node_rewards(node_rewards)

        # ── Pre-compute sink mask for BUG-5 fix ───────────────────────────
        # A sink node has no outbound edges in the adjacency mask.
        adj = self.kernel.topo.adjacency_mask        # (N, N) bool
        self._sink_mask = ~adj.any(axis=1)           # (N,) True where node is a sink

        # ── Cumulative analytics (Phase 3) ─────────────────────────────────
        N = self.kernel.topo.N
        self.total_node_visits    = np.zeros(N, dtype=np.int64)
        self.total_edge_crossings = np.zeros((N, N), dtype=np.int64)
        self.tick_count = 0
        self._metric_history_maxlen = 2048
        self._metric_history = {
            "tick": deque(maxlen=self._metric_history_maxlen),
            "mean_entropy": deque(maxlen=self._metric_history_maxlen),
            "mixing_time": deque(maxlen=self._metric_history_maxlen),
            "active_transit_pct": deque(maxlen=self._metric_history_maxlen),
            "edge_current_norm": deque(maxlen=self._metric_history_maxlen),
            "entropy_production": deque(maxlen=self._metric_history_maxlen),
            "mean_opportunity_cost": deque(maxlen=self._metric_history_maxlen),
            "stale_flow_share": deque(maxlen=self._metric_history_maxlen),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Random source — instance Generator when injected, legacy global otherwise
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_randint(self, lo: int, hi: int, size: int) -> np.ndarray:
        if self._rng is not None:
            return self._rng.integers(lo, hi, size=size)
        return np.random.randint(lo, hi, size=size)

    def _draw_uniform(self, shape: Tuple[int, ...]) -> np.ndarray:
        if self._rng is not None:
            return self._rng.random(shape)
        return np.random.random(shape)

    # ─────────────────────────────────────────────────────────────────────────
    # Population setup
    # ─────────────────────────────────────────────────────────────────────────

    def set_population(self, groups: List[Tuple[float, List[float]]]):
        """
        Set agent distributions.
        groups: [(0.4, [1,0,0]), (0.6, [0,1,0])] → 40% intent A, 60% intent B

        Empty groups list is a no-op (BUG-2 safety guard).
        """
        if not groups:
            return
        start_idx = 0
        for pct, raw_tel in groups:
            count = int(self.K * pct)
            if count <= 0:
                continue
            end_idx = min(start_idx + count, self.K)
            tel = np.array(raw_tel, dtype=np.float64)
            norm = np.linalg.norm(tel)
            if norm > 0:
                tel /= norm
            self.telemetries[start_idx:end_idx] = tel
            self.state[start_idx:end_idx] = 0
            self.current_node[start_idx:end_idx] = 0
            self.ticks_remaining[start_idx:end_idx] = 0
            start_idx = end_idx

    def reset_history(self):
        """Clear cumulative analytics without reconstructing agent state."""
        self.total_node_visits[:] = 0
        self.total_edge_crossings[:] = 0
        self.tick_count = 0
        for values in self._metric_history.values():
            values.clear()

    def set_node_rewards(self, rewards: np.ndarray) -> None:
        """
        Set per-node realized rewards used by the kernel memory law.

        Reward-gated and adaptive-eta memory laws require these values. They
        may be updated by a domain adapter before each tick when ecology is
        nonstationary.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        if rewards.shape != (self.kernel.topo.N,):
            raise ValueError(
                f"rewards must have shape ({self.kernel.topo.N},). "
                f"Got {rewards.shape}"
            )
        if np.any(~np.isfinite(rewards)):
            raise ValueError("rewards must be finite")
        self.node_rewards = rewards.copy()

    def _apply_memory_law(self, completed_traffic: np.ndarray) -> None:
        state = self.kernel.memory_law_state()
        mode = state.get("mode", "static")
        if mode == "static":
            self.last_memory_update = {"mode": "static"}
            self.last_opportunity_cost = None
            return
        if mode in ("reward_gated", "adaptive_eta"):
            if self.node_rewards is None:
                raise ValueError(
                    f"memory law {mode!r} requires simulator.node_rewards"
                )
            self.last_memory_update = self.kernel.memory_law_step(
                completed_traffic, node_reward=self.node_rewards
            )
            opp = self.kernel.memory_law_state().get("last_opportunity_diagnostic")
            self.last_opportunity_cost = opp
        else:
            self.last_memory_update = self.kernel.memory_law_step(completed_traffic)
            self.last_opportunity_cost = None

    def _memory_law_state_json(self) -> dict:
        state = self.kernel.memory_law_state()
        out = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif isinstance(value, dict):
                out[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                out[key] = value
        return out

    def _record_metrics(self, active_transit: int) -> None:
        diag = self._ensemble_diagnostics()
        mixing = diag["mixing_time"]
        mixing_value = -1.0 if np.isinf(mixing) else float(mixing)
        self._metric_history["tick"].append(int(self.tick_count))
        self._metric_history["mean_entropy"].append(float(diag["mean_entropy"]))
        self._metric_history["mixing_time"].append(mixing_value)
        self._metric_history["active_transit_pct"].append(float(active_transit / max(self.K, 1)))
        self._metric_history["edge_current_norm"].append(float(diag["edge_current_norm"]))
        self._metric_history["entropy_production"].append(float(diag["entropy_production"]))
        if self.last_opportunity_cost is None:
            self._metric_history["mean_opportunity_cost"].append(0.0)
            self._metric_history["stale_flow_share"].append(0.0)
        else:
            self._metric_history["mean_opportunity_cost"].append(
                float(self.last_opportunity_cost["mean_opportunity_cost"])
            )
            self._metric_history["stale_flow_share"].append(
                float(self.last_opportunity_cost["stale_flow_share"])
            )

    def get_metric_history(self) -> Dict[str, List[float]]:
        return {key: list(values) for key, values in self._metric_history.items()}

    def _ensemble_diagnostics(self) -> Dict[str, object]:
        """Diagnostics over the actual telemetry population, not mean telemetry."""
        N = self.kernel.topo.N
        if self.K <= 0:
            zeros = np.zeros((N, N), dtype=np.float64)
            return {
                "mean_entropy": 0.0,
                "mixing_time": np.inf,
                "expected_edge_flow": zeros,
                "expected_edge_current": zeros,
                "edge_current_norm": 0.0,
                "entropy_production": 0.0,
                "irreversible_flux": 0.0,
            }
        K = max(self.K, 1)
        P_all = self.kernel.transition_matrix_batch(self.telemetries, step=self.tick_count)

        agent_rows = P_all[np.arange(self.K), self.current_node, :]
        entropy_terms = np.zeros_like(agent_rows)
        positive = agent_rows > 0
        entropy_terms[positive] = -agent_rows[positive] * np.log2(agent_rows[positive])
        agent_entropy = entropy_terms.sum(axis=1)

        expected_edge_flow = np.zeros((N, N), dtype=np.float64)
        np.add.at(expected_edge_flow, self.current_node, agent_rows)
        expected_edge_flow /= K

        expected_edge_current = expected_edge_flow - expected_edge_flow.T
        edge_current_norm = float(np.linalg.norm(expected_edge_current, ord="fro"))

        eps = 1e-12
        reverse_flow = expected_edge_flow.T
        bidirectional = (expected_edge_flow > eps) & (reverse_flow > eps)
        entropy_production = 0.0
        if np.any(bidirectional):
            entropy_production = 0.5 * float(np.sum(
                expected_edge_flow[bidirectional]
                * np.log(expected_edge_flow[bidirectional] / reverse_flow[bidirectional])
            ))
        one_way = (expected_edge_flow > eps) & (reverse_flow <= eps)
        irreversible_flux = float(np.sum(expected_edge_flow[one_way]))

        avg_transition = P_all.mean(axis=0)
        eigenvalues = np.linalg.eigvals(avg_transition)
        mags = np.sort(np.abs(eigenvalues))[::-1]
        if len(mags) < 2 or 1.0 - mags[1] < 1e-12:
            mixing_time = np.inf
        else:
            mixing_time = float(1.0 / (1.0 - mags[1]))

        return {
            "mean_entropy": float(np.mean(agent_entropy)) if len(agent_entropy) else 0.0,
            "mixing_time": mixing_time,
            "expected_edge_flow": expected_edge_flow,
            "expected_edge_current": expected_edge_current,
            "edge_current_norm": edge_current_norm,
            "entropy_production": entropy_production,
            "irreversible_flux": irreversible_flux,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation tick
    # ─────────────────────────────────────────────────────────────────────────

    def tick(self) -> Dict[str, object]:
        """Advance the simulation by one discrete tick."""
        N = self.kernel.topo.N
        completed_traffic = np.zeros((N, N), dtype=np.float64)

        # ── 1. Process In-Transit agents ─────────────────────────────────────
        in_transit = (self.state == 1)
        if np.any(in_transit):
            self.ticks_remaining[in_transit] -= 1

            arrived = in_transit & (self.ticks_remaining <= 0)
            if np.any(arrived):
                self.state[arrived] = 0
                prev_nodes = self.current_node[arrived].copy()
                self.current_node[arrived] = self.target_node[arrived]

                # Record cumulative edge crossings and node arrivals
                arrived_idx = np.where(arrived)[0]
                np.add.at(self.total_edge_crossings,
                          (prev_nodes, self.current_node[arrived_idx]), 1)
                np.add.at(completed_traffic,
                          (prev_nodes, self.current_node[arrived_idx]), 1.0)
                np.add.at(self.total_node_visits, self.current_node[arrived_idx], 1)

                # Random dwell time 20–60 ticks
                self.ticks_remaining[arrived_idx] = self._draw_randint(
                    self.dwell_range[0], self.dwell_range[1], len(arrived_idx)
                )

                # Apply feedback on arrival
                arrived_features = self.kernel.topo.node_features[
                    self.current_node[arrived_idx]
                ]
                for i, idx in enumerate(arrived_idx):
                    self.telemetries[idx] = self.kernel._feedback_fn(
                        self.telemetries[idx], arrived_features[i]
                    )

        # Canonical V1 memory loop:
        # completed traversals -> traffic matrix -> reward observation -> memory update.
        self._apply_memory_law(completed_traffic)

        # ── 2. Process At-Node agents ─────────────────────────────────────────
        at_node = (self.state == 0)
        if np.any(at_node):
            self.ticks_remaining[at_node] -= 1

        ready_to_jump = at_node & (self.ticks_remaining <= 0)
        at_node_idx = np.where(ready_to_jump)[0]

        if len(at_node_idx) > 0:
            tels_batch = self.telemetries[at_node_idx]
            P_all = self.kernel.transition_matrix_batch(tels_batch, step=0)  # (M, N, N)

            current_nodes_batch = self.current_node[at_node_idx]
            rows = P_all[np.arange(len(at_node_idx)), current_nodes_batch, :]

            # Vectorized multinomial sampling
            cumsum = np.cumsum(rows, axis=1)
            u = self._draw_uniform((len(at_node_idx), 1))
            next_nodes = np.argmax(cumsum >= u, axis=1)

            # ── BUG-5 FIX: sink node recovery ────────────────────────────────
            # Agents sitting on a sink node have row_sum == 0 (no outbound
            # edges). Instead of freezing them in place, teleport them back to
            # node 0 (entry) so directed topologies stay dynamic.
            row_sums = rows.sum(axis=1)
            isolated = (row_sums < 1e-12)
            at_sink_nodes = self._sink_mask[current_nodes_batch]
            stuck = isolated & at_sink_nodes
            if np.any(stuck):
                # Reset to entry node; they'll dwell briefly then re-route
                next_nodes[stuck] = 0
                # Give them a short dwell before re-entering (10–25 ticks)
                stuck_global_idx = at_node_idx[stuck]
                self.ticks_remaining[stuck_global_idx] = self._draw_randint(
                    self.sink_dwell_range[0], self.sink_dwell_range[1], int(np.sum(stuck))
                )
            elif np.any(isolated):
                # Isolated but not a structural sink (e.g. self-loop only)
                # — just stay in place
                next_nodes[isolated] = current_nodes_batch[isolated]

            # Only move agents that actually chose a different node
            moving_mask = (next_nodes != current_nodes_batch)
            moving_agent_idx = at_node_idx[moving_mask]

            if len(moving_agent_idx) > 0:
                dest_nodes = next_nodes[moving_mask]
                self.state[moving_agent_idx] = 1
                self.target_node[moving_agent_idx] = dest_nodes

                distances = self.kernel.topo.distance_matrix[
                    self.current_node[moving_agent_idx], dest_nodes
                ]
                distances = np.where(np.isposinf(distances), 10.0, distances)
                self.ticks_remaining[moving_agent_idx] = np.maximum(
                    1, (distances * self.time_multiplier).astype(int)
                )

        # ── 3. Aggregate stats ────────────────────────────────────────────────
        node_counts = np.bincount(self.current_node[self.state == 0], minlength=N)

        edge_counts = np.zeros((N, N), dtype=int)
        in_transit_idx = np.where(self.state == 1)[0]
        if len(in_transit_idx) > 0:
            np.add.at(
                edge_counts,
                (self.current_node[in_transit_idx], self.target_node[in_transit_idx]),
                1
            )

        # Sponsorship decay is time semantics: exactly once per simulation
        # tick, regardless of how many agents moved. The simulator drives
        # transition_matrix_batch directly, so without this call configured
        # sponsor_decay would never take effect in live sessions.
        self.kernel.tick_decay()

        self.tick_count += 1
        self._record_metrics(len(in_transit_idx))

        return {
            "node_counts":       node_counts.tolist(),
            "edge_counts":       edge_counts.tolist(),
            "active_transit":    int(len(in_transit_idx)),
            "active_stationary": int(self.K - len(in_transit_idx)),
            "tick":              int(self.tick_count),
            "memory_update":     self.last_memory_update,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Report snapshot (Phase 3A)
    # ─────────────────────────────────────────────────────────────────────────

    def get_report_snapshot(self) -> Dict[str, object]:
        """
        Return a rich analytics snapshot for the export endpoint.
        """
        N = self.kernel.topo.N
        labels = self.kernel.topo.labels
        elapsed = max(self.tick_count, 1)

        visit_rates   = (self.total_node_visits / elapsed / self.K).tolist()
        node_counts   = np.bincount(self.current_node[self.state == 0], minlength=N)

        beta_inbound_mean     = self.kernel._beta.mean(axis=0)
        friction_inbound_mean = self.kernel._sponsor_friction.mean(axis=0)
        visits = self.total_node_visits

        beta_roi = np.where(
            beta_inbound_mean > 0,
            visits / (beta_inbound_mean * elapsed + 1e-9),
            0.0
        ).tolist()
        friction_roi = np.where(
            friction_inbound_mean > 0,
            visits / (friction_inbound_mean * elapsed + 1e-9),
            0.0
        ).tolist()

        mean_tel = self.telemetries.mean(axis=0).tolist()
        ensemble = self._ensemble_diagnostics()

        return {
            "topology": {
                "labels": labels,
                "N": N,
                "F": self.kernel.topo.F,
                "undirected": bool(np.allclose(
                    self.kernel.topo.distance_matrix,
                    self.kernel.topo.distance_matrix.T,
                    equal_nan=True
                )),
            },
            "simulation": {
                "K":               self.K,
                "tick_count":      self.tick_count,
                "active_transit":  int(np.sum(self.state == 1)),
                "active_stationary": int(np.sum(self.state == 0)),
                "mean_telemetry":  mean_tel,
            },
            "ensemble_diagnostics": {
                "mean_entropy": float(ensemble["mean_entropy"]),
                "mixing_time": -1.0 if np.isinf(ensemble["mixing_time"]) else float(ensemble["mixing_time"]),
                "edge_current_norm": float(ensemble["edge_current_norm"]),
                "entropy_production": float(ensemble["entropy_production"]),
                "irreversible_flux": float(ensemble["irreversible_flux"]),
                "expected_edge_flow": ensemble["expected_edge_flow"].tolist(),
                "expected_edge_current": ensemble["expected_edge_current"].tolist(),
            },
            "node_analytics": [
                {
                    "label":               labels[i],
                    "current_occupancy":   int(node_counts[i]),
                    "total_visits":        int(self.total_node_visits[i]),
                    "visit_rate_per_tick": round(visit_rates[i], 6),
                    "beta_roi":            round(beta_roi[i], 4),
                    "friction_roi":        round(friction_roi[i], 4),
                }
                for i in range(N)
            ],
            "edge_analytics": [
                {
                    "from":            labels[i],
                    "to":              labels[j],
                    "total_crossings": int(self.total_edge_crossings[i, j]),
                    "flow_per_tick":   round(float(self.total_edge_crossings[i, j]) / elapsed, 4),
                }
                for i in range(N) for j in range(N)
                if self.kernel.topo.adjacency_mask[i, j]
            ],
            "sponsorship": {
                "beta_tensor":     self.kernel._beta.tolist(),
                "friction_tensor": self.kernel._sponsor_friction.tolist(),
                "sponsor_decay":   self.kernel._sponsor_decay,
                "temperature":     self.kernel.temperature,
            },
            "memory_law": {
                "state": self._memory_law_state_json(),
                "last_update": self.last_memory_update,
                "node_rewards": None if self.node_rewards is None else self.node_rewards.tolist(),
            },
        }

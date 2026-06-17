import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from adapters import ADAPTERS, DomainAdapter
from kernel import DynamicTopologyKernel, topology_from_edges
from simulator import PopulationSimulator


DEFAULT_SESSION_ID = "default"


@dataclass
class SessionState:
    session_id: str
    kernel: DynamicTopologyKernel
    sim: PopulationSimulator
    adapter: DomainAdapter
    last_access: float
    optimizer: Any = None

    def touch(self) -> "SessionState":
        self.last_access = time.monotonic()
        return self


def build_from_adapter(adapter: DomainAdapter, session_id: str = DEFAULT_SESSION_ID) -> SessionState:
    kernel = adapter.build_kernel()
    sim = PopulationSimulator(kernel, K=1000, time_multiplier=adapter.time_multiplier)
    sim.set_population([(1.0, [1.0 / adapter.F] * adapter.F)])
    return SessionState(
        session_id=session_id,
        kernel=kernel,
        sim=sim,
        adapter=adapter,
        last_access=time.monotonic(),
    )


def build_custom_session(
    nodes: dict,
    edges: list,
    undirected: bool,
    session_id: str,
) -> SessionState:
    nodes_np = {k: np.array(v, dtype=np.float64) for k, v in nodes.items()}
    edges_typed = [(a, b, float(d)) for a, b, d in edges]
    topo = topology_from_edges(nodes=nodes_np, edges=edges_typed, undirected=undirected)
    n = topo.N
    kernel = DynamicTopologyKernel(
        topology=topo,
        alpha=1.0,
        beta=np.full((n, n), 5.0),
        feedback_rate=0.15,
        temperature=1.0,
        feedback_noise=0.02,
    )
    sim = PopulationSimulator(kernel, K=1000, time_multiplier=4.0)
    sim.set_population([(1.0, [1.0 / topo.F] * topo.F)])
    adapter = DomainAdapter(
        key="custom",
        name="Custom",
        description="User-defined topology",
        icon="custom",
        accent="#7a82a0",
        undirected=undirected,
        nodes=nodes,
        edges=edges,
        feature_labels=[f"Feature {i}" for i in range(topo.F)],
        intent_presets={"Neutral": [1.0 / topo.F] * topo.F},
    )
    return SessionState(session_id, kernel, sim, adapter, time.monotonic())


class SessionManager:
    def __init__(self, ttl_seconds: int = 30 * 60):
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, SessionState] = {
            DEFAULT_SESSION_ID: build_from_adapter(ADAPTERS["mall"], DEFAULT_SESSION_ID)
        }

    def get(self, session_id: str | None = None) -> SessionState:
        sid = session_id or DEFAULT_SESSION_ID
        if sid not in self._sessions:
            self._sessions[sid] = build_from_adapter(ADAPTERS["mall"], sid)
        return self._sessions[sid].touch()

    def create(self, adapter: DomainAdapter, session_id: str | None = None) -> SessionState:
        sid = session_id or self.new_session_id()
        state = build_from_adapter(adapter, sid)
        self._sessions[sid] = state
        return state.touch()

    def set(self, state: SessionState) -> SessionState:
        self._sessions[state.session_id] = state
        return state.touch()

    def replace_default(self, state: SessionState) -> SessionState:
        state.session_id = DEFAULT_SESSION_ID
        return self.set(state)

    def all_active(self) -> Iterable[SessionState]:
        self.prune()
        return list(self._sessions.values())

    def prune(self) -> None:
        now = time.monotonic()
        expired = [
            sid for sid, state in self._sessions.items()
            if sid != DEFAULT_SESSION_ID and now - state.last_access > self.ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

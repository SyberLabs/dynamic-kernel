import io
import csv
import asyncio
import json
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from kernel import DynamicTopologyKernel, topology_from_edges
from simulator import PopulationSimulator
from adapters import ADAPTERS, DomainAdapter, get_adapter
from optimizer import SymmetryOptimizer, SymmetryMode
from session import (
    DEFAULT_SESSION_ID,
    SessionManager,
    SessionState,
)


# ---------------------------------------------------------------------------
# Global state — built from adapters
# ---------------------------------------------------------------------------

session_manager = SessionManager()
_ROOT = Path(__file__).resolve().parent
_DEMO_SEMICONDUCTOR_PATH = _ROOT / "semiconductor_onshoring_falsification_output.json"


def _sync_default_globals() -> None:
    global kernel, sim, active_adapter
    state = session_manager.get(DEFAULT_SESSION_ID)
    kernel = state.kernel
    sim = state.sim
    active_adapter = state.adapter


def _get_session_id(request: Request) -> str:
    return (
        request.query_params.get("session_id")
        or request.cookies.get("session_id")
        or DEFAULT_SESSION_ID
    )


def _load_semiconductor_demo_payload() -> dict:
    if not _DEMO_SEMICONDUCTOR_PATH.exists():
        raise HTTPException(404, detail="semiconductor demo artifact not found")
    try:
        with _DEMO_SEMICONDUCTOR_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise HTTPException(500, detail=f"invalid semiconductor demo artifact: {exc}") from exc

    return {
        "config": raw.get("config", {}),
        "summary": raw.get("summary", {}),
        "choicePointRelocation": raw.get("choice_point_relocation", {}).get("grouped", []),
        "topologySurgery": raw.get("topology_surgery", {}).get("grouped", []),
        "feedbackContinuum": raw.get("feedback_continuum", {}).get("grouped", []),
    }


def _topology_payload(state: SessionState, status: str | None = None) -> dict:
    D = state.kernel.topo.distance_matrix
    D_clean = np.where(np.isposinf(D), 9999.9, D)
    payload = {
        "labels":          state.kernel.topo.labels,
        "nodesConfig":     state.kernel.topo.node_features.tolist(),
        "distanceMatrix":  D_clean.tolist(),
        "adjacencyMask":   state.kernel.topo.adjacency_mask.astype(int).tolist(),
        "N":               state.kernel.topo.N,
        "F":               state.kernel.topo.F,
        "presetName":      state.adapter.name,
        "featureLabels":   state.adapter.feature_labels,
        "intentPresets":   state.adapter.intent_presets,
        "icon":            state.adapter.icon,
        "accent":          state.adapter.accent,
        "undirected":      state.adapter.undirected,
        "sessionId":       state.session_id,
        # F3: expose the kernel's actual node_bias so the visualizer can stay
        # faithful to whichever preset is loaded, instead of hardcoding
        # node_bias[0] = 0.3 (which only made sense for the Mall adapter).
        "nodeBias":        state.kernel._node_bias.tolist(),
    }
    if status is not None:
        payload["status"] = status
    return payload


_sync_default_globals()

# ---------------------------------------------------------------------------
# Connection registry — maps websocket → asyncio.Queue for backpressure
# BUG-3 FIX: each client gets its own bounded queue; slow clients are dropped
#            rather than stalling the broadcast loop or accumulating messages.
# ---------------------------------------------------------------------------

# client_ws → asyncio.Queue(maxsize)
_client_queues: dict[WebSocket, tuple[str, asyncio.Queue]] = {}

_SIM_TICK_HZ   = 10          # simulation ticks per second
_MAX_QUEUE_SZ  = 5           # max unread frames per client before dropping


@asynccontextmanager
async def lifespan(application: FastAPI):
    asyncio.create_task(_simulation_tick_loop())
    asyncio.create_task(_broadcast_loop())
    asyncio.create_task(_neural_tick_loop())
    asyncio.create_task(_neural_broadcast_loop())
    yield


app = FastAPI(title="Dynamic Topology Kernel API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Simulation loop — produces ticks, places them on each client's queue
# BUG-3 FIX: tick loop is decoupled from the send loop.  If a client queue is
#            full we simply skip that frame (drop) rather than blocking.
# BUG-4 FIX: we snapshot `sim` at the top of every tick so that a concurrent
#            K-change swap in the WS handler doesn't cause a torn read.
# ---------------------------------------------------------------------------

# Shared "latest stats" slot — the broadcast loop reads this
_latest_stats: dict[str, dict] = {}
_stats_event = asyncio.Event()
_neural_stats: dict[str, dict] = {}
_neural_event = asyncio.Event()
_neural_queues: dict[WebSocket, tuple[str, asyncio.Queue]] = {}
_NEURAL_TICK_HZ = 4


async def _simulation_tick_loop():
    global _latest_stats
    interval = 1.0 / _SIM_TICK_HZ
    while True:
        try:
            loop = asyncio.get_event_loop()
            ticked: dict[str, dict] = {}
            for state in session_manager.all_active():
                stats = await loop.run_in_executor(None, state.sim.tick)
                ticked[state.session_id] = stats
            _latest_stats = ticked
            _stats_event.set()
        except Exception:
            traceback.print_exc()
        await asyncio.sleep(interval)


async def _broadcast_loop():
    """Distribute latest tick stats to all connected client queues."""
    while True:
        await _stats_event.wait()
        _stats_event.clear()
        stats_by_session = _latest_stats
        if not stats_by_session:
            continue
        dead = []
        for ws, (session_id, q) in list(_client_queues.items()):
            try:
                stats = stats_by_session.get(session_id)
                if stats is None:
                    continue
                q.put_nowait(stats)    # non-blocking; drops if queue is full
            except asyncio.QueueFull:
                pass                   # slow client — skip this frame
            except Exception:
                dead.append(ws)
        for ws in dead:
            _client_queues.pop(ws, None)


async def _neural_tick_loop():
    global _neural_stats
    interval = 1.0 / _NEURAL_TICK_HZ
    while True:
        try:
            loop = asyncio.get_event_loop()
            ticked: dict[str, dict] = {}
            for state in session_manager.all_active():
                if state.optimizer is None:
                    continue
                stats = await loop.run_in_executor(None, state.optimizer.step)
                ticked[state.session_id] = stats
            if ticked:
                _neural_stats = ticked
                _neural_event.set()
        except Exception:
            traceback.print_exc()
        await asyncio.sleep(interval)


async def _neural_broadcast_loop():
    while True:
        await _neural_event.wait()
        _neural_event.clear()
        dead = []
        for ws, (session_id, q) in list(_neural_queues.items()):
            try:
                stats = _neural_stats.get(session_id)
                if stats is None:
                    continue
                q.put_nowait(stats)
            except asyncio.QueueFull:
                pass
            except Exception:
                dead.append(ws)
        for ws in dead:
            _neural_queues.pop(ws, None)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DiagnosticRequest(BaseModel):
    telemetry: List[float]
    beta: List[List[float]]
    sponsor_friction: List[List[float]]
    node_bias: List[float]
    temperature: float

class TopologyLoadRequest(BaseModel):
    preset: Optional[str] = None
    nodes: Optional[dict] = None
    edges: Optional[List] = None
    undirected: bool = True

class NeuralLoadRequest(BaseModel):
    n_neurons: int = 12
    density: float = 1.0
    directed: bool = False
    telemetry: Optional[List[float]] = None
    beta_init: float = 3.0

class NeuralOptimizerRequest(BaseModel):
    mode: Optional[str] = None
    eta: Optional[float] = None
    eps: Optional[float] = None
    beta_max: Optional[float] = None
    noise_sigma: Optional[float] = None
    telemetry: Optional[List[float]] = None
    target_pi: Optional[List[float]] = None
    composite_lambda: Optional[float] = None
    composite_utility_type: Optional[str] = None   # "l2" | "kl"
    normalize_gradient: Optional[bool] = None

class NeuralPhaseRequest(BaseModel):
    """Run a sequence of optimizer modes on the same kernel without reset."""
    phases: List[dict]   # [{"mode": str, "steps": int}, ...]
    reset_between: bool = False
    beta_init: float = 3.0

class NeuralPauseRequest(BaseModel):
    paused: bool

class NeuralResetRequest(BaseModel):
    beta_init: float = 3.0

# ---------------------------------------------------------------------------
# Neural helpers
# ---------------------------------------------------------------------------

def _build_neural_session(req: NeuralLoadRequest, session_id: str) -> SessionState:
    n = max(2, min(int(req.n_neurons), 16))
    density = max(0.0, min(float(req.density), 1.0))
    feature_count = len(req.telemetry) if req.telemetry else 4
    telemetry = np.array(req.telemetry or ([1.0 / feature_count] * feature_count), dtype=np.float64)
    if telemetry.ndim != 1 or len(telemetry) == 0:
        telemetry = np.full(4, 0.25)
        feature_count = 4

    nodes = {
        f"Neuron {i + 1}": [1.0 / feature_count] * feature_count
        for i in range(n)
    }

    rng = np.random.default_rng(42)
    edges = []
    if req.directed:
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() <= density:
                    edges.append((f"Neuron {i + 1}", f"Neuron {j + 1}", 1.0))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() <= density:
                    edges.append((f"Neuron {i + 1}", f"Neuron {j + 1}", 1.0))

    if not edges:
        edges = [(f"Neuron {i + 1}", f"Neuron {((i + 1) % n) + 1}", 1.0) for i in range(n)]

    topo = topology_from_edges(
        nodes={label: np.array(vec, dtype=np.float64) for label, vec in nodes.items()},
        edges=edges,
        undirected=not req.directed,
    )
    beta = np.zeros((topo.N, topo.N), dtype=np.float64)
    beta[topo.adjacency_mask] = float(req.beta_init)
    kernel = DynamicTopologyKernel(
        topology=topo,
        alpha=1.0,
        beta=beta,
        feedback_rate=0.0,
        temperature=1.0,
        feedback_noise=0.0,
        node_bias=np.zeros(topo.N),
    )
    sim = PopulationSimulator(kernel, K=1, time_multiplier=1.0)
    adapter = DomainAdapter(
        key="neural_custom",
        name="Neural Optimizer",
        description="Abstract self-organizing beta topology",
        icon="neural",
        accent="#9b6cf7",
        undirected=not req.directed,
        nodes=nodes,
        edges=edges,
        feature_labels=[f"Feature {i}" for i in range(topo.F)],
        intent_presets={"Uniform": (telemetry / max(float(telemetry.sum()), 1e-12)).tolist()},
        default_beta=float(req.beta_init),
        node_biases={},
        time_multiplier=1.0,
    )
    optimizer = SymmetryOptimizer(
        kernel=kernel,
        mode=SymmetryMode.ENTROPY_PI,
        eta=None,           # use MODE_DEFAULT_ETA
        eps=1e-3,
        beta_max=20.0,
        noise_sigma=0.0,    # investigation showed noise is harmful — default off
        telemetry=telemetry,
    )
    return SessionState(session_id, kernel, sim, adapter, 0.0, optimizer=optimizer)


def _ensure_neural_session(session_id: str) -> SessionState:
    state = session_manager.get(session_id)
    if state.optimizer is None:
        state = session_manager.set(_build_neural_session(NeuralLoadRequest(), session_id))
    return state


def _neural_payload(state: SessionState) -> dict:
    if state.optimizer is None:
        raise HTTPException(400, detail="Session has no neural optimizer")
    payload = state.optimizer.snapshot()
    payload["sessionId"] = state.session_id
    payload["topology"] = _topology_payload(state)
    payload["config"] = {
        "eta": state.optimizer.eta,
        "eps": state.optimizer.eps,
        "beta_max": state.optimizer.beta_max,
        "noise_sigma": state.optimizer.noise_sigma,
        "composite_lambda": state.optimizer.composite_lambda,
        "target_pi": None if state.optimizer.target_pi is None else state.optimizer.target_pi.tolist(),
    }
    return payload

# ---------------------------------------------------------------------------
# Topology endpoints
# ---------------------------------------------------------------------------

@app.get("/api/topology")
def get_topology(request: Request):
    state = session_manager.get(_get_session_id(request))
    return _topology_payload(state)

@app.get("/api/topology/presets")
def list_presets():
    return {key: adapter.to_api_meta() for key, adapter in ADAPTERS.items()}


@app.get("/api/topology/stationary")
def get_stationary_distribution(request: Request, telemetry: str | None = None):
    state = session_manager.get(_get_session_id(request))
    if telemetry is None:
        tel = state.sim.telemetries.mean(axis=0)
    else:
        try:
            tel = np.array([float(x) for x in telemetry.split(",")], dtype=np.float64)
        except ValueError:
            raise HTTPException(400, detail="telemetry must be comma-separated floats")
    if len(tel) != state.kernel.topo.F:
        raise HTTPException(400, detail=f"telemetry must have length {state.kernel.topo.F}")
    pi = state.kernel.stationary_distribution(tel)
    return {
        "sessionId": state.session_id,
        "labels": state.kernel.topo.labels,
        "stationary": pi.tolist(),
        "sum": float(pi.sum()),
    }


@app.get("/api/metrics/history")
def get_metrics_history(request: Request):
    state = session_manager.get(_get_session_id(request))
    return {
        "sessionId": state.session_id,
        "history": state.sim.get_metric_history(),
    }


@app.get("/api/demo/semiconductor")
def get_semiconductor_demo():
    return _load_semiconductor_demo_payload()


@app.post("/api/neural/load")
def neural_load(req: NeuralLoadRequest, request: Request):
    session_id = _get_session_id(request)
    state = session_manager.set(_build_neural_session(req, session_id))
    return _neural_payload(state)


@app.get("/api/neural/state")
def neural_state(request: Request):
    state = _ensure_neural_session(_get_session_id(request))
    return _neural_payload(state)


@app.post("/api/neural/optimizer")
def neural_optimizer(req: NeuralOptimizerRequest, request: Request):
    state = _ensure_neural_session(_get_session_id(request))
    try:
        state.optimizer.configure(
            mode=req.mode,
            eta=req.eta,
            eps=req.eps,
            beta_max=req.beta_max,
            noise_sigma=req.noise_sigma,
            telemetry=req.telemetry,
            target_pi=req.target_pi,
            composite_lambda=req.composite_lambda,
            composite_utility_type=req.composite_utility_type,
            normalize_gradient=req.normalize_gradient,
            paused=False,
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return _neural_payload(state)


@app.get("/api/neural/gradient")
def neural_gradient(request: Request):
    """Return the current N×N gradient matrix ∂Σ/∂β_ij for the active mode.
    Expensive — call on-demand, not every tick.
    """
    state = _ensure_neural_session(_get_session_id(request))
    if state.optimizer is None:
        raise HTTPException(400, detail="No neural optimizer on this session")
    grad = state.optimizer.compute_gradient_matrix()
    return {
        "sessionId": state.session_id,
        "mode": state.optimizer.mode.value,
        "tick": state.optimizer.tick,
        "gradient": grad,
        "N": state.kernel.topo.N,
    }


@app.post("/api/neural/phase_chain")
def neural_phase_chain(req: NeuralPhaseRequest, request: Request):
    """Run a sequence of optimizer modes on the same kernel.

    Each phase: {"mode": str, "steps": int}.
    The kernel state carries over between phases (weights accumulate).
    Returns the final snapshot plus a per-phase history list.
    """
    state = _ensure_neural_session(_get_session_id(request))
    if state.optimizer is None:
        raise HTTPException(400, detail="No neural optimizer on this session")
    if req.reset_between:
        state.optimizer.reset_beta(beta_init=req.beta_init)

    phase_summaries = []
    for phase in req.phases:
        p_mode  = phase.get("mode", "ENTROPY_PI")
        p_steps = max(1, int(phase.get("steps", 50)))
        state.optimizer.configure(mode=p_mode, paused=False)
        start_sigma = state.optimizer._eval_sigma(state.optimizer.kernel._beta)
        for _ in range(p_steps):
            state.optimizer.step()
        end_sigma = state.optimizer._eval_sigma(state.optimizer.kernel._beta)
        phase_summaries.append({
            "mode": p_mode,
            "steps": p_steps,
            "sigma_start": float(start_sigma),
            "sigma_end": float(end_sigma),
            "delta_sigma": float(end_sigma - start_sigma),
            "final_tick": state.optimizer.tick,
        })

    payload = _neural_payload(state)
    payload["phase_summaries"] = phase_summaries
    return payload


@app.post("/api/neural/pause")
def neural_pause(req: NeuralPauseRequest, request: Request):
    state = _ensure_neural_session(_get_session_id(request))
    state.optimizer.configure(paused=req.paused)
    return _neural_payload(state)


@app.post("/api/neural/reset")
def neural_reset(req: NeuralResetRequest, request: Request):
    state = _ensure_neural_session(_get_session_id(request))
    state.optimizer.reset_beta(beta_init=req.beta_init)
    return _neural_payload(state)

@app.post("/api/topology/load")
def load_topology(req: TopologyLoadRequest, request: Request):
    session_id = _get_session_id(request)

    if req.preset is not None:
        try:
            adapter = get_adapter(req.preset)
        except KeyError as e:
            raise HTTPException(404, detail=str(e))
        state = session_manager.create(adapter, session_id=session_id)

    elif req.nodes is not None and req.edges is not None:
        try:
            nodes  = {k: np.array(v, dtype=np.float64) for k, v in req.nodes.items()}
            edges  = [(a, b, float(d)) for a, b, d in req.edges]
            topo   = topology_from_edges(nodes=nodes, edges=edges, undirected=req.undirected)
            N      = topo.N
            kernel = DynamicTopologyKernel(
                topology=topo, alpha=1.0, beta=np.full((N, N), 5.0),
                feedback_rate=0.15, temperature=1.0, feedback_noise=0.02,
            )
            sim = PopulationSimulator(kernel, K=1000, time_multiplier=4.0)
            sim.set_population([(1.0, [1.0 / topo.F] * topo.F)])
            active_adapter = DomainAdapter(
                key="custom", name="Custom", description="User-defined topology",
                icon="🔧", accent="#7a82a0", undirected=req.undirected,
                nodes=req.nodes, edges=req.edges,
                feature_labels=[f"Feature {i}" for i in range(topo.F)],
                intent_presets={"Neutral": [1 / topo.F] * topo.F},
            )
            state = session_manager.set(SessionState(session_id, kernel, sim, active_adapter, 0.0))
        except Exception as e:
            raise HTTPException(400, detail=str(e))
    else:
        raise HTTPException(400, detail="Provide 'preset' or 'nodes'+'edges'.")

    if session_id == DEFAULT_SESSION_ID:
        _sync_default_globals()
    return _topology_payload(state, status="ok")

# ---------------------------------------------------------------------------
# Diagnostic endpoint
# ---------------------------------------------------------------------------

@app.post("/api/diagnostic")
def post_diagnostic(req: DiagnosticRequest, request: Request):
    state = session_manager.get(_get_session_id(request))
    kern = state.kernel
    N = kern.topo.N
    if len(req.telemetry) != kern.topo.F:
        raise HTTPException(400, detail=f"telemetry must have length {kern.topo.F}")
    if len(req.beta) != N or any(len(r) != N for r in req.beta):
        raise HTTPException(400, detail=f"beta must be {N}x{N}")
    if len(req.sponsor_friction) != N or any(len(r) != N for r in req.sponsor_friction):
        raise HTTPException(400, detail=f"sponsor_friction must be {N}x{N}")
    if len(req.node_bias) != N:
        raise HTTPException(400, detail=f"node_bias must be length {N}")

    kern.temperature         = max(req.temperature, 1e-8)
    kern._beta               = np.array(req.beta, dtype=np.float64)
    kern._sponsor_friction   = np.array(req.sponsor_friction, dtype=np.float64)
    kern._node_bias          = np.array(req.node_bias, dtype=np.float64)

    tel  = np.array(req.telemetry, dtype=np.float64)
    diag = kern.get_diagnostic(tel)

    def clean(obj):
        if isinstance(obj, np.ndarray):
            o = np.where(np.isposinf(obj), 9999.9, obj)
            o = np.where(np.isneginf(o), -9999.9, o)
            o = np.where(np.isnan(o), 0.0, o)
            return o.tolist()
        return obj

    try:
        return {
            "alignment":         clean(diag["alignment"]),
            "weight_matrix_raw": clean(diag["weight_matrix_raw"]),
            "weight_matrix":     clean(diag["weight_matrix"]),
            "transition_matrix": clean(diag["transition_matrix"]),
            "row_entropy":       clean(diag["row_entropy"]),
            "effective_rank":    clean(diag["effective_rank"]),
            "mixing_time": float(diag["mixing_time"])
                           if not np.isinf(diag["mixing_time"]) else -1.0,
            "edge_flux":         clean(diag["edge_flux"]),
            "edge_current":      clean(diag["edge_current"]),
            "entropy_production": float(diag["entropy_production"]),
            "irreversible_flux":  float(diag["irreversible_flux"]),
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# ---------------------------------------------------------------------------
# Export endpoint
# ---------------------------------------------------------------------------

@app.get("/api/export")
def export_report(request: Request, format: str = "json", reset: bool = False):
    """
    Export a full simulation analytics report.

    Query params:
        format : "json" (default) | "csv"
        reset  : if true, reset cumulative counters after export
    """
    state = session_manager.get(_get_session_id(request))
    snapshot = state.sim.get_report_snapshot()

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["# Dynamic Topology Kernel — Simulation Report"])
        writer.writerow(["# Topology",   snapshot["topology"]["labels"]])
        writer.writerow(["# Ticks",      snapshot["simulation"]["tick_count"]])
        writer.writerow(["# Agents (K)", snapshot["simulation"]["K"]])
        writer.writerow([])

        writer.writerow(["NODE ANALYTICS"])
        writer.writerow(["Node", "Current Occupancy", "Total Visits",
                         "Visit Rate / Tick", "Beta ROI", "Friction ROI"])
        for row in snapshot["node_analytics"]:
            writer.writerow([
                row["label"], row["current_occupancy"], row["total_visits"],
                row["visit_rate_per_tick"], row["beta_roi"], row["friction_roi"],
            ])
        writer.writerow([])

        writer.writerow(["EDGE ANALYTICS"])
        writer.writerow(["From", "To", "Total Crossings", "Flow / Tick"])
        for row in snapshot["edge_analytics"]:
            writer.writerow([row["from"], row["to"],
                             row["total_crossings"], row["flow_per_tick"]])

        if reset:
            state.sim.reset_history()

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition":
                     f'attachment; filename="dtk_report_{state.adapter.key}.csv"'},
        )

    if reset:
        state.sim.reset_history()

    return snapshot

@app.post("/api/export/reset")
def reset_analytics(request: Request):
    """Reset cumulative visit counters without modifying the simulation state."""
    state = session_manager.get(_get_session_id(request))
    state.sim.reset_history()
    return {"status": "ok", "message": "Analytics counters reset."}

# ---------------------------------------------------------------------------
# WebSocket — Population stream
# BUG-3 FIX: client reads from its own bounded queue; the tick loop never
#            blocks on a slow send.
# BUG-4 FIX: K-change rebuilds sim atomically, then re-registers the queue.
# ---------------------------------------------------------------------------

@app.websocket("/api/mall/stream")
async def mall_stream(websocket: WebSocket):
    session_id = (
        websocket.query_params.get("session_id")
        or websocket.cookies.get("session_id")
        or DEFAULT_SESSION_ID
    )
    state = session_manager.get(session_id)
    await websocket.accept()

    # Register a bounded queue for this client
    q: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE_SZ)
    _client_queues[websocket] = (state.session_id, q)

    async def _sender():
        """Drain this client's queue and forward frames over the WebSocket."""
        while True:
            stats = await q.get()
            await websocket.send_json(stats)

    sender_task = asyncio.create_task(_sender())

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception as e:
                print("WS receive error:", e)
                break

            # ── Apply control updates ──────────────────────────────────────
            try:
                if "groups" in data and data["groups"]:
                    state.sim.set_population(data["groups"])

                if "K" in data:
                    new_k = int(data["K"])
                    if new_k != state.sim.K:
                        # BUG-4 FIX: build new sim, then atomically swap
                        new_sim = PopulationSimulator(
                            state.kernel, K=new_k,
                            time_multiplier=state.sim.time_multiplier,
                        )
                        new_sim.set_population([(1.0, [1.0 / state.kernel.topo.F] * state.kernel.topo.F)])
                        state.sim = new_sim           # single reference assignment
                        if state.session_id == DEFAULT_SESSION_ID:
                            _sync_default_globals()

                if "beta" in data:
                    new_beta = np.array(data["beta"], dtype=np.float64)
                    if new_beta.shape == state.kernel._beta.shape:
                        state.kernel._beta = new_beta
                        state.kernel._beta_baseline = state.kernel._beta.copy()

                if "sponsor_friction" in data:
                    new_sf = np.array(data["sponsor_friction"], dtype=np.float64)
                    if new_sf.shape == state.kernel._sponsor_friction.shape:
                        state.kernel._sponsor_friction = new_sf
                        state.kernel._friction_baseline = state.kernel._sponsor_friction.copy()

                if "temperature" in data:
                    state.kernel.temperature = max(float(data["temperature"]), 1e-8)

                if "sponsor_decay" in data:
                    state.kernel._sponsor_decay = max(0.0, min(float(data["sponsor_decay"]), 1.0))

            except Exception as e:
                print("WS control error:", e)
                traceback.print_exc()

    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        _client_queues.pop(websocket, None)


@app.websocket("/api/neural/stream")
async def neural_stream(websocket: WebSocket):
    session_id = (
        websocket.query_params.get("session_id")
        or websocket.cookies.get("session_id")
        or "neural_default"
    )
    state = _ensure_neural_session(session_id)
    await websocket.accept()

    q: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE_SZ)
    _neural_queues[websocket] = (state.session_id, q)
    await websocket.send_json(_neural_payload(state))

    async def _sender():
        while True:
            stats = await q.get()
            await websocket.send_json(stats)

    sender_task = asyncio.create_task(_sender())

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception as e:
                print("Neural WS receive error:", e)
                break

            try:
                if "paused" in data:
                    state.optimizer.configure(paused=bool(data["paused"]))
                if "reset" in data:
                    state.optimizer.reset_beta(float(data.get("beta_init", 3.0)))
                config = {
                    key: data[key]
                    for key in [
                        "mode", "eta", "eps", "beta_max", "noise_sigma",
                        "telemetry", "target_pi", "composite_lambda",
                    ]
                    if key in data
                }
                if config:
                    state.optimizer.configure(**config)
            except Exception as e:
                print("Neural WS control error:", e)
                traceback.print_exc()

    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        _neural_queues.pop(websocket, None)

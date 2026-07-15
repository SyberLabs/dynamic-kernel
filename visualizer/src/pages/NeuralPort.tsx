import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Download, Pause, Play, RotateCcw } from 'lucide-react';

type SymmetryMode =
  | 'ENTROPY_PI'
  | 'ROW_ENTROPY'
  | 'DETAILED_BALANCE'
  | 'SPECTRAL_GAP'
  | 'WEIGHT_SYMMETRY'
  | 'COMPOSITE';

interface NeuralTopology {
  labels: string[];
  adjacencyMask: number[][];
  N: number;
}

interface NeuralFrame {
  tick: number;
  beta: number[][];
  sigma_value: number;
  grad_norm: number;
  pi: number[];
  row_entropy: number[];
  mixing_time: number;
  mode: SymmetryMode;
  converged: boolean;
  paused: boolean;
  history: {
    tick: number[];
    sigma: number[];
    grad_norm: number[];
  };
  topology?: NeuralTopology;
  config?: {
    eta: number;
    eps: number;
    beta_max: number;
    noise_sigma: number;
    composite_lambda: number;
    target_pi: number[] | null;
  };
  // Enhancement additions
  health?: 'HEALTHY' | 'SLOW' | 'STALLED';
  health_msg?: string;
  recommended_eta?: number;
  composite_utility_type?: 'l2' | 'kl';
  normalize_gradient?: boolean;
  target_feasibility?: {
    status: 'REACHABLE' | 'PARTIAL' | 'CONSTRAINED';
    message: string;
    estimated: boolean;
    l1_error: number;
    max_abs_error: number;
    initial_l1_error: number;
    improvement: number;
    achieved_pi: number[];
    target_focus_mass: number;
    probe_steps: number;
  } | null;
}

const SESSION_ID = 'neural_port';

const MODES: Array<{ mode: SymmetryMode; title: string; formula: string }> = [
  { mode: 'ENTROPY_PI', title: 'Stationary Entropy', formula: '-sum pi*log(pi)' },
  { mode: 'ROW_ENTROPY', title: 'Row Entropy', formula: '-Var(H_i)' },
  { mode: 'DETAILED_BALANCE', title: 'Detailed Balance', formula: '-KL(fwd||rev flux)' },
  { mode: 'SPECTRAL_GAP', title: 'Spectral Gap', formula: '1 - soft ||P-1pi||' },
  { mode: 'WEIGHT_SYMMETRY', title: 'Weight Mirror', formula: '-||beta - beta^T||' },
  { mode: 'COMPOSITE', title: 'Composite', formula: 'lambda*sym + utility' },
];

const MODE_DEFAULT_ETA: Record<SymmetryMode, number> = {
  ENTROPY_PI: 0.50,
  ROW_ENTROPY: 0.30,
  DETAILED_BALANCE: 0.80,
  SPECTRAL_GAP: 0.02,
  WEIGHT_SYMMETRY: 0.05,
  COMPOSITE: 0.20,
};

function normalize(values: number[]): number[] {
  const clipped = values.map(value => Math.max(0, value));
  const sum = clipped.reduce((acc, value) => acc + value, 0);
  if (sum <= 0) return values.map(() => 1 / values.length);
  return clipped.map(value => value / sum);
}

function valueRange(values: number[]): [number, number] {
  if (values.length === 0) return [0, 1];
  const min = Math.min(...values);
  const max = Math.max(...values);
  return min === max ? [min - 1, max + 1] : [min, max];
}

const NeuralPort: React.FC = () => {
  const graphRef = useRef<any>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const [frame, setFrame] = useState<NeuralFrame | null>(null);
  const [topology, setTopology] = useState<NeuralTopology | null>(null);
  const [nNeurons, setNNeurons] = useState(12);
  const [density, setDensity] = useState(1);
  const [mode, setMode] = useState<SymmetryMode>('ENTROPY_PI');
  const [eta, setEta] = useState(MODE_DEFAULT_ETA['ENTROPY_PI']);
  const [eps, setEps] = useState(0.001);
  const [noiseSigma, setNoiseSigma] = useState(0.0);  // default off per investigation
  const [betaMax, setBetaMax] = useState(20);
  const [compositeLambda, setCompositeLambda] = useState(0.5);
  const [compositeUtilityType, setCompositeUtilityType] = useState<'l2'|'kl'>('kl');
  const [targetPi, setTargetPi] = useState<number[]>(Array.from({ length: 12 }, () => 1 / 12));
  const [paused, setPaused] = useState(false);
  const [connectionError, setConnectionError] = useState(false);
  const [gradientData, setGradientData] = useState<number[][] | null>(null);
  const [gradientLoading, setGradientLoading] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Phase chain state
  const [phases, setPhases] = useState<Array<{mode: SymmetryMode; steps: number}>>([{mode: 'WEIGHT_SYMMETRY', steps: 150}, {mode: 'ENTROPY_PI', steps: 150}]);
  const [phaseRunning, setPhaseRunning] = useState(false);
  const [phaseSummaries, setPhaseSummaries] = useState<any[]>([]);

  const loadNetwork = useCallback(async (nextN = nNeurons, nextDensity = density) => {
    const res = await fetch(`/api/neural/load?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        n_neurons: nextN,
        density: nextDensity,
        directed: false,
        beta_init: 3.0,
        telemetry: [0.25, 0.25, 0.25, 0.25],
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json() as NeuralFrame;
    setFrame(data);
    setTopology(data.topology ?? null);
    setTargetPi(Array.from({ length: data.beta.length }, () => 1 / data.beta.length));
    setPaused(false);
  }, [density, nNeurons]);

  const configureOptimizer = useCallback(async (patch: Record<string, unknown>) => {
    const res = await fetch(`/api/neural/optimizer?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json() as NeuralFrame;
    setFrame(data);
  }, []);

  const fetchGradient = useCallback(async () => {
    setGradientLoading(true);
    try {
      const res = await fetch(`/api/neural/gradient?session_id=${SESSION_ID}`);
      if (res.ok) {
        const data = await res.json();
        setGradientData(data.gradient);
      }
    } finally {
      setGradientLoading(false);
    }
  }, []);

  const runPhaseChain = useCallback(async () => {
    setPhaseRunning(true);
    setPhaseSummaries([]);
    try {
      const res = await fetch(`/api/neural/phase_chain?session_id=${SESSION_ID}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phases, reset_between: true, beta_init: 3.0 }),
      });
      if (res.ok) {
        const data = await res.json();
        setFrame(data);
        setPhaseSummaries(data.phase_summaries ?? []);
      }
    } finally {
      setPhaseRunning(false);
    }
  }, [phases]);

  // Auto-set η when mode changes
  const handleModeChange = useCallback((nextMode: SymmetryMode) => {
    setMode(nextMode);
    setEta(MODE_DEFAULT_ETA[nextMode]);
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function start() {
      try {
        await loadNetwork();
        if (cancelled) return;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${protocol}//${window.location.host}/api/neural/stream?session_id=${SESSION_ID}`);
        socketRef.current = ws;
        ws.onmessage = event => {
          const data = JSON.parse(event.data) as NeuralFrame;
          setFrame(data);
          if (data.topology) setTopology(data.topology);
          setPaused(data.paused);
          setConnectionError(false);
        };
        ws.onerror = () => setConnectionError(true);
        ws.onclose = () => {
          if (!cancelled) setConnectionError(true);
        };
      } catch {
        if (!cancelled) setConnectionError(true);
      }
    }

    start();
    return () => {
      cancelled = true;
      socketRef.current?.close();
    };
  }, [loadNetwork]);

  useEffect(() => {
    configureOptimizer({
      mode,
      eta,
      eps,
      beta_max: betaMax,
      noise_sigma: noiseSigma,
      composite_lambda: compositeLambda,
      composite_utility_type: compositeUtilityType,
      target_pi: targetPi,
    }).catch(() => setConnectionError(true));
  }, [mode, eta, eps, betaMax, noiseSigma, compositeLambda, compositeUtilityType, targetPi, configureOptimizer]);

  const setPause = async (nextPaused: boolean) => {
    const res = await fetch(`/api/neural/pause?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paused: nextPaused }),
    });
    if (res.ok) {
      const data = await res.json() as NeuralFrame;
      setFrame(data);
      setPaused(nextPaused);
    }
  };

  const reset = async () => {
    const res = await fetch(`/api/neural/reset?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ beta_init: 3.0 }),
    });
    if (res.ok) {
      const data = await res.json() as NeuralFrame;
      setFrame(data);
      setPaused(false);
    }
  };

  const exportBeta = () => {
    if (!frame) return;
    const blob = new Blob([JSON.stringify({
      mode,
      tick: frame.tick,
      sigma_value: frame.sigma_value,
      grad_norm: frame.grad_norm,
      beta: frame.beta,
      pi: frame.pi,
    }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neural_beta_${mode.toLowerCase()}_${frame.tick}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Live frame data in a ref — never triggers graph layout restart ──────────
  const frameRef = useRef<NeuralFrame | null>(null);
  useEffect(() => {
    frameRef.current = frame;
  });

  // ── Structural graph: only depends on topology (stable after load) ───────────
  // fx/fy pin nodes in a ring — D3 layout runs once then stops. frame excluded.
  const graphData = useMemo(() => {
    if (!topology) return { nodes: [], links: [] };
    const nodes = topology.labels.map((label, idx) => {
      const angle = (idx / topology.N) * Math.PI * 2;
      const radius = 200;
      return {
        id: idx,
        name: label,
        fx: Math.cos(angle) * radius,
        fy: Math.sin(angle) * radius,
      };
    });
    // Links are structural — weight is painted live via getLinkWidth
    const links: { source: number; target: number }[] = [];
    for (let i = 0; i < topology.N; i++) {
      for (let j = i + 1; j < topology.N; j++) {
        if (topology.adjacencyMask[i][j] || topology.adjacencyMask[j][i]) {
          links.push({ source: i, target: j });
        }
      }
    }
    return { nodes, links };
  }, [topology]); // ← frame intentionally excluded

  // ── Canvas painter — reads from frameRef every animation frame ───────────────
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const f = frameRef.current;
    const piVals = f?.pi ?? [];
    const entropy = f?.row_entropy ?? [];
    const maxPi = Math.max(...piVals, 1e-9);
    const maxEntropy = Math.max(...entropy, 1e-9);
    const piVal = piVals[node.id] ?? 0;
    const entropyVal = entropy[node.id] ?? 0;
    const entropyRatio = entropyVal / maxEntropy;
    const radius = 4 + (piVal / maxPi) * 12;
    // Low entropy (constrained/hub-like) → amber/gold (hue ~35)
    // High entropy (free/distributed)   → teal/cyan  (hue ~180)
    const hue = 35 + entropyRatio * 145;
    const sat = 65 + (piVal / maxPi) * 25;   // high-mass nodes more vivid
    const lit = 42 + entropyRatio * 22;       // deeper amber, brighter teal
    const nodeColor = `hsl(${hue}, ${sat}%, ${lit}%)`;

    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = `hsla(${hue}, ${sat}%, ${lit}%, 0.20)`;
    ctx.fill();
    ctx.strokeStyle = nodeColor;
    ctx.lineWidth = 2 / globalScale;
    ctx.stroke();

    const fontSize = Math.max(10, 11) / globalScale;
    ctx.font = `600 ${fontSize}px Inter`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#f0f2f8';
    ctx.fillText(node.name, node.x, node.y - radius - 4 / globalScale);
    ctx.font = `500 ${9 / globalScale}px JetBrains Mono`;
    ctx.fillStyle = nodeColor;
    ctx.fillText(`${(piVal * 100).toFixed(1)}%`, node.x, node.y + radius + 5 / globalScale);
  }, []); // stable — reads via ref

  // ── Link width — reads beta from frameRef, never causes graph reset ──────────
  const getLinkWidth = useCallback((link: any) => {
    const beta = frameRef.current?.beta;
    if (!beta) return 0.5;
    const si = typeof link.source === 'object' ? link.source.id : link.source;
    const ti = typeof link.target === 'object' ? link.target.id : link.target;
    const weight = Math.max(beta[si]?.[ti] ?? 0, beta[ti]?.[si] ?? 0);
    return 0.5 + Math.min(7, weight / 2);
  }, []); // stable

  const getLinkColor = useCallback((link: any) => {
    const beta = frameRef.current?.beta;
    if (!beta) return 'rgba(255,255,255,0.08)';
    const si = typeof link.source === 'object' ? link.source.id : link.source;
    const ti = typeof link.target === 'object' ? link.target.id : link.target;
    const maxBeta = 20;
    const weight = Math.max(beta[si]?.[ti] ?? 0, beta[ti]?.[si] ?? 0);
    const intensity = Math.min(weight / maxBeta, 1);
    return `rgba(52, 211, 153, ${0.06 + intensity * 0.45})`;
  }, []); // stable

  const updateTarget = (index: number, raw: number) => {
    setTargetPi(prev => normalize(prev.map((value, idx) => idx === index ? raw : value)));
  };

  if (!frame || !topology) {
    return (
      <div className="loading-screen">
        <div className="loading-ring" />
        <span>{connectionError ? 'Adaptive routing backend unavailable.' : 'Initializing adaptive routing lab...'}</span>
      </div>
    );
  }

  return (
    <div className="neural-shell">
      <aside className="neural-sidebar">
        <div className="sidebar-brand">
          <h1>Adaptive Routing Lab</h1>
          <p>Routing memory dynamics</p>
        </div>

        <div className="stats-bar">
          <div className="stat-pill">
            <div className="label">Sigma</div>
            <div className="value purple">{frame.sigma_value.toFixed(4)}</div>
          </div>
          <div className="stat-pill">
            <div className="label">Grad Norm</div>
            <div className="value green">{frame.grad_norm.toExponential(2)}</div>
          </div>
          {frame.health && (
            <div className={`health-badge health-${frame.health.toLowerCase()}`}>
              {frame.health === 'HEALTHY' ? 'Active' : frame.health === 'SLOW' ? 'Slow' : 'Stalled'}
            </div>
          )}
        </div>
        {frame.health === 'STALLED' && frame.health_msg && (
          <div className="health-advice">{frame.health_msg}</div>
        )}

        <div className="sidebar-content">
          <section>
            <div className="section-heading">Network Load</div>
            <div className="hyperparams">
              <Slider label="Neurons" min={4} max={16} step={1} value={nNeurons} onChange={setNNeurons} />
              <Slider label="Density" min={0.25} max={1} step={0.05} value={density} onChange={setDensity} />
              <button type="button" className="neural-action full" onClick={() => loadNetwork(nNeurons, density)}>
                Reload Network
              </button>
            </div>
          </section>

          <section>
            <div className="section-heading">Symmetry Mode</div>
            <div className="mode-grid">
            {MODES.map(item => (
              <button
                type="button"
                key={item.mode}
                className={`mode-card ${mode === item.mode ? 'active' : ''}`}
                onClick={() => handleModeChange(item.mode)}
                aria-pressed={mode === item.mode}
              >
                <strong>{item.title}</strong>
                <span>{item.formula}</span>
              </button>
            ))}
          </div>
          </section>

          <section>
            <div className="neural-button-row">
              <button type="button" className="neural-action" onClick={() => setPause(!paused)}>
                {paused ? <Play size={14} /> : <Pause size={14} />}
                {paused ? 'Run' : 'Pause'}
              </button>
              <button type="button" className="neural-action" onClick={reset}>
                <RotateCcw size={14} />
                Reset
              </button>
              <button type="button" className="neural-action" onClick={exportBeta}>
                <Download size={14} />
                Export
              </button>
            </div>
            {frame.converged && <div className="converged-badge">Converged at tick {frame.tick}</div>}
            {connectionError && <div className="neural-warning">Stream disconnected. Refresh to reconnect.</div>}
          </section>

          <details
            className="advanced-disclosure"
            open={advancedOpen}
            onToggle={event => setAdvancedOpen(event.currentTarget.open)}
          >
            <summary>Advanced Tuning</summary>

            <section>
              <div className="section-heading">Hyperparameters</div>
              <div className="hyperparams">
                <Slider label={`Eta (rec: ${MODE_DEFAULT_ETA[mode]})`} min={0} max={1.0} step={0.01} value={eta} onChange={setEta} />
                <Slider label="Finite Diff" min={0.0002} max={0.01} step={0.0002} value={eps} onChange={setEps} />
                <div>
                  <Slider label="Noise" min={0} max={0.08} step={0.002} value={noiseSigma} onChange={setNoiseSigma} />
                  {noiseSigma > 0.02 && (
                    <div className="noise-warning">
                      Noise above 0.02 degrades the gradient estimate. Optimizer may walk backwards.
                    </div>
                  )}
                </div>
                <Slider label="Beta Max" min={3} max={30} step={1} value={betaMax} onChange={setBetaMax} />
              </div>
            </section>

            {mode === 'COMPOSITE' && (
              <section>
                <div className="section-heading">Composite Target</div>
                <Slider label="Lambda" min={0} max={1} step={0.02} value={compositeLambda} onChange={setCompositeLambda} />
                <div className="utility-type-row">
                  <span>Utility</span>
                  {(['kl', 'l2'] as const).map(t => (
                    <button
                      type="button"
                      key={t}
                      className={`utility-type-btn ${compositeUtilityType === t ? 'active' : ''}`}
                      onClick={() => setCompositeUtilityType(t)}
                    >{t.toUpperCase()}</button>
                  ))}
                </div>
                {frame.target_feasibility && (
                  <div className={`feasibility-card feasibility-${frame.target_feasibility.status.toLowerCase()}`}>
                    <strong>{frame.target_feasibility.status}</strong>
                    <span>{frame.target_feasibility.message}</span>
                    <em>
                      L1 {frame.target_feasibility.l1_error.toFixed(3)}
                      {' / '}
                      max {frame.target_feasibility.max_abs_error.toFixed(3)}
                    </em>
                  </div>
                )}
                <div className="target-pi-list">
                  {targetPi.map((value, idx) => (
                    <div key={idx} className="target-pi-row">
                      <span>N{idx + 1}</span>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.01}
                        value={value}
                        onChange={event => updateTarget(idx, parseFloat(event.target.value))}
                      />
                      <strong>{Math.round(value * 100)}%</strong>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section>
              <div className="section-heading">Phase Chain</div>
              <div className="phase-chain">
                {phases.map((ph, idx) => (
                  <div key={idx} className="phase-row">
                    <span>Phase {idx + 1}</span>
                    <select
                      value={ph.mode}
                      onChange={e => setPhases(prev => prev.map((p, i) => i === idx ? { ...p, mode: e.target.value as SymmetryMode } : p))}
                    >
                      {MODES.map(m => <option key={m.mode} value={m.mode}>{m.title}</option>)}
                    </select>
                    <input
                      type="number" min={10} max={500} step={10}
                      value={ph.steps}
                      onChange={e => setPhases(prev => prev.map((p, i) => i === idx ? { ...p, steps: parseInt(e.target.value) } : p))}
                    />
                  </div>
                ))}
                <button type="button" className="neural-action full" onClick={runPhaseChain} disabled={phaseRunning}>
                  {phaseRunning ? 'Running...' : 'Run Chain'}
                </button>
                {phaseSummaries.map((ph, i) => (
                  <div key={i} className="phase-summary">
                    <strong>{ph.mode}</strong>: delta Sigma={ph.delta_sigma?.toFixed(4)}
                  </div>
                ))}
              </div>
            </section>
          </details>
        </div>
      </aside>

      <main className="neural-main">
        <section className="neural-panel neural-heatmap-panel">
          <header className="neural-panel-title">
            <h2>Topology Metrics</h2>
            <div className="panel-actions">
              <button 
                type="button"
                className={`gradient-fetch-btn ${gradientLoading ? 'loading' : ''}`}
                onClick={fetchGradient}
                disabled={gradientLoading}
              >
                {gradientLoading ? 'Computing...' : 'Fetch Gradient dSigma/dbeta'}
              </button>
            </div>
          </header>
          
          <div className="heatmap-container">
            <div className="heatmap-box">
              <span className="heatmap-label">Beta Weights</span>
              <WeightHeatmap beta={frame.beta} />
            </div>
            {gradientData && (
              <div className="heatmap-box">
                <span className="heatmap-label">dSigma/dbeta Gradient</span>
                <GradientHeatmap gradient={gradientData} />
              </div>
            )}
          </div>

          <LossCurve 
            sigma={frame.history.sigma} 
            grad={frame.history.grad_norm} 
            phases={phaseSummaries}
          />
        </section>

        <section className="neural-panel neural-graph-panel">
          <PanelTitle
            title="Network Graph"
            subtitle={`tick ${frame.tick} | mixing ${frame.mixing_time < 0 ? 'inf' : frame.mixing_time.toFixed(1)}`}
          />
          <div className="neural-graph">
            <ForceGraph2D
              ref={graphRef}
              graphData={graphData}
              nodeRelSize={4}
              nodeCanvasObjectMode={() => 'replace'}
              nodeCanvasObject={paintNode}
              linkColor={getLinkColor}
              linkWidth={getLinkWidth}
              linkDirectionalParticles={0}
              d3AlphaDecay={0.09}
              d3VelocityDecay={0.55}
              cooldownTicks={40}
              onEngineStop={() => graphRef.current?.zoomToFit(250, 60)}
              backgroundColor="transparent"
            />
          </div>
          <PiBars labels={topology.labels} pi={frame.pi} rowEntropy={frame.row_entropy} />
        </section>
      </main>
    </div>
  );
};

const Slider: React.FC<{
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
}> = ({ label, min, max, step, value, onChange }) => {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <label className="neural-slider">
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ '--pct': `${pct}%` } as React.CSSProperties}
        onChange={event => onChange(parseFloat(event.target.value))}
      />
      <strong>{value < 0.01 ? value.toFixed(4) : value.toFixed(3)}</strong>
    </label>
  );
};

const PanelTitle: React.FC<{ title: string; subtitle: string }> = ({ title, subtitle }) => (
  <header className="neural-panel-title">
    <h2>{title}</h2>
    <span>{subtitle}</span>
  </header>
);

const WeightHeatmap: React.FC<{ beta: number[][] }> = ({ beta }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const n = beta.length;
    const dpr = window.devicePixelRatio || 1;
    const size = canvas.clientWidth;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);
    const max = Math.max(...beta.flat(), 1);
    const cell = size / n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const value = beta[i][j] / max;
        ctx.fillStyle = i === j
          ? 'rgba(255,255,255,0.03)'
          : `rgba(${Math.round(50 + value * 70)}, ${Math.round(95 + value * 160)}, ${Math.round(120 + value * 80)}, ${0.18 + value * 0.78})`;
        ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= n; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cell, 0);
      ctx.lineTo(i * cell, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * cell);
      ctx.lineTo(size, i * cell);
      ctx.stroke();
    }
  }, [beta]);

  return <canvas ref={canvasRef} className="neural-heatmap" />;
};

const GradientHeatmap: React.FC<{ gradient: number[][] }> = ({ gradient }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const n = gradient.length;
    const dpr = window.devicePixelRatio || 1;
    const size = canvas.clientWidth;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);
    
    // Find absolute max for normalization (diverging scale)
    let maxAbs = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        maxAbs = Math.max(maxAbs, Math.abs(gradient[i][j]));
      }
    }
    if (maxAbs === 0) maxAbs = 1;

    const cell = size / n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const val = gradient[i][j] / maxAbs;
        if (i === j) {
          ctx.fillStyle = 'rgba(255,255,255,0.03)';
        } else {
          // Red for positive (push up), Blue for negative (push down)
          if (val > 0) {
            ctx.fillStyle = `rgba(239, 68, 68, ${0.1 + val * 0.8})`; // Red-500
          } else {
            ctx.fillStyle = `rgba(59, 130, 246, ${0.1 + Math.abs(val) * 0.8})`; // Blue-500
          }
        }
        ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }
    
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= n; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(size, i * cell); ctx.stroke();
    }
  }, [gradient]);

  return <canvas ref={canvasRef} className="neural-heatmap gradient-heatmap" />;
};

const LossCurve: React.FC<{ sigma: number[]; grad: number[]; phases: any[] }> = ({ sigma, grad, phases }) => {
  const w = 520;
  const h = 150;
  const sigmaPoints = toPathPoints(sigma, w, h, 18);
  const gradPoints = toPathPoints(grad, w, h, 18);
  
  // Find phase boundary X coordinates
  const lastTick = sigma.length > 0 ? sigma.length : 1;
  const phaseMarkers = phases.map(p => {
    const x = 18 + (p.final_tick / lastTick) * (w - 36);
    return { x, label: p.mode };
  }).filter(p => p.x < w - 20);

  return (
    <div className="loss-curve">
      <div className="loss-title">
        <span>Sigma(t)</span>
        <span>Gradient norm</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        {/* Phase Markers */}
        {phaseMarkers.map((m, i) => (
          <g key={i}>
            <line x1={m.x} y1={5} x2={m.x} y2={h-5} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 2" />
            <text x={m.x + 4} y={15} fill="rgba(255,255,255,0.4)" fontSize="9" fontWeight="600">{m.label.split('_')[0]}</text>
          </g>
        ))}
        
        <path d={sigmaPoints} fill="none" stroke="#34d399" strokeWidth="2.5" />
        <path d={gradPoints} fill="none" stroke="#9b6cf7" strokeWidth="2" opacity="0.85" />
      </svg>
    </div>
  );
};

function toPathPoints(values: number[], width: number, height: number, pad: number): string {
  if (values.length < 2) return '';
  const [min, max] = valueRange(values);
  const scaleY = (value: number) => height - pad - ((value - min) / (max - min)) * (height - pad * 2);
  return values
    .map((value, index) => {
      const x = pad + (index / (values.length - 1)) * (width - pad * 2);
      const y = scaleY(value);
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

const PiBars: React.FC<{ labels: string[]; pi: number[]; rowEntropy: number[] }> = ({ labels, pi, rowEntropy }) => {
  const maxPi = Math.max(...pi, 1e-9);
  return (
    <div className="pi-bars">
      {labels.map((label, idx) => (
        <div key={label} className="pi-row">
          <span>{label}</span>
          <div>
            <i style={{ width: `${(pi[idx] / maxPi) * 100}%` }} />
          </div>
          <strong>{(pi[idx] * 100).toFixed(1)}%</strong>
          <em>{rowEntropy[idx]?.toFixed(2)}</em>
        </div>
      ))}
    </div>
  );
};

export default NeuralPort;

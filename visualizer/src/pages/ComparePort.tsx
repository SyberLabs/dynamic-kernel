import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { GitCompareArrows } from 'lucide-react';
import { ControlSection, SliderField } from '../components/UIControls';

interface TopologyResponse {
  labels: string[];
  adjacencyMask: number[][];
  N: number;
  F: number;
  presetName: string;
  intentPresets: Record<string, number[]>;
  accent?: string;
  undirected?: boolean;
}

interface SimState {
  node_counts: number[];
  edge_counts: number[][];
  active_transit: number;
  active_stationary: number;
  tick?: number;
}

interface MetricHistory {
  tick: number[];
  mean_entropy: number[];
  mixing_time: number[];
  active_transit_pct: number[];
}

interface CityPanelState {
  topology: TopologyResponse | null;
  sim: SimState | null;
  history: MetricHistory | null;
  error: boolean;
}

interface DemoGroup {
  name: string;
  val: number[];
  pct: number;
  color: string;
}

type SponsorMode = 'none' | 'beta' | 'friction';
type CityKey = 'wheel' | 'rhizome';

const CITY_SESSIONS: Record<CityKey, { sessionId: string; preset: string; color: string }> = {
  wheel: { sessionId: 'compare_wheel', preset: 'wheel_city', color: '#38bdf8' },
  rhizome: { sessionId: 'compare_rhizome', preset: 'rhizome_city', color: '#34d399' },
};

const GROUP_COLORS = ['#38bdf8', '#34d399', '#f5a623', '#9b6cf7', '#f87171'];

function normalize(values: number[]): number[] {
  const sum = values.reduce((acc, value) => acc + value, 0);
  if (sum <= 0) return values.map(() => 1 / values.length);
  return values.map(value => value / sum);
}

function sponsorTarget(labels: string[]): number {
  const keywords = ['cbd', 'market', 'square', 'station', 'cultural'];
  for (const keyword of keywords) {
    const found = labels.findIndex(label => label.toLowerCase().includes(keyword));
    if (found >= 0) return found;
  }
  return Math.max(0, Math.floor(labels.length / 2));
}

function sponsorSource(labels: string[], target: number): number {
  if (target > 0) return 0;
  return Math.min(1, labels.length - 1);
}

function formatMixing(value?: number): string {
  if (value === undefined) return '--';
  if (value < 0) return 'inf';
  return value.toFixed(1);
}

function latestMetric(history: MetricHistory | null, key: keyof MetricHistory): number | undefined {
  const values = history?.[key];
  if (!values || values.length === 0) return undefined;
  return values[values.length - 1];
}

function buildControlPayload(
  topology: TopologyResponse,
  groups: DemoGroup[],
  totalAgents: number,
  temperature: number,
  sponsor: SponsorMode,
) {
  const n = topology.N;
  const beta = Array.from({ length: n }, () => Array(n).fill(5.0));
  const sponsor_friction = Array.from({ length: n }, () => Array(n).fill(0.0));
  const target = sponsorTarget(topology.labels);
  const source = sponsorSource(topology.labels, target);

  if (sponsor === 'beta') beta[source][target] = 15.0;
  if (sponsor === 'friction') sponsor_friction[source][target] = 8.0;

  return {
    K: totalAgents,
    temperature,
    groups: groups.map(group => [group.pct, group.val]),
    beta,
    sponsor_friction,
    sponsor_decay: 0,
  };
}

const ComparePort: React.FC = () => {
  const wheelGraphRef = useRef<any>(null);
  const rhizomeGraphRef = useRef<any>(null);
  const socketsRef = useRef<Partial<Record<CityKey, WebSocket>>>({});

  const [cities, setCities] = useState<Record<CityKey, CityPanelState>>({
    wheel: { topology: null, sim: null, history: null, error: false },
    rhizome: { topology: null, sim: null, history: null, error: false },
  });
  const [groups, setGroups] = useState<DemoGroup[]>([]);
  const [totalAgents, setTotalAgents] = useState(1200);
  const [temperature, setTemperature] = useState(1.0);
  const [sponsor, setSponsor] = useState<SponsorMode>('none');
  const [socketReadyTick, setSocketReadyTick] = useState(0);

  useEffect(() => {
    let cancelled = false;

    async function bootCity(key: CityKey) {
      const city = CITY_SESSIONS[key];
      try {
        const load = await fetch(`/api/topology/load?session_id=${city.sessionId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ preset: city.preset }),
        });
        if (!load.ok) throw new Error(`HTTP ${load.status}`);
        const topology: TopologyResponse = await load.json();
        if (cancelled) return;

        setCities(prev => ({
          ...prev,
          [key]: { ...prev[key], topology, error: false },
        }));

        if (key === 'wheel' && topology.intentPresets) {
          const presetGroups = Object.entries(topology.intentPresets).map(([name, val], index) => ({
            name,
            val,
            pct: 1 / Object.keys(topology.intentPresets).length,
            color: GROUP_COLORS[index % GROUP_COLORS.length],
          }));
          setGroups(presetGroups);
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${protocol}//${window.location.host}/api/mall/stream?session_id=${city.sessionId}`);
        socketsRef.current[key] = ws;
        ws.onopen = () => setSocketReadyTick(tick => tick + 1);
        ws.onmessage = event => {
          try {
            const sim = JSON.parse(event.data) as SimState;
            setCities(prev => ({ ...prev, [key]: { ...prev[key], sim } }));
          } catch {
            // Ignore malformed frames; the next tick will replace it.
          }
        };
        ws.onerror = () => {
          setCities(prev => ({ ...prev, [key]: { ...prev[key], error: true } }));
        };
      } catch {
        if (!cancelled) {
          setCities(prev => ({ ...prev, [key]: { ...prev[key], error: true } }));
        }
      }
    }

    bootCity('wheel');
    bootCity('rhizome');

    return () => {
      cancelled = true;
      Object.values(socketsRef.current).forEach(ws => ws?.close());
    };
  }, []);

  const wheelTopology = cities.wheel.topology;
  const rhizomeTopology = cities.rhizome.topology;

  useEffect(() => {
    if (groups.length === 0) return;
    const topologies: Partial<Record<CityKey, TopologyResponse>> = {
      wheel: wheelTopology ?? undefined,
      rhizome: rhizomeTopology ?? undefined,
    };
    (Object.keys(CITY_SESSIONS) as CityKey[]).forEach(key => {
      const ws = socketsRef.current[key];
      const topology = topologies[key];
      if (!ws || ws.readyState !== WebSocket.OPEN || !topology) return;
      ws.send(JSON.stringify(buildControlPayload(topology, groups, totalAgents, temperature, sponsor)));
    });
  }, [groups, totalAgents, temperature, sponsor, socketReadyTick, wheelTopology, rhizomeTopology]);

  useEffect(() => {
    const interval = window.setInterval(async () => {
      await Promise.all((Object.keys(CITY_SESSIONS) as CityKey[]).map(async key => {
        const city = CITY_SESSIONS[key];
        try {
          const res = await fetch(`/api/metrics/history?session_id=${city.sessionId}`);
          if (!res.ok) return;
          const data = await res.json();
          setCities(prev => ({
            ...prev,
            [key]: { ...prev[key], history: data.history },
          }));
        } catch {
          // Keep the last known metric frame.
        }
      }));
    }, 1200);

    return () => window.clearInterval(interval);
  }, []);

  const handleGroupChange = useCallback((index: number, rawPct: number) => {
    setGroups(prev => {
      const adjusted = prev.map((group, idx) => idx === index ? { ...group, pct: rawPct } : group);
      const normalized = normalize(adjusted.map(group => group.pct));
      return adjusted.map((group, idx) => ({ ...group, pct: normalized[idx] }));
    });
  }, []);

  const wheelEntropy = latestMetric(cities.wheel.history, 'mean_entropy');
  const rhizomeEntropy = latestMetric(cities.rhizome.history, 'mean_entropy');
  const wheelMixing = latestMetric(cities.wheel.history, 'mixing_time');
  const rhizomeMixing = latestMetric(cities.rhizome.history, 'mixing_time');
  const wheelHub = cities.wheel.sim?.node_counts[0] ?? 0;
  const rhizomePeak = cities.rhizome.sim ? Math.max(...cities.rhizome.sim.node_counts, 0) : 0;

  if (!cities.wheel.topology || !cities.rhizome.topology) {
    return (
      <div className="loading-screen">
        <div className="loading-ring" />
        <span>{cities.wheel.error || cities.rhizome.error ? 'Comparison backend unavailable.' : 'Preparing city comparison...'}</span>
      </div>
    );
  }

  return (
    <div className="compare-shell">
      <aside className="compare-sidebar">
        <div className="sidebar-brand">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <GitCompareArrows size={18} color="var(--green)" />
            <div>
              <h1>Topology Comparison</h1>
              <p>Same agents, different graph</p>
            </div>
          </div>
        </div>

        <div className="stats-bar">
          <div className="stat-pill">
            <div className="label">Entropy Delta</div>
            <div className="value green">
              {wheelEntropy !== undefined && rhizomeEntropy !== undefined
                ? (rhizomeEntropy - wheelEntropy).toFixed(3)
                : '--'}
            </div>
          </div>
          <div className="stat-pill">
            <div className="label">Peak Load Gap</div>
            <div className="value amber">{Math.max(0, wheelHub - rhizomePeak)}</div>
          </div>
        </div>

        <div className="sidebar-content">
          <ControlSection title="Agent Count">
            <SliderField
              label="Shared population"
              min={100}
              max={3000}
              step={100}
              value={totalAgents}
              valueLabel={totalAgents.toLocaleString()}
              minLabel="100"
              maxLabel="3,000"
              tone="green"
              onChange={value => setTotalAgents(Math.round(value))}
            />
          </ControlSection>

          <ControlSection title="Exploration Tau">
            <SliderField
              label="Route softness"
              min={0.1}
              max={4}
              step={0.05}
              value={temperature}
              valueLabel={`tau ${temperature.toFixed(2)}`}
              minLabel="Greedy"
              maxLabel="Explore"
              onChange={setTemperature}
            />
          </ControlSection>

          <ControlSection title="Shared Agent Mix">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {groups.map((group, index) => {
                const pctStyle = {
                  '--pct': `${group.pct * 100}%`,
                  background: `linear-gradient(to right, ${group.color} 0%, ${group.color} ${group.pct * 100}%, var(--border-mid) ${group.pct * 100}%)`,
                } as React.CSSProperties;
                return (
                  <div key={group.name}>
                    <div className="compare-group-row">
                      <span style={{ color: group.color }}>{group.name}</span>
                      <strong>{Math.round(group.pct * 100)}%</strong>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={group.pct}
                      style={pctStyle}
                      onChange={event => handleGroupChange(index, parseFloat(event.target.value))}
                    />
                  </div>
                );
              })}
            </div>
          </ControlSection>

          <ControlSection title="Synchronized Intervention">
            <div className="sponsor-cards">
              {(['none', 'beta', 'friction'] as SponsorMode[]).map(mode => (
                <button
                  type="button"
                  key={mode}
                  className={`compare-mode-button ${sponsor === mode ? `active-${mode}` : ''}`}
                  onClick={() => setSponsor(mode)}
                  aria-pressed={sponsor === mode}
                >
                  {mode === 'none' ? 'Baseline' : mode === 'beta' ? 'Preference Bias' : 'Friction Reduction'}
                </button>
              ))}
            </div>
          </ControlSection>
        </div>
      </aside>

      <main className="compare-main">
        <CityPanel
          cityKey="wheel"
          title="WHEEL City"
          subtitle="Radial hierarchy"
          state={cities.wheel}
          graphRef={wheelGraphRef}
          color={CITY_SESSIONS.wheel.color}
        />
        <CityPanel
          cityKey="rhizome"
          title="RHIZOME City"
          subtitle="Distributed mesh"
          state={cities.rhizome}
          graphRef={rhizomeGraphRef}
          color={CITY_SESSIONS.rhizome.color}
        />
        <div className="compare-metric-strip">
          <MetricBlock label="WHEEL Entropy" value={wheelEntropy?.toFixed(3) ?? '--'} color="#38bdf8" />
          <MetricBlock label="RHIZOME Entropy" value={rhizomeEntropy?.toFixed(3) ?? '--'} color="#34d399" />
          <MetricBlock label="WHEEL Mixing" value={formatMixing(wheelMixing)} color="#38bdf8" />
          <MetricBlock label="RHIZOME Mixing" value={formatMixing(rhizomeMixing)} color="#34d399" />
          <MetricBlock label="CBD Occupancy" value={wheelHub.toString()} color="#f5a623" />
          <MetricBlock label="Mesh Peak" value={rhizomePeak.toString()} color="#9b6cf7" />
        </div>
      </main>
    </div>
  );
};

interface CityPanelProps {
  cityKey: CityKey;
  title: string;
  subtitle: string;
  state: CityPanelState;
  graphRef: React.MutableRefObject<any>;
  color: string;
}

const CityPanel: React.FC<CityPanelProps> = ({ cityKey, title, subtitle, state, graphRef, color }) => {
  // ── Live sim data lives in a ref — updating it never restarts the D3 layout ──
  const simRef = useRef<SimState | null>(null);
  useEffect(() => {
    simRef.current = state.sim;
  });

  // ── Structural graph: built ONLY from topology (stable after first load) ──────
  // Deliberately excludes state.sim so the force engine never reinitialises on tick.
  const graphData = useMemo(() => {
    const topology = state.topology;
    if (!topology) return { nodes: [], links: [] };

    const target = sponsorTarget(topology.labels);
    const nodes = topology.labels.map((label, idx) => ({
      id: idx,
      name: label,
      isTarget: idx === target,
    }));

    const links: { source: number; target: number }[] = [];
    for (let i = 0; i < topology.N; i++) {
      for (let j = topology.undirected === false ? 0 : i + 1; j < topology.N; j++) {
        if (i === j) continue;
        if (topology.adjacencyMask[i][j] || topology.adjacencyMask[j][i]) {
          links.push({ source: i, target: j });
        }
      }
    }
    return { nodes, links };
  }, [state.topology]); // ← sim intentionally excluded

  // ── Canvas painter — reads from simRef, called every animation frame ──────────
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const counts = simRef.current?.node_counts;
    const maxCount = counts ? Math.max(...counts, 1) : 1;
    const count = counts?.[node.id] ?? 0;

    // Draw filled circle scaled by occupancy
    const radius = 4 + (count / maxCount) * 7;
    const nodeColor = node.isTarget ? '#f5a623' : color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = nodeColor + '22';
    ctx.fill();
    ctx.strokeStyle = nodeColor;
    ctx.lineWidth = 1.5 / globalScale;
    ctx.stroke();

    // Label + count
    const fontSize = 11 / globalScale;
    ctx.font = `600 ${fontSize}px Inter`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#f0f2f8';
    ctx.fillText(node.name, node.x, node.y - radius - 4 / globalScale);
    ctx.font = `500 ${10 / globalScale}px JetBrains Mono`;
    ctx.fillStyle = count > 0 ? '#34d399' : '#7a82a0';
    ctx.fillText(String(count), node.x, node.y + radius + 6 / globalScale);
  }, [color]); // stable — sim reads via ref

  // ── Link width — reads from simRef, never causes graph reset ─────────────────
  const getLinkWidth = useCallback((link: any) => {
    const ec = simRef.current?.edge_counts;
    if (!ec) return 1;
    const si = typeof link.source === 'object' ? link.source.id : link.source;
    const ti = typeof link.target === 'object' ? link.target.id : link.target;
    const flow = (ec[si]?.[ti] ?? 0) + (ec[ti]?.[si] ?? 0);
    return 1 + Math.min(5, flow / 40);
  }, []); // stable

  return (
    <section className={`city-panel city-panel-${cityKey}`}>
      <div className="city-panel-header">
        <div>
          <h2>{title}</h2>
          <span>{subtitle}</span>
        </div>
        <div className="city-status">
          <span style={{ color }}>{state.sim?.active_transit ?? '--'}</span>
          transit
        </div>
      </div>
      <div className="city-graph">
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          nodeRelSize={4}
          nodeColor={(n: any) => n.isTarget ? '#f5a623' : color}
          linkColor={() => 'rgba(255,255,255,0.12)'}
          linkWidth={getLinkWidth}
          nodeCanvasObjectMode={() => 'replace'}
          nodeCanvasObject={paintNode}
          d3AlphaDecay={0.028}
          d3VelocityDecay={0.38}
          cooldownTicks={120}
          onEngineStop={() => graphRef.current?.zoomToFit(350, 44)}
          backgroundColor="transparent"
        />
      </div>
    </section>
  );
};

const MetricBlock: React.FC<{ label: string; value: string; color: string }> = ({ label, value, color }) => (
  <div className="compare-metric">
    <span>{label}</span>
    <strong style={{ color }}>{value}</strong>
  </div>
);

export default ComparePort;

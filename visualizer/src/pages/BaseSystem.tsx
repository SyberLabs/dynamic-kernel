import React, { useState, useEffect, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

interface DiagnosticState {
  alignment: number[];
  transition_matrix: number[][];
  row_entropy: number[];
  effective_rank: number[];
  mixing_time: number;
}

/**
 * F3: Pick a sponsor-target node by label keyword. Mirrors the helper of
 * the same name in MallPort.tsx so both ports use the same heuristic
 * regardless of which adapter is loaded. Falls back to the last node.
 */
function deriveSponsorTarget(labels: string[]): number {
  const keywords = ['outlet', 'gate', 'retail', 'distribution', 'cbd', 'market', 'square'];
  for (let i = labels.length - 1; i >= 0; i--) {
    const lc = labels[i].toLowerCase();
    if (keywords.some(k => lc.includes(k))) return i;
  }
  return labels.length - 1;
}

interface SelectedNodeInfo {
  id: number;
  name: string;
  alignment: number;
  entropy: number;
  effectiveRank: number;
  outbound: { label: string; prob: number }[];
}

export default function BaseSystem() {
  const fgRef = useRef<any>(null);
  const [topology, setTopology] = useState<any>(null);
  const [diagnostic, setDiagnostic] = useState<DiagnosticState | null>(null);
  const [selectedNode, setSelectedNode] = useState<SelectedNodeInfo | null>(null);

  // Controls
  const [activeIntent, setActiveIntent] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(1.0);
  const [sponsor, setSponsor] = useState<'none' | 'beta' | 'friction'>('none');

  useEffect(() => {
    fetch('/api/topology')
      .then(r => r.json())
      .then(d => {
        setTopology(d);
        if (d.intentPresets) {
          setActiveIntent(Object.keys(d.intentPresets)[0]);
        }
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!topology || !activeIntent || !topology.intentPresets) return;

    let tel = topology.intentPresets[activeIntent];
    if (!tel) return;

    const N = topology.labels.length;
    const beta = Array.from({ length: N }, () => Array(N).fill(5.0));
    const sponsor_friction = Array.from({ length: N }, () => Array(N).fill(0.0));

    // F3: source node_bias from the loaded preset (MALL gets [0.3, 0, 0, 0, 0],
    // others get whatever their adapter declares). Falls back to zeros if the
    // backend doesn't yet populate `nodeBias` (older deploys).
    const node_bias: number[] = topology.nodeBias
      ? topology.nodeBias.slice()
      : Array.from({ length: N }, () => 0.0);

    // F3: pick the sponsor target by label keyword instead of N-1, matching
    // MallPort.tsx's deriveSponsorTarget. Keywords cover the supplied adapters
    // (mall, airport, museum, supply_chain, wheel/rhizome cities).
    const sponsorDest = deriveSponsorTarget(topology.labels);
    const sponsorSrc = Math.min(1, N - 1);

    if (sponsor === 'beta') beta[sponsorSrc][sponsorDest] = 15.0;
    if (sponsor === 'friction') sponsor_friction[sponsorSrc][sponsorDest] = 8.0;


    fetch('/api/diagnostic', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ telemetry: tel, temperature, beta, sponsor_friction, node_bias })
    })
      .then(r => r.json())
      .then((d: DiagnosticState) => {
        setDiagnostic(d);
        // Update selected node info if one is selected
        setSelectedNode(prev => {
          if (!prev) return null;
          return buildNodeInfo(prev.id, topology, d);
        });
      })
      .catch(console.error);
  }, [activeIntent, temperature, sponsor, topology]);

  function buildNodeInfo(nodeId: number, topo: any, diag: DiagnosticState): SelectedNodeInfo {
    const P = diag.transition_matrix;
    const outbound = topo.labels
      .map((label: string, j: number) => ({ label, prob: P[nodeId][j] }))
      .filter((x: any) => x.prob > 0.0005)
      .sort((a: any, b: any) => b.prob - a.prob);

    return {
      id: nodeId,
      name: topo.labels[nodeId],
      alignment: diag.alignment[nodeId],
      entropy: diag.row_entropy[nodeId],
      effectiveRank: diag.effective_rank[nodeId],
      outbound,
    };
  }

  const handleNodeClick = useCallback((node: any) => {
    if (!topology || !diagnostic) return;
    if (selectedNode && selectedNode.id === node.id) {
      setSelectedNode(null);
      return;
    }
    setSelectedNode(buildNodeInfo(node.id, topology, diagnostic));
  }, [topology, diagnostic, selectedNode]);

  const graphData = React.useMemo(() => {
    if (!topology || !diagnostic) return { nodes: [], links: [] };

    const nodes = topology.labels.map((label: string, idx: number) => {
      const align = diagnostic.alignment[idx];
      const entropy = diagnostic.row_entropy[idx];
      const isSelected = selectedNode?.id === idx;
      let color = '#4b5563';
      if (isSelected)       color = '#f5a623';
      else if (align > 0.5) color = '#4f8ef5';
      else if (align > 0.1) color = '#9b6cf7';
      return {
        id: idx, name: label,
        val: 1.0 + Math.max(0, Math.min(align, 1)) * 1.8 + (isSelected ? 0.8 : 0),
        color, align, entropy
      };
    });

    const links: any[] = [];
    const P = diagnostic.transition_matrix;
    for (let i = 0; i < topology.labels.length; i++) {
      for (let j = 0; j < topology.labels.length; j++) {
        if (topology.adjacencyMask[i][j]) {
          const prob = P[i][j];
          if (prob > 0.001) {
            links.push({
              source: i, target: j,
              width: Math.max(1, prob * 4),
              particles: Math.floor(prob * 10),
              prob
            });
          }
        }
      }
    }
    return { nodes, links };
  }, [topology, diagnostic, selectedNode]);

  const sliderStyle = {
    '--pct': `${((temperature - 0.01) / (5.0 - 0.01)) * 100}%`
  } as React.CSSProperties;

  if (!topology || !diagnostic) return (
    <div className="loading-screen">
      <div className="loading-ring"/>
      <span>Loading kernel...</span>
    </div>
  );

  const meanEntropy = diagnostic.row_entropy.reduce((a, b) => a + b, 0) / topology.N;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1>Dynamic Topology Engine</h1>
            <p>Mathematical Kernel View</p>
          </div>
          <button
            onClick={() => window.open('/api/export?format=csv', '_blank')}
            style={{
              background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
              color: '#f0f2f8', padding: '6px 10px', borderRadius: 6, fontSize: 10,
              fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter',
              textTransform: 'uppercase', letterSpacing: '0.04em'
            }}
          >
            ↓ Export CSV
          </button>
        </div>
        <div className="stats-bar">
          <div className="stat-pill">
            <div className="label">Mean Entropy</div>
            <div className="value blue">{meanEntropy.toFixed(3)}</div>
          </div>
          <div className="stat-pill">
            <div className="label">Mixing Time</div>
            <div className="value amber">
              {diagnostic.mixing_time < 0 ? '∞' : diagnostic.mixing_time.toFixed(0)}
            </div>
          </div>
        </div>
        <div className="sidebar-content">

          <section>
            <div className="section-heading">Consumer Intent Profile</div>
            <div className="chip-row">
              {topology.intentPresets && Object.keys(topology.intentPresets).map(it => (
                <div
                  key={it}
                  className={`chip ${activeIntent === it ? 'active' : ''}`}
                  onClick={() => setActiveIntent(it)}
                >
                  {it}
                </div>
              ))}
            </div>
          </section>

          <section>
            <div className="section-heading">Exploration τ</div>
            <div className="slider-group">
              <input
                type="range" min="0.01" max="5.0" step="0.01"
                value={temperature}
                style={sliderStyle}
                onChange={e => setTemperature(parseFloat(e.target.value))}
              />
              <div className="slider-meta">
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Deterministic</span>
                <span className="slider-badge">τ = {temperature.toFixed(2)}</span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Exploratory</span>
              </div>
            </div>
          </section>

          <section>
            <div className="section-heading">Sponsor Channel</div>
            <div className="sponsor-cards">
              <div
                className={`sponsor-card ${sponsor === 'none' ? 'active-blue' : ''}`}
                onClick={() => setSponsor('none')}
              >
                <div className="sponsor-dot"/>
                <div className="sponsor-text">
                  <strong>Baseline</strong>
                  <span>Topology only.</span>
                </div>
              </div>
              <div
                className={`sponsor-card ${sponsor === 'beta' ? 'active-purple' : ''}`}
                onClick={() => setSponsor('beta')}
              >
                <div className="sponsor-dot"/>
                <div className="sponsor-text">
                  <strong>Relevance Bid (β)</strong>
                  <span>Alignment-coupled amplification.</span>
                </div>
              </div>
              <div
                className={`sponsor-card ${sponsor === 'friction' ? 'active-amber' : ''}`}
                onClick={() => setSponsor('friction')}
              >
                <div className="sponsor-dot"/>
                <div className="sponsor-text">
                  <strong>Friction Bid (S)</strong>
                  <span>Alignment-independent reduction.</span>
                </div>
              </div>
            </div>
          </section>

          <section>
            <div className="section-heading">Node Alignment Scores</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {topology.labels.map((label: string, i: number) => {
                const align = diagnostic.alignment[i];
                const pct = Math.max(0, Math.min(align, 1)) * 100;
                return (
                  <div key={i} style={{ fontSize: 11 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                      <span style={{ fontFamily: 'JetBrains Mono', color: 'var(--blue)', fontSize: 10 }}>
                        {align.toFixed(3)}
                      </span>
                    </div>
                    <div style={{ height: 3, background: 'var(--border-mid)', borderRadius: 2 }}>
                      <div style={{
                        height: '100%', width: `${pct}%`,
                        background: 'var(--blue)', borderRadius: 2,
                        transition: 'width 0.3s ease'
                      }}/>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <div className="insight-box">
            Click any node to inspect its transition probabilities, entropy, and effective rank.
          </div>
        </div>
      </aside>

      <main className="graph-area">
        <ForceGraph2D
          ref={fgRef}
          graphData={graphData}
          nodeLabel="name"
          nodeColor="color"
          nodeRelSize={4}
          linkColor={() => 'rgba(255,255,255,0.08)'}
          linkWidth={link => link.width as number}
          linkDirectionalParticles={link => link.particles as number}
          linkDirectionalParticleSpeed={0.005}
          linkDirectionalParticleWidth={2}
          d3AlphaDecay={0.03}
          d3VelocityDecay={0.3}
          backgroundColor="transparent"
          onNodeClick={handleNodeClick}
          onBackgroundClick={() => setSelectedNode(null)}
        />

        {/* ── Node Inspector Overlay ─────────────────────── */}
        {selectedNode && (
          <div style={{
            position: 'absolute', top: 20, left: 20, zIndex: 100,
            background: 'var(--bg-panel)',
            border: '1px solid var(--border-bright)',
            borderRadius: 'var(--radius-md)',
            padding: '16px 18px',
            minWidth: 220,
            boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
            animation: 'fadeIn 0.15s ease',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--amber)' }}>
                {selectedNode.name}
              </h3>
              <button
                onClick={() => setSelectedNode(null)}
                style={{
                  background: 'none', border: 'none', color: 'var(--text-muted)',
                  cursor: 'pointer', fontSize: 16, lineHeight: 1, padding: 0,
                }}
              >×</button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 14 }}>
              {[
                ['Alignment', selectedNode.alignment.toFixed(4), 'var(--blue)'],
                ['Row Entropy', `${selectedNode.entropy.toFixed(4)} bits`, 'var(--purple)'],
                ['Effective Rank', selectedNode.effectiveRank.toFixed(2), 'var(--green)'],
              ].map(([label, val, col]) => (
                <div key={label as string} style={{
                  display: 'flex', justifyContent: 'space-between',
                  fontSize: 11, padding: '3px 0',
                  borderBottom: '1px solid var(--border-dim)'
                }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                  <span style={{ fontFamily: 'JetBrains Mono', color: col as string, fontSize: 11 }}>{val}</span>
                </div>
              ))}
            </div>

            <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 8 }}>
              Outbound P
            </div>
            {selectedNode.outbound.map(({ label, prob }) => (
              <div key={label} style={{ marginBottom: 5 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 2 }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                  <span style={{ fontFamily: 'JetBrains Mono', color: 'var(--text-primary)' }}>
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ height: 3, background: 'var(--border-mid)', borderRadius: 2 }}>
                  <div style={{
                    height: '100%',
                    width: `${prob * 100}%`,
                    background: `linear-gradient(90deg, var(--blue), var(--purple))`,
                    borderRadius: 2,
                    transition: 'width 0.25s ease',
                  }}/>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

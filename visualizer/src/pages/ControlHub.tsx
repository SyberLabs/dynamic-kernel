import { useState, useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';

interface PresetMeta {
  name: string;
  description: string;
  undirected: boolean;
  nodeCount: number;
}

const PRESET_CONFIGS: Record<string, { accent: string; icon: string }> = {
  mall:          { accent: '#4f8ef5', icon: '🏬' },
  airport:       { accent: '#34d399', icon: '✈️'  },
  museum:        { accent: '#9b6cf7', icon: '🏛️'  },
  supply_chain:  { accent: '#f5a623', icon: '⛓️'  },
  wheel_city:    { accent: '#f5a623', icon: '🕸️'  },
  rhizome_city:  { accent: '#34d399', icon: '🌐'  },
  neural_dense:  { accent: '#9b6cf7', icon: '🧠'  },
  social_media:  { accent: '#ef4444', icon: '#' },
};

interface PortCard {
  route: string;
  kicker: string;
  title: string;
  accent: string;
  glow: string;
  body: ReactNode;
}

// Ports are grouped by whether they consume the Step-1 topology selection.
// "Topology Explorers" render whatever environment is chosen above; the
// "Standalone Studies" carry their own fixed graphs and ignore the selector.
const TOPOLOGY_EXPLORERS: PortCard[] = [
  {
    route: '/base',
    kicker: 'Explorer',
    title: 'Mathematical Kernel',
    accent: '#4f8ef5',
    glow: 'rgba(79,142,245,0.2)',
    body: (
      <>
        Visualize pure Markovian transition probabilities P<sub>ij</sub>, node alignment scores,
        system entropy, and mixing time. Click any node for a live diagnostic breakdown.
      </>
    ),
  },
  {
    route: '/mall',
    kicker: 'Explorer',
    title: 'Population Simulator',
    accent: '#34d399',
    glow: 'rgba(52,211,153,0.2)',
    body: (
      <>
        Observe 1,000+ discrete agents traversing the topology in real time. Control crowd
        demographics, temperature, and sponsorship channels with live occupancy and edge traffic.
      </>
    ),
  },
];

const STANDALONE_STUDIES: PortCard[] = [
  {
    route: '/demo',
    kicker: 'Portfolio',
    title: 'Semiconductor Demo',
    accent: '#34d399',
    glow: 'rgba(52,211,153,0.2)',
    body: (
      <>
        Explore the strongest current evidence case: where policy budget changes routing, where
        it fails, and why topology decides the feasible frontier.
      </>
    ),
  },
  {
    route: '/compare',
    kicker: 'Study',
    title: 'City Comparison',
    accent: '#f5a623',
    glow: 'rgba(245,166,35,0.2)',
    body: (
      <>
        Run WHEEL and RHIZOME side by side with matched demographics. Compare entropy, mixing
        time, hub saturation, and sponsor response.
      </>
    ),
  },
];

function PortCardView({ card }: { card: PortCard }) {
  const navigate = useNavigate();
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => navigate(card.route)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          navigate(card.route);
        }
      }}
      style={{
        background: '#111520', border: '1px solid rgba(255,255,255,0.08)',
        padding: '28px', borderRadius: 16, cursor: 'pointer',
        transition: 'all 0.2s', boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
        flex: '1 1 300px', maxWidth: 360, outline: 'none',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = card.accent;
        e.currentTarget.style.boxShadow = `0 0 24px ${card.glow}`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
        e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.5)';
      }}
      onFocus={(e) => {
        e.currentTarget.style.borderColor = card.accent;
        e.currentTarget.style.boxShadow = `0 0 24px ${card.glow}`;
      }}
      onBlur={(e) => {
        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
        e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.5)';
      }}
    >
      <div style={{
        fontSize: 11, color: card.accent, fontWeight: 600,
        letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8,
      }}>
        {card.kicker}
      </div>
      <h2 style={{ fontSize: 18, marginBottom: 8, color: card.accent, fontWeight: 700 }}>
        {card.title}
      </h2>
      <p style={{ fontSize: 12, color: '#7a82a0', lineHeight: 1.6, marginBottom: 16 }}>
        {card.body}
      </p>
      <div style={{
        fontSize: 11, color: card.accent, fontWeight: 600,
        display: 'flex', alignItems: 'center', gap: 4,
      }}>
        Open →
      </div>
    </div>
  );
}

function GroupLabel({ children }: { children: ReactNode }) {
  return (
    <div style={{
      fontSize: 10, fontWeight: 600, letterSpacing: '0.12em',
      color: '#4a5070', textTransform: 'uppercase', marginBottom: 16,
      textAlign: 'center',
    }}>
      {children}
    </div>
  );
}

export default function ControlHub() {
  const [presets, setPresets] = useState<Record<string, PresetMeta>>({});
  const [activePreset, setActivePreset] = useState<string>('mall');
  const [loading, setLoading] = useState<string | null>(null);
  const [activePresetName, setActivePresetName] = useState<string>('Enterprise Mall');
  const [presetsLoading, setPresetsLoading] = useState(true);
  const [presetsError, setPresetsError] = useState(false);

  // ── Fetch presets with retry ──────────────────────────────────────────────
  const fetchPresets = useCallback((attempt = 0) => {
    setPresetsLoading(true);
    setPresetsError(false);

    fetch('/api/topology/presets')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(data => {
        setPresets(data);
        setPresetsLoading(false);
      })
      .catch(() => {
        if (attempt < 4) {
          // Exponential back-off: 500ms, 1s, 2s, 4s
          setTimeout(() => fetchPresets(attempt + 1), 500 * Math.pow(2, attempt));
        } else {
          setPresetsLoading(false);
          setPresetsError(true);
        }
      });
  }, []);

  useEffect(() => {
    fetchPresets();

    fetch('/api/topology')
      .then(r => r.json())
      .then(d => {
        if (d.presetName) setActivePresetName(d.presetName);
        if (d.preset) setActivePreset(d.preset);
      })
      .catch(() => {});
  }, [fetchPresets]);

  const handleSelectPreset = async (key: string) => {
    if (key === activePreset && loading === null) return;
    setLoading(key);
    try {
      const res = await fetch('/api/topology/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset: key }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.presetName) setActivePresetName(data.presetName);
      setActivePreset(key);
    } catch (e) {
      console.error('Topology load failed:', e);
    } finally {
      setLoading(null);
    }
  };

  // ── Skeleton placeholder while loading ───────────────────────────────────
  const renderPresetArea = () => {
    if (presetsLoading) {
      return (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
          gap: 12,
        }}>
          {[1, 2, 3, 4].map(i => (
            <div key={i} style={{
              background: '#111520',
              border: '1px solid rgba(255,255,255,0.06)',
              padding: '18px 16px',
              borderRadius: 14,
              height: 130,
              animation: 'pulse 1.5s ease infinite',
            }}/>
          ))}
        </div>
      );
    }

    if (presetsError) {
      return (
        <div style={{
          textAlign: 'center',
          padding: '32px 16px',
          background: '#111520',
          border: '1px solid rgba(255,60,60,0.2)',
          borderRadius: 14,
          color: '#f87171',
        }}>
          <div style={{ fontSize: 24, marginBottom: 10 }}>⚠️</div>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>Backend Unreachable</div>
          <div style={{ fontSize: 11, color: '#7a82a0', marginBottom: 16 }}>
            Could not load topology presets. Is the FastAPI server running?
          </div>
          <button
            onClick={() => fetchPresets()}
            style={{
              background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.15)',
              color: '#f0f2f8', padding: '8px 18px', borderRadius: 8,
              fontSize: 12, fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter',
            }}
          >
            Retry Connection
          </button>
        </div>
      );
    }

    if (Object.keys(presets).length === 0) {
      return (
        <div style={{ textAlign: 'center', color: '#4a5070', fontSize: 13, padding: 32 }}>
          No presets returned from server.
        </div>
      );
    }

    return (
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
        gap: 12,
      }}>
        {Object.entries(presets).map(([key, meta]) => {
          const cfg = PRESET_CONFIGS[key] ?? { accent: '#7a82a0', icon: '🔵' };
          const isActive = activePreset === key;
          const isLoading = loading === key;

          return (
            <div
              key={key}
              onClick={() => handleSelectPreset(key)}
              style={{
                background: isActive ? `${cfg.accent}14` : '#111520',
                border: `1px solid ${isActive ? cfg.accent : 'rgba(255,255,255,0.08)'}`,
                boxShadow: isActive ? `0 0 20px ${cfg.accent}30` : 'none',
                padding: '18px 16px',
                borderRadius: 14,
                cursor: loading ? 'default' : 'pointer',
                transition: 'all 0.2s ease',
                position: 'relative',
                overflow: 'hidden',
                opacity: loading && !isLoading ? 0.5 : 1,
              }}
              onMouseEnter={e => {
                if (!isActive && !loading) e.currentTarget.style.borderColor = `${cfg.accent}88`;
              }}
              onMouseLeave={e => {
                if (!isActive) e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
              }}
            >
              {/* Accent glow top bar */}
              {isActive && (
                <div style={{
                  position: 'absolute', top: 0, left: 0, right: 0,
                  height: 2, background: cfg.accent,
                  borderRadius: '14px 14px 0 0',
                }}/>
              )}

              <div style={{ fontSize: 24, marginBottom: 8 }}>{cfg.icon}</div>
              <div style={{
                fontSize: 13, fontWeight: 600, color: isActive ? cfg.accent : '#f0f2f8',
                marginBottom: 5,
              }}>
                {meta.name}
              </div>
              <div style={{ fontSize: 10, color: '#7a82a0', lineHeight: 1.5, marginBottom: 10 }}>
                {meta.description}
              </div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <span style={{
                  fontSize: 9, fontWeight: 600, padding: '2px 7px',
                  borderRadius: 4, border: `1px solid ${cfg.accent}44`,
                  color: cfg.accent, background: `${cfg.accent}18`,
                  textTransform: 'uppercase', letterSpacing: '0.06em',
                }}>
                  {meta.nodeCount} nodes
                </span>
                <span style={{
                  fontSize: 9, fontWeight: 600, padding: '2px 7px',
                  borderRadius: 4,
                  border: '1px solid rgba(255,255,255,0.12)',
                  color: '#7a82a0',
                  textTransform: 'uppercase', letterSpacing: '0.06em',
                }}>
                  {meta.undirected ? 'undirected' : 'directed ↗'}
                </span>
              </div>

              {isLoading && (
                <div style={{
                  position: 'absolute', inset: 0,
                  background: 'rgba(8,11,15,0.7)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  borderRadius: 14,
                }}>
                  <div style={{
                    width: 20, height: 20, border: `2px solid ${cfg.accent}`,
                    borderTopColor: 'transparent', borderRadius: '50%',
                    animation: 'spin 0.6s linear infinite',
                  }}/>
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div style={{
      width: '100vw', minHeight: '100vh',
      background: 'radial-gradient(circle at 50% 0%, #151a27 0%, #080b0f 80%)',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      color: '#f0f2f8', fontFamily: 'Inter',
      gap: 48, padding: '40px 20px',
    }}>

      {/* ── Hero Header ── */}
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontSize: 11, fontWeight: 600, letterSpacing: '0.18em',
          color: '#4f8ef5', textTransform: 'uppercase', marginBottom: 12,
        }}>
          SyberLabs Intelligence Field
        </div>
        <h1 style={{
          fontSize: 36, fontWeight: 700, letterSpacing: '-0.02em',
          background: 'linear-gradient(90deg, #fff 30%, #4f8ef5)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          marginBottom: 10,
        }}>
          Topology Control Hub
        </h1>
        <p style={{ color: '#7a82a0', fontSize: 13, maxWidth: 480, lineHeight: 1.6 }}>
          Choose a topology to load into the explorers, or jump straight into a
          self-contained study. Each port observes or analyses the routing kernel in real time.
        </p>
      </div>

      {/* ── Topology Selector ── */}
      <div style={{ width: '100%', maxWidth: 780 }}>
        <div style={{
          fontSize: 10, fontWeight: 600, letterSpacing: '0.12em',
          color: '#4a5070', textTransform: 'uppercase', marginBottom: 6,
          textAlign: 'center',
        }}>
          Choose Topology
        </div>
        <div style={{
          fontSize: 11, color: '#7a82a0', textAlign: 'center',
          marginBottom: 16, lineHeight: 1.5,
        }}>
          Loads into the <span style={{ color: '#4f8ef5', fontWeight: 600 }}>Topology Explorers</span> below.
          The standalone studies use their own fixed graphs.
        </div>

        {renderPresetArea()}

        {/* Active badge */}
        {!presetsError && (
          <div style={{
            textAlign: 'center', marginTop: 14,
            fontSize: 11, color: '#4a5070',
          }}>
            Active environment:{' '}
            <span style={{ color: '#4f8ef5', fontWeight: 600 }}>{activePresetName}</span>
          </div>
        )}
      </div>

      {/* ── Topology Explorers (consume the Step-1 selection) ── */}
      <div style={{ width: '100%', maxWidth: 780 }}>
        <GroupLabel>Topology Explorers · use the selection above</GroupLabel>
        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', justifyContent: 'center' }}>
          {TOPOLOGY_EXPLORERS.map((card) => (
            <PortCardView key={card.route} card={card} />
          ))}
        </div>
      </div>

      {/* ── Standalone Studies (self-contained fixed graphs) ── */}
      <div style={{ width: '100%', maxWidth: 780 }}>
        <GroupLabel>Standalone Studies · self-contained</GroupLabel>
        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', justifyContent: 'center' }}>
          {STANDALONE_STUDIES.map((card) => (
            <PortCardView key={card.route} card={card} />
          ))}
        </div>
      </div>

    </div>
  );
}

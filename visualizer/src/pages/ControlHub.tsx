import { useState, useEffect, useCallback } from 'react';
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
  social_media:  { accent: '#ef4444', icon: '#' },
};

export default function ControlHub() {
  const navigate = useNavigate();
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
        <p style={{ color: '#7a82a0', fontSize: 13, maxWidth: 460, lineHeight: 1.6 }}>
          Select an environment, then enter a port to observe or analyse the
          routing kernel in real-time.
        </p>
      </div>

      {/* ── Topology Selector ── */}
      <div style={{ width: '100%', maxWidth: 780 }}>
        <div style={{
          fontSize: 10, fontWeight: 600, letterSpacing: '0.12em',
          color: '#4a5070', textTransform: 'uppercase', marginBottom: 16,
          textAlign: 'center',
        }}>
          Step 1 — Choose Topology
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

      {/* ── Port Cards ── */}
      <div style={{ width: '100%', maxWidth: 780 }}>
        <div style={{
          fontSize: 10, fontWeight: 600, letterSpacing: '0.12em',
          color: '#4a5070', textTransform: 'uppercase', marginBottom: 16,
          textAlign: 'center',
        }}>
          Step 2 — Enter a Port
        </div>

        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', justifyContent: 'center' }}>
          {/* Mathematical Kernel Port */}
          <div
            onClick={() => navigate('/base')}
            style={{
              background: '#111520', border: '1px solid rgba(255,255,255,0.08)',
              padding: '28px 28px', borderRadius: 16, width: 340, cursor: 'pointer',
              transition: 'all 0.2s', boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
              flex: '1 1 300px', maxWidth: 360,
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = '#4f8ef5';
              e.currentTarget.style.boxShadow = '0 0 24px rgba(79,142,245,0.2)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
              e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.5)';
            }}
          >
            <div style={{
              fontSize: 11, color: '#4f8ef5', fontWeight: 600,
              letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8,
            }}>
              Port 1
            </div>
            <h2 style={{ fontSize: 18, marginBottom: 8, color: '#4f8ef5', fontWeight: 700 }}>
              Mathematical Kernel
            </h2>
            <p style={{ fontSize: 12, color: '#7a82a0', lineHeight: 1.6, marginBottom: 16 }}>
              Visualize pure Markovian transition probabilities P<sub>ij</sub>, node alignment
              scores, system entropy, and mixing time. Click any node for a live diagnostic
              breakdown.
            </p>
            <div style={{
              fontSize: 11, color: '#4f8ef5', fontWeight: 600,
              display: 'flex', alignItems: 'center', gap: 4,
            }}>
              Open → 
            </div>
          </div>

          {/* Mall Port */}
          <div
            onClick={() => navigate('/mall')}
            style={{
              background: '#111520', border: '1px solid rgba(255,255,255,0.08)',
              padding: '28px 28px', borderRadius: 16, width: 340, cursor: 'pointer',
              transition: 'all 0.2s', boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
              flex: '1 1 300px', maxWidth: 360,
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = '#34d399';
              e.currentTarget.style.boxShadow = '0 0 24px rgba(52,211,153,0.2)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
              e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.5)';
            }}
          >
            <div style={{
              fontSize: 11, color: '#34d399', fontWeight: 600,
              letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8,
            }}>
              Port 2
            </div>
            <h2 style={{ fontSize: 18, marginBottom: 8, color: '#34d399', fontWeight: 700 }}>
              Population Simulator
            </h2>
            <p style={{ fontSize: 12, color: '#7a82a0', lineHeight: 1.6, marginBottom: 16 }}>
              Observe 1,000+ discrete agents traversing the topology in real-time.
              Control crowd demographics, temperature, and sponsorship channels.
              Live node occupancy and edge traffic counts.
            </p>
            <div style={{
              fontSize: 11, color: '#34d399', fontWeight: 600,
              display: 'flex', alignItems: 'center', gap: 4,
            }}>
              Open →
            </div>
          </div>

          <div
            onClick={() => navigate('/compare')}
            style={{
              background: '#111520', border: '1px solid rgba(255,255,255,0.08)',
              padding: '28px 28px', borderRadius: 16, width: 340, cursor: 'pointer',
              transition: 'all 0.2s', boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
              flex: '1 1 300px', maxWidth: 360,
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = '#f5a623';
              e.currentTarget.style.boxShadow = '0 0 24px rgba(245,166,35,0.2)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
              e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.5)';
            }}
          >
            <div style={{
              fontSize: 11, color: '#f5a623', fontWeight: 600,
              letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8,
            }}>
              Port 3
            </div>
            <h2 style={{ fontSize: 18, marginBottom: 8, color: '#f5a623', fontWeight: 700 }}>
              City Comparison
            </h2>
            <p style={{ fontSize: 12, color: '#7a82a0', lineHeight: 1.6, marginBottom: 16 }}>
              Run WHEEL and RHIZOME side by side with matched demographics.
              Compare entropy, mixing time, hub saturation, and sponsor response.
            </p>
            <div style={{
              fontSize: 11, color: '#f5a623', fontWeight: 600,
              display: 'flex', alignItems: 'center', gap: 4,
            }}>
              Open -&gt;
            </div>
          </div>

          <div
            onClick={() => navigate('/neural')}
            style={{
              background: '#111520', border: '1px solid rgba(255,255,255,0.08)',
              padding: '28px 28px', borderRadius: 16, width: 340, cursor: 'pointer',
              transition: 'all 0.2s', boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
              flex: '1 1 300px', maxWidth: 360,
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = '#9b6cf7';
              e.currentTarget.style.boxShadow = '0 0 24px rgba(155,108,247,0.2)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
              e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.5)';
            }}
          >
            <div style={{
              fontSize: 11, color: '#9b6cf7', fontWeight: 600,
              letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8,
            }}>
              Port 4
            </div>
            <h2 style={{ fontSize: 18, marginBottom: 8, color: '#9b6cf7', fontWeight: 700 }}>
              Neural Optimizer
            </h2>
            <p style={{ fontSize: 12, color: '#7a82a0', lineHeight: 1.6, marginBottom: 16 }}>
              Watch a fully connected network reorganize its own beta weights in real time
              to maximize a chosen structural symmetry.
            </p>
            <div style={{
              fontSize: 11, color: '#9b6cf7', fontWeight: 600,
              display: 'flex', alignItems: 'center', gap: 4,
            }}>
              Open -&gt;
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}

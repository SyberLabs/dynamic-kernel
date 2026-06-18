import { useEffect, useMemo, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  Factory,
  GitBranch,
  Layers,
  RotateCcw,
  SlidersHorizontal,
  Target,
} from 'lucide-react';

type DemoMode = 'relocation' | 'surgery' | 'feedback';
type ChoiceLocation = 'upstream_choice' | 'route_commitment' | 'downstream_serial';
type SurgeryTopology = 'serial' | 'reconsideration_exits';

interface ChoiceRow {
  location: ChoiceLocation;
  budget: number;
  viable_rate: number;
  mean_onshore_share: number;
  mean_completion: number;
}

interface SurgeryRow {
  topology: SurgeryTopology;
  budget: number;
  viable_rate: number;
  mean_onshore_share: number;
  mean_completion: number;
}

interface FeedbackRow {
  feedback_rate: number;
  domestic_pull: number;
  viable_rate: number;
  mean_onshore_share: number;
  mean_completion: number;
}

interface DemoPayload {
  config: {
    agents?: number;
    steps?: number;
    seeds?: number[];
  };
  summary: {
    relocation_max_share_lift?: Record<ChoiceLocation, number>;
    surgery_max_share_lift?: Record<SurgeryTopology, number>;
    nulls_with_high_share_but_no_robust_transition?: number;
  };
  choicePointRelocation: ChoiceRow[];
  topologySurgery: SurgeryRow[];
  feedbackContinuum: FeedbackRow[];
}

interface ActiveMetrics {
  share: number;
  completion: number;
  viable: number;
  lift: number;
  // Baseline the current cell is compared against (zero-budget, or no-pull for
  // feedback). Surfacing both endpoints turns a flat result into visible evidence
  // ("0.502 → 0.502, +0.000") rather than a silent non-change.
  baseShare: number;
  viableBase: number;
  viableLift: number;
  status: 'active' | 'limited' | 'blocked';
  // Which metric actually carries this experiment's signal. Relocation/surgery
  // are share-driven; feedback's share is flat noise and its story is robustness.
  headline: 'share' | 'robustness';
  verdict: string;
  mechanism: string;
}

const BUDGETS = [0, 4, 8, 12];
const FEEDBACK_RATES = [0, 0.05, 0.1, 0.15, 0.3, 0.5];

const STATUS_COPY: Record<ActiveMetrics['status'], string> = {
  active: 'Lever is live',
  limited: 'Partial response',
  blocked: 'Lever is inert',
};

const MODE_COPY: Record<DemoMode, { eyebrow: string; title: string; subtitle: string }> = {
  relocation: {
    eyebrow: 'Experiment 01',
    title: 'Policy Location',
    subtitle: 'Hold the subsidy fixed and move it to different positions in the graph.',
  },
  surgery: {
    eyebrow: 'Experiment 02',
    title: 'Topology Surgery',
    subtitle: 'Compare a serial corridor against a graph with reconsideration exits.',
  },
  feedback: {
    eyebrow: 'Experiment 03',
    title: 'Preference Memory',
    subtitle: 'Vary the feedback rate and domestic pull under fixed constraints.',
  },
};

function pct(value: number, digits = 1) {
  return `${(100 * value).toFixed(digits)}%`;
}

function fmt(value: number, digits = 3) {
  return value.toFixed(digits);
}

function signed(value: number, digits = 3) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

function near(a: number, b: number) {
  return Math.abs(a - b) < 1e-9;
}

function metricStatus(lift: number, viable: number): ActiveMetrics['status'] {
  if (lift > 0.04 && viable >= 0.8) return 'active';
  if (lift > 0.01 || viable > 0.4) return 'limited';
  return 'blocked';
}

function MetricCard({
  label,
  value,
  context,
  fill,
  tone,
  primary = false,
}: {
  label: string;
  value: string;
  context: string;
  fill: number;
  tone: 'green' | 'amber' | 'blue' | 'red';
  primary?: boolean;
}) {
  return (
    <div className={`demo-metric demo-metric-${tone} ${primary ? 'primary' : ''}`}>
      <span className="demo-metric-label">{label}</span>
      <strong className="demo-metric-value flash">{value}</strong>
      <div className="demo-metric-bar">
        <i style={{ width: `${Math.max(2, Math.min(100, fill * 100))}%` }} />
      </div>
      <span className="demo-metric-context">{context}</span>
    </div>
  );
}

function SegmentButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: ReactNode;
  onClick: () => void;
}) {
  return (
    <button className={`demo-segment ${active ? 'active' : ''}`} onClick={onClick} type="button">
      {children}
    </button>
  );
}

function ResponseChart({
  rows,
  active,
  unit,
  baseline,
  inert,
}: {
  rows: Array<{ budget: number; mean_onshore_share: number; viable_rate: number }>;
  active: number;
  unit: string;
  baseline: number;
  inert: boolean;
}) {
  const width = 460;
  const height = 188;
  const padL = 40;
  const padR = 18;
  const padT = 22;
  const padB = 30;
  const sorted = [...rows].sort((a, b) => a.budget - b.budget);
  const values = sorted.map((row) => row.mean_onshore_share);
  const min = Math.min(...values, 0.45);
  const max = Math.max(...values, 0.76);
  const span = Math.max(max - min, 0.01);

  const points = sorted.map((row, index) => {
    const x = padL + (index / Math.max(sorted.length - 1, 1)) * (width - padL - padR);
    const y = height - padB - ((row.mean_onshore_share - min) / span) * (height - padT - padB);
    return { ...row, x, y };
  });

  const line = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
  const area = points.length
    ? `${line} L ${points[points.length - 1].x} ${height - padB} L ${points[0].x} ${height - padB} Z`
    : '';
  const activePoint = points.find((p) => near(p.budget, active));
  const gridY = [0, 0.5, 1].map((t) => height - padB - t * (height - padT - padB));
  const yOf = (v: number) => height - padB - ((v - min) / span) * (height - padT - padB);
  const baselineY = yOf(baseline);

  return (
    <svg className="demo-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Onshore share response curve">
      <defs>
        <linearGradient id="demo-area" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(52,211,153,0.30)" />
          <stop offset="100%" stopColor="rgba(52,211,153,0)" />
        </linearGradient>
      </defs>
      {gridY.map((y, i) => (
        <line key={i} className="demo-grid" x1={padL} y1={y} x2={width - padR} y2={y} />
      ))}
      <text className="demo-axis" x={padL - 8} y={gridY[2] + 3} textAnchor="end">{pct(max, 0)}</text>
      <text className="demo-axis" x={padL - 8} y={gridY[0] + 3} textAnchor="end">{pct(min, 0)}</text>
      {/* zero-budget baseline — a flat curve sitting ON this line reads as
          "no lift", making the null result legible instead of ambiguous */}
      <line className="demo-baseline" x1={padL} y1={baselineY} x2={width - padR} y2={baselineY} />
      <text className="demo-axis demo-baseline-label" x={padL + 2} y={baselineY - 4}>baseline</text>
      {area && <path className="demo-area" d={area} fill="url(#demo-area)" />}
      <path className={`demo-line ${inert ? 'inert' : ''}`} d={line} />
      {inert && (
        <text className="demo-inert-note" x={(padL + width - padR) / 2} y={baselineY + 16} textAnchor="middle">
          flat — lever inert at this position
        </text>
      )}
      {points.map((p) => (
        <g key={p.budget}>
          <circle className={near(p.budget, active) ? 'active' : ''} cx={p.x} cy={p.y} r={near(p.budget, active) ? 5.5 : 4} />
          <text className="demo-axis" x={p.x} y={height - 9} textAnchor="middle">{p.budget}</text>
        </g>
      ))}
      {activePoint && (
        <text className="demo-callout" x={activePoint.x} y={activePoint.y - 12} textAnchor="middle">
          {pct(activePoint.mean_onshore_share)}
        </text>
      )}
      <text className="demo-axis demo-axis-unit" x={width - padR} y={height - 9} textAnchor="end">{unit}</text>
    </svg>
  );
}

// ── Production graph model ───────────────────────────────────────────────────
// One linear production pipeline: Inputs → Commit → Packaging → US Demand.
// These four are the only "real" places — a chip flows through them. The policy
// experiments are NOT extra nodes; they are operations ON this pipeline:
//   • relocation  — a subsidy LEVER injected at one pipeline position
//   • surgery     — whether the downstream position has a live EXIT back to an
//                   alternative route, or is a serial dead-end
//   • feedback    — a preference-MEMORY field that modifies the Commit decision
// The finding is single-axis: a lever only does work where a live alternative
// exists. So the picture shows position + whether the choice at it is live.

type StageId = 'inputs' | 'commit' | 'packaging' | 'demand';

interface Stage {
  id: StageId;
  name: string;
  role: string;
  x: number;
}

// SVG user-space is 720 × 300. Pipeline runs along y = 132.
const LANE_Y = 132;
const STAGES: Stage[] = [
  { id: 'inputs', name: 'Inputs', role: 'raw + tooling', x: 96 },
  { id: 'commit', name: 'Route Commit', role: 'fab decision', x: 288 },
  { id: 'packaging', name: 'Packaging', role: 'assembly + test', x: 468 },
  { id: 'demand', name: 'US Demand', role: 'onshore target', x: 636 },
];

// Where the lever sits for each relocation site / surgery topology.
const LEVER_AT: Record<string, StageId> = {
  upstream_choice: 'inputs',
  route_commitment: 'commit',
  downstream_serial: 'packaging',
  serial: 'packaging',
  reconsideration_exits: 'packaging',
};

function stageX(id: StageId): number {
  return STAGES.find((s) => s.id === id)!.x;
}

function ProductionGraph({
  mode,
  location,
  topology,
  status,
  share,
}: {
  mode: DemoMode;
  location: ChoiceLocation;
  topology: SurgeryTopology;
  status: ActiveMetrics['status'];
  share: number;
}) {
  // Lever position only applies to the two pipeline experiments.
  const leverStage: StageId | null =
    mode === 'relocation' ? LEVER_AT[location] : mode === 'surgery' ? LEVER_AT[topology] : null;
  const leverX = leverStage ? stageX(leverStage) : null;

  // Is there a live alternative route at the lever's position?
  //  - relocation: only the upstream position has competing outgoing routes.
  //  - surgery: only the reconsideration topology re-opens a downstream exit.
  const choiceLive =
    mode === 'relocation'
      ? location === 'upstream_choice'
      : mode === 'surgery'
      ? topology === 'reconsideration_exits'
      : false;

  const flowing = status === 'active';
  // Feedback halo sits on the Commit decision; intensity by feedback regime.
  const memoryOn = mode === 'feedback';

  const W = 720;
  const H = 300;
  const inputX = stageX('inputs');
  const commitX = stageX('commit');
  const packX = stageX('packaging');
  const demandX = stageX('demand');

  // The serial spine (always present) and the alternative branch that only
  // carries flow when a live choice exists at the relevant position.
  const branchY = LANE_Y - 64;

  const subLabel =
    mode === 'relocation'
      ? 'subsidy lever · relocation'
      : mode === 'surgery'
      ? 'topology surgery · downstream exit'
      : 'preference memory · decision field';

  return (
    <div className={`demo-map ${flowing ? 'flowing' : ''}`}>
      <div className="demo-map-legend">
        <span className="demo-map-eyebrow">{subLabel}</span>
        <span className="demo-map-flow">
          <i className={flowing ? 'on' : ''} /> {flowing ? 'Routing responds' : 'Routing pinned'}
        </span>
      </div>

      <svg className="demo-pipe" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Semiconductor production pipeline">
        <defs>
          <linearGradient id="pipe-flow" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgba(52,211,153,0)" />
            <stop offset="50%" stopColor="rgba(52,211,153,0.9)" />
            <stop offset="100%" stopColor="rgba(52,211,153,0)" />
          </linearGradient>
        </defs>

        {/* ── alternative branch route (commit → up-and-over → demand) ── */}
        <path
          className={`demo-pipe-alt ${choiceLive ? 'live' : ''}`}
          d={`M ${commitX} ${LANE_Y} C ${commitX} ${branchY}, ${packX} ${branchY}, ${demandX} ${LANE_Y}`}
        />
        {choiceLive && (
          <text className="demo-pipe-alt-label" x={(commitX + demandX) / 2} y={branchY - 8} textAnchor="middle">
            live alternative route
          </text>
        )}

        {/* ── the serial spine: the pipeline itself ── */}
        <line className="demo-pipe-spine" x1={inputX} y1={LANE_Y} x2={demandX} y2={LANE_Y} />
        {flowing && (
          <line className="demo-pipe-spine-flow" x1={inputX} y1={LANE_Y} x2={demandX} y2={LANE_Y} />
        )}

        {/* directional chevrons between stages */}
        {STAGES.slice(0, -1).map((s, i) => {
          const next = STAGES[i + 1];
          const mx = (s.x + next.x) / 2;
          return (
            <path key={s.id} className="demo-pipe-chevron" d={`M ${mx - 4} ${LANE_Y - 5} L ${mx + 4} ${LANE_Y} L ${mx - 4} ${LANE_Y + 5}`} />
          );
        })}

        {/* ── feedback / memory field on the Commit decision ── */}
        {memoryOn && (
          <g className="demo-pipe-memory">
            <circle cx={commitX} cy={LANE_Y} r="46" />
            <circle cx={commitX} cy={LANE_Y} r="34" />
            <text x={commitX} y={LANE_Y - 58} textAnchor="middle">preference memory</text>
          </g>
        )}

        {/* ── stage nodes ── */}
        {STAGES.map((s) => {
          const isDemand = s.id === 'demand';
          const isLever = s.id === leverStage;
          const cls = [
            'demo-stage-node',
            isDemand ? 'demand' : '',
            isLever ? 'lever-site' : '',
          ]
            .filter(Boolean)
            .join(' ');
          const r = isDemand ? 24 : 20;
          return (
            <g key={s.id} className={cls}>
              {isDemand && (
                <circle
                  className="demo-stage-fill"
                  cx={s.x}
                  cy={LANE_Y}
                  r={r - 3}
                  style={{ opacity: 0.12 + share * 0.55 }}
                />
              )}
              <circle className="demo-stage-ring" cx={s.x} cy={LANE_Y} r={r} />
              <text className="demo-stage-name" x={s.x} y={LANE_Y + r + 18} textAnchor="middle">
                {s.name}
              </text>
              <text className="demo-stage-role" x={s.x} y={LANE_Y + r + 31} textAnchor="middle">
                {isDemand ? pct(share, 0) + ' onshore' : s.role}
              </text>
            </g>
          );
        })}

        {/* ── the subsidy lever marker, parked under its pipeline position ── */}
        {leverX !== null && (
          <g
            className={`demo-lever ${choiceLive ? 'live' : 'inert'}`}
            style={{ transform: `translateX(${leverX}px)` }}
          >
            {/* injection arrow up into the lane */}
            <line className="demo-lever-stem" x1={0} y1={LANE_Y + 58} x2={0} y2={LANE_Y + 26} />
            <path className="demo-lever-head" d={`M -5 ${LANE_Y + 32} L 0 ${LANE_Y + 24} L 5 ${LANE_Y + 32}`} />
            <rect className="demo-lever-body" x={-44} y={LANE_Y + 58} width={88} height={34} rx={8} />
            <text className="demo-lever-title" x={0} y={LANE_Y + 72} textAnchor="middle">
              SUBSIDY
            </text>
            <text className="demo-lever-state" x={0} y={LANE_Y + 84} textAnchor="middle">
              {choiceLive ? 'lever is live' : 'no live choice'}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}

export default function DemoPort() {
  const [payload, setPayload] = useState<DemoPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<DemoMode>('surgery');
  const [budget, setBudget] = useState(8);
  const [location, setLocation] = useState<ChoiceLocation>('upstream_choice');
  const [topology, setTopology] = useState<SurgeryTopology>('reconsideration_exits');
  const [feedbackRate, setFeedbackRate] = useState(0.15);
  const [domesticPull, setDomesticPull] = useState(true);

  // Counts every control change so the readouts can flash an acknowledgement on
  // each interaction — including the inert case where the number doesn't move
  // (which is itself the point: "you moved the lever; nothing changed").
  const [pulse, setPulse] = useState(0);
  useEffect(() => {
    setPulse((n) => n + 1);
  }, [budget, location, topology, feedbackRate, domesticPull, mode]);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/demo/semiconductor', { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((data: DemoPayload) => {
        setPayload(data);
        setError(null);
      })
      .catch((err: Error) => {
        if (err.name !== 'AbortError') setError(err.message);
      });
    return () => controller.abort();
  }, []);

  const active = useMemo<ActiveMetrics | null>(() => {
    if (!payload) return null;
    if (mode === 'relocation') {
      const row = payload.choicePointRelocation.find((item) => item.location === location && near(item.budget, budget));
      const base = payload.choicePointRelocation.find((item) => item.location === location && near(item.budget, 0));
      if (!row || !base) return null;
      const lift = row.mean_onshore_share - base.mean_onshore_share;
      const status = metricStatus(lift, row.viable_rate);
      return {
        share: row.mean_onshore_share,
        completion: row.mean_completion,
        viable: row.viable_rate,
        lift,
        baseShare: base.mean_onshore_share,
        viableBase: base.viable_rate,
        viableLift: row.viable_rate - base.viable_rate,
        status,
        headline: 'share',
        verdict: location === 'upstream_choice' ? 'Budget is live before commitment.' : 'Budget is inert after commitment.',
        mechanism: location === 'upstream_choice'
          ? 'The intervention changes a row with competing outgoing routes.'
          : 'The selected row has no meaningful alternative, so softmax cannot express the policy.',
      };
    }
    if (mode === 'surgery') {
      const row = payload.topologySurgery.find((item) => item.topology === topology && near(item.budget, budget));
      const base = payload.topologySurgery.find((item) => item.topology === topology && near(item.budget, 0));
      if (!row || !base) return null;
      const lift = row.mean_onshore_share - base.mean_onshore_share;
      const status = metricStatus(lift, row.viable_rate);
      return {
        share: row.mean_onshore_share,
        completion: row.mean_completion,
        viable: row.viable_rate,
        lift,
        baseShare: base.mean_onshore_share,
        viableBase: base.viable_rate,
        viableLift: row.viable_rate - base.viable_rate,
        status,
        headline: 'share',
        verdict: topology === 'reconsideration_exits' ? 'Topology unlocks the policy lever.' : 'Serial topology blocks the lever.',
        mechanism: topology === 'reconsideration_exits'
          ? 'A downstream penalty becomes effective only after the graph is given exits back to alternative routes.'
          : 'A single-file corridor gives the agent nowhere to redirect, regardless of budget.',
      };
    }
    const row = payload.feedbackContinuum.find(
      (item) => near(item.feedback_rate, feedbackRate) && near(item.domestic_pull, domesticPull ? 1 : 0),
    );
    const base = payload.feedbackContinuum.find(
      (item) => near(item.feedback_rate, feedbackRate) && near(item.domestic_pull, 0),
    );
    if (!row || !base) return null;
    const lift = row.mean_onshore_share - base.mean_onshore_share;
    const viableLift = row.viable_rate - base.viable_rate;
    // Feedback's share is flat noise; its real lever is robustness. Status keys
    // off the viability gain from domestic pull, not the (negligible) share lift.
    const status: ActiveMetrics['status'] =
      viableLift >= 0.4 ? 'active' : viableLift > 0 ? 'limited' : 'blocked';
    return {
      share: row.mean_onshore_share,
      completion: row.mean_completion,
      viable: row.viable_rate,
      lift,
      baseShare: base.mean_onshore_share,
      viableBase: base.viable_rate,
      viableLift,
      status,
      headline: 'robustness',
      verdict: domesticPull ? 'Memory buys reliability, not production.' : 'Without pull, outcomes stay fragile.',
      mechanism: domesticPull
        ? 'Domestic pull barely moves mean share, but it lifts the viable-seed rate sharply — the same policy completes far more often.'
        : 'Feedback alone changes route propensity; without domestic pull, few seeds reach a robust onshore outcome.',
    };
  }, [budget, domesticPull, feedbackRate, location, mode, payload, topology]);

  const chartRows = useMemo(() => {
    if (!payload) return [];
    if (mode === 'relocation') return payload.choicePointRelocation.filter((row) => row.location === location);
    if (mode === 'surgery') return payload.topologySurgery.filter((row) => row.topology === topology);
    return payload.feedbackContinuum
      .filter((row) => near(row.domestic_pull, domesticPull ? 1 : 0))
      .map((row) => ({
        budget: row.feedback_rate,
        mean_onshore_share: row.mean_onshore_share,
        viable_rate: row.viable_rate,
      }));
  }, [domesticPull, location, mode, payload, topology]);

  const currentBudgetForChart = mode === 'feedback' ? feedbackRate : budget;
  const seeds = payload?.config.seeds?.length ?? 0;
  const maxSurgeryLift = payload?.summary.surgery_max_share_lift?.reconsideration_exits ?? 0;
  const nullCount = payload?.summary.nulls_with_high_share_but_no_robust_transition ?? 0;

  return (
    <div className="demo-shell">
      <aside className="demo-rail">
        <div className="demo-brand">
          <span>Portfolio Demo</span>
          <h1>Semiconductor Policy Lab</h1>
          <p>Where a subsidy lands in the graph decides whether it does anything at all.</p>
        </div>

        <div className="demo-section">
          <div className="demo-section-title"><SlidersHorizontal size={13} /> Experiment lane</div>
          <div className="demo-segments">
            <SegmentButton active={mode === 'relocation'} onClick={() => setMode('relocation')}>Location</SegmentButton>
            <SegmentButton active={mode === 'surgery'} onClick={() => setMode('surgery')}>Topology</SegmentButton>
            <SegmentButton active={mode === 'feedback'} onClick={() => setMode('feedback')}>Memory</SegmentButton>
          </div>
        </div>

        {mode === 'relocation' && (
          <div className="demo-section">
            <div className="demo-section-title"><GitBranch size={13} /> Intervention site</div>
            <div className="demo-option-list">
              <SegmentButton active={location === 'upstream_choice'} onClick={() => setLocation('upstream_choice')}>Upstream choice</SegmentButton>
              <SegmentButton active={location === 'route_commitment'} onClick={() => setLocation('route_commitment')}>Route commit</SegmentButton>
              <SegmentButton active={location === 'downstream_serial'} onClick={() => setLocation('downstream_serial')}>Downstream serial</SegmentButton>
            </div>
          </div>
        )}

        {mode === 'surgery' && (
          <div className="demo-section">
            <div className="demo-section-title"><Layers size={13} /> Graph structure</div>
            <div className="demo-option-list">
              <SegmentButton active={topology === 'serial'} onClick={() => setTopology('serial')}>Serial corridor</SegmentButton>
              <SegmentButton active={topology === 'reconsideration_exits'} onClick={() => setTopology('reconsideration_exits')}>Reconsideration exits</SegmentButton>
            </div>
          </div>
        )}

        {mode !== 'feedback' && (
          <div className="demo-section">
            <div className="demo-section-title">
              <Factory size={13} /> Policy budget
              <strong className="demo-section-readout">{budget}</strong>
            </div>
            <input
              aria-label="Policy budget"
              max={12}
              min={0}
              onChange={(event) => setBudget(Number(event.target.value))}
              step={4}
              style={{ '--pct': `${(budget / 12) * 100}%` } as CSSProperties}
              type="range"
              value={budget}
            />
            <div className="demo-budget-row">
              {BUDGETS.map((item) => (
                <button className={item === budget ? 'active' : ''} key={item} onClick={() => setBudget(item)} type="button">
                  {item}
                </button>
              ))}
            </div>
          </div>
        )}

        {mode === 'feedback' && (
          <div className="demo-section">
            <div className="demo-section-title"><Activity size={13} /> Feedback regime</div>
            <select value={feedbackRate} onChange={(event) => setFeedbackRate(Number(event.target.value))}>
              {FEEDBACK_RATES.map((rate) => (
                <option key={rate} value={rate}>feedback rate · {rate.toFixed(2)}</option>
              ))}
            </select>
            <button
              className={`demo-toggle ${domesticPull ? 'active' : ''}`}
              onClick={() => setDomesticPull((value) => !value)}
              type="button"
            >
              {domesticPull ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
              Domestic pull {domesticPull ? 'on' : 'off'}
            </button>
          </div>
        )}

        <button
          className="demo-reset"
          onClick={() => {
            setMode('surgery');
            setBudget(8);
            setLocation('upstream_choice');
            setTopology('reconsideration_exits');
            setFeedbackRate(0.15);
            setDomesticPull(true);
          }}
          type="button"
        >
          <RotateCcw size={14} /> Reset scenario
        </button>
      </aside>

      <main className="demo-main">
        <header className="demo-header">
          <div className="demo-header-copy">
            <span>{MODE_COPY[mode].eyebrow}</span>
            <h2>{MODE_COPY[mode].title}</h2>
            <p>{MODE_COPY[mode].subtitle}</p>
          </div>
          <div className="demo-run-meta">
            <div><strong>{payload?.config.agents ?? '--'}</strong><span>agents</span></div>
            <div><strong>{payload?.config.steps ?? '--'}</strong><span>steps</span></div>
            <div><strong>{seeds || '--'}</strong><span>seeds</span></div>
          </div>
        </header>

        {error && (
          <div className="demo-error">
            <AlertTriangle size={18} />
            Demo artifact unavailable: {error}
          </div>
        )}

        {!payload && !error && (
          <div className="demo-loading">
            <span className="demo-spinner" />
            Loading semiconductor evidence…
          </div>
        )}

        {payload && active && (
          <>
            <section className="demo-stage">
              <ProductionGraph mode={mode} location={location} topology={topology} status={active.status} share={active.share} />
              <div className={`demo-verdict ${active.status}`}>
                <div className="demo-verdict-badge">
                  <span className="demo-verdict-dot" />
                  {STATUS_COPY[active.status]}
                </div>
                <h3>{active.verdict}</h3>
                <p>{active.mechanism}</p>

                {/* Measured response: turns a flat/null result into explicit
                    evidence — shows both endpoints and the delta, so "nothing
                    moved" reads as a measurement, not a frozen UI. */}
                <div className={`demo-response ${active.status}`}>
                  <div className="demo-response-head">
                    <span>Measured response</span>
                    {active.headline === 'share' ? (
                      <em>{mode === 'feedback' ? 'feedback' : `budget ${budget}`}</em>
                    ) : (
                      <em>domestic pull {domesticPull ? 'on' : 'off'}</em>
                    )}
                  </div>
                  {active.headline === 'share' ? (
                    <div className="demo-response-row">
                      <span className="demo-response-from">{pct(active.baseShare)}</span>
                      <ArrowRight size={13} />
                      <span className="demo-response-to">{pct(active.share)}</span>
                      <strong key={pulse} className="demo-response-delta flash">
                        {signed(active.lift)}
                      </strong>
                    </div>
                  ) : (
                    <div className="demo-response-row">
                      <span className="demo-response-from">{pct(active.viableBase, 0)} viable</span>
                      <ArrowRight size={13} />
                      <span className="demo-response-to">{pct(active.viable, 0)} viable</span>
                      <strong key={pulse} className="demo-response-delta flash">
                        {signed(active.viableLift, 2)}
                      </strong>
                    </div>
                  )}
                  <span className="demo-response-note">
                    {active.status === 'blocked'
                      ? 'lever moved · outcome unchanged — inert at this position'
                      : active.headline === 'robustness'
                      ? 'share ≈ flat; robustness is where the policy acts'
                      : 'measured lift vs. the zero-budget baseline'}
                  </span>
                </div>
              </div>
            </section>

            <section className="demo-metric-grid" key={pulse}>
              {active.headline === 'robustness' ? (
                <>
                  {/* Feedback mode leads with robustness — its real signal. */}
                  <MetricCard
                    primary
                    label="Viable seeds"
                    value={pct(active.viable, 0)}
                    context={
                      active.viableLift > 0
                        ? `+${pct(active.viableLift, 0)} vs. no pull — far more robust`
                        : 'fragile — few seeds reach onshore'
                    }
                    fill={active.viable}
                    tone={active.viable >= 0.8 ? 'green' : 'amber'}
                  />
                  <MetricCard
                    label="Onshore share"
                    value={pct(active.share)}
                    context={`${signed(active.lift)} vs. no pull — share stays ≈ flat`}
                    fill={active.share}
                    tone="amber"
                  />
                  <MetricCard
                    label="Completion"
                    value={pct(active.completion, 1)}
                    context={active.completion >= 0.98 ? 'lots clear the line' : 'lots stall before finish'}
                    fill={active.completion}
                    tone={active.completion >= 0.98 ? 'blue' : 'red'}
                  />
                </>
              ) : (
                <>
                  {/* Relocation / surgery lead with share + measured lift. */}
                  <MetricCard
                    primary
                    label="Onshore share"
                    value={pct(active.share)}
                    context="fraction of demand met domestically"
                    fill={active.share}
                    tone="green"
                  />
                  <MetricCard
                    label="Share lift"
                    value={signed(active.lift)}
                    context={
                      active.status === 'blocked'
                        ? 'lever inert at this position'
                        : active.lift > 0.03
                        ? 'meaningful policy response'
                        : 'small response'
                    }
                    fill={Math.min(1, active.lift / 0.1)}
                    tone={active.status === 'blocked' ? 'red' : active.lift > 0.03 ? 'green' : 'amber'}
                  />
                  <MetricCard
                    label="Viable seeds"
                    value={pct(active.viable, 0)}
                    context={active.viable >= 0.8 ? 'robust across seeds' : 'fragile across seeds'}
                    fill={active.viable}
                    tone={active.viable >= 0.8 ? 'green' : 'amber'}
                  />
                  <MetricCard
                    label="Completion"
                    value={pct(active.completion, 1)}
                    context={active.completion >= 0.98 ? 'lots clear the line' : 'lots stall before finish'}
                    fill={active.completion}
                    tone={active.completion >= 0.98 ? 'blue' : 'red'}
                  />
                </>
              )}
            </section>

            <section className="demo-bottom">
              <div className="demo-evidence">
                <div className="demo-panel-title">
                  <span>Response curve</span>
                  <strong>
                    {mode === 'feedback' ? 'feedback rate' : 'budget'} <ArrowRight size={13} /> onshore share
                  </strong>
                </div>
                <ResponseChart
                  rows={chartRows}
                  active={currentBudgetForChart}
                  unit={mode === 'feedback' ? 'feedback rate' : 'budget'}
                  baseline={mode === 'feedback' ? (chartRows[0]?.mean_onshore_share ?? active.share) : active.baseShare}
                  inert={mode !== 'feedback' && active.status === 'blocked'}
                />
              </div>
              <div className="demo-evidence demo-evidence-claim">
                <div className="demo-panel-title">
                  <span>Paper claim</span>
                  <strong><Target size={13} /> Effectiveness is graph-conditional</strong>
                </div>
                <p className="demo-claim">
                  The same intervention has <em>zero lift</em> in a serial corridor and up to{' '}
                  <em>{fmt(maxSurgeryLift)} lift</em> when reconsideration exits create a live routing choice.
                </p>
                <div className="demo-claim-null">
                  <AlertTriangle size={14} />
                  <span>
                    The null study still found <strong>{nullCount}</strong> high-share cases without a robust transition —
                    static share alone is not evidence of onshoring.
                  </span>
                </div>
                <Link className="demo-claim-link" to="/methodology">
                  Read the methodology <ArrowUpRight size={13} />
                </Link>
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

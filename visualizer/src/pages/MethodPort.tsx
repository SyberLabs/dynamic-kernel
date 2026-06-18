import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, ArrowRight, FlaskConical, GitBranch, Layers, Quote, Scale, X } from 'lucide-react';

// Focused methodology companion to the Semiconductor Policy Lab demo. Content is
// drawn verbatim from DTE_PAPER_DRAFT.md and the falsification artifact; every
// number here matches the live demo's data. Figures are the reproducible SVGs
// emitted by paper_figures.py, served statically from /figures.

interface ResultRow {
  label: string;
  detail: string;
  value: string;
  tone: 'green' | 'red' | 'amber';
}

const RELOCATION_ROWS: ResultRow[] = [
  { label: 'Upstream choice', detail: 'competing outgoing routes exist', value: '+0.084', tone: 'green' },
  { label: 'Route commitment', detail: 'single committed edge', value: '0.000', tone: 'red' },
  { label: 'Downstream serial corridor', detail: 'single-file, no alternative', value: '0.000', tone: 'red' },
];

const SURGERY_ROWS: Array<{ topology: string; budget: string; viable: string; share: string; tone: 'green' | 'red' }> = [
  { topology: 'Serial corridor', budget: '0', viable: '40%', share: '0.502', tone: 'red' },
  { topology: 'Serial corridor', budget: '12', viable: '40%', share: '0.502', tone: 'red' },
  { topology: 'Reconsideration exits', budget: '0', viable: '40%', share: '0.508', tone: 'red' },
  { topology: 'Reconsideration exits', budget: '12', viable: '100%', share: '0.743', tone: 'green' },
];

const LIMITATIONS = [
  ['No external calibration claim.', 'The semiconductor graph is a public, role-bearing abstraction — not calibrated to confidential firm operations or predictive trade shares.'],
  ['Discrete-time routing abstraction.', 'Agents move in synchronous ticks; no continuous-time lead-time distributions, queue disciplines, or production-cycle duration.'],
  ['Simplified production mechanics.', 'Bill-of-material gates and service capacities represent dependency feasibility, not yields, rework, price formation, or inventory policy.'],
  ['Finite seed and grid evidence.', 'Robustness is empirical over five seeds and sampled grids. No sampled threshold is a mathematical lower bound.'],
  ['Feedback model is stylized.', 'Telemetry updates via an exponential moving average; real procurement is governed by contracts and strategic optimization.'],
  ['No welfare objective.', 'Higher domestic share is not automatically socially optimal — cost, innovation, and resilience are not combined into one objective.'],
];

const FIGURES: Array<{ src: string; caption: string }> = [
  { src: '/figures/semiconductor_choice_point_relocation.svg', caption: 'Equal-cost relocation: lift only at the upstream choice point.' },
  { src: '/figures/semiconductor_topology_surgery.svg', caption: 'Topology surgery: the same downstream penalty acts only once an exit exists.' },
  { src: '/figures/semiconductor_feedback_continuum.svg', caption: 'Feedback continuum: non-monotone, a phase-shaper rather than an amplifier.' },
  { src: '/figures/semiconductor_topology_nulls.svg', caption: 'Topology nulls: high route share can coexist with non-viable production.' },
  { src: '/figures/dte_mechanism_schematic.svg', caption: 'DTE runtime mechanism.' },
  { src: '/figures/semiconductor_topology_schematic.svg', caption: 'Semiconductor production graph used in the case study.' },
];

export default function MethodPort() {
  // Click/keyboard-selected figure shown enlarged in a lightbox. null = closed.
  const [zoomed, setZoomed] = useState<{ src: string; caption: string } | null>(null);

  useEffect(() => {
    if (!zoomed) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setZoomed(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [zoomed]);

  return (
    <div className="method-shell">
      <div className="method-doc">
        <Link to="/demo" className="method-back">
          <ArrowLeft size={14} /> Back to the Policy Lab
        </Link>

        <header className="method-head">
          <span className="method-eyebrow">Methodology · Dynamic Topology Engine</span>
          <h1>Choice Points, Feasibility, and Adaptation in Dynamic Circulation Networks</h1>
          <p className="method-lede">
            Interventions in networked systems are often evaluated as if changing an edge cost <em>must</em> change
            flow. That assumption fails when agents adapt, when downstream production needs synchronized inputs, or
            when an intervention lands after route commitment. The Dynamic Topology Engine exposes the phase
            boundaries where a policy lever is live versus inert.
          </p>
        </header>

        <blockquote className="method-claim">
          <Quote size={16} />
          <p>
            Adaptive circulation interventions are effective only when applied to a transition with a genuine
            alternative, supported by feasible downstream production, and evaluated under the resulting agent
            adaptation.
          </p>
        </blockquote>

        <section className="method-section">
          <div className="method-section-title"><GitBranch size={15} /> The Choice-Point Principle</div>
          <p>
            Let node <i>i</i> have exactly one admissible outgoing neighbor <i>j</i>. Under row-wise softmax routing,
            the transition probability is fixed regardless of any finite intervention on that edge:
          </p>
          <div className="method-math">
            P<sub>ij</sub> = exp(−W<sub>ij</sub>/τ) / exp(−W<sub>ij</sub>/τ) = 1
          </div>
          <p>
            The softmax denominator contains a single term, so any finite change to that edge&rsquo;s cost,
            preference, or alignment leaves the next-state distribution unchanged. <strong>Intervention
            effectiveness is a property of the intervention&ndash;topology pair</strong>, not of magnitude alone.
            A subsidy has no route-steering power unless the edge it controls participates in a genuine choice.
          </p>
        </section>

        <section className="method-section">
          <div className="method-section-title"><FlaskConical size={15} /> Falsification: equal-cost relocation</div>
          <p>
            The same intervention budget was applied at three positions along the production pipeline (320 agents,
            40 steps, 5 paired seeds). Only the upstream position — where competing routes still exist — responds.
          </p>
          <div className="method-result-table">
            {RELOCATION_ROWS.map((row) => (
              <div className={`method-result-row ${row.tone}`} key={row.label}>
                <div className="method-result-label">
                  <strong>{row.label}</strong>
                  <span>{row.detail}</span>
                </div>
                <div className="method-result-value">{row.value}</div>
              </div>
            ))}
          </div>
          <p className="method-foot">Maximum mean onshore-share lift, by intervention location.</p>
        </section>

        <section className="method-section">
          <div className="method-section-title"><Layers size={15} /> Falsification: topology surgery</div>
          <p>
            Adding latent reconsideration exits back to the allocation desk makes the previously inert downstream
            penalty effective — lift of <strong>0.235</strong> and viability rising from 40% to 100%.
          </p>
          <div className="method-grid-table">
            <div className="method-grid-head">
              <span>Topology</span>
              <span>Budget</span>
              <span>Viable</span>
              <span>Share</span>
            </div>
            {SURGERY_ROWS.map((row, i) => (
              <div className={`method-grid-row ${row.tone}`} key={i}>
                <span>{row.topology}</span>
                <span>{row.budget}</span>
                <span>{row.viable}</span>
                <span>{row.share}</span>
              </div>
            ))}
          </div>
          <p className="method-foot">
            An alternative is <em>necessary</em> for rerouting; intervention strength decides whether it activates.
          </p>
        </section>

        <section className="method-section">
          <div className="method-section-title"><Scale size={15} /> Benchmark contrast</div>
          <p>
            A static expected-flow model predicts majority domestic share in all 60 tested policy cells — including
            24 that are <strong>not robust</strong> under DTE. Degree-preserving topology nulls show high route share
            coexisting with insufficient completed production.
          </p>
          <p className="method-foot">Route share alone is not evidence of an institutionally viable transition.</p>
        </section>

        <section className="method-section">
          <div className="method-section-title">Figures</div>
          <div className="method-figures">
            {FIGURES.map((fig) => (
              <figure className="method-figure" key={fig.src}>
                <button
                  type="button"
                  className="method-figure-btn"
                  onClick={() => setZoomed(fig)}
                  aria-label={`Enlarge figure: ${fig.caption}`}
                >
                  <img src={fig.src} alt={fig.caption} loading="lazy" />
                  <span className="method-figure-zoom">Click to enlarge</span>
                </button>
                <figcaption>{fig.caption}</figcaption>
              </figure>
            ))}
          </div>
        </section>

        <section className="method-section">
          <div className="method-section-title">Limitations</div>
          <ol className="method-limits">
            {LIMITATIONS.map(([title, body]) => (
              <li key={title}>
                <strong>{title}</strong> {body}
              </li>
            ))}
          </ol>
          <p className="method-foot">
            Results derive from frozen experiment artifacts via paired common-random-number comparisons; figures are
            regenerated by <code>paper_figures.py</code>.
          </p>
        </section>

        <Link to="/demo" className="method-cta">
          Explore the interactive Policy Lab <ArrowRight size={15} />
        </Link>
      </div>

      {zoomed && (
        <div
          className="method-lightbox"
          role="dialog"
          aria-modal="true"
          aria-label={zoomed.caption}
          onClick={() => setZoomed(null)}
        >
          <button type="button" className="method-lightbox-close" aria-label="Close enlarged figure">
            <X size={20} />
          </button>
          <figure className="method-lightbox-figure" onClick={(e) => e.stopPropagation()}>
            <img src={zoomed.src} alt={zoomed.caption} />
            <figcaption>{zoomed.caption}</figcaption>
          </figure>
        </div>
      )}
    </div>
  );
}

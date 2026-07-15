# Dynamic Kernel UI/UX Audit

## Executive Summary

Dynamic Kernel already has a strong product direction: it feels like a
research instrument for inspecting adaptive graph dynamics. The recent move
to a Hub-first experience is correct. It makes the system easier to understand
as software before asking the viewer to parse any specific research case.

The main UX gap is not visual ambition. The gap is control-system coherence.
The UI has several good local patterns, but they are implemented differently
across pages, often with inline styles, page-specific labels, and inconsistent
control semantics. A visitor can see that the project is technically serious;
the next polish pass should make every surface feel like one professional
simulation platform.

Recommended product framing:

> Dynamic Kernel is an interactive graph-based simulation platform for
> adaptive routing, population movement, and topology-aware memory dynamics.

Recommended software framing:

> The visualizer should behave like a cockpit: choose a topology, adjust
> a small set of well-labeled controls, observe the graph response, and
> inspect evidence without having to read the paper first.

## Scorecard

| Area | Rating | Assessment |
|---|---:|---|
| First impression | 7.5/10 | Strong dark technical aesthetic; Hub-first structure is right. |
| Information architecture | 7/10 | Clearer after semiconductor trim, but page categories still need sharper language. |
| Control ergonomics | 6.5/10 | Good controls exist, but patterns differ across pages and many click targets are custom divs. |
| Visual consistency | 6/10 | Shared tokens exist, but the Hub and overlays use many inline styles. |
| Accessibility | 4.5/10 | Canvas-heavy app needs stronger keyboard, ARIA, focus, and text alternatives. |
| Engineering maintainability | 5.5/10 | Working product, but duplicated controls and dead CSS increase future friction. |
| Portfolio clarity | 7.5/10 | The current public surface tells a better story than the semiconductor-first version. |

## Audit Scope

Inspected:

- `visualizer/src/pages/ControlHub.tsx`
- `visualizer/src/pages/BaseSystem.tsx`
- `visualizer/src/pages/MallPort.tsx`
- `visualizer/src/pages/ComparePort.tsx`
- `visualizer/src/pages/NeuralPort.tsx`
- `visualizer/src/App.tsx`
- `visualizer/src/index.css`
- `README.md`

Recent validation state:

- Backend suite passed after the demo trim: `144 passed, 1 warning`.
- Frontend production build passed after installing dependencies from the lockfile.

## Implementation Status

Applied in the July 15, 2026 polish pass:

- Removed the dead `.demo-*` and methodology CSS residue from the public stylesheet.
- Added shared `ControlSection`, `SliderField`, and `StatusBanner` UI primitives.
- Rebuilt the Hub as a clearer topology-first launch surface with native buttons.
- Renamed public labels from narrow retail/ad-tech terms to agent/intervention language.
- Converted selectable chips/cards from clickable `div` elements to semantic buttons.
- Folded Adaptive Routing Lab hyperparameters, composite targets, and phase chains behind an advanced disclosure.
- Verified the visualizer with `npm run build`.

Remaining useful cleanup:

- Replace the force-graph wrapper `any` types with local typed graph adapters.
- Add fuller canvas accessibility descriptions for graph state.
- Run a screenshot QA pass against a live backend session before the next public demo recording.

## Top Findings

### P0: Remove Dead Semiconductor Demo CSS

The semiconductor demo pages were removed from routing and source, but
`visualizer/src/index.css` still contains a large block of `.demo-*` styles.
These now belong to deleted surfaces.

Why it matters:

- It makes the public package look less intentionally curated.
- It increases CSS scan cost.
- It can create accidental selector collisions later.

Recommendation:

- Delete the `.demo-*` CSS block from `visualizer/src/index.css`.
- Re-run `npm run build`.

### P1: Build A Shared Control System

The app has recurring control types:

- topology cards,
- port cards,
- intent chips,
- sponsor mode cards,
- sliders,
- metric pills,
- status banners,
- graph panels,
- export/reset/run buttons.

Today these are implemented inconsistently across pages. `ControlHub.tsx`
uses extensive inline styling; `BaseSystem.tsx` and `MallPort.tsx` use custom
card-like divs for interactive controls; `ComparePort.tsx` uses real buttons
for sponsor modes; `NeuralPort.tsx` has its own `Slider` component.

Recommendation:

Create a small UI kit inside `visualizer/src/components/`:

- `PageShell`
- `Sidebar`
- `MetricPill`
- `SliderField`
- `SegmentedControl`
- `ModeCard`
- `StatusBanner`
- `IconButton`
- `GraphPanel`

This should not become a design-system project. Keep it practical: enough
shared components to make every page feel intentionally related.

### P1: Reframe Page Names Around User Mental Models

Current names are close, but a portfolio viewer needs faster semantic entry.

Recommended labels:

| Current | Recommended | Reason |
|---|---|---|
| Mathematical Kernel | Kernel Inspector | More direct, less abstract. |
| Population Simulator | Agent Flow Simulator | Says what moves. |
| Adaptive Router | Adaptive Routing Lab | Avoids implying a neural network product. |
| City Comparison | Topology Comparison | Generalizes beyond city metaphor. |
| Consumer Intent Profile | Agent Intent | Works across mall, airport, museum, supply chain. |
| Consumer Demographics | Agent Mix | Works outside retail. |
| Sponsor Channel | Intervention Channel | More general and research-accurate. |
| Sponsor Automation | Intervention Mode | Less ad-tech-specific. |

This is a high-ROI wording pass. It will make the same UI feel more mature.

### P1: Add One-Sentence Task Framing Per Port

Each port should answer, without documentation:

1. What am I looking at?
2. What can I change?
3. What should I notice?

Recommended pattern:

```text
Kernel Inspector
Change agent intent and exploration temperature to see how the transition
matrix reshapes local route probabilities.
```

```text
Agent Flow Simulator
Adjust population size, agent mix, and intervention channel to watch live
occupancy and edge traffic respond.
```

```text
Adaptive Routing Lab
Tune the optimizer objective and hyperparameters to observe how transition
memory reorganizes stationary mass.
```

```text
Topology Comparison
Run the same agent mix through two graph structures to compare congestion,
mixing, and entropy.
```

### P1: Convert Clickable Divs To Native Buttons

Several interactive controls are visually button-like but implemented as
`div` with `onClick`, especially chips and sponsor cards. This creates
keyboard, focus, and semantics issues.

Examples:

- `ControlHub.tsx`: `PortCardView` handles keyboard manually.
- `BaseSystem.tsx`: intent chips and sponsor cards are clickable divs.
- `MallPort.tsx`: sponsor cards are clickable divs.
- `ComparePort.tsx`: sponsor controls are already real buttons and should
  become the model.

Recommendation:

- Use native `<button>` for all interactive cards, chips, and mode controls.
- Preserve card styling through classes.
- Add `aria-pressed` for mutually exclusive mode selections.
- Add visible focus states using the same accent color as hover.

### P1: Make The Hub More Product-Like And Less Prototype-Like

The Hub is the portfolio front door, but much of it is inline-styled. It works,
but it does not yet communicate a stable product system.

Recommended changes:

- Move Hub styling into CSS classes.
- Use lucide icons instead of emoji in topology cards.
- Add a compact top-level orientation line: "Select a graph, then inspect
  transition probabilities or live population flow."
- Make "Topology Explorers" and "Standalone Studies" visually distinct but
  semantically precise:
  - "Topology-Driven Views"
  - "Fixed Scenario Labs"

The Hub should feel like a professional control surface, not a one-off landing
screen.

### P1: Reduce NeuralPort Cognitive Load

`NeuralPort.tsx` is the most impressive but also the densest surface.
It exposes many controls at once:

- network size,
- density,
- mode cards,
- eta,
- finite difference,
- noise,
- beta max,
- composite target controls,
- target pi sliders,
- pause/reset/export,
- phase chain,
- gradient fetch.

Recommendation:

- Split the sidebar into disclosure groups:
  - Basic: network load, mode, run controls.
  - Tuning: eta, finite difference, beta max, noise.
  - Advanced: composite target, phase chain, gradient fetch.
- Default the advanced groups collapsed.
- Rename "Neural Optimizer" to "Adaptive Routing Lab" or "Routing Memory Lab"
  unless the visual is explicitly meant to represent neural systems.

This page should remain powerful, but the first impression should be
"I can operate this" rather than "I must understand every control now."

### P2: Establish A Control Vocabulary

Use consistent control types:

| User Need | Control |
|---|---|
| Choose one mode | segmented control or mode card with `aria-pressed` |
| Set numeric continuous value | slider with min/max labels and current value |
| Set count or bounded integer | slider plus stepper/input if precision matters |
| Trigger command | icon+text button |
| Export/download | download icon button with label |
| Pause/run/reset | icon buttons with tooltips or visible labels |
| Show health | status banner or badge |
| Explain unfamiliar math | tooltip or compact helper text |

Avoid using a rounded text card when a standard control would be clearer.

### P2: Add Canvas Accessibility Companions

The app relies heavily on canvas through `react-force-graph-2d`. Canvas is
appropriate for this domain, but it needs textual companions.

Recommendation:

- Add `aria-label` or adjacent descriptive region for each graph canvas.
- Provide a compact "Top changes" list beside or under the graph:
  - highest occupancy node,
  - highest outbound transition,
  - highest entropy node,
  - current intervention target.
- Ensure node click information is reachable without pointer-only interaction.

This also improves portfolio comprehension, not just accessibility.

### P2: Make Loading And Error States Consistent

There are good states already:

- Hub retry state,
- backend unreachable states,
- stream disconnected warning,
- Neural optimizer unavailable.

But each page phrases and styles them differently.

Recommendation:

- Use one `StatusBanner` component with variants: `loading`, `warning`,
  `error`, `success`.
- Include action language:
  - "Retry"
  - "Reload"
  - "Start backend: uvicorn api:app --reload --port 8000"

### P2: Clarify Export Semantics

`BaseSystem` and `MallPort` both expose "Export CSV", but it is unclear what
is being exported: topology, current transition matrix, current simulation
state, or global app state.

Recommendation:

- Rename to "Export Current State".
- Add a small secondary label or tooltip:
  - "CSV of current topology and metrics"
  - or "JSON snapshot" if more accurate.

## Page-Specific Notes

### Control Hub

Strengths:

- The Hub-first direction is right.
- Topology selection before exploration is a strong mental model.
- The split between topology-driven views and fixed studies is useful.

Weaknesses:

- Heavy inline styling makes the Hub less maintainable.
- Emoji topology icons may render inconsistently and feel less premium than
  the rest of the interface.
- The user still has to infer the relationship between topology selection and
  the individual ports.

Recommended next edit:

- Move Hub styles to CSS.
- Rename groups to "Topology-Driven Views" and "Fixed Scenario Labs".
- Add a one-line "How this works" strip:
  "Topology selection feeds the Kernel Inspector and Agent Flow Simulator."

### Kernel Inspector

Strengths:

- Good conceptual fit for the core mathematical kernel.
- Node inspector is a strong interaction.
- Entropy and mixing time are useful top-level metrics.

Weaknesses:

- "Consumer Intent Profile" is too domain-specific.
- Sponsor terminology narrows the perceived platform.
- Node alignment list can become long and dense for larger topologies.

Recommended next edit:

- Rename to "Agent Intent" and "Intervention Channel".
- Convert intent chips to native buttons.
- Add a "Selected Node" empty state before click.

### Agent Flow Simulator

Strengths:

- This is currently the clearest public demo of "graph-based simulator of
  agents."
- Population controls, intent mix, live occupancy, and edge traffic are
  immediately legible.

Weaknesses:

- "Consumer Demographics" makes non-retail presets feel accidental.
- Sponsor wording reads too ad-tech-specific.
- Demographic sliders auto-normalize, which is powerful but slightly
  surprising.

Recommended next edit:

- Rename "Consumer Demographics" to "Agent Mix".
- Add a small note to the distribution bar: "Shares auto-normalize to 100%."
- Consider a lock icon later if users want to hold one group fixed while
  adjusting another.

### Adaptive Routing Lab

Strengths:

- Technically rich and visually distinctive.
- Heatmaps, loss curve, and graph view together communicate real system
  depth.
- Mode cards are a good pattern.

Weaknesses:

- The page is too dense for a first-time portfolio viewer.
- Some labels are research shorthand rather than UI labels.
- Advanced controls compete with the primary demo.

Recommended next edit:

- Rename page and Hub card to "Adaptive Routing Lab".
- Collapse advanced controls.
- Add a top-level readout explaining the current objective in plain language.

### Topology Comparison

Strengths:

- Strong comparative layout.
- Side-by-side topology contrast is easy to understand.
- Good candidate for the academic deck because it directly asks what graph
  structure changes.

Weaknesses:

- The "WHEEL vs RHIZOME" metaphor needs one sentence of interpretation.
- Synchronized controls are useful but could state the experimental rule:
  "same population, same intervention, different graph."

Recommended next edit:

- Add a hypothesis strip:
  "Same agents and controls, different topology: compare concentration,
  entropy, and congestion."

## Recommended Implementation Plan

### Pass 1: Public Polish Cleanup

Objective: remove obvious public-surface debt.

Tasks:

1. Remove dead `.demo-*` CSS.
2. Rename Hub labels and port labels.
3. Replace domain-specific words:
   - consumer -> agent,
   - sponsor -> intervention where appropriate,
   - Neural Optimizer -> Adaptive Routing Lab.
4. Add one-sentence task framing to each port.
5. Re-run backend tests and frontend build.

Expected effort: 2-4 hours.

### Pass 2: Shared Controls

Objective: make the app feel like one designed system.

Tasks:

1. Create shared UI components:
   - `SliderField`
   - `MetricPill`
   - `ModeButton`
   - `StatusBanner`
   - `ControlSection`
2. Replace repeated inline controls in `BaseSystem`, `MallPort`, and
   `ComparePort`.
3. Move Hub styling out of inline style objects.
4. Add consistent focus states and `aria-pressed`.

Expected effort: 1-2 days.

### Pass 3: First-Run Comprehension

Objective: make every screen understandable without coaching.

Tasks:

1. Add task framing to each page header.
2. Add graph companion summaries.
3. Add selected-node empty states.
4. Collapse advanced Neural controls by default.

Expected effort: 1 day.

### Pass 4: Accessibility And Responsive QA

Objective: make the demo robust across devices and review contexts.

Tasks:

1. Keyboard navigation through all controls.
2. Canvas text alternative summaries.
3. Reduced-motion check.
4. Desktop and mobile viewport screenshots.
5. Text overflow pass.

Expected effort: 1 day.

## Design Principles To Preserve

- Keep the app as an instrument, not a marketing page.
- Keep the graph visual primary.
- Keep controls close to the thing they affect.
- Prefer short operational labels over explanatory prose.
- Preserve the dark technical aesthetic, but use it with restraint.
- Make uncertainty and boundaries visible in docs, not in the main interaction
  flow unless the user needs them to operate the system.

## Bottom Line

Dynamic Kernel is already credible as a technical artifact. The next UI/UX
step is to make it feel intentionally designed as a platform. The highest ROI
is not adding new features. It is removing dead demo residue, unifying controls,
renaming domain-specific labels, and making each page announce its task in one
sentence.

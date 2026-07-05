# Choice Points, Feasibility, and Adaptation in Dynamic Circulation Networks

## A Dynamic Topology Engine for Intervention Analysis

**Status:** First manuscript draft  
**Target positioning:** stochastic processes, complex systems, or applied network science  
**Primary case study:** semiconductor onshoring and supply-chain circulation  
**Citation status:** related-work and calibration citations remain to be added

---

## Abstract

Interventions in networked systems are often evaluated as if changing an edge cost, increasing a route preference, or expanding capacity must produce a corresponding change in flow. This assumption fails when agents adapt to visited states, when downstream production requires multiple synchronized inputs, or when an intervention is applied after route commitment. We introduce the Dynamic Topology Engine (DTE), a vectorized routing kernel for non-stationary Markov processes on weighted, feature-decorated graphs. In DTE, transition probabilities depend jointly on physical edge friction, alignment between agent telemetry and destination features, and intervention-specific edge controls. Agent telemetry co-evolves with visited nodes, allowing route choice and agent state to influence one another.

We establish a structural invariance result: an intervention on an edge leaving a node with a single admissible outgoing neighbor cannot change the next-state distribution under row-wise softmax routing. Intervention effectiveness therefore depends on the presence and location of genuine alternatives, not only on intervention magnitude. We evaluate this principle in a semiconductor onshoring model with explicit bill-of-material gates, production-lot accounting, node and gate service capacities, heterogeneous demand agents, and paired common-random-number experiments. Physical resource scaling removes material infeasibility but does not guarantee robust majority domestic capture. At high load, a domestic procurement preference increases robust viable-transition frequency from 40% to 80%, while import friction applied on a serial offshore corridor has zero effect on route share. Relocating equal-cost penalties to upstream choice edges raises mean domestic share by 0.084, whereas penalties at route-commitment or downstream serial edges produce no change. Adding latent reconsideration exits makes the same downstream intervention effective, producing a maximum mean-share lift of 0.235.

Benchmark comparisons show that a static expected-flow model predicts majority domestic share in all 60 tested policy cells, including 24 cells that are not robust under DTE. Degree-preserving randomized topology nulls further show that high route share can coexist with insufficient completed production. These results support a general conclusion: intervention analysis in adaptive circulation systems must distinguish route attractiveness, downstream feasibility, and agent adaptation. Route share alone is not evidence of an institutionally viable transition.

A separate Neural V2 adaptive-routing benchmark tests the same kernel as a
controlled decision system rather than a supply-chain case study. Across 30
paired seeds, DTE-native EXP3 with an importance-weighted (EXP3-IX) gain
estimator improves over local-regret DTE in clean, corrupted-delayed-
attribution, and adversarial-switching regimes, and is the strongest
DTE-native lane under corrupted attribution. An earlier version of this lane
multiplied rewards by realized selection frequency instead of importance
weighting; it appeared brittle under corrupted attribution, and that
brittleness is now traced to the biased estimator reproducing deadly
familiarity inside the lane itself, not to attribution noise. External
UCB/EXP3 remain stronger when the problem reduces to clean policy ownership.
This preserves a policy-arbitration boundary in revised form: DTE is not a
universal bandit replacement, but a topology-memory governor whose learning
lanes require unbiased gain estimation.

---

## 1. Introduction

Many policy and operational interventions are expressed as changes to a network. A tariff increases the cost of an import path. A procurement contract strengthens a preferred supplier route. A subsidy expands capacity. A recommendation system increases the probability of a transition toward selected content. A resilience policy creates an alternate logistics corridor. In each case, the intervention is commonly evaluated by asking whether it makes a desired route more attractive or a disfavored route more expensive.

That question is incomplete.

An intervention can be economically large and dynamically inert. If it is applied after an agent has already committed to a route, and if the current node has no alternative outgoing transition, changing the cost of the only available edge cannot change the next move. Conversely, an intervention that successfully redirects flow can still fail institutionally if the receiving route lacks capacity, required inputs do not arrive together, or adaptive agents change their preferences in response to the altered topology. The relevant object is not route share alone. It is the coupled transition among feasibility, allocation, and adaptation.

This paper introduces the Dynamic Topology Engine (DTE), a general routing kernel designed to study that coupled transition. DTE represents a system as a directed weighted graph with feature-decorated nodes. Agents carry telemetry vectors describing their current intent, preference, or state. Edge weights depend on physical friction, destination alignment, and intervention channels. After each transition, the visited node updates agent telemetry. The resulting augmented process is Markov on the joint space of discrete position and continuous telemetry, while the position process alone is generally non-stationary.

Figure 1 summarizes the DTE runtime mechanism, and Figure 2 shows the layered
memory state needed for Markov closure.

The paper makes four contributions.

First, we state and prove a structural invariance result for softmax routing. At a node with one admissible outgoing neighbor, any finite change to the weight of that edge leaves the next-state distribution unchanged. This **choice-point principle** gives a precise condition under which edge-level interventions cannot reroute flow.

Second, we extend DTE from route preference into operational feasibility. The semiconductor case study includes bill-of-material gates, production-lot accounting, input replenishment, node capacity, gate service capacity, and heterogeneous agent populations. This allows us to distinguish a route that appears attractive from a route that can complete production at sufficient scale.

Third, we subject the principal case-study conclusions to adversarial falsification. We relocate equal-cost interventions, add latent downstream alternatives, sweep the feedback rate, compare against static and frozen-agent baselines, vary the classification definition, and rewire the choice topology while preserving directed degree and the production backbone. Some claims survive and others fail. Procurement preference is effective in the specified semiconductor topology but is not topology invariant. Feedback reshapes the phase boundary but is not a universal amplifier. The choice-point principle and the distinction between route share and viable transition survive.

Fourth, we evaluate DTE policy lanes in a controlled Neural V2 routing bench.
This benchmark isolates adaptive inference routing under clean labels, corrupted
delayed rewards, and adversarial switching reward surfaces. With
importance-weighted gain estimation, DTE-EXP3 is the strongest DTE-native lane
in all three regimes, while full policy-owning bandits dominate when topology
memory is not needed. The benchmark also contributes a cautionary estimator
result: a traffic-weighted gain — reward multiplied by realized selection
frequency — re-creates deadly familiarity inside the learning lane, and in an
earlier draft this estimator bias masqueraded as attribution fragility.

The resulting claim is deliberately narrower than a policy forecast:

> Adaptive circulation interventions are effective only when applied to a transition with a genuine alternative, supported by feasible downstream production, and evaluated under the resulting agent adaptation.

This paper does not claim that DTE replaces inventory optimization, discrete-event simulation, equilibrium trade models, or domain-specific forecasting. Its purpose is different: to expose intervention phase boundaries that disappear when route attractiveness, production feasibility, and adaptive state are modeled separately.

---

## 2. Dynamic Topology Engine

### 2.1 Graph, features, and agent telemetry

Let \(G=(V,E)\) be a directed graph with \(|V|=N\). Each node \(j\in V\) has a feature vector \(N_j\in\mathbb{R}^F\), and each directed edge \((i,j)\in E\) has a base traversal cost \(D_{ij}>0\). Non-edges are represented by \(D_{ij}=\infty\).

An agent at time \(t\) has:

- position \(X_t\in V\),
- telemetry \(a_t\in\mathbb{R}^F\), normalized to the unit sphere after each update.

Telemetry represents the agent's current state or intent. Its interpretation is domain dependent: demand urgency in a supply chain, interest state in a recommendation system, or preference state in an abstract routing process.

### 2.2 Dynamic weights and transition probabilities

The alignment of telemetry \(a_t\) with destination node \(j\) is

\[
A_j(a_t)=\langle a_t,N_j\rangle+b_j,
\]

where \(b_j\) is a node-specific bias.

The dynamic edge weight is

\[
W_{ij}(a_t)
=
\alpha D_{ij}
-
\beta_{ij}A_j(a_t)
-
S_{ij},
\]

where:

- \(\alpha>0\) scales physical friction,
- \(\beta_{ij}\) is an alignment-coupled edge preference,
- \(S_{ij}\) is an alignment-independent friction reduction.

The transition matrix is a row-wise softmax over negative weights:

\[
P_{ij}(a_t)
=
\frac{\exp(-W_{ij}(a_t)/\tau)}
{\sum_{k:(i,k)\in E}\exp(-W_{ik}(a_t)/\tau)}
\quad \text{for } (i,j)\in E,
\]

with \(P_{ij}=0\) for non-edges. The temperature \(\tau>0\) controls exploration relative to greedy route selection.

### 2.3 Telemetry feedback

After the agent visits node \(j^*=X_{t+1}\), telemetry updates by

\[
a_{t+1}
=
\operatorname{norm}
\left(
(1-\lambda)a_t+\lambda N_{j^*}+\varepsilon_t
\right),
\]

where \(\lambda\in[0,1]\) is the feedback rate, \(\varepsilon_t\) is optional noise, and \(\operatorname{norm}\) projects the vector to unit norm.

When \(\lambda=0\), each agent follows a time-homogeneous Markov chain conditional on its initial telemetry. When \(\lambda>0\), the position process alone is non-stationary because route history changes future transition probabilities. The joint process \((X_t,a_t)\) remains Markov.

### 2.4 Intervention channels

DTE separates two intervention primitives:

1. **Alignment-coupled preference** through \(\beta_{ij}\).  
   This changes how strongly a route attracts agents whose telemetry aligns with the destination.

2. **Alignment-independent friction** through \(S_{ij}\).  
   This changes traversal cost for all agents using the edge.

This distinction matters because the same nominal budget can have different effects across heterogeneous agents and different locations in the topology.

Both channels act on the weight matrix before a softplus floor is applied, and
this makes intervention effect size nonlinear in a documented way. Because the
floor precedes the row softmax, it breaks softmax shift-invariance: the
derivative of the floored weight with respect to the raw weight is
\(\sigma(k\,(W_{ij}-f))\), so the marginal effect of additional sponsorship
decays smoothly to zero once an edge is pushed below the floor. Sponsorship
therefore has intentional diminishing returns rather than unbounded linear
effect. Any first-order sensitivity analysis of DTE interventions must include
this sigmoid gate; dropping it overestimates leverage on heavily sponsored
edges by orders of magnitude (numerically, roughly 250x one unit below the
floor at the default sharpness).

Figure 1 gives the corresponding runtime loop: graph features and agent
telemetry determine dynamic edge weights; softmax routing selects a move; the
realized transition updates telemetry and preference memory.

### 2.5 Proposition-to-evidence map

The paper uses DTE as both a formal object and an experimental instrument. The
following table fixes the evidentiary role of each major claim.

| Proposition or Boundary | Claim | Evidence |
|---|---|---|
| Markov closure | The process is Markov on the augmented state including position, telemetry, structural memory, preference memory, delayed buffers, and ecology variables; the position process alone is generally non-Markovian. | Layered-memory schematic, kernel definition, memory-law tests |
| Choice-point invariance | Singleton-outdegree interventions cannot change finite row-wise softmax next-state probabilities. | Proposition 1, choice relocation figure, topology surgery figure |
| Estimator-bias fragility | A multiplicative lane whose gain multiplies reward by realized selection frequency re-creates deadly familiarity internally; the attribution-fragility reading of an earlier draft was this estimator bias. Importance-weighted (EXP3-IX) gains remove it. | Two-route switching regression test, Neural V2 hard benchmark pre/post estimator fix (archived pre-fix artifacts) |
| Policy-arbitration boundary | DTE-native learning lanes improve DTE across tested regimes once gains are importance-weighted, but do not dominate full policy owners on clean contextual bandit tasks. | Neural V2 seed-validation table and policy-lane figure |
| Feasibility-allocation distinction | Majority route share can coexist with failed completed production. | BOM gates, model benchmark figure, topology nulls figure |

### Proposition 0: Markov closure under layered memory

Let the augmented DTE state be

\[
Z_t=(X_t,A_t,M_s(t),M_p(t),B_t,R_t),
\]

where \(X_t\) is position or occupancy, \(A_t\) is telemetry state memory,
\(M_s(t)\) is structural memory, \(M_p(t)\) is preference memory, \(B_t\) is
the delayed-feedback buffer, and \(R_t\) contains ecology variables used by the
update rules. If movement, telemetry update, memory update, and delayed
attribution consume only \(Z_t\), the control \(u_t\), and fresh randomness,
then \(\{Z_t\}\) is Markov. Projections such as \(\{X_t\}\) alone are generally
not Markov when feedback rate or reward delay is nonzero.

Proof sketch. Conditional on \(Z_t\), the transition matrix is determined by
current topology, edge controls, telemetry, preference memory, and ecology
variables. Telemetry and memory updates depend on the realized transition,
current rewards, and pending records in \(B_t\). Thus no earlier history is
needed once these variables are included. If telemetry or delayed buffers are
omitted, two histories with identical visible position can produce different
alignment scores or future memory updates, so the projected process need not be
Markov.

---

## 3. Choice-Point Principle

### Proposition 1: Singleton-outdegree intervention invariance

Let node \(i\) have exactly one admissible outgoing neighbor \(j\). For any finite edge weight \(W_{ij}\) and any temperature \(\tau>0\),

\[
P_{ij}=1.
\]

Therefore, any finite intervention that changes only \(D_{ij}\), \(\beta_{ij}\), \(S_{ij}\), or the destination alignment term leaves the next-state distribution from node \(i\) unchanged.

### Proof

Since \(j\) is the only admissible outgoing neighbor of \(i\), the softmax denominator contains one term:

\[
P_{ij}
=
\frac{\exp(-W_{ij}/\tau)}
{\exp(-W_{ij}/\tau)}
=1.
\]

No finite modification of \(W_{ij}\) changes this ratio. \(\square\)

### Corollary 1: Serial-corridor cost changes cannot reroute flow

Consider a directed path in which each intermediate node has outdegree one. Edge-cost interventions applied strictly inside that serial corridor cannot alter route choice until they:

1. affect an upstream node with more than one admissible outgoing edge,
2. add or activate a competing downstream transition,
3. make an edge infeasible or remove it entirely.

### Implication

The effectiveness of an intervention is a property of the intervention-topology pair. A tariff, subsidy, recommendation boost, or friction reduction does not possess route-steering power in isolation. Its effect depends on whether the controlled edge participates in a genuine choice.

---

## 4. Operational Feasibility Extension

### 4.1 Why route choice is insufficient

A route can attract agents while failing to produce useful output. In supply chains, production requires synchronized inputs, processing capacity, and terminal delivery. A domestic semiconductor route is not viable merely because agents visit domestic fabrication nodes. Completed lots must pass through fabrication, packaging, and demand fulfillment without excessive overflow or dependency starvation.

Figure 5 gives the conceptual distinction used throughout the case study:
attractiveness, allocation, feasibility, and completed transition are separate
stages.

### 4.2 Bill-of-material gates

The semiconductor adapter introduces gates that require multiple input categories before an agent may traverse a production transition. The principal gates are:

- U.S. wafer fabrication gate,
- U.S. advanced packaging gate,
- Taiwan export packaging gate.

Each gate tracks:

- required parts,
- part arrivals,
- inventory,
- per-lot consumption,
- service-capacity limits,
- blocked attempts,
- completed transitions.

The U.S. fabrication gate requires domestic wafers, materials, tooling and design support, and power/labor continuity. The U.S. advanced packaging gate separately requires packaged U.S. wafers and U.S. packaging inputs. This separation prevents domestic fabrication from being counted as finished domestic output before packaging is feasible.

### 4.3 Production-lot accounting

Each agent can complete at most one U.S. demand lot. Completed lots are attributed to:

- domestic finished packaged chips,
- imported finished chips through the U.S. port,
- strategic reserve release.

The principal allocation metric is

\[
\text{Onshore Share}
=
\frac{\text{Domestic Completed U.S. Lots}}
{\text{Total Completed U.S. Lots}}.
\]

This avoids interpreting repeated circulation through a demand edge as repeated production.

### 4.4 Viable transition criterion

A policy cell is classified as a viable onshoring transition when:

\[
\text{Onshore Share}\geq \theta_{\text{share}},
\]

\[
\text{Overflow}\leq \theta_{\text{overflow}},
\]

\[
\text{Dependency Pressure}\leq \theta_{\text{dependency}},
\]

\[
\text{Finished-Lot Ratio}\geq \theta_{\text{flow}}.
\]

The primary thresholds are:

- \(\theta_{\text{share}}=0.50\),
- \(\theta_{\text{overflow}}=0.10\),
- \(\theta_{\text{dependency}}=0.65\),
- \(\theta_{\text{flow}}=0.85\).

A cell is robust when at least two-thirds of seeds satisfy the viable-transition criterion. With five seeds, this requires at least four viable runs.

---

## 5. Semiconductor Onshoring Case Study

### 5.1 Scope and interpretation

The semiconductor topology is a public, non-sensitive abstraction of U.S., Taiwan, China, Japan, Korea, and European roles in advanced semiconductor production. Nodes represent role-bearing industrial functions rather than forecasts about private firm operations. The graph includes demand, design/IP, tools, materials, fabrication, advanced packaging, export review, logistics, ports, and strategic reserve functions.

Figure 4 summarizes this abstraction, emphasizing upstream route-choice points,
domestic production gates, and the serial offshore corridor.

Agents represent circulating demand and procurement intents, not physical wafers or firms. Their feature vectors encode urgency, reliability, speed, cost sensitivity, advanced-node criticality, capacity preference, policy compatibility, China exposure, and strategic-buffer preference.

The adapter is intended to test mechanisms, not estimate real-world market shares.

### 5.2 Resource-scaling regime

Prior experiments separated four physical resource classes:

- consumable stock renewal,
- domestic wafer inflow,
- fabrication and packaging node capacity,
- gate service capacity.

Scaling only one resource class did not produce a robust high-load transition. Scaling all resource classes reduced dependency pressure, removed overflow, and raised completion toward 100%, but did not guarantee majority domestic capture. A sampled all-resource exponent of 1.25 produced the only robust high-load cell at 160 agents in the resource frontier.

This result motivates a distinction:

> Physical feasibility determines whether domestic production can complete. Allocation determines whether completed demand selects the domestic route.

### 5.3 Feasibility-preference surface

Holding the all-resource scaling regime fixed, we varied:

- agents: 160, 240, 320,
- domestic procurement pull: 0.0 to 4.0,
- offshore import friction: 0.0 to 3.0,
- five paired seeds,
- 40 steps per run.

At 320 agents, the no-pull policy has a robust viable-transition rate of 40% and mean onshore share of 0.502. Positive domestic procurement pull raises the viable-transition rate to 80% and mean share to 0.522.

Import friction produces no observed change in mean onshore share:

\[
\max_{\text{tariff levels}}
\Delta \text{Mean Onshore Share}
=0.
\]

This is not evidence that tariffs are universally ineffective. In the modeled topology, the tariff friction is applied on serial offshore corridor edges after route commitment. Proposition 1 predicts that these cost changes cannot reroute flow.

### 5.4 Classification robustness

We reclassified the same simulation rows over 54 threshold configurations:

- share threshold: 0.45, 0.50, 0.55,
- maximum dependency pressure: 0.55, 0.65, 0.75,
- maximum overflow: 0.05, 0.10,
- minimum finished-flow ratio: 0.80, 0.85, 0.95.

At a 0.45 share threshold, capacity-only high-load cells are robust in all 18 admissible configurations. At the literal majority threshold of 0.50, all 18 admissible configurations require allocation pressure for high-load robustness. At 0.55, no tested policy is robust.

The defensible claim is therefore not that allocation pressure is universally necessary. It is:

> Allocation pressure is necessary to cross from capable domestic participation into robust majority capture at high load in the specified topology.

---

## 6. Falsification Experiments

### 6.1 Equal-cost intervention relocation

We applied equal total intervention budgets at three locations:

1. upstream route-choice edges leading toward offshore fabrication,
2. the route-commitment edge entering the offshore packaging chain,
3. downstream serial offshore corridor edges.

At 320 agents, the upstream intervention raises mean onshore share by as much as 0.084 and raises viable-transition frequency from 40% to 100%. The route-commitment and downstream serial interventions produce exactly zero share change at every tested budget.

Figure 6 plots this relocation experiment.

| Intervention Location | Maximum Mean-Share Lift |
|---|---:|
| Upstream choice | 0.084 |
| Route commitment | 0.000 |
| Downstream serial corridor | 0.000 |

This experiment supports the choice-point principle without relying on the policy label assigned to an edge.

### 6.2 Topology surgery

We added latent reconsideration exits from downstream offshore nodes back to the allocation desk. The added edges have sufficiently high base distance that they remain marginal without an intervention.

In the original serial topology, downstream penalties produce no share lift. After reconsideration exits are added, the same downstream penalty produces a maximum mean-share lift of 0.235 and raises viable-transition frequency from 40% to 100%.

Figure 7 shows the contrast between the serial topology and the topology with
reconsideration exits.

| Topology | Downstream Budget | Viable Rate | Mean Share |
|---|---:|---:|---:|
| Serial corridor | 0.0 | 40% | 0.502 |
| Serial corridor | 12.0 | 40% | 0.502 |
| Reconsideration exits | 0.0 | 40% | 0.508 |
| Reconsideration exits | 12.0 | 100% | 0.743 |

The result demonstrates that an alternative is necessary for cost-based rerouting, while intervention strength determines whether the alternative becomes active.

### 6.3 Feedback-rate continuum

We swept telemetry feedback rate

\[
\lambda\in\{0.00,0.05,0.10,0.15,0.30,0.50\}
\]

with and without domestic procurement pull. Positive-pull viable rates are:

\[
100\%, 100\%, 60\%, 80\%, 60\%, 80\%.
\]

Feedback is non-monotone. The frozen-telemetry case is not dominated by every adaptive case. DTE feedback should therefore be described as a phase-shaping mechanism, not a universal amplifier.

Figure 8 visualizes the feedback continuum.

### 6.4 Degree-preserving topology nulls

We generated eight randomized choice-topology nulls. The production backbone, bill-of-material arrivals, terminal-accounting edges, and directed in/out degree are preserved; other route-choice edges are rewired.

Procurement pull improves viable-transition frequency in only one null. Several nulls are robust regardless of pull, while others remain nonviable despite mean onshore share above 0.80. Four nulls exhibit majority domestic share without a robust viable transition.

Figure 10 plots the topology-null distinction between onshore share, completed
production, and robust transition status.

This falsifies a topology-invariant procurement-pull claim, while preserving the more general distinction between route share and institutional feasibility.

---

## 7. Benchmark Models

We compare DTE against two reduced models.

### 7.1 Static expected-flow Markov baseline

The static baseline computes expected flow using one mean telemetry vector. It contains route attractiveness but no bill-of-material dynamics, capacity blocking, or telemetry adaptation.

### 7.2 Frozen-telemetry heterogeneous-agent baseline

The frozen-agent baseline preserves heterogeneous agents, bill-of-material gates, node and gate capacities, and stochastic routing, but sets \(\lambda=0\). It isolates the effect of telemetry adaptation.

### 7.3 Results

Across 60 policy cells:

| Metric | Result |
|---|---:|
| DTE robust cells | 36 |
| Frozen-agent robust cells | 16 |
| Cells where feedback changes robust classification | 20 |
| Static majority-onshore predictions | 60 |
| Static majority predictions not robust under DTE | 24 |

The static model systematically overstates successful policy cells because it cannot test production feasibility. The frozen-agent model captures hard constraints but misses a substantial feedback-dependent region of the phase surface.

Figure 9 summarizes the static, frozen-agent, and DTE benchmark comparison.

The benchmark supports an institutional-validity claim:

> DTE is useful when the decision question is not merely whether a route is attractive, but whether an intervention remains feasible, majority-capturing, and robust after heterogeneous agents adapt.

### 7.4 Neural V2 adaptive-routing validation

To test whether the kernel's memory and policy-lane mechanisms generalize
beyond the semiconductor case, we use a Neural V2 routing benchmark. Agents are
task batches routed among language, symbolic, and generalist modules. The
benchmark compares local-regret DTE, reliability-arbitrated UCB, DTE-EXP3,
reliability-gated DTE-EXP3, and external UCB/EXP3 baselines.

We evaluate three regimes over 30 paired seeds:

1. a clean adaptive-routing regime,
2. a corrupted/delayed hard regime with noisy context labels and delayed
   rewards,
3. an adversarial-switching regime where the favored module family alternates.

The DTE-EXP3 lane uses an importance-weighted gain with implicit exploration
(EXP3-IX): the per-edge gain is realized reward divided by the realized
selection frequency plus a smoothing term, evaluated on traffic support.
The validation table reports paired regret reduction relative to local-regret
DTE:

| Regime | Router | Regret | 95% CI | Delta vs Local | Delta 95% CI |
|---|---|---:|---:|---:|---:|
| Clean | DTE-EXP3 | 0.1081 | 0.0016 | 0.0233 | 0.0016 |
| Clean | Reliability-UCB DTE | 0.1275 | 0.0009 | 0.0039 | 0.0011 |
| Clean | External UCB | 0.0010 | 0.0000 | 0.1304 | 0.0008 |
| Hard | DTE-EXP3 | 0.1375 | 0.0013 | 0.0334 | 0.0017 |
| Hard | Reliability-gated DTE-EXP3 | 0.1478 | 0.0013 | 0.0231 | 0.0021 |
| Hard | Reliability-UCB DTE | 0.1681 | 0.0015 | 0.0028 | 0.0020 |
| Hard | External EXP3 | 0.1445 | 0.0029 | 0.0264 | 0.0033 |
| Adversarial switching | DTE-EXP3 | 0.1646 | 0.0015 | 0.0240 | 0.0017 |
| Adversarial switching | Reliability-gated DTE-EXP3 | 0.1659 | 0.0017 | 0.0226 | 0.0016 |
| Adversarial switching | Reliability-UCB DTE | 0.1831 | 0.0015 | 0.0055 | 0.0017 |
| Adversarial switching | External UCB | 0.0448 | 0.0004 | 0.1438 | 0.0016 |

These results revise the policy-arbitration boundary reported in an earlier
draft of this benchmark. DTE-EXP3 is now the strongest DTE-native lane in all
three regimes, including corrupted delayed attribution (paired delta +0.0334),
where it also runs slightly ahead of external EXP3 in absolute regret (0.1375
vs 0.1445). Reliability gating no longer earns its complexity: it reduces the
hard-regime gain (+0.0231 vs +0.0334 ungated) and helps at only one of eight
adversarial-switching grid coordinates. Reliability-arbitrated UCB is no
longer the hard-regime default. External UCB and EXP3 still dominate whenever
the task reduces to full policy ownership (clean paired delta +0.1304 for
external UCB against +0.0233 for DTE-EXP3; switching +0.1438 against +0.0240).
The implication is unchanged in kind: DTE does not beat bandits at their own
problem; it hosts learning policies while preserving topology and memory as
first-class dynamics — provided the hosted lane uses an unbiased gain
estimator.

The revision itself is a falsification result worth recording. The earlier
lane computed its gain as reward multiplied by realized selection frequency,
which throttles the update of a high-reward, rarely-taken edge by that edge's
own unpopularity — a multiplicative rich-get-richer loop in log-weight space,
and precisely the deadly-familiarity pathology DTE is built to diagnose. In a
two-route switching scenario the pre-fix lane never re-preferred a revived
sparse route within 2,800 post-switch steps; the importance-weighted lane
recovers in about two steps, and its hard-regime paired delta flips from
-0.0167 (worsens) to +0.0334 (largest DTE-native improvement). What the
earlier draft reported as attribution fragility was estimator bias. The
pre-fix artifacts are frozen in `archive/pre_estimator_fix_neural_v2/` for
comparison, and the two-route scenario is a permanent regression test.

Figure 3 visualizes the same policy-lane boundary as paired regret reduction
relative to local-regret DTE.

---

## 8. Discussion

### 8.1 General result versus case-study result

The choice-point principle is a general property of row-wise softmax routing on directed graphs. It does not depend on semiconductors, feature labels, or calibration.

The effectiveness of domestic procurement pull is not general. It is a topology-dependent case-study result. The randomized nulls show that the same intervention can help, do nothing, or coexist with failed production depending on route arrangement.

This distinction should be preserved throughout interpretation:

- **General:** interventions on singleton-outdegree edges cannot reroute flow.
- **General:** route share and feasible completion are distinct observables.
- **General:** adaptive feedback can change a robustness classification.
- **Case specific:** procurement pull produces majority capture in the specified semiconductor topology.
- **Case specific:** downstream import friction is inert because of where it is placed in this topology.

### 8.2 Policy interpretation

The semiconductor experiments suggest that policy should be represented at the point where decision makers retain a meaningful alternative. A cost imposed after route commitment may change accounting without changing allocation. A procurement rule or qualification decision applied before route commitment can change allocation, provided domestic production is feasible.

The model does not establish the magnitude of real-world tariffs, procurement commitments, or capacity investments required for onshoring. It establishes a structural question that should precede those estimates:

> Does the intervention act at a point where the system can still choose?

### 8.3 Cybernetic circulation

DTE treats supply chains as circulation systems in which demand signals, production capabilities, policy preferences, and agent state co-evolve. This cybernetic framing is useful because interventions do not merely push flow through a fixed graph. They alter the environment that shapes future routing behavior.

The same framing may apply to recommendation systems, knowledge navigation, transportation, and institutional information flow. Those applications require separate validation; they are not evidence in the present paper.

### 8.4 Policy-lane boundary

The Neural V2 benchmark shows that DTE should not be positioned as a universal
replacement for contextual bandits. When labels are clean, rewards are
immediate, and the router owns the full action distribution, external UCB and
EXP3 are the right baselines and should win. DTE's contribution is different:
it makes topology, memory, delayed feedback, and structural feasibility
explicit. With importance-weighted gain estimation, the EXP3 lane improves
DTE in every tested regime, so the boundary that remains is against full
policy ownership, not between reward regimes. The benchmark also carries a
design lesson for hosted learning lanes generally: an estimator that scales
updates by realized selection frequency imports the very lock-in dynamics the
governor exists to expose, and it did so here undetected until the gain rule
was audited. Learning lanes hosted inside adaptive-routing systems should be
held to the same falsification discipline as the routing claims themselves.

---

## 9. Limitations

This study has important limitations.

1. **No external calibration claim.**  
   The semiconductor graph is a public, role-bearing abstraction. It is not calibrated to confidential firm operations, real contract volumes, or predictive trade shares.

2. **Discrete-time routing abstraction.**  
   Agents move in synchronous ticks. The model does not yet represent continuous-time lead-time distributions, queue disciplines, or production-cycle duration.

3. **Simplified production mechanics.**  
   Bill-of-material gates and service capacities represent dependency feasibility, but not yields, rework, quality grades, price formation, or multi-period inventory policy.

4. **Finite seed and finite grid evidence.**  
   Robustness is empirical over five seeds and sampled intervention grids. No sampled threshold is a mathematical lower bound.

5. **Feedback model is stylized.**  
   The telemetry update is an exponential moving average. Real procurement behavior may be governed by contracts, organizational inertia, and strategic optimization rather than feature assimilation.

6. **Topology nulls are constrained abstractions.**  
   Degree-preserving rewiring tests sensitivity to route arrangement but does not produce economically realistic alternative semiconductor industries.

7. **No welfare objective.**  
   A higher domestic share is not automatically socially optimal. Cost, innovation, alliance resilience, consumer welfare, and geopolitical risk are not combined into one normative objective.

---

## 10. Related Work Positioning

**TODO: add verified citations and write this section after a focused literature review.**

The paper should position DTE relative to:

- stochastic shortest-path and Markov routing models,
- dynamic and non-stationary Markov processes,
- agent-based supply-chain simulation,
- discrete-event simulation and queueing networks,
- network interdiction and resilience optimization,
- input-output and production-network models,
- reinforcement learning and adaptive routing,
- logit choice and entropy-regularized transport,
- cybernetics and adaptive control.

The central distinction to preserve is that DTE is not proposed as a replacement for these methods. It contributes a compact joint model of adaptive route preference, topology-local intervention, and operational feasibility.

---

## 11. Reproducibility

The implementation is written in Python and NumPy. Experiments use paired common-random-number comparisons through a shared randomization key. The Neural V2 validation bench, semiconductor falsification suite, feasibility-preference surface, threshold robustness analysis, and benchmark reports are generated by:

- `neural_v2_seed_validation.py`
- `neural_v2_seed_validation_figure.py`
- `neural_v2_router_benchmark.py`
- `semiconductor_onshoring_feasibility_preference.py`
- `semiconductor_onshoring_classification_robustness.py`
- `semiconductor_onshoring_model_benchmark.py`
- `semiconductor_onshoring_falsification.py`

The Neural V2 EXP3 results in this draft were generated after the
2026-07-05 gain-estimator correction (`exp3_ix`, importance weighting with
implicit exploration, per-source weight renormalization). The superseded
pre-fix outputs are frozen in `archive/pre_estimator_fix_neural_v2/` and are
cited only as the estimator-bias case study in Section 7.4. The two-route
switching scenario that exposed the bias runs as a permanent regression test
in `tests/test_review_fixes.py`.

The public reproduction suite packaged in this repository currently passes:

```text
149 passed
```

**TODO before submission:**

- add commit hash and environment versions to the frozen experiment manifest,
- export remaining paper figures from scripts rather than manual plotting,
- add public calibration source citations,
- publish a minimal reproduction command sequence,
- archive raw JSON outputs used in tables.

---

## 12. Conclusion

This paper introduced the Dynamic Topology Engine as a model of adaptive circulation on feature-decorated directed graphs. The principal structural result is simple but consequential: changing the cost or attractiveness of an edge cannot reroute flow when that edge is the only admissible next transition. Intervention effectiveness depends on the location of choice.

The semiconductor case study shows why this matters operationally. Physical resource scaling can make domestic production feasible without ensuring robust majority capture. Upstream procurement preference can change allocation in the specified topology, while import friction applied inside a serial offshore corridor is dynamically inert. Adding latent reconsideration exits activates the same downstream intervention. Static expected-flow models overstate successful policy cells, and high route share can coexist with failed completed production.

The broader lesson is that adaptive network interventions should be evaluated as coupled feasibility, allocation, and adaptation transitions. A route that looks attractive is not necessarily usable. A route that is usable is not necessarily selected. A route that is selected is not necessarily robust after agents adapt.

---

## Appendix A. Claim-Evidence Boundary

| Claim | Status | Evidence |
|---|---|---|
| Singleton-outdegree edge interventions cannot reroute flow | General theorem | Proposition 1 |
| Downstream serial penalties are inert in the semiconductor topology | Supported case-study result | Relocation experiment |
| Adding latent alternatives can activate downstream penalties | Supported mechanism result | Topology surgery |
| Procurement pull guarantees onshoring | Rejected | Randomized topology nulls |
| Feedback universally amplifies interventions | Rejected | Feedback-rate continuum |
| Route share implies viable production | Rejected | BOM gates, benchmark, topology nulls |
| DTE universally beats UCB/EXP3 | Rejected | Neural V2 seed validation |
| DTE-EXP3 with traffic-weighted gains is robust to corrupted attribution | Rejected — failure traced to gain-estimator bias, not attribution noise | Neural V2 hard benchmark (archived pre-fix artifacts), two-route switching regression |
| DTE-EXP3 with importance-weighted (EXP3-IX) gains improves under corrupted attribution | Supported at 30 paired seeds | Neural V2 hard benchmark and seed validation |
| DTE-EXP3 is useful under adversarial switching | Supported regime result | Neural V2 adversarial-switch validation |
| DTE replaces domain-specific supply-chain models | Not claimed | Scope and limitations |

## Appendix B. Planned Figures

1. DTE schematic: `figures/dte_mechanism_schematic.svg`.
2. Layered-memory state schematic: `figures/dte_layered_memory_schematic.svg`.
3. Neural V2 policy-lane boundary: `figures/neural_v2_seed_validation_delta.svg`.
4. Semiconductor topology with production gates and route-choice points: `figures/semiconductor_topology_schematic.svg`.
5. Feasibility-allocation-adaptation distinction: `figures/feasibility_allocation_adaptation.svg`.
6. Choice-point relocation: `figures/semiconductor_choice_point_relocation.svg`.
7. Topology surgery: `figures/semiconductor_topology_surgery.svg`.
8. Feedback-rate continuum: `figures/semiconductor_feedback_continuum.svg`.
9. Benchmark comparison: `figures/semiconductor_model_benchmark.svg`.
10. Randomized topology nulls: `figures/semiconductor_topology_nulls.svg`.

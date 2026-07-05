# DTE V1 Theory And Kernel Closure

## Status

This note defines the V1 kernel contract after the design-space critique and
Axis-5 promotion. It is not a broad application memo. It records what the core
kernel now means, what is first-class API, and what remains outside V1.

## V1 Thesis

DTE is a kernel for adaptive circulation under layered memory. Its distinctive
object is not shortest-path routing, but the interaction among:

- $M_s(t)$: structural memory, including topology, gates, contracts, and feasible edges.
- $M_p(t)$: preference memory, including $\beta$ and friction deltas over edges.
- $M_x(t)$: state memory, including agent telemetry, intent, or role.
- $R_t$: ecology variables, including realized reward, capacity, stock, hazard, and demand.

The V1 failure mode is memory-ecology mismatch:

$M_p$ persists after $R_t$ changes, so circulation follows a once-valid path
that is no longer locally best.

The V1 mitigation is adaptive forgetting:

$\eta_t$ increases when realized reward falls below expectation or when chosen
traffic incurs local opportunity cost.

## Formal Object

For V1, treat DTE as a controlled adaptive process on the augmented state:

$$
Z_t = (X_t, A_t, M_s(t), M_p(t), B_t, R_t)
$$

where:

- $X_t$: discrete agent position or population occupancy.
- $A_t$: agent telemetry or state-memory distribution.
- $M_s(t)$: structural memory, including topology, capacities, gates, and contracts.
- $M_p(t)$: preference memory, including $\beta$, friction deltas, and policy lanes.
- $B_t$: delayed-feedback buffers and pending attribution records.
- $R_t$: realized ecology, including reward, demand, capacity, stock, and hazard.

The one-step transition has the form:

$$
\Pr[Z_{t+1}\mid Z_t,u_t]
= K_{\mathrm{move}}(P(M_s,M_p,A_t,R_t,u_t))
  K_{\mathrm{state}}(A_{t+1}\mid A_t,X_{t+1})
  K_{\mathrm{memory}}(M_s(t+1),M_p(t+1),B_{t+1},R_{t+1}\mid Z_t,X_{t+1})
$$

The kernel's distinctive claim is not that this factorization is unique. The
claim is that position, state memory, preference memory, structural memory, and
delayed attribution must be present in the state whenever the application
contains endogenous adaptation. If any of these are projected away, the
observed process may remain useful but is generally not Markov.

## V1 Propositions

### Proposition 1: Markov Closure Under Layered Memory

If the augmented state includes current position or occupancy, telemetry,
structural memory, preference memory, delayed-feedback buffers, and current
ecology variables used by the update rules, then the V1 DTE process is Markov.
If telemetry or delayed-feedback buffers are omitted while feedback rate or
reward delay is nonzero, the observed position process is generally
non-Markovian.

Proof sketch. The transition probabilities are computed from current topology,
edge controls, telemetry, and ecology variables. Memory-law updates depend on
current traffic, reward, and any pending delayed records. Once those quantities
are included in `Z_t`, no earlier history is needed. If telemetry is omitted,
two histories with the same position can induce different alignment scores and
therefore different next-state distributions. If delayed buffers are omitted,
two histories with the same visible state can produce different future memory
updates when pending rewards mature.

### Proposition 2: Choice-Point Invariance

At any node with exactly one admissible outgoing edge, every finite
edge-weight intervention leaves the next-state distribution unchanged under
row-wise softmax routing.

Proof sketch. The softmax denominator contains a single admissible term, so the
transition probability is one regardless of finite changes to cost, beta,
friction reduction, or alignment. This is the paper's structural invariance
result and explains why downstream serial-corridor interventions can be
dynamically inert until a genuine alternative is created.

### Proposition 3: Estimator Bias In Traffic-Weighted Multiplicative Policy Lanes

Consider an EXP3-style policy lane whose weights update multiplicatively from
delayed rewards assigned to realized traffic. If the gain estimator multiplies
reward by realized selection frequency, high-reward low-traffic edges are
discounted by their own current unpopularity. This can recreate stale
familiarity inside the learning lane itself. An importance-weighted EXP3-IX
estimator removes this traffic-weighted bias.

Proof sketch. EXP3 updates are additive in log-space:

$$
\log w_{c,a}(t+1)=\log w_{c,a}(t)+\eta\,\hat r(c,a,t).
$$

With the biased estimator, $\hat r$ is proportional to reward times realized
traffic share. The update therefore punishes sparse edges even when their
counterfactual reward is high. The corrected EXP3-IX lane instead estimates
edge gain as reward divided by the realized selection probability plus implicit
exploration, so the hard-regime deficit in the earlier benchmark is treated as
estimator bias rather than intrinsic attribution fragility.

### Proposition 4: Arbitration Boundary

On clean contextual bandit tasks with immediate rewards and independent module
choices, an external policy that owns the full action distribution can dominate
an arbitrated DTE policy lane. DTE's advantage is expected only when topology,
memory, feasibility, or delayed correction are part of the task.

Proof sketch. A full policy owner chooses directly from its learned action
distribution. An arbitrated DTE lane mixes the learned distribution with a
topology-memory governor:

$$
P = (1-\lambda_t)P_{\mathrm{DTE}}+\lambda_t P_{\mathrm{policy}}.
$$

Unless $\lambda_t=1$, the arbitrated policy is constrained by DTE's existing
memory and topology pressure. This constraint is useful when those pressures
encode real system dynamics, but it is a liability on a clean bandit benchmark
where independent action choice is the whole problem.

## Canonical Runtime Loop

Kernel V1 uses a four-stage population loop:

```text
agents move
completed traversals accumulate traffic_ij
node rewards R_j are observed
memory_law_step(traffic, R) updates beta or friction deltas
```

This loop is now represented in `PopulationSimulator.tick()` and should be the
default integration pattern for new adapters. Bespoke simulators may still
exist when they need domain-specific mechanics, but they should report how
their loop maps to this contract.

## Preference-Memory Law

The kernel default remains static:

$$
M_p(t+1)=M_p(t)
$$

Nonstationary experiments may opt in to:

Traffic-only update:

$$
\delta_{ij}(t+1)=(1-\eta)\delta_{ij}(t)+\rho\,\mathrm{traffic}_{ij}(t)
$$

Reward-gated update:

$$
\delta_{ij}(t+1)=(1-\eta)\delta_{ij}(t)+\rho\,\mathrm{traffic}_{ij}(t)R_j(t)
$$

Adaptive-$\eta$ update:

$$
\delta_{ij}(t+1)=(1-\eta_j(t))\delta_{ij}(t)+\rho\,\mathrm{traffic}_{ij}(t)R_j(t)
$$

with:

$$
\eta_j(t)=\operatorname{clip}\left(
\eta
+ g_s\,\mathrm{traffic\_share}_j\,\max(0,\hat R_j(t)-R_j(t))
+ g_o\,\mathrm{destination\_regret}_j(t),
\eta,
\eta_{\max}
\right)
$$

Only admissible edges may accumulate preference memory. Non-edge traffic is
masked out by the kernel.

## Opportunity Cost

Surprise detects collapse:

$$
R_j(t)<\hat R_j(t)
$$

But the design-space experiments exposed a harder regime: marginal stale
grazing. A route may remain mildly productive and therefore generate little
surprise while still being worse than a reachable alternative.

V1 therefore adds local opportunity cost:

$$
\mathrm{regret}_{ij}(t)=
\max\left(0,\max_{k\in\operatorname{Out}(i)}R_k(t)-R_j(t)\right)
$$

and destination-level regret:

$$
\mathrm{destination\_regret}_j(t)
=
\frac{\sum_i \mathrm{traffic}_{ij}(t)\,\mathrm{regret}_{ij}(t)}
{\sum_{i,k}\mathrm{traffic}_{ik}(t)}
$$

This is not a full counterfactual planner. It is a local diagnostic for stale
preference memory: "traffic is still flowing here, but a better reachable
choice exists from the same source."

## What The Design-Space Critique Established

1. Choice-point invariance is structural. If the admissible set has one edge,
   normalized routing gives probability one regardless of softmax details.

2. Stale lock-in is structural across monotone link functions, but its severity
   is controlled by the preference-memory update law.

3. The default kernel could not express the headline stale-memory pathology
   without experiment-level reinforcement loops. Promoting Axis-5 memory laws
   as opt-in API fixes that architectural gap.

4. Telemetry update variants matter, but they did not carry the central
   phenomenon in the first design-space pass. The current EMA telemetry update
   remains the V1 default.

5. Adaptive evaporation is a mitigation, not a universal cure. It needs
   opportunity-cost diagnostics when the stale route remains weakly productive.

## V1 API Commitments

- `DynamicTopologyKernel.configure_memory_law(...)`
- `DynamicTopologyKernel.memory_law_step(traffic, node_reward=None)`
- `DynamicTopologyKernel.memory_law_state()`
- `DynamicTopologyKernel.opportunity_cost_diagnostic(traffic, node_reward)`
- `PopulationSimulator.set_node_rewards(rewards)`
- `PopulationSimulator.tick()` applies the canonical memory loop when a
  memory law is configured.

## Out Of Scope For V1

- Learned telemetry update laws.
- Topology editing as an endogenous action.
- Global counterfactual planning over all unchosen paths.
- A general equilibrium sponsor market.
- Full UI doctrine for every application adapter.

These are research directions, not kernel-closure requirements.

## V1 Completion Criterion

Kernel V1 is complete when:

1. Static behavior is backward-compatible.
2. Endogenous preference memory is opt-in and tested.
3. Traffic/reward/memory integration has one canonical simulator path.
4. Surprise and opportunity-cost diagnostics are available.
5. A visualization can expose memory help, stale lock-in, and adaptive
   forgetting without adapter-specific hacks.

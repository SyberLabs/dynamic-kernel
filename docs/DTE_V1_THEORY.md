# DTE V1 Theory And Kernel Closure

## Status

This note defines the V1 kernel contract after the design-space critique and
Axis-5 promotion. It is not a broad application memo. It records what the core
kernel now means, what is first-class API, and what remains outside V1.

## V1 Thesis

DTE is a kernel for adaptive circulation under layered memory. Its distinctive
object is not shortest-path routing, but the interaction among:

```text
M_s(t): structural memory   topology, gates, contracts, feasible edges
M_p(t): preference memory   beta and friction deltas over edges
M_x(t): state memory        agent telemetry / intent / role
R_t:    ecology             realized reward, capacity, stock, hazard, demand
```

The V1 failure mode is memory-ecology mismatch:

```text
M_p persists after R_t changes, so circulation follows a once-valid path
that is no longer locally best.
```

The V1 mitigation is adaptive forgetting:

```text
eta_t increases when realized reward falls below expectation or when chosen
traffic incurs local opportunity cost.
```

## Formal Object

For V1, treat DTE as a controlled adaptive process on the augmented state:

```text
Z_t = (X_t, A_t, M_s(t), M_p(t), B_t, R_t)
```

where:

```text
X_t: discrete agent position or population occupancy
A_t: agent telemetry / state-memory distribution
M_s(t): structural memory, including topology, capacities, gates, contracts
M_p(t): preference memory, including beta, friction deltas, and policy lanes
B_t: delayed-feedback buffers and pending attribution records
R_t: realized ecology, including reward, demand, capacity, stock, hazard
```

The one-step transition has the form:

```text
Pr[Z_{t+1} | Z_t, u_t]
    = K_move(P(M_s, M_p, A_t, R_t, u_t))
      K_state(A_{t+1} | A_t, X_{t+1})
      K_memory(M_s(t+1), M_p(t+1), B_{t+1}, R_{t+1} | Z_t, X_{t+1})
```

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

### Proposition 3: Attribution Fragility Of Multiplicative Policy Lanes

Consider an EXP3-style policy lane whose weights update multiplicatively from
delayed rewards assigned to observed context labels. If observed labels differ
from true reward contexts with nonzero probability, then expected log-weight
growth can favor an action in the wrong context whenever the misattributed
reward advantage exceeds the true-context advantage.

Proof sketch. EXP3 updates are additive in log-space:

```text
log w_{c,a}(t+1) = log w_{c,a}(t) + eta * r_hat(c,a,t).
```

With label noise, the estimator for context `c` is a mixture of true rewards
from `c` and rewards from other contexts. A wrong action becomes reinforced
when the mixture expectation for that action exceeds the expectation of the
correct action under the true context. Because the update is multiplicative in
weight space, persistent misattribution compounds rather than merely averaging
out over short horizons. This explains why DTE-EXP3 improves under isolated
switching rewards but becomes brittle in the default corrupted/delayed hard
benchmark.

### Proposition 4: Arbitration Boundary

On clean contextual bandit tasks with immediate rewards and independent module
choices, an external policy that owns the full action distribution can dominate
an arbitrated DTE policy lane. DTE's advantage is expected only when topology,
memory, feasibility, or delayed correction are part of the task.

Proof sketch. A full policy owner chooses directly from its learned action
distribution. An arbitrated DTE lane mixes the learned distribution with a
topology-memory governor:

```text
P = (1 - lambda_t) P_DTE + lambda_t P_policy.
```

Unless `lambda_t = 1`, the arbitrated policy is constrained by DTE's existing
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

```text
M_p(t+1) = M_p(t)
```

Nonstationary experiments may opt in to:

```text
traffic:
delta_ij(t+1) = (1 - eta) delta_ij(t) + rho traffic_ij(t)

reward_gated:
delta_ij(t+1) = (1 - eta) delta_ij(t) + rho traffic_ij(t) R_j(t)

adaptive_eta:
delta_ij(t+1) = (1 - eta_j(t)) delta_ij(t) + rho traffic_ij(t) R_j(t)
```

with:

```text
eta_j(t) = clip(
    eta
    + surprise_gain * traffic_share_j * max(0, Rhat_j(t) - R_j(t))
    + opportunity_gain * destination_regret_j(t),
    eta,
    eta_max
)
```

Only admissible edges may accumulate preference memory. Non-edge traffic is
masked out by the kernel.

## Opportunity Cost

Surprise detects collapse:

```text
R_j(t) < Rhat_j(t)
```

But the design-space experiments exposed a harder regime: marginal stale
grazing. A route may remain mildly productive and therefore generate little
surprise while still being worse than a reachable alternative.

V1 therefore adds local opportunity cost:

```text
regret_ij(t) = max(0, max_{k in Out(i)} R_k(t) - R_j(t))
```

and destination-level regret:

```text
destination_regret_j(t)
    = sum_i traffic_ij(t) regret_ij(t) / sum_{i,k} traffic_ik(t)
```

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

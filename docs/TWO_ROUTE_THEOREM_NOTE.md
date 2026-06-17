# Two-Route Memory-Ecology Theorem Note

## Purpose

This note derives the minimal mathematical mechanism behind the
Memory-Ecology Mismatch Principle. The goal is not to prove the full DTE theory
yet. The goal is to prove that stale-memory lock-in exists in the simplest
possible adaptive circulation system.

The model has two substitutable routes:

```text
Choice -> h  high-reward finite route
       -> l  lower-reward persistent route
```

The high route is initially better, then depletes. The low route remains
productive. Preference memory accumulates on successful routes and evaporates
geometrically.

## Model

Let:

```text
r_h(t) = reward of high route h at time t
r_l    = reward of persistent route l, with r_l > 0
m_h(t) = preference memory on h
m_l(t) = preference memory on l
rho    = reinforcement rate
eta    = evaporation rate
kappa  = inverse temperature
epsilon = exploration rate
T      = depletion time
```

Assume:

```text
r_h(t) = r_h > r_l for t < T
r_h(t) = 0       for t >= T
```

The route scores are:

```text
s_h(t) = r_h(t) + m_h(t)
s_l(t) = r_l + m_l(t)
```

The non-exploratory softmax probability of choosing the high route is:

```text
q_h(t) = exp(kappa s_h(t)) / (exp(kappa s_h(t)) + exp(kappa s_l(t)))
```

and:

```text
q_l(t) = 1 - q_h(t)
```

With exploration:

```text
p_h(t) = (1 - epsilon) q_h(t) + epsilon / 2
p_l(t) = (1 - epsilon) q_l(t) + epsilon / 2
```

The memory updates are:

```text
m_h(t+1) = (1 - eta) m_h(t) + rho p_h(t) r_h(t)
m_l(t+1) = (1 - eta) m_l(t) + rho p_l(t) r_l
```

Define the memory gap:

```text
Delta_t = m_h(t) - m_l(t)
```

## Exact Gap Dynamics

Subtracting the memory equations gives:

```text
Delta_{t+1}
= (1 - eta) Delta_t
  + rho [p_h(t) r_h(t) - p_l(t) r_l]
```

Before depletion:

```text
Delta_{t+1}
= (1 - eta) Delta_t
  + rho [p_h(t) r_h - p_l(t) r_l]
```

After depletion:

```text
Delta_{t+1}
= (1 - eta) Delta_t
  - rho p_l(t) r_l
```

This post-depletion equation is the key. The high route receives no new reward,
but its memory does not disappear instantly. Sparse-route memory continues to
grow. Recovery is therefore driven by both high-route evaporation and sparse
route reinforcement.

## Proposition 1: Exact Lock-In Condition

For `epsilon < 1`, the high route has greater transition probability than the
low route if and only if its softmax score is greater:

```text
p_h(t) > p_l(t)
iff
s_h(t) > s_l(t)
```

Because exploration mixes both routes equally, it does not change the ordering
unless `epsilon = 1`, in which case both routes are forced to probability
`1/2`.

After depletion, `r_h(t) = 0`. Therefore stale lock-in occurs exactly when:

```text
m_h(t) > r_l + m_l(t)
```

or equivalently:

```text
Delta_t > r_l
```

This is the cleanest form of the theorem. The high route is currently worthless,
but it remains preferred while its memory advantage exceeds the persistent
route's reward.

## Proposition 2: Existence Of A Positive Stale Interval

Suppose that at depletion time `T`,

```text
Delta_T > r_l
```

and `0 <= epsilon < 1`. Then there exists at least one post-depletion step for
which:

```text
p_h(T) > p_l(T)
```

If the update rule is continuous in `Delta_t`, then there exists a positive
interval of stale lock-in until the first time `tau` such that:

```text
Delta_{T+tau} <= r_l
```

Proof: By Proposition 1, `Delta_T > r_l` implies `p_h(T) > p_l(T)` after
depletion. The memory update is finite and continuous in the current memory
state and probabilities, so the preference cannot jump from strict high-route
dominance to strict low-route dominance without crossing the boundary
`Delta = r_l`.

This proves the existence of deadly familiarity in the two-route system.

## Proposition 3: Conservative Recovery Bound

After depletion:

```text
Delta_{t+1}
= (1 - eta) Delta_t - rho p_l(t) r_l
```

Since `rho p_l(t) r_l >= 0`, we have:

```text
Delta_{t+1} <= (1 - eta) Delta_t
```

By induction:

```text
Delta_{T+s} <= Delta_T (1 - eta)^s
```

A sufficient condition for recovery is:

```text
Delta_T (1 - eta)^s <= r_l
```

Solving for `s` gives:

```text
s >= log(r_l / Delta_T) / log(1 - eta)
```

when:

```text
Delta_T > r_l
0 < eta < 1
```

Because `log(1 - eta) < 0`, the inequality direction is preserved by taking the
ceiling carefully:

```text
T_recover <= ceil(log(r_l / Delta_T) / log(1 - eta))
```

This is an upper bound on recovery time under the full model because the sparse
route also accumulates memory after depletion, accelerating recovery. It is a
conservative decay-only estimate.

The simulator reports this quantity as `decay_only_recovery_bound`. It is the
sufficient recovery time under evaporation-only decay. Sparse-route
reinforcement can only accelerate recovery relative to this bound in the full
two-memory model.

## Proposition 4: No Recovery Under Zero Evaporation And Insufficient Sparse Reinforcement

If `eta = 0`, post-depletion gap dynamics become:

```text
Delta_{t+1} = Delta_t - rho p_l(t) r_l
```

If `rho = 0` as well, then:

```text
Delta_{t+1} = Delta_t
```

Thus if:

```text
Delta_T > r_l
rho = 0
eta = 0
```

then stale lock-in never recovers.

If `eta = 0` but `rho > 0`, sparse reinforcement can still reduce the gap:

```text
Delta_{t+s} = Delta_T - rho r_l sum_{u=0}^{s-1} p_l(T+u)
```

Since exploration gives:

```text
p_l(t) >= epsilon / 2
```

for `epsilon > 0`, recovery occurs in finite time bounded by:

```text
s >= 2(Delta_T - r_l) / (rho r_l epsilon)
```

This shows why exploration is not just noise. It is a recovery mechanism.

## Proposition 5: Exploration Weakens But Does Not Eliminate Lock-In

For `0 < epsilon < 1`:

```text
p_h(t) - p_l(t) = (1 - epsilon)(q_h(t) - q_l(t))
```

Thus exploration shrinks the probability gap but preserves its sign.

Consequences:

1. Exploration reduces empty-route mass during lock-in.
2. Exploration increases sparse-route reinforcement after depletion.
3. Exploration does not by itself change the exact score-ordering threshold
   unless `epsilon = 1`.

Therefore exploration is a release mechanism through cumulative sparse
experience, not through immediate reversal of preference ordering.

## Relation To The Simulator

The simulator implements the exact equations above with:

```text
r_h = 1.0 before depletion
r_h = 0.0 after depletion
r_l = 0.62
kappa = 5.0
```

The quick sweep observed:

| Classification | Count |
|---|---:|
| `recovered_stale_lockin` | 11 |
| `unrecovered_stale_lockin` | 7 |

The first lock-in appeared at:

```text
rho / eta = 1.00
```

The first unrecovered lock-in appeared at:

```text
rho / eta = 12.00
```

The strongest quick-grid lock-in case:

```text
rho = 0.12
eta = 0.01
epsilon = 0.02
rho / eta = 12.00
stale_lockin_duration = 60 cycles
post_depletion_empty_rate = 0.989
```

This is consistent with the propositions:

- before depletion, high reward builds `Delta_T`;
- after depletion, lock-in occurs while `Delta_t > r_l`;
- lower `eta` extends the stale interval;
- higher `epsilon` reduces empty-route mass and helps sparse memory recover.

## Bound Semantics

The decay-only bound should not be described as a lower bound on full-system
recovery. In the full model:

```text
Delta_{t+1}
= (1 - eta) Delta_t - rho p_l(t) r_l
```

sparse reinforcement speeds recovery relative to pure evaporation. Therefore:

```text
ceil(log(r_l / Delta_T) / log(1 - eta))
```

is a sufficient recovery time under evaporation-only decay:

```text
decay_only_recovery_timescale
```

## Paper-Ready Theorem Statement

**Theorem.**  
Consider the two-route adaptive-memory process above with `0 <= epsilon < 1`,
`0 < eta < 1`, `rho >= 0`, and `r_l > 0`. Suppose the high route depletes at
time `T`, so `r_h(t) = 0` for `t >= T`. If:

```text
Delta_T = m_h(T) - m_l(T) > r_l
```

then the process exhibits stale lock-in at time `T`:

```text
p_h(T) > p_l(T)
```

even though:

```text
r_h(T) = 0 < r_l
```

Furthermore, if `rho = 0` after depletion, recovery occurs no later than:

```text
ceil(log(r_l / Delta_T) / log(1 - eta))
```

steps after depletion. If `rho > 0`, sparse-route reinforcement can only
accelerate recovery relative to this evaporation-only bound.

## Interpretation

The theorem says that stale lock-in does not require a complex graph. It does
not require heterogeneous agents, social influence, or adversarial behavior.
It appears as soon as three conditions coexist:

1. a route can accumulate memory from past success;
2. the route's reward can vanish faster than memory evaporates;
3. transition probability still consults the memory after the reward vanishes.

This is the minimal mathematical form of:

```text
memory remains operationally true after it becomes ecologically false
```

The richer DTE architecture matters because it adds ways to detect and correct
that mismatch: state memory, semantic features, feasibility diagnostics,
exploration controls, memory decay, and structural rewiring.

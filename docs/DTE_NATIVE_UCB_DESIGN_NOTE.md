# DTE-Native UCB Design Note

## Scope

This note records the first kernel-level implementation of DTE-native UCB.
The purpose is not to claim victory over ordinary UCB. The purpose is to test
whether UCB-style optimism can be internalized into the DTE transition kernel
without abandoning topology, memory law, and local-regret diagnostics.

## Kernel Form

The baseline transition weight is:

```text
W_ij = alpha * D_ij - beta_ij * alignment_j - S_ij
```

DTE-native UCB adds an edge-learning potential before the softmax:

```text
W^UCB_ij = W_ij
           - reward_gain * rhat_ij
           - uncertainty_gain * c * sqrt(log(1 + n_i) / (1 + n_ij))
```

where:

- `rhat_ij` is the learned reward estimate for edge `i -> j`
- `n_ij` is the edge visit count
- `n_i` is the source visit count
- invalid topology edges are masked out

The feature is opt-in:

```python
kernel.configure_edge_learning(mode="ucb")
kernel.edge_learning_step(traffic, node_reward=reward)
```

Default kernel behavior is unchanged when edge learning is not configured.

## First Empirical Result

The first aggressive DTE-UCB parameterization failed. It over-valued
under-sampled edges and reduced symbolic routing in Neural V2. After tuning,
the conservative configuration became nearly neutral but still did not improve
plain local-regret DTE:

```text
hard_dte_local_regret regret = 0.1669
hard_dte_ucb regret          = 0.1683
```

Across the frontier, DTE-native UCB did not improve local-regret DTE at any
swept value. The typical loss was small after conservative tuning, usually
around 0.001 to 0.003 regret, but it remained a loss.

## Apex Inference

This is not evidence against UCB. It is evidence that the current DTE-native
UCB is missing the important state variable: context-conditioned reward memory.

Ordinary UCB wins because it estimates:

```text
rhat(context, module)
```

The first DTE-native UCB estimates only:

```text
rhat(edge)
```

That global edge estimate is too blunt for Neural V2, where the same edge can
be excellent for one task telemetry and mediocre for another. The next design
should therefore use telemetry-conditioned edge statistics:

```text
rhat_ij(z)
```

or a small set of learned telemetry bins:

```text
rhat_ij^k,  k = cluster(z)
```

## Next Design

The next version tested was DTE-contextual UCB:

```text
U_ij(z_t) = c * sqrt(log(1 + n_i^k) / (1 + n_ij^k))
```

with updates assigned to the active telemetry/context bin `k`.

This preserved DTE's topology and memory law while adding the key thing the
external UCB baseline has: context-specific reward estimation.

## Contextual UCB Result

The kernel now supports contextual edge-learning state through
`context_centroids`. Mechanically, this works: the same edge can carry different
reward estimates and UCB bonuses in different telemetry bins.

The Neural V2 benchmark improved, but only modestly. In the default hard
benchmark, contextual UCB is nearly tied with local-regret DTE:

```text
hard_dte_local_regret regret     = 0.1669
hard_dte_contextual_ucb regret  = 0.1669
```

The frontier result is more informative. DTE-contextual UCB improves
local-regret DTE at nearly every swept value, usually by about 0.002 to 0.004
regret. The effect is real enough to validate context-conditioned edge memory,
but too small to close the gap against policy-owning UCB/EXP3 baselines.
Stronger contextual reward/uncertainty gains still hurt performance, which
means the gap is not solved by simply turning up the additive UCB potential.

The likely reason is architectural: DTE's local-regret memory already exerts a
strong topology-conditioned routing pressure. Injecting UCB as another additive
potential inside `W` can disturb that pressure instead of complementing it.
External UCB succeeds because it owns the whole action policy; DTE-contextual
UCB merely perturbs an already structured stochastic policy.

The next serious design is therefore not "more UCB gain." It is policy
arbitration:

```text
P = (1 - lambda_t) * P_DTE + lambda_t * P_UCB
```

where `lambda_t` is itself adaptive, rising when the kernel detects high
uncertainty or stale-route regret and falling once local-regret memory has
stabilized. That would let UCB explore without constantly interfering with the
DTE memory law.

## Policy Arbitration Result

The kernel now supports this architecture through:

```python
kernel.configure_edge_learning(mode="ucb", policy="arbitrated")
```

In arbitrated mode, the edge-learning potential is no longer subtracted from
the DTE weight matrix. Instead, the kernel computes:

```text
P_mix = (1 - lambda_i) * P_DTE + lambda_i * P_UCB
lambda_i = lambda_min
           + (lambda_max - lambda_min) * u_i / (u_i + s)
```

where `u_i` is the row-level UCB uncertainty signal and `s` is a scale
parameter. This keeps DTE's topology, telemetry, and local-regret policy intact
while letting uncertainty open a separate exploration channel.

The first Neural V2 result validates the direction:

```text
clean dte_local_regret regret      = 0.1312
clean dte_arbitrated_ucb regret   = 0.1283

hard dte_local_regret regret      = 0.1669
hard dte_arbitrated_ucb regret   = 0.1652
```

Frontier behavior is mixed but informative. Arbitration beats local-regret DTE
on many slices and often beats additive contextual UCB, especially when verifier
bonus is high. It can over-explore under high context noise, where additive
contextual UCB is safer. The next refinement should make `lambda_t` depend not
only on UCB uncertainty, but also on observed context reliability.

## Context-Reliability Gating Result

The next implementation adds reliability-gated arbitration:

```text
lambda_i <- lambda_i * rho(z)
rho(z) = rho_min + (1 - rho_min) * margin(z) / (margin(z) + s_r)
margin(z) = top_1(<z, c_k>) - top_2(<z, c_k>)
```

where `z` is the observed telemetry/context vector and `c_k` are the context
centroids. When the observed context is close to one centroid and far from the
others, `rho` is high. When the context is ambiguous, `rho` suppresses the UCB
policy lane.

The first hard benchmark result is positive:

```text
hard dte_local_regret regret                    = 0.1669
hard dte_arbitrated_ucb regret                  = 0.1652
hard dte_reliability_arbitrated_ucb regret      = 0.1641
```

The clean benchmark shows the expected cost of the gate:

```text
clean dte_arbitrated_ucb regret                 = 0.1283
clean dte_reliability_arbitrated_ucb regret     = 0.1291
```

This is the right qualitative behavior. Reliability gating is not a universal
performance knob; it is a robustness mechanism. It helps when observed context
is corrupt enough that ungated arbitration over-explores the wrong UCB lane.
It hurts slightly when context is already clean because it withholds useful
exploration. In the frontier, this shows up most clearly under high
`context_noise`: gating loses to ungated arbitration at low noise, then improves
it once the context surface becomes unreliable.

The next refinement should make the reliability gate adaptive rather than
static: estimate calibration error online from delayed reward residuals, then
learn `rho_min` and `s_r` per context family.

## DTE-EXP3 Result

The next policy lane tested was DTE-EXP3:

```text
P_mix = (1 - lambda_i) * P_DTE + lambda_i * P_EXP3
w_ij <- w_ij * exp(eta * observed_reward_ij)
P_EXP3 = (1 - gamma) * normalize(w_i*) + gamma * Uniform(valid edges)
```

This is the correct architectural placement for EXP3: a separate adversarial
policy lane, not an additive term inside `W`.

The first clean benchmark result is strong:

```text
clean dte_local_regret regret      = 0.1312
clean dte_arbitrated_ucb regret    = 0.1283
clean dte_exp3 regret              = 0.0989
```

However, the default hard benchmark sharply falsifies the naive H2 claim:

```text
hard dte_local_regret regret                    = 0.1669
hard dte_reliability_arbitrated_ucb regret      = 0.1641
hard dte_exp3 regret                            = 0.2202
```

Interpretation: EXP3 is powerful when rewards are observed cleanly enough for
multiplicative weights to compound in the right context. Under delayed reward
updates plus corrupted context labels, DTE-EXP3 compounds the wrong evidence.
That is exactly the failure mode UCB reliability gating was designed to avoid.

So H2 should be narrowed:

```text
H2': DTE-EXP3 should outperform DTE-UCB under adversarial or rapidly switching
reward landscapes only when context attribution is reliable enough, or when
EXP3 is itself reliability-gated by delayed reward residuals.
```

The next proper experiment is not more tuning on the current hard benchmark.
It is a dedicated adversarial-switching benchmark that varies two axes
separately:

- reward adversariality / switching speed
- context attribution reliability

That will tell us whether EXP3 is failing because the landscape is not
adversarial enough, or because the update channel is too misattributed.

## Adversarial-Switching Benchmark Result

The dedicated H2 benchmark now separates reward nonstationarity from context
misattribution. The reward surface alternates which module family is favored,
while label noise is swept independently.

Local sweep:

```text
switch_periods = 2, 4, 8, 16
label_noise    = 0.00, 0.14, 0.28
reward_delay   = 4
```

Result:

```text
DTE-EXP3 beats reliability-arbitrated DTE-UCB at all 12 coordinates.
DTE-EXP3 beats external EXP3 at 0 of 12 coordinates.
Best DTE-EXP3 gain over reliability-UCB = +0.029 at switch_period=8, label_noise=0.00.
Worst DTE-EXP3 gain over reliability-UCB = +0.003 at switch_period=4, label_noise=0.28.
```

This refines H2 again:

```text
H2'': DTE-EXP3 is the correct DTE-native policy lane for genuine switching
reward surfaces, but it does not close the gap to policy-owning UCB/EXP3
baselines unless DTE is allowed stronger policy ownership or cleaner
counterfactual attribution.
```

The result is structurally important. The earlier hard-benchmark collapse was
not simply "EXP3 is bad." It was EXP3 plus delayed, corrupted context
attribution. When the adversarial property is isolated, EXP3 improves DTE's
own architecture. But external bandits still win because they own the whole
selection policy, whereas DTE-EXP3 is still an arbitrated lane inside a
topology-and-memory governor.

## Reliability-Gated DTE-EXP3 Result

The remaining H2 gap was whether the same reliability gate used for arbitrated
UCB can protect DTE-EXP3 from attribution poisoning.

Default hard benchmark:

```text
hard dte_local_regret regret                    = 0.1669
hard dte_reliability_arbitrated_ucb regret      = 0.1641
hard dte_exp3 regret                            = 0.2202
hard dte_reliability_arbitrated_exp3 regret     = 0.1926
hard external exp3 regret                       = 0.1547
```

The gate helps substantially in the default corrupted/delayed regime, reducing
DTE-EXP3 regret by about `0.028`. But it does not rescue EXP3 past local-regret
DTE or reliability-arbitrated UCB.

Adversarial-switching sweep:

```text
reliability-gated DTE-EXP3 beats reliability-UCB at all 12 switch/noise coordinates
reliability gating improves ungated DTE-EXP3 at 3 of 12 coordinates
best gate gain  = +0.006 at switch_period=16, label_noise=0.00
worst gate gain = -0.007 at switch_period=8, label_noise=0.00
```

Conclusion:

```text
Reliability gating is a selective poison-control brake for EXP3, not a universal
performance improvement. It is justified when attribution reliability is
suspect, but it withholds useful EXP3 pressure when context is clean enough.
```

This leaves a crisp architectural boundary. DTE can host UCB and EXP3 as
native policy lanes, and EXP3 is the better DTE-native lane under isolated
switching rewards. But closing the gap to external bandits requires either
stronger policy ownership inside DTE or counterfactual/off-policy attribution,
not just another reliability scalar.

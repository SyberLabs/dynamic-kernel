# Neural V2 Router Benchmark Report

## Scope

Shared task stream benchmark for adaptive inference routing. All routers
see the same language-heavy to symbolic-heavy shift per seed.

## Results

| Router | Runs | Reward | Regret | Cost | Language Share | Symbolic Share | Recovery Tick |
|---|---:|---:|---:|---:|---:|---:|---:|
| dte_surprise_only | 5 | 0.327 | 0.178 | 0.204 | 0.312 | 0.315 | 57.6 |
| dte_local_regret | 5 | 0.373 | 0.131 | 0.245 | 0.134 | 0.573 | 56.2 |
| dte_ucb | 5 | 0.371 | 0.134 | 0.242 | 0.145 | 0.556 | 56.2 |
| dte_contextual_ucb | 5 | 0.373 | 0.131 | 0.243 | 0.136 | 0.565 | 55.8 |
| dte_arbitrated_ucb | 5 | 0.376 | 0.128 | 0.240 | 0.139 | 0.555 | 55.6 |
| dte_reliability_arbitrated_ucb | 5 | 0.376 | 0.129 | 0.242 | 0.140 | 0.560 | 55.2 |
| dte_exp3 | 5 | 0.406 | 0.099 | 0.260 | 0.130 | 0.625 | 56.4 |
| static_contextual | 5 | 0.457 | 0.047 | 0.267 | 0.112 | 0.656 | 50.0 |
| epsilon_bandit | 5 | 0.482 | 0.023 | 0.295 | 0.095 | 0.763 | 50.0 |
| ucb | 5 | 0.504 | 0.000 | 0.310 | 0.080 | 0.819 | 50.0 |
| exp3 | 5 | 0.478 | 0.027 | 0.289 | 0.096 | 0.742 | 50.0 |
| oracle | 5 | 0.505 | 0.000 | 0.310 | 0.079 | 0.820 | 50.0 |

## Interpretation

DTE local regret improves over DTE surprise-only on post-shift regret.

The contextual and bandit baselines are intentionally strong. If they
match or beat DTE, the conclusion is not failure; it identifies the
conditions where a simpler router is sufficient. DTE's distinctive
claim is strongest when explicit memory, topology, and stale-route
diagnostics matter.

## Boundary Finding

This benchmark does not support the claim that DTE is the best flat
router when task labels are clean, rewards are immediate, and module
choices are independent. Under those assumptions, a contextual router
or contextual bandit should win, and here it does.

The result does support a narrower Neural V2 claim: local-regret memory
is a real corrective mechanism for stale adaptive routing. The next
validity test should remove the clean-label advantage by adding noisy
task context, delayed rewards, graph-constrained multi-stage routing,
and nonstationary module degradation.

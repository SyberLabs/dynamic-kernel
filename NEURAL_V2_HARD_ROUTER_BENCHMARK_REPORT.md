# Neural V2 Hard Router Benchmark Report

## Scope

Stress benchmark for adaptive inference routing under noisy task context,
corrupted context labels, delayed reward updates, post-shift module
degradation, and verifier-gated symbolic tasks. Rewards are computed
from the true task, while routers observe the noisy task surface.

## Results

| Router | Runs | Reward | Regret | Cost | Language Share | Symbolic Share | Recovery Tick |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard_dte_surprise_only | 5 | 0.257 | 0.326 | 0.202 | 0.311 | 0.311 | 63.2 |
| hard_dte_local_regret | 5 | 0.416 | 0.167 | 0.273 | 0.107 | 0.687 | 62.8 |
| hard_dte_ucb | 5 | 0.415 | 0.168 | 0.272 | 0.113 | 0.681 | 62.4 |
| hard_dte_contextual_ucb | 5 | 0.416 | 0.167 | 0.272 | 0.106 | 0.687 | 62.8 |
| hard_dte_arbitrated_ucb | 5 | 0.418 | 0.165 | 0.271 | 0.112 | 0.678 | 62.2 |
| hard_dte_reliability_arbitrated_ucb | 5 | 0.419 | 0.164 | 0.273 | 0.111 | 0.682 | 62.6 |
| hard_dte_exp3 | 5 | 0.452 | 0.132 | 0.277 | 0.080 | 0.715 | 61.2 |
| hard_dte_reliability_arbitrated_exp3 | 5 | 0.441 | 0.142 | 0.277 | 0.089 | 0.710 | 62.0 |
| hard_static_contextual | 5 | 0.400 | 0.183 | 0.227 | 0.153 | 0.501 | 50.0 |
| hard_epsilon_bandit | 5 | 0.420 | 0.163 | 0.295 | 0.282 | 0.634 | 62.0 |
| hard_ucb | 5 | 0.429 | 0.154 | 0.264 | 0.208 | 0.574 | 68.6 |
| hard_exp3 | 5 | 0.429 | 0.155 | 0.272 | 0.196 | 0.613 | 61.2 |
| hard_oracle | 5 | 0.583 | 0.000 | 0.310 | 0.082 | 0.817 | 50.0 |

## Interpretation

Local-regret DTE changes post-shift regret by `0.159` relative to surprise-only DTE.
Its regret gap versus the noisy contextual router is `-0.016`; its gap versus the delayed bandit is `0.003`.
Its regret gap versus delayed UCB is `0.013`; its gap versus delayed EXP3 is `0.012`.
DTE-native UCB changes regret by `-0.001` relative to local-regret DTE.
DTE-contextual UCB changes regret by `0.000` relative to local-regret DTE.
DTE-arbitrated UCB changes regret by `0.002` relative to local-regret DTE.
DTE-reliability-arbitrated UCB changes regret by `0.003` relative to local-regret DTE.
DTE-EXP3 regret gain is `0.035` relative to local-regret DTE; negative values mean higher regret.
DTE-reliability-arbitrated EXP3 regret gain is `0.025` relative to local-regret DTE.
Reliability gating changes DTE-EXP3 regret by `-0.010`; positive values mean the gate reduced regret.

This is the validity test the clean benchmark asked for. It does not
ask whether DTE beats a perfect contextual router. It asks whether
explicit stale-memory correction becomes more valuable once the
routing surface is corrupted and the reward signal arrives late.

## Institutional Boundary

A DTE advantage is institutionally meaningful only if it survives
strong baselines after cost, delay, noisy context, and topology-gated
utility are included. Otherwise DTE should be positioned as a
diagnostic and governance layer over adaptive routers, not as a
replacement for standard contextual decision algorithms.

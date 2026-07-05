# Neural V2 Adversarial-Switching Benchmark Report

## Scope

Controlled H2 test for DTE-EXP3. The reward surface alternates which
module family is favored while label noise is varied independently.
This separates adversarial reward nonstationarity from attribution
corruption.

## Matrix

| Switch Period | Label Noise | Local DTE | Rel-UCB DTE | DTE-EXP3 | Rel-EXP3 DTE | UCB | EXP3 | EXP3 Gain vs Rel-UCB | Gate Gain | DTE-EXP3 - EXP3 | Winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.00 | 0.165 | 0.161 | 0.146 | 0.149 | 0.041 | 0.066 | 0.015 | -0.002 | 0.081 | ucb |
| 4 | 0.00 | 0.155 | 0.151 | 0.134 | 0.138 | 0.049 | 0.066 | 0.017 | -0.004 | 0.068 | ucb |
| 8 | 0.00 | 0.172 | 0.166 | 0.155 | 0.153 | 0.059 | 0.071 | 0.011 | 0.002 | 0.084 | ucb |
| 16 | 0.00 | 0.174 | 0.168 | 0.147 | 0.148 | 0.183 | 0.064 | 0.022 | -0.001 | 0.083 | exp3 |
| 2 | 0.28 | 0.165 | 0.165 | 0.150 | 0.158 | 0.089 | 0.110 | 0.015 | -0.008 | 0.040 | ucb |
| 4 | 0.28 | 0.154 | 0.152 | 0.141 | 0.143 | 0.163 | 0.110 | 0.011 | -0.003 | 0.030 | exp3 |
| 8 | 0.28 | 0.172 | 0.166 | 0.163 | 0.163 | 0.089 | 0.107 | 0.003 | -0.000 | 0.056 | ucb |
| 16 | 0.28 | 0.172 | 0.172 | 0.151 | 0.153 | 0.186 | 0.106 | 0.021 | -0.002 | 0.045 | exp3 |

## Summary

DTE-EXP3 beats reliability-arbitrated UCB at coordinates: (2, 0.00), (4, 0.00), (8, 0.00), (16, 0.00), (2, 0.28), (4, 0.28), (8, 0.28), (16, 0.28).
DTE-EXP3 beats external EXP3 at coordinates: none.
Reliability-gated DTE-EXP3 beats reliability-arbitrated UCB at coordinates: (2, 0.00), (4, 0.00), (8, 0.00), (16, 0.00), (2, 0.28), (4, 0.28), (8, 0.28), (16, 0.28).
Reliability gating improves DTE-EXP3 at coordinates: (8, 0.00).
Best DTE-EXP3 coordinate: {'switch_period': 16, 'label_noise': 0.0, 'gain': 0.021625633210931944}.
Worst DTE-EXP3 coordinate: {'switch_period': 8, 'label_noise': 0.28, 'gain': 0.002856455832948218}.
Best EXP3 reliability-gate coordinate: {'switch_period': 8, 'label_noise': 0.0, 'gain': 0.0022970400875232666}.
Worst EXP3 reliability-gate coordinate: {'switch_period': 2, 'label_noise': 0.28, 'gain': -0.007573317904472965}.
Winner counts: {'ucb': 5, 'exp3': 3}.

## Interpretation

This benchmark is not a general hard-routing score. It is a diagnostic
for the H2 mechanism. DTE-EXP3 winning inside the DTE family means
multiplicative weights are useful for genuine adversarial
nonstationarity. Reliability gating then answers a narrower
question: whether attribution filtering protects the EXP3 lane from
poisoned context updates. In this sweep, the gate helps selectively
rather than universally, so it should be treated as a brake, not a
new default policy.

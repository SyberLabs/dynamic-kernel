# Neural V2 Adversarial-Switching Benchmark Report

## Scope

Controlled H2 test for DTE-EXP3. The reward surface alternates which
module family is favored while label noise is varied independently.
This separates adversarial reward nonstationarity from attribution
corruption.

## Matrix

| Switch Period | Label Noise | Local DTE | Rel-UCB DTE | DTE-EXP3 | Rel-EXP3 DTE | UCB | EXP3 | EXP3 Gain vs Rel-UCB | Gate Gain | DTE-EXP3 - EXP3 | Winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.00 | 0.165 | 0.159 | 0.138 | 0.139 | 0.041 | 0.068 | 0.021 | -0.001 | 0.070 | ucb |
| 4 | 0.00 | 0.161 | 0.155 | 0.133 | 0.134 | 0.050 | 0.067 | 0.022 | -0.001 | 0.066 | ucb |
| 8 | 0.00 | 0.188 | 0.175 | 0.146 | 0.153 | 0.049 | 0.069 | 0.029 | -0.007 | 0.076 | ucb |
| 16 | 0.00 | 0.180 | 0.169 | 0.149 | 0.143 | 0.160 | 0.091 | 0.020 | 0.006 | 0.058 | exp3 |
| 2 | 0.14 | 0.164 | 0.160 | 0.152 | 0.154 | 0.067 | 0.095 | 0.008 | -0.001 | 0.057 | ucb |
| 4 | 0.14 | 0.157 | 0.151 | 0.145 | 0.147 | 0.139 | 0.095 | 0.007 | -0.002 | 0.050 | exp3 |
| 8 | 0.14 | 0.186 | 0.174 | 0.160 | 0.166 | 0.073 | 0.096 | 0.014 | -0.006 | 0.064 | ucb |
| 16 | 0.14 | 0.176 | 0.167 | 0.162 | 0.162 | 0.180 | 0.105 | 0.005 | -0.000 | 0.057 | exp3 |
| 2 | 0.28 | 0.164 | 0.160 | 0.157 | 0.157 | 0.087 | 0.112 | 0.003 | 0.001 | 0.045 | ucb |
| 4 | 0.28 | 0.161 | 0.154 | 0.151 | 0.150 | 0.170 | 0.109 | 0.003 | 0.001 | 0.042 | exp3 |
| 8 | 0.28 | 0.189 | 0.178 | 0.167 | 0.170 | 0.087 | 0.112 | 0.011 | -0.003 | 0.055 | ucb |
| 16 | 0.28 | 0.177 | 0.170 | 0.166 | 0.166 | 0.159 | 0.127 | 0.004 | -0.001 | 0.039 | exp3 |

## Summary

DTE-EXP3 beats reliability-arbitrated UCB at coordinates: (2, 0.00), (4, 0.00), (8, 0.00), (16, 0.00), (2, 0.14), (4, 0.14), (8, 0.14), (16, 0.14), (2, 0.28), (4, 0.28), (8, 0.28), (16, 0.28).
DTE-EXP3 beats external EXP3 at coordinates: none.
Reliability-gated DTE-EXP3 beats reliability-arbitrated UCB at coordinates: (2, 0.00), (4, 0.00), (8, 0.00), (16, 0.00), (2, 0.14), (4, 0.14), (8, 0.14), (16, 0.14), (2, 0.28), (4, 0.28), (8, 0.28), (16, 0.28).
Reliability gating improves DTE-EXP3 at coordinates: (16, 0.00), (2, 0.28), (4, 0.28).
Best DTE-EXP3 coordinate: {'switch_period': 8, 'label_noise': 0.0, 'gain': 0.029028017866811057}.
Worst DTE-EXP3 coordinate: {'switch_period': 4, 'label_noise': 0.28, 'gain': 0.0030267775781816497}.
Best EXP3 reliability-gate coordinate: {'switch_period': 16, 'label_noise': 0.0, 'gain': 0.005631640296820367}.
Worst EXP3 reliability-gate coordinate: {'switch_period': 8, 'label_noise': 0.0, 'gain': -0.00681711294100551}.
Winner counts: {'ucb': 7, 'exp3': 5}.

## Interpretation

This benchmark is not a general hard-routing score. It is a diagnostic
for the H2 mechanism. DTE-EXP3 winning inside the DTE family means
multiplicative weights are useful for genuine adversarial
nonstationarity. Reliability gating then answers a narrower
question: whether attribution filtering protects the EXP3 lane from
poisoned context updates. In this sweep, the gate helps selectively
rather than universally, so it should be treated as a brake, not a
new default policy.

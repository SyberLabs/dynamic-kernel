# Neural V2 Parameter Frontier Report

## Scope

One-axis frontier sweep for the hard Neural V2 router benchmark. Each
axis varies around the same baseline while all routers see the same
seeded task streams per scenario.

Negative DTE-minus-baseline regret means DTE local-regret outperformed
that baseline. Positive values mean the baseline had lower regret.

## context_noise

| Value | DTE Regret | DTE-UCB Gain | Ctx-UCB Gain | Arb-UCB Gain | Rel-Arb Gain | DTE-EXP3 Gain | EXP3-vs-Rel | Gain vs Surprise | DTE - Contextual | DTE - Bandit | DTE - UCB | DTE - EXP3 | DTE Symbolic |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.168 | -0.002 | 0.003 | 0.005 | 0.004 | 0.040 | 0.035 | 0.159 | 0.043 | 0.013 | 0.034 | 0.027 | 0.680 |
| 0.1 | 0.169 | -0.002 | 0.003 | 0.006 | 0.005 | 0.040 | 0.036 | 0.157 | 0.027 | 0.015 | 0.035 | 0.031 | 0.677 |
| 0.22 | 0.170 | -0.001 | 0.003 | 0.003 | 0.002 | 0.034 | 0.032 | 0.156 | -0.013 | -0.001 | 0.017 | 0.013 | 0.677 |
| 0.34 | 0.169 | -0.002 | 0.003 | -0.006 | -0.004 | 0.025 | 0.029 | 0.158 | -0.055 | -0.039 | -0.071 | -0.014 | 0.681 |
| 0.46 | 0.168 | -0.002 | 0.002 | -0.010 | -0.006 | 0.025 | 0.031 | 0.159 | -0.086 | -0.055 | -0.201 | -0.019 | 0.684 |

DTE beats contextual at values: [0.22, 0.34, 0.46]
DTE beats bandit at values: [0.22, 0.34, 0.46]
DTE beats UCB at values: [0.34, 0.46]
DTE beats EXP3 at values: [0.34, 0.46]
DTE-native UCB improves DTE at values: []
DTE-contextual UCB improves DTE at values: [0.0, 0.1, 0.22, 0.34, 0.46]
DTE-arbitrated UCB improves DTE at values: [0.0, 0.1, 0.22]
DTE-reliability-arbitrated UCB improves DTE at values: [0.0, 0.1, 0.22]
Reliability gating improves arbitration at values: [0.34, 0.46]
DTE-EXP3 improves DTE at values: [0.0, 0.1, 0.22, 0.34, 0.46]
DTE-EXP3 improves reliability-arbitrated UCB at values: [0.0, 0.1, 0.22, 0.34, 0.46]
Best contextual gap: -0.086 at 0.46.
Best bandit gap: -0.055 at 0.46.
Best UCB gap: -0.201 at 0.46.
Best EXP3 gap: -0.019 at 0.46.
Best DTE-native UCB gain: -0.001 at 0.22.
Best DTE-contextual UCB gain: 0.003 at 0.34.
Best DTE-arbitrated UCB gain: 0.006 at 0.1.
Best arbitration gain over additive contextual UCB: 0.004 at 0.1.
Best DTE-reliability-arbitrated UCB gain: 0.005 at 0.1.
Best reliability-gating gain over arbitration: 0.004 at 0.46.
Best DTE-EXP3 gain: 0.040 at 0.1.
Best DTE-EXP3 gain over reliability-arbitrated UCB: 0.036 at 0.1.

## label_noise

| Value | DTE Regret | DTE-UCB Gain | Ctx-UCB Gain | Arb-UCB Gain | Rel-Arb Gain | DTE-EXP3 Gain | EXP3-vs-Rel | Gain vs Surprise | DTE - Contextual | DTE - Bandit | DTE - UCB | DTE - EXP3 | DTE Symbolic |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.170 | -0.002 | 0.002 | 0.009 | 0.008 | 0.036 | 0.027 | 0.155 | -0.015 | 0.102 | 0.140 | 0.106 | 0.679 |
| 0.14 | 0.170 | -0.003 | 0.002 | 0.004 | 0.005 | 0.035 | 0.030 | 0.157 | -0.012 | 0.047 | 0.076 | 0.057 | 0.680 |
| 0.28 | 0.170 | -0.001 | 0.003 | 0.003 | 0.002 | 0.034 | 0.032 | 0.156 | -0.013 | -0.001 | 0.017 | 0.013 | 0.677 |
| 0.42 | 0.170 | -0.001 | 0.002 | -0.003 | -0.002 | 0.032 | 0.034 | 0.157 | -0.015 | -0.029 | -0.073 | -0.030 | 0.678 |
| 0.56 | 0.170 | -0.004 | 0.002 | -0.008 | -0.007 | 0.032 | 0.039 | 0.156 | -0.014 | -0.091 | -0.253 | -0.073 | 0.680 |

DTE beats contextual at values: [0.0, 0.14, 0.28, 0.42, 0.56]
DTE beats bandit at values: [0.28, 0.42, 0.56]
DTE beats UCB at values: [0.42, 0.56]
DTE beats EXP3 at values: [0.42, 0.56]
DTE-native UCB improves DTE at values: []
DTE-contextual UCB improves DTE at values: [0.0, 0.14, 0.28, 0.42, 0.56]
DTE-arbitrated UCB improves DTE at values: [0.0, 0.14, 0.28]
DTE-reliability-arbitrated UCB improves DTE at values: [0.0, 0.14, 0.28]
Reliability gating improves arbitration at values: [0.14, 0.42, 0.56]
DTE-EXP3 improves DTE at values: [0.0, 0.14, 0.28, 0.42, 0.56]
DTE-EXP3 improves reliability-arbitrated UCB at values: [0.0, 0.14, 0.28, 0.42, 0.56]
Best contextual gap: -0.015 at 0.0.
Best bandit gap: -0.091 at 0.56.
Best UCB gap: -0.253 at 0.56.
Best EXP3 gap: -0.073 at 0.56.
Best DTE-native UCB gain: -0.001 at 0.42.
Best DTE-contextual UCB gain: 0.003 at 0.28.
Best DTE-arbitrated UCB gain: 0.009 at 0.0.
Best arbitration gain over additive contextual UCB: 0.007 at 0.0.
Best DTE-reliability-arbitrated UCB gain: 0.008 at 0.0.
Best reliability-gating gain over arbitration: 0.001 at 0.14.
Best DTE-EXP3 gain: 0.036 at 0.0.
Best DTE-EXP3 gain over reliability-arbitrated UCB: 0.039 at 0.56.

## reward_delay

| Value | DTE Regret | DTE-UCB Gain | Ctx-UCB Gain | Arb-UCB Gain | Rel-Arb Gain | DTE-EXP3 Gain | EXP3-vs-Rel | Gain vs Surprise | DTE - Contextual | DTE - Bandit | DTE - UCB | DTE - EXP3 | DTE Symbolic |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.167 | -0.002 | 0.004 | 0.004 | 0.004 | 0.029 | 0.025 | 0.159 | -0.016 | 0.038 | 0.118 | 0.020 | 0.686 |
| 4 | 0.168 | -0.002 | 0.003 | 0.004 | 0.003 | 0.031 | 0.028 | 0.159 | -0.016 | 0.010 | 0.016 | 0.017 | 0.685 |
| 8 | 0.170 | -0.001 | 0.003 | 0.003 | 0.002 | 0.034 | 0.032 | 0.156 | -0.013 | -0.001 | 0.017 | 0.013 | 0.677 |
| 12 | 0.180 | -0.002 | 0.002 | 0.002 | 0.001 | 0.034 | 0.033 | 0.147 | -0.004 | -0.001 | 0.000 | 0.010 | 0.652 |
| 16 | 0.193 | -0.003 | -0.000 | 0.002 | 0.001 | 0.033 | 0.031 | 0.135 | 0.010 | 0.012 | -0.151 | 0.023 | 0.618 |

DTE beats contextual at values: [0, 4, 8, 12]
DTE beats bandit at values: [8, 12]
DTE beats UCB at values: [16]
DTE beats EXP3 at values: []
DTE-native UCB improves DTE at values: []
DTE-contextual UCB improves DTE at values: [0, 4, 8, 12]
DTE-arbitrated UCB improves DTE at values: [0, 4, 8, 12, 16]
DTE-reliability-arbitrated UCB improves DTE at values: [0, 4, 8, 12, 16]
Reliability gating improves arbitration at values: []
DTE-EXP3 improves DTE at values: [0, 4, 8, 12, 16]
DTE-EXP3 improves reliability-arbitrated UCB at values: [0, 4, 8, 12, 16]
Best contextual gap: -0.016 at 0.
Best bandit gap: -0.001 at 12.
Best UCB gap: -0.151 at 16.
Best EXP3 gap: 0.010 at 12.
Best DTE-native UCB gain: -0.001 at 8.
Best DTE-contextual UCB gain: 0.004 at 0.
Best DTE-arbitrated UCB gain: 0.004 at 0.
Best arbitration gain over additive contextual UCB: 0.002 at 16.
Best DTE-reliability-arbitrated UCB gain: 0.004 at 0.
Best reliability-gating gain over arbitration: -0.000 at 0.
Best DTE-EXP3 gain: 0.034 at 12.
Best DTE-EXP3 gain over reliability-arbitrated UCB: 0.033 at 12.

## language_degradation

| Value | DTE Regret | DTE-UCB Gain | Ctx-UCB Gain | Arb-UCB Gain | Rel-Arb Gain | DTE-EXP3 Gain | EXP3-vs-Rel | Gain vs Surprise | DTE - Contextual | DTE - Bandit | DTE - UCB | DTE - EXP3 | DTE Symbolic |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.172 | -0.002 | 0.002 | 0.002 | 0.001 | 0.034 | 0.032 | 0.169 | -0.021 | -0.011 | 0.012 | 0.002 | 0.681 |
| 0.45 | 0.171 | -0.002 | 0.003 | 0.002 | 0.001 | 0.034 | 0.032 | 0.166 | -0.018 | -0.008 | 0.014 | 0.007 | 0.680 |
| 0.62 | 0.170 | -0.001 | 0.003 | 0.003 | 0.002 | 0.034 | 0.032 | 0.156 | -0.013 | -0.001 | 0.017 | 0.013 | 0.677 |
| 0.82 | 0.174 | -0.002 | 0.003 | 0.004 | 0.002 | 0.033 | 0.031 | 0.145 | -0.008 | 0.009 | 0.024 | 0.023 | 0.674 |
| 1.0 | 0.177 | -0.001 | 0.003 | 0.004 | 0.003 | 0.034 | 0.032 | 0.136 | -0.004 | 0.016 | 0.031 | 0.025 | 0.670 |

DTE beats contextual at values: [0.3, 0.45, 0.62, 0.82, 1.0]
DTE beats bandit at values: [0.3, 0.45, 0.62]
DTE beats UCB at values: []
DTE beats EXP3 at values: []
DTE-native UCB improves DTE at values: []
DTE-contextual UCB improves DTE at values: [0.3, 0.45, 0.62, 0.82, 1.0]
DTE-arbitrated UCB improves DTE at values: [0.3, 0.45, 0.62, 0.82, 1.0]
DTE-reliability-arbitrated UCB improves DTE at values: [0.3, 0.45, 0.62, 0.82, 1.0]
Reliability gating improves arbitration at values: []
DTE-EXP3 improves DTE at values: [0.3, 0.45, 0.62, 0.82, 1.0]
DTE-EXP3 improves reliability-arbitrated UCB at values: [0.3, 0.45, 0.62, 0.82, 1.0]
Best contextual gap: -0.021 at 0.3.
Best bandit gap: -0.011 at 0.3.
Best UCB gap: 0.012 at 0.3.
Best EXP3 gap: 0.002 at 0.3.
Best DTE-native UCB gain: -0.001 at 1.0.
Best DTE-contextual UCB gain: 0.003 at 1.0.
Best DTE-arbitrated UCB gain: 0.004 at 1.0.
Best arbitration gain over additive contextual UCB: 0.001 at 1.0.
Best DTE-reliability-arbitrated UCB gain: 0.003 at 1.0.
Best reliability-gating gain over arbitration: -0.001 at 0.3.
Best DTE-EXP3 gain: 0.034 at 1.0.
Best DTE-EXP3 gain over reliability-arbitrated UCB: 0.032 at 0.3.

## verifier_bonus

| Value | DTE Regret | DTE-UCB Gain | Ctx-UCB Gain | Arb-UCB Gain | Rel-Arb Gain | DTE-EXP3 Gain | EXP3-vs-Rel | Gain vs Surprise | DTE - Contextual | DTE - Bandit | DTE - UCB | DTE - EXP3 | DTE Symbolic |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.143 | -0.002 | 0.003 | -0.002 | -0.003 | 0.028 | 0.030 | 0.115 | 0.000 | -0.008 | 0.008 | 0.009 | 0.661 |
| 0.06 | 0.156 | -0.002 | 0.003 | -0.001 | -0.001 | 0.030 | 0.031 | 0.136 | -0.007 | -0.004 | 0.013 | 0.008 | 0.671 |
| 0.12 | 0.170 | -0.001 | 0.003 | 0.003 | 0.002 | 0.034 | 0.032 | 0.156 | -0.013 | -0.001 | 0.017 | 0.013 | 0.677 |
| 0.18 | 0.184 | -0.001 | 0.004 | 0.006 | 0.005 | 0.037 | 0.032 | 0.177 | -0.020 | 0.005 | 0.017 | 0.019 | 0.682 |
| 0.24 | 0.198 | -0.001 | 0.004 | 0.010 | 0.008 | 0.039 | 0.030 | 0.197 | -0.026 | 0.010 | 0.015 | 0.024 | 0.686 |

DTE beats contextual at values: [0.06, 0.12, 0.18, 0.24]
DTE beats bandit at values: [0.0, 0.06, 0.12]
DTE beats UCB at values: []
DTE beats EXP3 at values: []
DTE-native UCB improves DTE at values: []
DTE-contextual UCB improves DTE at values: [0.0, 0.06, 0.12, 0.18, 0.24]
DTE-arbitrated UCB improves DTE at values: [0.12, 0.18, 0.24]
DTE-reliability-arbitrated UCB improves DTE at values: [0.12, 0.18, 0.24]
Reliability gating improves arbitration at values: []
DTE-EXP3 improves DTE at values: [0.0, 0.06, 0.12, 0.18, 0.24]
DTE-EXP3 improves reliability-arbitrated UCB at values: [0.0, 0.06, 0.12, 0.18, 0.24]
Best contextual gap: -0.026 at 0.24.
Best bandit gap: -0.008 at 0.0.
Best UCB gap: 0.008 at 0.0.
Best EXP3 gap: 0.008 at 0.06.
Best DTE-native UCB gain: -0.001 at 0.18.
Best DTE-contextual UCB gain: 0.004 at 0.18.
Best DTE-arbitrated UCB gain: 0.010 at 0.24.
Best arbitration gain over additive contextual UCB: 0.006 at 0.24.
Best DTE-reliability-arbitrated UCB gain: 0.008 at 0.24.
Best reliability-gating gain over arbitration: -0.000 at 0.06.
Best DTE-EXP3 gain: 0.039 at 0.24.
Best DTE-EXP3 gain over reliability-arbitrated UCB: 0.032 at 0.12.

## Interpretation

This frontier is the first map of Neural V2's institutional validity.
The meaningful claim is not universal dominance. The meaningful claim
is boundary-sensitive: DTE local-regret has value when stale routing
memory is a first-class failure mode and the observed decision surface
is imperfect.

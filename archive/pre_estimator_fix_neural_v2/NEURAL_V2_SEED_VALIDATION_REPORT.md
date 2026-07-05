# Neural V2 Seed Validation Report

## Scope

Paired-seed validation for the Neural V2 controlled benchmark. Regret
deltas are paired against local-regret DTE within the same task stream
and seed.

## Configuration

- Seeds: `(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)`
- Ticks: `90`
- Batch size: `100`
- Hard label noise: `0.28`
- Hard reward delay: `8`
- Adversarial switch cell: period `8`, label noise `0.0`

## Results

| Regime | Router | Runs | Regret | CI95 | Reward | Delta vs Local | Delta CI95 |
|---|---|---:|---:|---:|---:|---:|---:|
| clean | dte_exp3 | 30 | 0.0991 | 0.0010 | 0.4053 | 0.0323 | 0.0012 |
| clean | dte_local_regret | 30 | 0.1314 | 0.0008 | 0.3731 | 0.0000 | 0.0000 |
| clean | dte_reliability_arbitrated_ucb | 30 | 0.1275 | 0.0009 | 0.3769 | 0.0039 | 0.0011 |
| clean | exp3 | 30 | 0.0268 | 0.0005 | 0.4777 | 0.1046 | 0.0009 |
| clean | ucb | 30 | 0.0010 | 0.0000 | 0.5034 | 0.1304 | 0.0008 |
| hard | hard_dte_exp3 | 30 | 0.1876 | 0.0026 | 0.4006 | -0.0167 | 0.0026 |
| hard | hard_dte_local_regret | 30 | 0.1709 | 0.0012 | 0.4173 | 0.0000 | 0.0000 |
| hard | hard_dte_reliability_arbitrated_exp3 | 30 | 0.1784 | 0.0019 | 0.4098 | -0.0075 | 0.0025 |
| hard | hard_dte_reliability_arbitrated_ucb | 30 | 0.1681 | 0.0015 | 0.4201 | 0.0028 | 0.0020 |
| hard | hard_exp3 | 30 | 0.1445 | 0.0029 | 0.4437 | 0.0264 | 0.0033 |
| hard | hard_ucb | 30 | 0.1486 | 0.0058 | 0.4396 | 0.0223 | 0.0059 |
| adversarial_switch | hard_dte_exp3 | 30 | 0.1503 | 0.0018 | 0.4125 | 0.0382 | 0.0020 |
| adversarial_switch | hard_dte_local_regret | 30 | 0.1886 | 0.0016 | 0.3743 | 0.0000 | 0.0000 |
| adversarial_switch | hard_dte_reliability_arbitrated_exp3 | 30 | 0.1581 | 0.0015 | 0.4048 | 0.0305 | 0.0017 |
| adversarial_switch | hard_dte_reliability_arbitrated_ucb | 30 | 0.1831 | 0.0015 | 0.3798 | 0.0055 | 0.0017 |
| adversarial_switch | hard_exp3 | 30 | 0.0645 | 0.0014 | 0.4984 | 0.1241 | 0.0021 |
| adversarial_switch | hard_ucb | 30 | 0.0448 | 0.0004 | 0.5181 | 0.1438 | 0.0016 |

## Reading

Positive paired deltas mean the router reduced regret relative to
local-regret DTE on the same seed. The table is designed to make the
policy-arbitration boundary visible: DTE-native lanes should improve
inside the DTE family in memory-sensitive regimes, while external
policy owners may still dominate when the task behaves like a clean
contextual bandit.

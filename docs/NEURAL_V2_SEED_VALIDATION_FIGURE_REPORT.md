# Neural V2 Seed Validation Figure Report

## Artifact

- SVG: `figures/neural_v2_seed_validation_delta.svg`

## Reading

The figure plots paired regret reduction relative to local-regret DTE.
Positive values improve on local-regret DTE; negative values are worse.

Best paired improvement: `adversarial_switch` / `hard_ucb` = 0.1438 +/- 0.0016.
Worst paired change: `hard` / `hard_dte_reliability_arbitrated_ucb` = 0.0028 +/- 0.0020.

## Paper Use

Use this as the Neural V2 regime-boundary figure. It visually supports
the paper claim that DTE-native policy lanes are regime-sensitive:
DTE-EXP3 is useful under adversarial switching, reliability-UCB is the
safer hard-regime default, and external policy owners still dominate
when the task reduces to contextual bandit selection.

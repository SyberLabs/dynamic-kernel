# Pre-estimator-fix Neural V2 artifacts (frozen)

These are the Neural V2 benchmark outputs generated BEFORE the EXP3 gain
estimator fix of 2026-07-05, retained for the paper's estimator-bias case
study. Do not regenerate into this directory.

The lane they exercised computed `gain = reward * (traffic / row_mass)` —
reward multiplied by realized selection frequency — instead of an
importance-weighted estimator. That biased update throttles high-reward,
low-traffic edges by their own unpopularity (multiplicative rich-get-richer
in log-weight space) and reproduced deadly familiarity inside the learning
lane itself. In a two-route switching scenario the pre-fix lane never
re-preferred a revived sparse route within 2,800 post-switch steps; the
EXP3-IX estimator recovers in about 2 steps.

Headline consequence recorded here: `hard_dte_exp3` post-shift regret 0.220
(second-worst lane) pre-fix vs 0.132 (best non-oracle lane) post-fix, and the
30-paired-seed hard-regime delta vs local-regret DTE flipping from -0.0167
(worsens) to +0.0334 (largest DTE-native improvement). The earlier
"DTE-EXP3 worsens under corrupted delayed attribution" conclusion was
estimator bias, not attribution fragility.

Current (post-fix) artifacts live in the repository root under the same
file names. The corrected estimator and its regression tests:
`kernel.py::_exp3_weight_update`, `tests/test_review_fixes.py`.

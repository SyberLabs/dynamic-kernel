# DTE Paper Internal Review

## Scope

Hard internal review after generating the Neural V2 and semiconductor paper
figures. Standard: every major claim must point to a theorem, table, figure, or
explicit limitation.

## Verdict

The manuscript is now structurally credible but not submission-ready. Its core
claims are mostly anchored. The remaining risk is not lack of experiments; it
is presentation discipline. The paper must avoid drifting from demonstrated
mechanism claims into policy, forecasting, or universal optimizer language.

## Claim-Evidence Audit

| Claim | Current Status | Required Anchor | Review |
|---|---|---|---|
| DTE is an adaptive routing kernel on feature-decorated directed graphs | Supported | Section 2 equations, `kernel.py` | Keep. This is definitional. |
| The augmented process is Markov while position alone is generally non-stationary | Supported | Proposition 0, proposition-to-evidence table, layered-memory figure | Keep. The manuscript now has the needed formal statement. |
| Singleton-outdegree interventions cannot reroute finite softmax flow | Supported | Proposition 1 | Keep. This is the cleanest theorem. |
| Downstream serial penalties are inert in the semiconductor topology | Supported | `figures/semiconductor_choice_point_relocation.svg` | Keep as case-study result, not universal tariff claim. |
| Adding reconsideration exits activates downstream penalties | Supported | `figures/semiconductor_topology_surgery.svg` | Keep. This is a strong mechanism figure. |
| Feedback reshapes robustness non-monotonically | Supported | `figures/semiconductor_feedback_continuum.svg` | Keep. Avoid saying feedback is good or bad in general. |
| Static expected-flow models overstate success | Supported | `figures/semiconductor_model_benchmark.svg` | Keep, but phrase as "in this benchmark" not general indictment. |
| High route share can coexist with failed completed production | Supported | `figures/semiconductor_topology_nulls.svg` | Keep. This is central and well supported. |
| Procurement pull is effective in the specified semiconductor topology | Supported | feasibility-preference table, falsification report | Keep with "specified topology" qualifier. |
| Procurement pull is topology invariant | Rejected | topology nulls | Good negative result. |
| DTE universally beats UCB/EXP3 | Rejected | Neural V2 seed validation | Good negative result. Mention directly. |
| DTE-EXP3 is useful under adversarial switching | Supported regime result | `figures/neural_v2_seed_validation_delta.svg` | Keep, with regime qualifier. |
| Traffic-weighted DTE-EXP3 gains are robust to corrupted delayed attribution | Rejected | hard Neural V2 validation, archived pre-fix artifacts | Failure traced to estimator bias, not attribution fragility. |
| DTE-EXP3 with EXP3-IX improves local-regret DTE under corrupted attribution | Supported at 30 paired seeds | corrected hard Neural V2 validation | Keep with boundary: external owners still dominate clean/switching. |
| DTE forecasts real semiconductor outcomes | Not supported | limitation section | Must remain explicitly not claimed. |
| DTE replaces domain-specific supply-chain tools | Not supported | limitation section | Must remain explicitly not claimed. |

## Major Findings

### F1 [Resolved]: Markov closure needs a formal statement in the manuscript

The theory note states Markov closure clearly, and the manuscript now includes
`Proposition 0: Markov closure under layered memory`:

```text
If Z_t = (X_t, A_t, M_s(t), M_p(t), B_t, R_t), and all update maps use only
Z_t plus fresh randomness, then {Z_t} is Markov. Projections such as X_t alone
need not be Markov.
```

This matters because the paper describes DTE as a non-stationary Markov
process. Reviewers will expect a precise state-space statement. This item is
closed unless the proposition is later moved to an appendix.

### F2 [Resolved]: Figure references are not yet numbered into the prose

The figures exist, are listed in Appendix B, and are now referenced in the
relevant body sections. This item is closed pending later renumbering during
LaTeX conversion.

### F3 [Significant]: Related work remains the largest submission blocker

The paper currently has a TODO instead of a literature section. This is the
biggest academic blocker. The minimum viable related-work spine must cover:

```text
non-stationary Markov chains
entropy/logit routing
adaptive routing and contextual bandits
agent-based supply-chain simulation
production networks / network interdiction
cybernetic control / adaptive systems
```

The paper cannot be submitted without verified citations.

### F4 [Resolved]: The DTE mechanism schematic is still missing

The DTE mechanism schematic now exists as
`figures/dte_mechanism_schematic.svg`.

### F5 [Resolved]: Feasibility-allocation-adaptation concept figure is still missing

The feasibility-allocation-adaptation schematic now exists as
`figures/feasibility_allocation_adaptation.svg`.

## Submission Readiness

Current level:

```text
Workshop / internal technical report: close
Applied network science / complex systems submission: plausible after citations and figure references
Top-tier CS/ML submission: not yet, unless theory is strengthened further
```

## Required Next Actions

1. Write related work with verified citations.
2. Run one final claim-evidence pass after citations and figure references.

## Bottom Line

The project is now strong because it rejects overclaims. The manuscript should
lean into that. The paper's core contribution is not universal superiority; it
is a rigorous way to expose when topology, memory, feasibility, and delayed
feedback change the answer that simpler routing or bandit models would give.

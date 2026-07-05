# DTE Paper Workplan

## Working Title

**Choice Points, Feasibility, and Adaptation in Dynamic Circulation Networks: A Dynamic Topology Engine for Intervention Analysis**

## Advisor-Approved V1 Path

The paper now follows a five-stage path:

1. Freeze the V1 kernel and define the theoretical object.
2. State a small set of proposition-level claims.
3. Use Neural V2 as the controlled adaptive-routing validation bench.
4. Use semiconductor onshoring as the applied feasibility case study.
5. Write the paper around boundaries, not universal dominance.

## Central Claim

Adaptive network interventions must be evaluated as coupled feasibility, allocation, and adaptation transitions. An intervention cannot reroute flow when it acts on a transition with no genuine alternative, and route share alone is not evidence of an institutionally viable transition.

## Contribution Hierarchy

### General Contributions

1. **Layered-memory Markov closure**
   - DTE is Markov only on the augmented state containing position,
     telemetry, structural memory, preference memory, delayed attribution
     buffers, and ecology variables.
   - Supported by `DTE_V1_THEORY.md` and kernel memory-law tests.

2. **Choice-point principle**
   - Singleton-outdegree edge interventions cannot change next-state routing probability under row-wise softmax.
   - Supported by Proposition 1, equal-cost relocation, and topology surgery.

3. **Joint intervention model**
   - DTE combines feature-aligned routing, topology-local control, agent telemetry feedback, and operational feasibility.

4. **Policy-arbitration boundary**
   - DTE-native UCB/EXP3 lanes improve DTE in specific regimes but do not
     erase the advantage of external policy owners on clean contextual bandit
     tasks.
   - Supported by Neural V2 clean, hard, frontier, and adversarial-switching
     benchmarks.

5. **Route-share versus viable-transition distinction**
   - Majority route share can coexist with inadequate completed production.
   - Supported by BOM gates, static benchmark failures, and randomized topology nulls.

### Controlled-Benchmark Contributions

1. Neural V2 demonstrates that local-regret memory improves stale adaptive
   routing relative to surprise-only DTE.
2. DTE-EXP3 with the EXP3-IX gain estimator is the strongest DTE-native lane
   under corrupted/delayed Neural V2 attribution in the corrected 30-seed run.
3. DTE-EXP3 is valid as a DTE-native lane under isolated adversarial switching
   reward surfaces.
4. Reliability-gated DTE-EXP3 no longer earns its complexity as the default
   hard-regime lane after the estimator correction.
5. External UCB/EXP3 remain stronger when the task reduces to clean policy
   ownership rather than topology-memory governance.

### Case-Study Contributions

1. Physical resource scaling removes semiconductor production infeasibility without guaranteeing majority domestic capture.
2. Upstream procurement preference can produce robust majority capture in the specified semiconductor topology.
3. Import friction applied inside the serial offshore corridor is dynamically inert.
4. Feedback reshapes robustness non-monotonically.

### Explicitly Rejected Claims

1. Procurement pull is a topology-invariant solution.
2. Feedback universally amplifies a desired intervention.
3. High domestic route share implies a viable production transition.
4. DTE replaces domain-specific supply-chain optimization or forecasting.

## Figure Plan

| Figure | Purpose | Source | Remaining Work |
|---|---|---|---|
| 1. DTE mechanism schematic | Explain position, telemetry, alignment, transition, and feedback | `figures/dte_mechanism_schematic.svg` | Generated SVG figure |
| 2. Layered-memory state schematic | Show `X_t`, `A_t`, `M_s`, `M_p`, buffers, ecology | `figures/dte_layered_memory_schematic.svg` | Generated SVG figure |
| 3. Neural V2 regime boundary | Clean/hard/adversarial-switch comparison | `neural_v2_seed_validation_output.json` | Generated table in `NEURAL_V2_SEED_VALIDATION_REPORT.md` |
| 4. DTE policy-lane boundary | Reliability-UCB versus EXP3 variants | `figures/neural_v2_seed_validation_delta.svg` | Generated SVG figure |
| 5. Semiconductor topology | Show route-choice points, serial offshore corridor, U.S. production gates | `figures/semiconductor_topology_schematic.svg` | Generated SVG figure |
| 6. Feasibility-allocation-adaptation concept | Distinguish capability, route capture, and feedback | `figures/feasibility_allocation_adaptation.svg` | Generated SVG figure |
| 7. Choice-point relocation | Equal-cost budget versus mean onshore share by location | `figures/semiconductor_choice_point_relocation.svg` | Generated SVG figure |
| 8. Topology surgery | Serial corridor versus reconsideration exits | `figures/semiconductor_topology_surgery.svg` | Generated SVG figure |
| 9. Model benchmark | Robust cells across static, frozen-agent, and DTE models | `figures/semiconductor_model_benchmark.svg` | Generated SVG figure |
| 10. Topology nulls | Onshore share versus lot completion, colored by robustness | `figures/semiconductor_topology_nulls.svg` | Generated SVG figure |

## Table Plan

| Table | Status | Source |
|---|---|---|
| DTE variables and intervention channels | Drafted in manuscript | `kernel.py` |
| V1 proposition-to-evidence map | Needed | `DTE_V1_THEORY.md`, this workplan |
| Neural V2 benchmark summary | Drafted | `NEURAL_V2_SEED_VALIDATION_REPORT.md` |
| Neural V2 policy-lane boundary | Drafted | `NEURAL_V2_SEED_VALIDATION_FIGURE_REPORT.md` |
| Viable-transition classification thresholds | Drafted in manuscript | `semiconductor_onshoring_frontier.py` |
| Feasibility-preference high-load result | Drafted in manuscript | `SEMICONDUCTOR_ONSHORING_FEASIBILITY_PREFERENCE_REPORT.md` |
| Classification robustness boundary | Drafted in manuscript | `SEMICONDUCTOR_ONSHORING_CLASSIFICATION_ROBUSTNESS_REPORT.md` |
| Benchmark comparison | Drafted in manuscript | `SEMICONDUCTOR_ONSHORING_MODEL_BENCHMARK_REPORT.md` |
| Claim-evidence boundary | Drafted in manuscript appendix | all reports |

## Submission Blockers

1. **Related-work review with verified citations**
   - stochastic and non-stationary Markov routing
   - logit choice and entropy-regularized transport
   - agent-based and discrete-event supply-chain models
   - production networks and network interdiction
   - cybernetics and adaptive control

2. **Public calibration and scope sources**
   - semiconductor production role categories
   - advanced packaging dependency
   - public trade and capacity context
   - explicit statement that no confidential operational data is used

3. **Figure generation**
   - all figures must be generated from versioned scripts and JSON outputs
   - no manual spreadsheet figures

4. **Reproducibility freeze**
   - commit hash
   - Python and dependency versions
   - one-command reproduction sequence
   - archived raw JSON outputs
   - fixed paper experiment manifest

5. **Statistical strengthening**
   - report seed-level uncertainty intervals where appropriate
   - decide whether five paired seeds are sufficient for the venue
   - run HPC replication only after the paper experiment manifest is frozen
   - for Neural V2, run paired seed-robust validation across clean, hard,
     frontier, and adversarial-switch regimes

6. **Venue decision**
   - applied network science venue if emphasizing the choice-point principle
   - complex systems venue if emphasizing adaptive circulation
   - ML theory/application venue only if optimizer or learning contributions are added

## Writing Sequence

1. Freeze the V1 theory note and experiment manifest.
2. Produce the Neural V2 validation table and regime-boundary figure.
3. Produce semiconductor Figures 5, 7, 8, 9, and 10.
4. Perform focused related-work review and add citations.
5. Add public calibration sources and a model-scope table.
6. Revise abstract and introduction after figures exist.
7. Convert the Markdown manuscript to LaTeX.
8. Run final local plus HPC reproduction from the frozen manifest.
9. Conduct an adversarial internal review using the claim-evidence appendix.

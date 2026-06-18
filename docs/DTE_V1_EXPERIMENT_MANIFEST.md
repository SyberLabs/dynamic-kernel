# DTE V1 Experiment Manifest

## Purpose

This manifest defines the V1 evidence set for the first paper. It separates
publication-critical experiments from exploratory applications. The goal is to
make every claim traceable to a versioned script, output file, and report.

## Freeze Principle

The V1 paper should not add new domains. The evidence set is:

```text
Theory object: DTE layered-memory process
Controlled bench: Neural V2 adaptive inference routing
Applied case: semiconductor onshoring and supply-chain feasibility
```

Social media, influence wargames, ants, microbiome, and consumer goods remain
future-work or demonstration modules unless explicitly moved into a later
paper.

## Proposition-To-Experiment Map

| Proposition | Claim | Primary Evidence | Required Validation |
|---|---|---|---|
| P1 Markov closure | DTE is Markov on augmented memory state, not generally on position alone | `DTE_V1_THEORY.md`, `kernel.py`, memory-law tests | Unit tests plus formal proof sketch |
| P2 Choice-point invariance | Singleton-outdegree interventions cannot reroute finite softmax flow | `semiconductor_onshoring_falsification.py`, `TWO_ROUTE_THEOREM_NOTE.md` | Figure from relocation and topology-surgery outputs |
| P3 Attribution fragility | Multiplicative policy lanes compound misattributed delayed reward | `neural_v2_router_benchmark.py` | Label-noise/reward-delay sweep and EXP3 gate-gain table |
| P4 Arbitration boundary | External policy owners dominate clean bandit tasks; DTE helps when memory/topology matter | Neural V2 clean, hard, frontier, adversarial-switch reports | Paired seed-robust benchmark with confidence intervals |
| P5 Feasibility-allocation distinction | Majority route share can coexist with infeasible completed production | semiconductor BOM, model benchmark, topology nulls | Scatter/table showing share versus completed lots |

## Publication-Critical Scripts

### Kernel And Theory Witnesses

| Script/Test | Purpose | Output |
|---|---|---|
| `test_kernel.py` and `tests/test_kernel.py` | Core transition and compatibility tests | pytest |
| `tests/test_memory_law.py` | Preference-memory API and diagnostics | pytest |
| `two_route_memory_theorem.py` | Minimal choice/memory theorem witness | `two_route_memory_theorem_output.json` |
| `kernel_local_regret_witness.py` | Local-regret stale-route witness | `kernel_local_regret_witness_output.json` |
| `paper_figures.py` | Layered-memory and semiconductor paper figures | `figures/*.svg`, `PAPER_FIGURE_GENERATION_REPORT.md` |

### Neural V2 Validation Bench

| Script | Purpose | Output | Report |
|---|---|---|---|
| `neural_v2_adaptive_routing.py` | Base Neural V2 topology and adaptive routing demo | `neural_v2_adaptive_routing_output.json` | `NEURAL_V2_ADAPTIVE_ROUTING_REPORT.md` |
| `neural_v2_router_benchmark.py` | Clean, hard, frontier, and adversarial-switch benchmark family | `neural_v2_*_output.json` | `NEURAL_V2_*_REPORT.md` |
| `neural_v2_seed_validation.py` | Paired-seed validation table with confidence intervals | `neural_v2_seed_validation_output.json` | `NEURAL_V2_SEED_VALIDATION_REPORT.md` |

V1 default policy after current evidence:

```text
Neural V2 default: reliability-arbitrated UCB
Experimental lane: DTE-EXP3 / reliability-gated DTE-EXP3
Boundary: external UCB/EXP3 still win when they own the full policy
```

### Semiconductor Case Study

| Script | Purpose | Output | Report |
|---|---|---|---|
| `semiconductor_onshoring.py` | Base topology and policy sweep | `semiconductor_onshoring_output.json` | `SEMICONDUCTOR_ONSHORING_REPORT.md` |
| `semiconductor_onshoring_frontier.py` | Feasibility frontier | `semiconductor_onshoring_frontier_output.json` | `SEMICONDUCTOR_ONSHORING_FRONTIER_REPORT.md` |
| `semiconductor_onshoring_input_phase_diagram.py` | Input constraint phase diagram | `semiconductor_onshoring_input_phase_output.json` | `SEMICONDUCTOR_ONSHORING_INPUT_PHASE_REPORT.md` |
| `semiconductor_onshoring_inventory_phase_diagram.py` | Inventory and reserve phase diagram | `semiconductor_onshoring_inventory_phase_output.json` | `SEMICONDUCTOR_ONSHORING_INVENTORY_PHASE_REPORT.md` |
| `semiconductor_onshoring_model_benchmark.py` | Static/frozen/DTE comparison | `semiconductor_onshoring_model_benchmark_output.json` | `SEMICONDUCTOR_ONSHORING_MODEL_BENCHMARK_REPORT.md` |
| `semiconductor_onshoring_falsification.py` | Choice-point relocation, topology surgery, feedback, nulls | `semiconductor_onshoring_falsification_output.json` | `SEMICONDUCTOR_ONSHORING_FALSIFICATION_REPORT.md` |
| `semiconductor_onshoring_classification_robustness.py` | Threshold robustness | `semiconductor_onshoring_classification_robustness_output.json` | `SEMICONDUCTOR_ONSHORING_CLASSIFICATION_ROBUSTNESS_REPORT.md` |

## Local Reproduction Commands

Use these before any HPC run:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe neural_v2_seed_validation.py
.venv\Scripts\python.exe paper_figures.py
.venv\Scripts\python.exe neural_v2_router_benchmark.py
.venv\Scripts\python.exe semiconductor_onshoring_model_benchmark.py
.venv\Scripts\python.exe semiconductor_onshoring_falsification.py
```

The full Neural V2 benchmark currently runs locally but is slower after adding
frontier and adversarial-switch families. Use local runs for correctness and
small seed validation; use SLURM only after the manifest is frozen.

## Seed-Robust Validation Targets

Minimum local validation:

```text
Neural V2: 5 paired seeds for clean and hard reports
Neural V2 adversarial-switch: at least 3 paired seeds per cell
Semiconductor falsification: existing 5 paired seeds
Semiconductor robustness/nulls: existing report plus one confirmatory rerun
```

Publication-strength validation:

```text
Neural V2: 20-30 paired seeds per main regime
Adversarial-switch grid: 10 paired seeds per cell
Semiconductor case: 20 paired seeds for main policy cells, 8-16 topology nulls
Report mean, 95% CI, and paired deltas where common random numbers are used
```

## Claim-Evidence Boundaries

Claims allowed in V1:

```text
DTE is a memory-aware adaptive routing kernel, not a universal optimizer.
DTE exposes intervention inertness at non-choice edges.
DTE distinguishes route share from completed feasible production.
DTE-native policy lanes help in specific regimes but do not dominate policy owners.
Semiconductor procurement preference is effective in the studied topology, not topology-invariant.
```

Claims not allowed in V1:

```text
DTE universally beats UCB, EXP3, or contextual bandits.
DTE forecasts real semiconductor markets.
DTE proves a national policy recommendation from public toy calibration alone.
DTE's other exploratory domains are validated case studies.
```

## Next Execution Step

The next concrete task is to build the Neural V2 seed-robust validation table:

```text
rows: clean, hard, adversarial-switch
columns: local-regret DTE, reliability-UCB DTE, DTE-EXP3, gated DTE-EXP3, external UCB, external EXP3
statistics: mean regret, CI95, paired regret delta versus local-regret DTE
```

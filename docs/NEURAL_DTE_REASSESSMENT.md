# Neural DTE Reassessment

## Verdict

The current Neural surface should be reclassified as an advanced optimizer
instrument, not as a validated neural-computation application. It is useful
for beta dynamics, phase chaining, feasibility reporting, gradient inspection,
and topology stress tests. It does not yet model task performance, module
specialization, loss reduction, activation routing, or adaptive computation.

## Current Construction

- Nodes are abstract neurons with identical feature vectors.
- Edges are dense or random unit-distance links.
- Telemetry is frozen in the optimizer backend.
- Beta is optimized against stationary-distribution symmetry objectives.
- There is no reward/ecology layer and no task-conditioned input stream.

## Probe Results

| Case | Mode | Delta Sigma | Beta Std | Pi Std | Mixing | Health |
|---|---|---:|---:|---:|---:|---|
| complete_undirected | ENTROPY_PI | 0.000000 | 0.0000 | 0.0000 | 1.10 | STALLED |
| complete_undirected | ROW_ENTROPY | 0.000000 | 0.0000 | 0.0000 | 1.10 | STALLED |
| complete_undirected | DETAILED_BALANCE | -0.000000 | 0.0000 | 0.0000 | 1.10 | STALLED |
| complete_undirected | SPECTRAL_GAP | 0.000000 | 0.0000 | 0.0000 | 1.10 | STALLED |
| complete_undirected | WEIGHT_SYMMETRY | 0.000000 | 0.0000 | 0.0000 | 1.10 | STALLED |
| sparse_undirected | ENTROPY_PI | 0.000125 | 0.0144 | 0.0300 | 6.47 | STALLED |
| sparse_undirected | ROW_ENTROPY | -0.000000 | 0.0000 | 0.0300 | 6.47 | STALLED |
| sparse_undirected | DETAILED_BALANCE | 0.000000 | 0.0000 | 0.0300 | 6.47 | STALLED |
| sparse_undirected | SPECTRAL_GAP | 0.000031 | 0.0014 | 0.0300 | 6.47 | STALLED |
| sparse_undirected | WEIGHT_SYMMETRY | 0.000000 | 0.0000 | 0.0300 | 6.47 | STALLED |
| sparse_directed | ENTROPY_PI | 0.000074 | 0.0099 | 0.0301 | 2.00 | STALLED |
| sparse_directed | ROW_ENTROPY | 0.000000 | 0.0000 | 0.0301 | 2.00 | STALLED |
| sparse_directed | DETAILED_BALANCE | 0.000058 | 0.0111 | 0.0302 | 2.00 | STALLED |
| sparse_directed | SPECTRAL_GAP | 0.000016 | 0.0009 | 0.0301 | 2.00 | STALLED |
| sparse_directed | WEIGHT_SYMMETRY | 12.000000 | 0.6532 | 0.0330 | 1.93 | HEALTHY |

## Composite Feasibility Probe

Targeting node 1 at probability `0.40` on the complete current Neural graph reached pi_1 `0.1243` after 120 steps.
The feasibility probe classifies the target as `PARTIAL` with L1 error `0.4698`.

## Interpretation

1. The complete undirected default is exactly symmetric, so all symmetry
   objectives stall. This is mathematically expected, not a frontend bug.

2. Sparse topology introduces structural asymmetry, but most objectives
   still move weakly because all node features and distances remain equal.

3. Directed sparse topology gives WEIGHT_SYMMETRY a real signal because
   beta can repair directional asymmetry. This makes Neural a useful
   optimizer stress test, but not yet a neural-domain model.

4. COMPOSITE targeting can partially move stationary mass, but this is
   routing-control over a graph, not learning or inference.

## What Would Make It A Serious Neural Application

Reframe nodes as modules or experts, not neurons. Then define:

- node features: module capabilities, cost, latency, modality, specialization;
- telemetry: task or input embedding;
- reward: loss reduction, confidence gain, energy efficiency, or latency-adjusted utility;
- beta memory: learned routing preference between modules;
- local regret: traffic sent to a weaker module while a better reachable module exists;
- ecology: task distribution drift, module degradation, or compute budget shocks.

The resulting object would be closer to mixture-of-experts routing or
adaptive-computation governance than to biological neural dynamics.

## Recommendation

Keep NeuralPort, but relabel it as `Optimizer Lab` or `Neural Routing Lab`.
Do not use it as the paper's main proof demo yet. Use it as the advanced
instrument surface after the ant/local-regret paper demo is clear.

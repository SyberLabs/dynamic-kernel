# Neural V2 Adaptive Routing Report

## Scope

Neural V2 models adaptive routing among computational modules / experts,
not biological neurons. Tasks are routed from a task router to language,
symbolic, vision, or fast-generalist modules. The task stream shifts from
language-heavy to symbolic-heavy, testing stale routing memory.

## Result

| Condition | Runs | Reward | Regret | Language Share | Symbolic Share | Lang Memory | Sym Memory | Recovery Tick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_forgetting | 5 | 0.325 | 0.179 | 0.307 | 0.308 | 6.478 | 5.307 | 52.4 |
| surprise_only | 5 | 0.327 | 0.177 | 0.309 | 0.311 | 4.621 | 5.176 | 56.2 |
| local_regret | 5 | 0.373 | 0.131 | 0.137 | 0.566 | 0.153 | 8.237 | 56.4 |

## Interpretation

Against surprise-only adaptation, local regret reduces post-shift mean regret by `0.046`.
It increases symbolic-expert routing by `0.255` and decreases stale language routing by `0.173`.

This is the Neural V2 claim in miniature: a computational router can
retain stale preference memory for a once-good module after the task
ecology shifts. Local regret supplies the missing counterfactual signal:
the chosen module still works, but a better reachable module exists.

## V2 Status

This is a rigorous minimal witness, not a neural-network benchmark.
The next maturity step is to replace synthetic task rewards with real
module performance traces: loss reduction, confidence gain, latency,
or cost on a task suite.

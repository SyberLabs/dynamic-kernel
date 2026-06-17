# Kernel Local-Regret Witness Report

## Scope

Tiny three-node topology using the built-in `PopulationSimulator` memory loop.
The learned stale route still pays reward `0.8`; the reachable alternative
pays `1.0`. Surprise-only adaptive evaporation receives no destination
reward-collapse signal, while local regret sees the opportunity cost.

## Result

| Condition | Runs | Final stale delta | Final P(Entry->Stale) | Last-10 stale share |
|---|---:|---:|---:|---:|
| Surprise only | 5 | 0.7283 | 0.9334 | 0.9487 |
| Local regret | 5 | 0.0157 | 0.5155 | 0.5291 |

## Interpretation

Local regret behaves as intended: it evaporates stale preference memory
even when the stale route remains mildly productive. The route probability
returns to the unbiased baseline near 0.5, while surprise-only adaptation
leaves the stale route dominant.

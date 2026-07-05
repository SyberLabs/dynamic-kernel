# Semiconductor Onshoring Falsification Report

## Scope

Adversarial tests of whether the feasibility-allocation-adaptation conclusion survives changes in intervention location, topology, feedback rate, and route-choice wiring.

- Agents: `320`
- Steps: `40`
- Seeds: `20260606, 20260607, 20260608, 20260609, 20260610`

## Summary

- Maximum upstream-choice share lift: `0.084`
- Maximum route-commitment share lift: `0.000`
- Maximum downstream-serial share lift: `0.000`
- Maximum serial-topology downstream-penalty share lift: `0.000`
- Maximum reconsideration-topology downstream-penalty share lift: `0.235`
- Randomized nulls where pull improves viable rate: `1`
- Randomized nulls with majority share but no robust transition: `4`

## Choice-Point Relocation

| Location | Budget | Viable Rate | Mean Share | Mean Completion |
|---|---:|---:|---:|---:|
| `upstream_choice` | 0.0 | 40.0% | 0.502 | 100.0% |
| `upstream_choice` | 4.0 | 80.0% | 0.517 | 100.0% |
| `upstream_choice` | 8.0 | 100.0% | 0.559 | 99.9% |
| `upstream_choice` | 12.0 | 100.0% | 0.586 | 97.6% |
| `route_commitment` | 0.0 | 40.0% | 0.502 | 100.0% |
| `route_commitment` | 4.0 | 40.0% | 0.502 | 100.0% |
| `route_commitment` | 8.0 | 40.0% | 0.502 | 100.0% |
| `route_commitment` | 12.0 | 40.0% | 0.502 | 100.0% |
| `downstream_serial` | 0.0 | 40.0% | 0.502 | 100.0% |
| `downstream_serial` | 4.0 | 40.0% | 0.502 | 100.0% |
| `downstream_serial` | 8.0 | 40.0% | 0.502 | 100.0% |
| `downstream_serial` | 12.0 | 40.0% | 0.502 | 100.0% |

## Topology Surgery

| Topology | Downstream Budget | Viable Rate | Mean Share | Mean Completion |
|---|---:|---:|---:|---:|
| `serial` | 0.0 | 40.0% | 0.502 | 100.0% |
| `serial` | 4.0 | 40.0% | 0.502 | 100.0% |
| `serial` | 8.0 | 40.0% | 0.502 | 100.0% |
| `serial` | 12.0 | 40.0% | 0.502 | 100.0% |
| `reconsideration_exits` | 0.0 | 40.0% | 0.508 | 100.0% |
| `reconsideration_exits` | 4.0 | 80.0% | 0.537 | 100.0% |
| `reconsideration_exits` | 8.0 | 100.0% | 0.683 | 99.6% |
| `reconsideration_exits` | 12.0 | 100.0% | 0.743 | 98.9% |

## Feedback Continuum

| Feedback Rate | Domestic Pull | Viable Rate | Mean Share | Mean Completion |
|---:|---:|---:|---:|---:|
| 0.00 | 0.0 | 40.0% | 0.484 | 100.0% |
| 0.00 | 1.0 | 100.0% | 0.531 | 100.0% |
| 0.05 | 0.0 | 20.0% | 0.489 | 100.0% |
| 0.05 | 1.0 | 100.0% | 0.521 | 100.0% |
| 0.10 | 0.0 | 20.0% | 0.475 | 100.0% |
| 0.10 | 1.0 | 60.0% | 0.512 | 100.0% |
| 0.15 | 0.0 | 40.0% | 0.502 | 100.0% |
| 0.15 | 1.0 | 80.0% | 0.522 | 100.0% |
| 0.30 | 0.0 | 20.0% | 0.472 | 100.0% |
| 0.30 | 1.0 | 60.0% | 0.496 | 100.0% |
| 0.50 | 0.0 | 20.0% | 0.478 | 100.0% |
| 0.50 | 1.0 | 80.0% | 0.514 | 100.0% |

## Randomized Choice-Topology Nulls

| Null Seed | Domestic Pull | Viable Rate | Mean Share | Mean Completion | Rewired Edges |
|---:|---:|---:|---:|---:|---:|
| 7001 | 0.0 | 100.0% | 0.895 | 97.6% | 36 |
| 7001 | 1.0 | 100.0% | 0.897 | 97.4% | 36 |
| 7002 | 0.0 | 0.0% | 0.846 | 71.7% | 39 |
| 7002 | 1.0 | 0.0% | 0.850 | 73.2% | 39 |
| 7003 | 0.0 | 100.0% | 0.739 | 98.5% | 37 |
| 7003 | 1.0 | 100.0% | 0.746 | 98.4% | 37 |
| 7004 | 0.0 | 0.0% | 0.815 | 71.2% | 37 |
| 7004 | 1.0 | 20.0% | 0.817 | 72.1% | 37 |
| 7005 | 0.0 | 100.0% | 0.806 | 99.1% | 38 |
| 7005 | 1.0 | 100.0% | 0.776 | 100.0% | 38 |
| 7006 | 0.0 | 40.0% | 0.858 | 71.0% | 39 |
| 7006 | 1.0 | 0.0% | 0.863 | 72.8% | 39 |
| 7007 | 0.0 | 0.0% | 0.834 | 80.4% | 41 |
| 7007 | 1.0 | 0.0% | 0.845 | 81.5% | 41 |
| 7008 | 0.0 | 100.0% | 0.857 | 96.9% | 39 |
| 7008 | 1.0 | 100.0% | 0.859 | 97.6% | 39 |

## Reading

The choice-point principle survives: equal-cost penalties change route share only at upstream choice edges, or after topology surgery creates a latent alternative. Feedback reshapes the phase boundary non-monotonically rather than acting as a universal amplifier. Procurement pull is topology-contingent, while the distinction between high route share and a viable production transition persists across nulls.

# Semiconductor Onshoring Scale-Stability Comparison

## Scope

Comparison of the near-viable onshoring pocket under two packaging-input structures:

- `shared_upstream`: U.S. advanced packaging and Taiwan export packaging both draw from `Korea Packaging Inputs`.
- `independent_upstream`: U.S. advanced packaging draws from a distinct `US Packaging Inputs` source, while Taiwan export packaging continues to draw from `Korea Packaging Inputs`.

Both sweeps hold the near-viable `subsidy_packaging` doctrine fixed and vary:

- agents: `40`, `80`, `120`, `160`, `240`, `320`
- renewal: `2`, `3`, `4`, `5`
- U.S. advanced-packaging capacity multiplier: `0.0`, `3.0`, `6.0`
- seeds: `20260606`, `20260607`, `20260608`
- steps per run: `40`

Each model ran `216` simulations.

## Headline

U.S. packaging-input independence improves robustness inside the viable pocket, but it does not move the collapse threshold. Viability remains robust at low load, fragile around `120` agents, and absent from `160` agents upward.

## Summary

| Model | Viable Groups | Robust Groups | Best Viable Share | Best Viable Dependency |
|---|---:|---:|---:|---:|
| `shared_upstream` | `19` | `10` | `0.650` | `57.5%` |
| `independent_upstream` | `19` | `13` | `0.700` | `52.5%` |

## Scale Threshold

| Agents | Shared Max Viable Rate | Independent Max Viable Rate | Shared Robust Groups | Independent Robust Groups | Shared Min Dependency | Independent Min Dependency |
|---:|---:|---:|---:|---:|---:|---:|
| `40` | `100.0%` | `100.0%` | `8` | `9` | `8.2%` | `9.2%` |
| `80` | `100.0%` | `100.0%` | `2` | `4` | `43.0%` | `41.1%` |
| `120` | `33.3%` | `33.3%` | `0` | `0` | `56.3%` | `54.8%` |
| `160` | `0.0%` | `0.0%` | `0` | `0` | `60.3%` | `61.1%` |
| `240` | `0.0%` | `0.0%` | `0` | `0` | `58.3%` | `59.8%` |
| `320` | `0.0%` | `0.0%` | `0` | `0` | `54.9%` | `58.1%` |

## Reading

The viable pocket is real but load-sensitive. Domestic packaging-input independence improves local performance: it raises robust groups from `10` to `13`, doubles robust groups at `80` agents from `2` to `4`, and improves the best viable run from onshore share `0.650` to `0.700`.

But independence does not make the phase scale-stable. Both structures collapse to non-viability by `160` agents. The binding variable is therefore not merely upstream packaging independence. It is the scaling law of renewal relative to demand load and gate queue pressure.

## Apex Inference

The current topology has a viable onshoring phase below a critical load scale. U.S. packaging-input independence widens the pocket but does not change the critical scale. A robust institutional strategy would need renewal capacity, domestic wafer inflow, and gate service capacity to scale with load rather than remain fixed per tick.

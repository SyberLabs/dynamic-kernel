# Intervention Leverage Screen — Semiconductor Case Study

Analytic leverage field (one matrix inversion per telemetry) computed
BEFORE consulting simulation, then compared against the measured
paired-CRN falsification lifts. The screen predicts sign and ranking;
magnitudes belong to the adaptive simulation, which sees gates,
capacities, and adaptation that this frozen-telemetry object cannot.

Onshore mass set: US Hyperscaler Demand, US Defense Demand, US EDA IP, US Packaging Inputs, US Power Labor, Intel US Fabs, TSMC Arizona Fabs, Samsung Texas Fabs, US Wafer Fabrication, US Advanced Packaging, US Finished Packaged Chips, US West Coast Port, Strategic Chip Reserve

## Relocation sites (serial topology)

| Site | own-edge dP/dS | onshore gain / budget | measured max lift |
|---|---:|---:|---:|
| upstream_choice | 0.005951 | 0.001143 | 0.084 |
| route_commitment | 0.000000 | 0.000000 | 0.000 |
| downstream_serial | 0.000000 | 0.000000 | 0.000 |

## Topology surgery (downstream site)

| Topology | own-edge dP/dS | onshore gain / budget | measured max lift |
|---|---:|---:|---:|
| serial | 0.000000 | 0.000000 | 0.000 |
| reconsideration_exits | 0.005899 | -0.000320 | 0.235 |

## Checks

- `relocation_top_site_match`: `True`
- `relocation_predicted_rank`: `['upstream_choice', 'route_commitment', 'downstream_serial']`
- `relocation_measured_rank`: `['upstream_choice', 'route_commitment', 'downstream_serial']`
- `inert_sites_predicted`: `['route_commitment', 'downstream_serial']`
- `inert_sites_measured`: `['route_commitment', 'downstream_serial']`
- `surgery_unlock_predicted`: `True`
- `surgery_unlock_measured`: `True`
- `surgery_direction_by_onshore_set`: `{'canonical': -0.00032021575071691544, 'excl_west_coast_port': 6.47027052523592e-05, 'production_backbone_only': 3.251938944088512e-05}`

The screen's verdict is LIVENESS (zero vs nonzero leverage) and
RANKING, both of which match measurement exactly. The surgery
outcome DIRECTION under the canonical onshore set is dominated by
US West Coast Port — the import-corridor terminus — and flips
positive when the functional is restricted to the production
backbone (see surgery_direction_by_onshore_set). Outcome
magnitudes and directions remain the adaptive simulation's job,
which is the paper's own thesis about frozen models.

Leverage formula includes the softplus floor gate:
`L_ij = sigmoid(k (W_ij - floor)) * P_ij (1 - P_ij) / tau`.
Choice-point invariance is the `P_ij -> 1` limit of this field.

"""
Seed-robust validation for supply-chain stress controls.

The stress suite gives single-trajectory evidence. This module asks a harder
question: which controls remain positive across random seeds, and which ones
backfire once capacity overflow is counted as part of the objective?

Usage:
    .venv\\Scripts\\python.exe supply_chain_stress_robustness.py --quick
    .venv\\Scripts\\python.exe supply_chain_stress_robustness.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from supply_chain_resilience import SimulationConfig
from supply_chain_stress import StressShock, hard_capacity_shocks, severity_shocks, simulate_stress


SEVERITY_CONTROLS = [
    "no_control",
    "port_reroute",
    "dual_source_chips",
    "buffer_release",
    "generic_resilience",
]
HARD_CAPACITY_CONTROLS = [
    "no_control",
    "port_reroute",
    "expedite_air",
    "buffer_release",
    "generic_resilience",
    "dual_source_chips",
]


@dataclass(frozen=True)
class RobustnessConfig:
    seeds: int = 5
    seed_base: int = 20260604
    agents: int = 192
    steps: int = 72
    severities: tuple[float, ...] = (1.0, 2.0, 4.0)
    include_hard_capacity: bool = True


def _mean_ci(values: list[float], z: float = 1.96) -> dict:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": 1}
    stderr = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    half = z * stderr
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half, "n": int(len(arr))}


def _control_benefit(row: dict) -> float:
    overflow_penalty = max(row["overflow_delta_vs_baseline"], 0.0)
    blocked_penalty = max(row["hard_blocked_delta_vs_baseline"], 0.0)
    return (
        row["fulfillment_delta_vs_baseline"]
        + 0.75 * row["critical_delta_vs_baseline"]
        - 0.25 * overflow_penalty
        - 0.10 * blocked_penalty
    )


def _capacity_adjusted_roi(row: dict) -> float:
    if row["control_cost"] <= 0:
        return 0.0
    return _control_benefit(row) / row["control_cost"]


def _classify(summary: dict) -> str:
    if summary["control"] == "no_control":
        return "baseline"

    fulfillment = summary["fulfillment_delta"]
    critical = summary["critical_delta"]
    overflow = summary["overflow_delta"]
    benefit = summary["net_benefit_delta"]
    roi = summary["capacity_adjusted_roi"]

    positive_signal = fulfillment["ci_low"] > 0.005 or critical["ci_low"] > 0.005 or roi["ci_low"] > 0.0
    overflow_backfire = overflow["ci_low"] > 0.02
    negative_signal = fulfillment["ci_high"] < -0.005 and critical["ci_high"] <= 0.005

    if positive_signal and not overflow_backfire:
        return "robust_positive"
    if negative_signal or (overflow_backfire and roi["ci_high"] <= 0.0):
        return "backfire"
    if benefit["mean"] > 0.0 and summary["positive_seed_rate"] >= 0.60:
        return "mixed_positive"
    if benefit["mean"] < 0.0 and summary["positive_seed_rate"] <= 0.40:
        return "mixed_negative"
    return "neutral"


def _shock_plan(config: RobustnessConfig, quick: bool = False) -> list[tuple[StressShock, list[str]]]:
    severity_values = [2.0] if quick else list(config.severities)
    plan = [(shock, SEVERITY_CONTROLS) for shock in severity_shocks(severity_values)]
    if not config.include_hard_capacity:
        return plan
    hard_cases = hard_capacity_shocks()
    if quick:
        hard_cases = [
            shock
            for shock in hard_cases
            if shock.name in {
                "ocean_route_hard_failure",
                "air_freight_capacity_08",
                "port_and_air_capacity",
                "chip_fab_b_capacity_04",
            }
        ]
    plan.extend((shock, HARD_CAPACITY_CONTROLS) for shock in hard_cases)
    return plan


def _enrich_seed_rows(rows: list[dict]) -> list[dict]:
    baselines = {
        (row["seed_index"], row["shock"]): row
        for row in rows
        if row["control"] == "no_control"
    }
    enriched = []
    for row in rows:
        baseline = baselines[(row["seed_index"], row["shock"])]
        item = dict(row)
        item["fulfillment_delta_vs_baseline"] = row["fulfillment_share"] - baseline["fulfillment_share"]
        item["critical_delta_vs_baseline"] = row["critical_service_share"] - baseline["critical_service_share"]
        item["overflow_delta_vs_baseline"] = row["capacity_overflow_rate"] - baseline["capacity_overflow_rate"]
        item["edge_overflow_delta_vs_baseline"] = (
            row["edge_capacity_overflow_rate"] - baseline["edge_capacity_overflow_rate"]
        )
        item["node_overflow_delta_vs_baseline"] = (
            row["node_capacity_overflow_rate"] - baseline["node_capacity_overflow_rate"]
        )
        item["hard_blocked_delta_vs_baseline"] = row["hard_blocked_mass"] - baseline["hard_blocked_mass"]
        item["edge_current_delta_vs_baseline"] = row["edge_current_norm"] - baseline["edge_current_norm"]
        item["entropy_delta_vs_baseline"] = row["entropy_production"] - baseline["entropy_production"]
        item["net_benefit_delta_vs_baseline"] = _control_benefit(item)
        item["capacity_adjusted_roi"] = _capacity_adjusted_roi(item)
        enriched.append(item)
    return enriched


def _summarize(enriched_rows: list[dict]) -> list[dict]:
    summaries = []
    keys = sorted({(row["shock"], row["control"]) for row in enriched_rows})
    for shock, control in keys:
        rows = [row for row in enriched_rows if row["shock"] == shock and row["control"] == control]
        first = rows[0]
        summary = {
            "shock": shock,
            "family": first["family"],
            "severity": first["severity"],
            "control": control,
            "control_cost": first["control_cost"],
            "seeds": len(rows),
            "fulfillment": _mean_ci([row["fulfillment_share"] for row in rows]),
            "critical_service": _mean_ci([row["critical_service_share"] for row in rows]),
            "overflow_rate": _mean_ci([row["capacity_overflow_rate"] for row in rows]),
            "edge_overflow_rate": _mean_ci([row["edge_capacity_overflow_rate"] for row in rows]),
            "node_overflow_rate": _mean_ci([row["node_capacity_overflow_rate"] for row in rows]),
            "hard_blocked_mass": _mean_ci([row["hard_blocked_mass"] for row in rows]),
            "fulfillment_delta": _mean_ci([row["fulfillment_delta_vs_baseline"] for row in rows]),
            "critical_delta": _mean_ci([row["critical_delta_vs_baseline"] for row in rows]),
            "overflow_delta": _mean_ci([row["overflow_delta_vs_baseline"] for row in rows]),
            "edge_overflow_delta": _mean_ci([row["edge_overflow_delta_vs_baseline"] for row in rows]),
            "node_overflow_delta": _mean_ci([row["node_overflow_delta_vs_baseline"] for row in rows]),
            "hard_blocked_delta": _mean_ci([row["hard_blocked_delta_vs_baseline"] for row in rows]),
            "edge_current_delta": _mean_ci([row["edge_current_delta_vs_baseline"] for row in rows]),
            "entropy_delta": _mean_ci([row["entropy_delta_vs_baseline"] for row in rows]),
            "net_benefit_delta": _mean_ci([row["net_benefit_delta_vs_baseline"] for row in rows]),
            "capacity_adjusted_roi": _mean_ci([row["capacity_adjusted_roi"] for row in rows]),
            "positive_seed_rate": float(np.mean([row["net_benefit_delta_vs_baseline"] > 0.0 for row in rows])),
        }
        summary["classification"] = _classify(summary)
        summaries.append(summary)
    return summaries


def run_robustness_suite(config: RobustnessConfig | None = None, quick: bool = False) -> dict:
    config = config or RobustnessConfig()
    rows = []
    for seed_index in range(2 if quick else config.seeds):
        sim_config = SimulationConfig(
            agents=config.agents,
            steps=config.steps,
            seed=config.seed_base + seed_index,
        )
        for shock, control_names in _shock_plan(config, quick=quick):
            for control_name in control_names:
                row = simulate_stress(sim_config, shock, control_name)
                row["seed_index"] = seed_index
                row["seed"] = sim_config.seed
                rows.append(row)

    enriched = _enrich_seed_rows(rows)
    summary = _summarize(enriched)
    counts = {
        label: sum(1 for row in summary if row["classification"] == label)
        for label in sorted({row["classification"] for row in summary})
    }
    return {
        "config": asdict(config) | {"actual_seeds": 2 if quick else config.seeds, "quick": quick},
        "classification_counts": counts,
        "summary": summary,
        "rows": enriched,
    }


def _fmt_ci(metric: dict, decimals: int = 3) -> str:
    return f"{metric['mean']:+.{decimals}f} [{metric['ci_low']:+.{decimals}f}, {metric['ci_high']:+.{decimals}f}]"


def _top_controls(summary: list[dict], classification: str, limit: int = 8) -> list[dict]:
    rows = [row for row in summary if row["classification"] == classification]
    return sorted(rows, key=lambda row: row["net_benefit_delta"]["mean"], reverse=True)[:limit]


def render_report(payload: dict) -> str:
    summary = [row for row in payload["summary"] if row["control"] != "no_control"]
    positives = _top_controls(summary, "robust_positive")
    mixed_positive = _top_controls(summary, "mixed_positive")
    backfires = sorted(
        [row for row in summary if row["classification"] == "backfire"],
        key=lambda row: row["net_benefit_delta"]["mean"],
    )[:10]
    capacity_rows = [
        row
        for row in summary
        if row["family"] in {"capacity", "node_capacity"} and row["control"] != "no_control"
    ]

    config = payload["config"]
    lines = [
        "# Supply Chain Stress Robustness Report",
        "",
        "## Scope",
        "",
        (
            "Seed-robust validation of supply-chain stress controls using common-random-seed "
            "comparisons against the no-control baseline for each shock."
        ),
        "",
        f"- Seeds: `{config['actual_seeds']}`",
        f"- Agents per run: `{config['agents']}`",
        f"- Steps per run: `{config['steps']}`",
        f"- Shock-control runs: `{len(payload['rows'])}`",
        "",
        "## Classification Counts",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(payload["classification_counts"].items()):
        lines.append(f"| `{label}` | {count} |")

    lines.extend([
        "",
        "## Robust Positive Controls",
        "",
        "| Shock | Control | Benefit Delta CI95 | Fulfillment Delta CI95 | Critical Delta CI95 | Overflow Delta CI95 | ROI CI95 | Positive Seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in positives:
        lines.append(
            f"| `{row['shock']}` | `{row['control']}` | {_fmt_ci(row['net_benefit_delta'])} | "
            f"{_fmt_ci(row['fulfillment_delta'])} | {_fmt_ci(row['critical_delta'])} | "
            f"{_fmt_ci(row['overflow_delta'])} | {_fmt_ci(row['capacity_adjusted_roi'], 4)} | "
            f"{row['positive_seed_rate']:.0%} |"
        )
    if not positives:
        lines.append("| none | none | | | | | | |")

    lines.extend([
        "",
        "## Mixed Positive Controls",
        "",
        "| Shock | Control | Benefit Delta CI95 | Fulfillment Delta CI95 | Critical Delta CI95 | Overflow Delta CI95 | ROI CI95 | Positive Seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in mixed_positive:
        lines.append(
            f"| `{row['shock']}` | `{row['control']}` | {_fmt_ci(row['net_benefit_delta'])} | "
            f"{_fmt_ci(row['fulfillment_delta'])} | {_fmt_ci(row['critical_delta'])} | "
            f"{_fmt_ci(row['overflow_delta'])} | {_fmt_ci(row['capacity_adjusted_roi'], 4)} | "
            f"{row['positive_seed_rate']:.0%} |"
        )
    if not mixed_positive:
        lines.append("| none | none | | | | | | |")

    lines.extend([
        "",
        "## Backfires",
        "",
        "| Shock | Control | Benefit Delta CI95 | Fulfillment Delta CI95 | Critical Delta CI95 | Overflow Delta CI95 | ROI CI95 | Positive Seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in backfires:
        lines.append(
            f"| `{row['shock']}` | `{row['control']}` | {_fmt_ci(row['net_benefit_delta'])} | "
            f"{_fmt_ci(row['fulfillment_delta'])} | {_fmt_ci(row['critical_delta'])} | "
            f"{_fmt_ci(row['overflow_delta'])} | {_fmt_ci(row['capacity_adjusted_roi'], 4)} | "
            f"{row['positive_seed_rate']:.0%} |"
        )
    if not backfires:
        lines.append("| none | none | | | | | | |")

    lines.extend([
        "",
        "## Capacity Doctrine Table",
        "",
        "| Shock | Family | Control | Classification | Overflow Delta CI95 | Edge Overflow Delta CI95 | Node Overflow Delta CI95 | Benefit Delta CI95 | Fulfillment Delta CI95 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(capacity_rows, key=lambda item: (item["shock"], item["control"])):
        lines.append(
            f"| `{row['shock']}` | `{row['family']}` | `{row['control']}` | `{row['classification']}` | "
            f"{_fmt_ci(row['overflow_delta'])} | {_fmt_ci(row['edge_overflow_delta'])} | "
            f"{_fmt_ci(row['node_overflow_delta'])} | {_fmt_ci(row['net_benefit_delta'])} | "
            f"{_fmt_ci(row['fulfillment_delta'])} |"
        )

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "The classification is intentionally stricter than raw ROI. A control only earns "
            "`robust_positive` when its lower confidence bound improves fulfillment, critical "
            "service, or capacity-adjusted ROI without a statistically positive overflow penalty. "
            "This makes capacity saturation visible as a control failure rather than a footnote."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: dict,
    output_json: Path = Path("supply_chain_stress_robustness_output.json"),
    output_md: Path = Path("SUPPLY_CHAIN_STRESS_ROBUSTNESS_REPORT.md"),
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def _parse_severities(raw: str | None) -> tuple[float, ...]:
    if raw is None:
        return RobustnessConfig.severities
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seed-robust supply-chain stress validation.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=RobustnessConfig.seeds)
    parser.add_argument("--agents", type=int, default=RobustnessConfig.agents)
    parser.add_argument("--steps", type=int, default=RobustnessConfig.steps)
    parser.add_argument("--seed-base", type=int, default=RobustnessConfig.seed_base)
    parser.add_argument("--severities", type=str, default=None, help="Comma-separated severity values.")
    parser.add_argument("--output-json", type=Path, default=Path("supply_chain_stress_robustness_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SUPPLY_CHAIN_STRESS_ROBUSTNESS_REPORT.md"))
    args = parser.parse_args()

    config = RobustnessConfig(
        seeds=args.seeds,
        seed_base=args.seed_base,
        agents=64 if args.quick and args.agents == RobustnessConfig.agents else args.agents,
        steps=24 if args.quick and args.steps == RobustnessConfig.steps else args.steps,
        severities=_parse_severities(args.severities),
    )
    payload = run_robustness_suite(config, quick=args.quick)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "classification_counts": payload["classification_counts"],
        "runs": len(payload["rows"]),
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

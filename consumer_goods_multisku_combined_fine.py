"""
Fine combined-stress frontier for the multi-SKU consumer-goods topology.

This narrows around the coarse frontier's only recommendable cell:
combined severity 2.00 with north_south_rebalance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consumer_goods_multisku_frontier import run_frontier


def _band(values: list[float]) -> dict:
    if not values:
        return {"start": None, "end": None, "count": 0}
    return {"start": min(values), "end": max(values), "count": len(set(values))}


def run_combined_fine_frontier(
    severities: tuple[float, ...] = (1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00),
    seeds: tuple[int, ...] = (
        20260617,
        20260618,
        20260619,
        20260620,
        20260621,
        20260622,
        20260623,
    ),
    agents: int = 160,
    steps: int = 44,
) -> dict:
    payload = run_frontier(
        families=("combined",),
        severities=severities,
        seeds=seeds,
        agents=agents,
        steps=steps,
    )
    best_rows = [
        payload["best_by_family_severity"][f"combined:{severity:.2f}"]
        for severity in severities
    ]
    recommendable = [
        row for row in payload["recommendable_controls"]
        if row["family"] == "combined"
    ]
    useful = [
        row for row in payload["useful_controls"]
        if row["family"] == "combined"
    ]
    score_positive = [
        row["severity"] for row in best_rows
        if row["mean_score"] >= 0.0
    ]
    payload["fine_summary"] = {
        "recommendable_band": _band([row["severity"] for row in recommendable]),
        "useful_band": _band([row["severity"] for row in useful]),
        "best_score_positive_band": _band(score_positive),
        "first_negative_best_score_severity": next(
            (row["severity"] for row in best_rows if row["mean_score"] < 0.0),
            None,
        ),
        "best_rows": best_rows,
    }
    return payload


def render_report(payload: dict) -> str:
    summary = payload["fine_summary"]
    rec_band = summary["recommendable_band"]
    useful_band = summary["useful_band"]
    positive_band = summary["best_score_positive_band"]
    lines = [
        "# Consumer Goods Multi-SKU Combined Fine Frontier",
        "",
        "## Scope",
        "",
        (
            "Fine sweep around the coarse frontier's combined-stress stability island, "
            "with more seeds and narrower severity spacing."
        ),
        "",
        f"- Severities: `{', '.join(f'{item:.2f}' for item in payload['config']['severities'])}`",
        f"- Seeds: `{', '.join(str(item) for item in payload['config']['seeds'])}`",
        f"- Agents: `{payload['config']['agents']}`",
        f"- Steps: `{payload['config']['steps']}`",
        f"- Runs: `{len(payload['rows'])}`",
        "",
        "## Stability Bands",
        "",
        "| Band | Start | End | Count |",
        "|---|---:|---:|---:|",
        f"| Recommendable controls | {_fmt_band_value(rec_band['start'])} | {_fmt_band_value(rec_band['end'])} | {rec_band['count']} |",
        f"| Useful non-backfire controls | {_fmt_band_value(useful_band['start'])} | {_fmt_band_value(useful_band['end'])} | {useful_band['count']} |",
        f"| Best-control score-positive | {_fmt_band_value(positive_band['start'])} | {_fmt_band_value(positive_band['end'])} | {positive_band['count']} |",
        "",
        "## Best Control By Severity",
        "",
        "| Severity | Best Control | Viable | Partial | Backfire | Service | Priority | Overflow | Starvation | Gate Capacity | Score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["best_rows"]:
        lines.append(
            f"| {row['severity']:.2f} | `{row['control']}` | "
            f"{row['viable_rate']:.1%} | {row['partial_rate']:.1%} | {row['backfire_rate']:.1%} | "
            f"{row['mean_service']:.1%} | {row['mean_priority']:.1%} | {row['mean_overflow']:.1%} | "
            f"{row['mean_starvation']:.1%} | {row['mean_gate_capacity']:.1%} | {row['mean_score']:.4f} |"
        )

    lines.extend([
        "",
        "## Recommendable Controls",
        "",
        "| Severity | Control | Viable+Partial | Backfire | Service | Priority | Overflow | Score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["recommendable_controls"]:
        lines.append(
            f"| {row['severity']:.2f} | `{row['control']}` | "
            f"{(row['viable_rate'] + row['partial_rate']):.1%} | {row['backfire_rate']:.1%} | "
            f"{row['mean_service']:.1%} | {row['mean_priority']:.1%} | "
            f"{row['mean_overflow']:.1%} | {row['mean_score']:.4f} |"
        )
    if not payload["recommendable_controls"]:
        lines.append("| none | none | none | none | none | none | none | none |")

    lines.extend([
        "",
        "## Reading",
        "",
        (
            "This report tests whether the coarse-grid recommendable cell is a stable "
            "island or a grid artifact. The recommendable band requires useful "
            "non-backfire behavior and non-negative score after service, priority, "
            "substitution, lost-demand, overflow, gate, and cost penalties."
        ),
    ])
    return "\n".join(lines) + "\n"


def _fmt_band_value(value: float | None) -> str:
    return "none" if value is None else f"{value:.2f}"


def write_outputs(payload: dict, output_json: Path, output_md: Path) -> None:
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("consumer_goods_multisku_combined_fine_output.json"))
    parser.add_argument("--output-md", type=Path, default=Path("CONSUMER_GOODS_MULTISKU_COMBINED_FINE_REPORT.md"))
    args = parser.parse_args()
    payload = run_combined_fine_frontier(
        severities=(1.50, 2.00, 2.50) if args.quick else (1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00),
        seeds=(20260617, 20260618) if args.quick else (
            20260617,
            20260618,
            20260619,
            20260620,
            20260621,
            20260622,
            20260623,
        ),
        agents=72 if args.quick else 160,
        steps=24 if args.quick else 44,
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({
        "runs": len(payload["rows"]),
        "recommendable_controls": len(payload["recommendable_controls"]),
        "recommendable_band": payload["fine_summary"]["recommendable_band"],
        "report": str(args.output_md),
    }, indent=2))


if __name__ == "__main__":
    main()

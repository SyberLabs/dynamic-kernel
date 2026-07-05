"""
Generate publication-facing Neural V2 seed-validation artifacts.

The figure is written as SVG using only the standard library so the paper
pipeline does not acquire a plotting dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "neural_v2_seed_validation_output.json"
FIGURE_DIR = ROOT / "figures"
SVG_PATH = FIGURE_DIR / "neural_v2_seed_validation_delta.svg"
REPORT_PATH = ROOT / "NEURAL_V2_SEED_VALIDATION_FIGURE_REPORT.md"


ROUTER_LABELS = {
    "dte_exp3": "DTE-EXP3",
    "dte_reliability_arbitrated_ucb": "Rel-UCB DTE",
    "exp3": "EXP3",
    "ucb": "UCB",
    "hard_dte_exp3": "DTE-EXP3",
    "hard_dte_reliability_arbitrated_exp3": "Gated DTE-EXP3",
    "hard_dte_reliability_arbitrated_ucb": "Rel-UCB DTE",
    "hard_exp3": "EXP3",
    "hard_ucb": "UCB",
}

REGIME_LABELS = {
    "clean": "Clean",
    "hard": "Corrupted/delayed",
    "adversarial_switch": "Adversarial switching",
}

REGIME_ORDER = ["clean", "hard", "adversarial_switch"]
ROUTER_ORDER = [
    "dte_reliability_arbitrated_ucb",
    "dte_exp3",
    "hard_dte_reliability_arbitrated_ucb",
    "hard_dte_exp3",
    "hard_dte_reliability_arbitrated_exp3",
    "ucb",
    "exp3",
    "hard_ucb",
    "hard_exp3",
]


def _load_payload(path: Path = INPUT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _figure_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for regime in REGIME_ORDER:
        regime_rows = [row for row in payload["summary"] if row["regime"] == regime]
        order = {router: idx for idx, router in enumerate(ROUTER_ORDER)}
        for row in sorted(regime_rows, key=lambda item: order.get(item["router"], 999)):
            if row["paired_delta_vs_local"] == 0.0:
                continue
            rows.append(row)
    return rows


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(payload: dict[str, Any]) -> str:
    rows = _figure_rows(payload)
    ticks = [-0.02, 0.0, 0.04, 0.08, 0.12, 0.16]
    min_x = min(
        row["paired_delta_vs_local"] - row["ci95_delta_vs_local"] for row in rows
    )
    max_x = max(
        row["paired_delta_vs_local"] + row["ci95_delta_vs_local"] for row in rows
    )
    min_x = min(min_x, min(ticks) - 0.005)
    max_x = max(max_x, max(ticks) + 0.005)
    width = 980
    left = 235
    right = 45
    top = 72
    row_h = 30
    gap_h = 20
    axis_h = 56
    plot_w = width - left - right
    height = top + axis_h + len(rows) * row_h + 2 * gap_h + 54

    def x_pos(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_w

    zero = x_pos(0.0)
    colors = {
        "clean": "#2563eb",
        "hard": "#b91c1c",
        "adversarial_switch": "#047857",
    }
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #111827; }",
        ".title { font-size: 20px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #4b5563; }",
        ".label { font-size: 12px; }",
        ".regime { font-size: 13px; font-weight: 700; }",
        ".tick { font-size: 11px; fill: #4b5563; }",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" class="title">Neural V2 policy-lane boundary</text>',
        (
            '<text x="24" y="52" class="subtitle">Bars show paired regret '
            "reduction versus local-regret DTE; whiskers show 95% CI.</text>"
        ),
    ]

    axis_y = top - 14
    lines.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{left + plot_w}" y2="{axis_y}" '
        'stroke="#d1d5db" stroke-width="1"/>'
    )
    for tick in ticks:
        x = x_pos(tick)
        lines.append(
            f'<line x1="{x:.1f}" y1="{axis_y - 5}" x2="{x:.1f}" '
            f'y2="{height - 42}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{axis_y - 11}" text-anchor="middle" '
            f'class="tick">{tick:.2f}</text>'
        )
    lines.append(
        f'<line x1="{zero:.1f}" y1="{axis_y - 5}" x2="{zero:.1f}" '
        f'y2="{height - 42}" stroke="#111827" stroke-width="1.4"/>'
    )

    y = top + 12
    previous_regime = None
    for row in rows:
        regime = row["regime"]
        if previous_regime != regime:
            if previous_regime is not None:
                y += gap_h
            lines.append(
                f'<text x="24" y="{y - 8}" class="regime">'
                f"{_svg_escape(REGIME_LABELS[regime])}</text>"
            )
            previous_regime = regime
        label = ROUTER_LABELS.get(row["router"], row["router"])
        value = row["paired_delta_vs_local"]
        ci = row["ci95_delta_vs_local"]
        x0 = x_pos(0.0)
        x1 = x_pos(value)
        bar_x = min(x0, x1)
        bar_w = abs(x1 - x0)
        color = colors[regime]
        cy = y + 8
        ci_l = x_pos(value - ci)
        ci_r = x_pos(value + ci)
        lines.extend(
            [
                f'<text x="42" y="{cy + 4}" class="label">{_svg_escape(label)}</text>',
                (
                    f'<rect x="{bar_x:.1f}" y="{cy - 8}" width="{bar_w:.1f}" '
                    f'height="16" rx="2" fill="{color}" opacity="0.82"/>'
                ),
                (
                    f'<line x1="{ci_l:.1f}" y1="{cy}" x2="{ci_r:.1f}" y2="{cy}" '
                    'stroke="#111827" stroke-width="1.2"/>'
                ),
                (
                    f'<line x1="{ci_l:.1f}" y1="{cy - 5}" x2="{ci_l:.1f}" '
                    f'y2="{cy + 5}" stroke="#111827" stroke-width="1.2"/>'
                ),
                (
                    f'<line x1="{ci_r:.1f}" y1="{cy - 5}" x2="{ci_r:.1f}" '
                    f'y2="{cy + 5}" stroke="#111827" stroke-width="1.2"/>'
                ),
                (
                    f'<text x="{left + plot_w + 8}" y="{cy + 4}" class="tick">'
                    f"{value:+.3f}</text>"
                ),
            ]
        )
        y += row_h

    lines.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 16}" '
        'text-anchor="middle" class="subtitle">Paired regret reduction versus '
        "local-regret DTE</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines)


def render_report(payload: dict[str, Any]) -> str:
    rows = _figure_rows(payload)
    best = max(rows, key=lambda row: row["paired_delta_vs_local"])
    worst = min(rows, key=lambda row: row["paired_delta_vs_local"])
    return "\n".join(
        [
            "# Neural V2 Seed Validation Figure Report",
            "",
            "## Artifact",
            "",
            f"- SVG: `figures/{SVG_PATH.name}`",
            "",
            "## Reading",
            "",
            "The figure plots paired regret reduction relative to local-regret DTE.",
            "Positive values improve on local-regret DTE; negative values are worse.",
            "",
            (
                "Best paired improvement: "
                f"`{best['regime']}` / `{best['router']}` = "
                f"{best['paired_delta_vs_local']:.4f} "
                f"+/- {best['ci95_delta_vs_local']:.4f}."
            ),
            (
                "Worst paired change: "
                f"`{worst['regime']}` / `{worst['router']}` = "
                f"{worst['paired_delta_vs_local']:.4f} "
                f"+/- {worst['ci95_delta_vs_local']:.4f}."
            ),
            "",
            "## Paper Use",
            "",
            "Use this as the Neural V2 regime-boundary figure. It visually supports",
            "the paper claim that DTE-native policy lanes are regime-sensitive:",
            "DTE-EXP3 is useful under adversarial switching, reliability-UCB is the",
            "safer hard-regime default, and external policy owners still dominate",
            "when the task reduces to contextual bandit selection.",
            "",
        ]
    )


def generate_seed_validation_figure(
    input_path: Path = INPUT_PATH,
    svg_path: Path = SVG_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, str]:
    payload = _load_payload(input_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(render_svg(payload), encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"svg": str(svg_path), "report": str(report_path)}


if __name__ == "__main__":
    result = generate_seed_validation_figure()
    print(json.dumps(result, indent=2))

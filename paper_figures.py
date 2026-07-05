"""
Generate paper-facing SVG figures from frozen DTE experiment outputs.

The generator intentionally uses only the Python standard library. Figures are
simple but reproducible and suitable for manuscript drafting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
FALSIFICATION_PATH = ROOT / "semiconductor_onshoring_falsification_output.json"
MODEL_BENCHMARK_PATH = ROOT / "semiconductor_onshoring_model_benchmark_output.json"
REPORT_PATH = ROOT / "PAPER_FIGURE_GENERATION_REPORT.md"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _base_svg(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #111827; }",
        ".title { font-size: 20px; font-weight: 700; }",
        ".subtitle { font-size: 12px; fill: #4b5563; }",
        ".axis { font-size: 11px; fill: #4b5563; }",
        ".label { font-size: 12px; }",
        ".small { font-size: 10px; fill: #4b5563; }",
        ".node { font-size: 11px; font-weight: 700; }",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" class="title">{_esc(title)}</text>',
    ]
    if subtitle:
        lines.append(f'<text x="24" y="50" class="subtitle">{_esc(subtitle)}</text>')
    return lines


def _close_svg(lines: list[str]) -> str:
    return "\n".join(lines + ["</svg>"])


def _line_plot(
    rows: list[dict[str, Any]],
    group_key: str,
    x_key: str,
    y_key: str,
    title: str,
    subtitle: str,
    y_label: str,
    colors: dict[str, str],
) -> str:
    width, height = 880, 430
    left, right, top, bottom = 76, 190, 76, 56
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = sorted({float(row[x_key]) for row in rows})
    y_values = [float(row[y_key]) for row in rows]
    y_min = min(0.0, min(y_values) - 0.04)
    y_max = max(1.0 if "rate" in y_key else max(y_values) + 0.04, max(y_values) + 0.04)
    if y_key == "mean_onshore_share":
        y_min = min(y_min, 0.45)
        y_max = max(y_max, 0.78)

    def x_pos(x: float) -> float:
        if len(xs) == 1:
            return left + plot_w / 2
        return left + (x - min(xs)) / (max(xs) - min(xs)) * plot_w

    def y_pos(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    lines = _base_svg(width, height, title, subtitle)
    lines.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#9ca3af"/>'
    )
    lines.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + plot_h}" stroke="#9ca3af"/>'
    )
    for tick in xs:
        x = x_pos(tick)
        lines.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" '
            f'y2="{top + plot_h + 5}" stroke="#9ca3af"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle" '
            f'class="axis">{tick:g}</text>'
        )
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yv = y_min + frac * (y_max - y_min)
        y = y_pos(yv)
        lines.append(
            f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" '
            'stroke="#9ca3af"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="axis">{yv:.2f}</text>'
        )
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" '
            f'y2="{y:.1f}" stroke="#f3f4f6"/>'
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)
    legend_y = top + 10
    for idx, (group, group_rows) in enumerate(sorted(grouped.items())):
        group_rows = sorted(group_rows, key=lambda row: float(row[x_key]))
        color = colors.get(group, "#374151")
        points = [(x_pos(float(row[x_key])), y_pos(float(row[y_key]))) for row in group_rows]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(
            f'<polyline points="{path}" fill="none" stroke="{color}" '
            'stroke-width="2.4"/>'
        )
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        lines.append(
            f'<rect x="{left + plot_w + 28}" y="{legend_y + idx * 22 - 9}" '
            f'width="12" height="12" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{left + plot_w + 48}" y="{legend_y + idx * 22 + 2}" '
            f'class="label">{_esc(group.replace("_", " "))}</text>'
        )
    lines.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" '
        'text-anchor="middle" class="subtitle">Intervention budget / feedback rate</text>'
    )
    lines.append(
        f'<text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 {top + plot_h / 2:.1f})" '
        f'text-anchor="middle" class="subtitle">{_esc(y_label)}</text>'
    )
    return _close_svg(lines)


def _bar_chart_model(summary: dict[str, Any]) -> str:
    width, height = 760, 390
    left, right, top, bottom = 76, 34, 74, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    bars = [
        ("DTE robust", summary["dte_robust_cells"], "#047857"),
        ("Frozen robust", summary["frozen_robust_cells"], "#2563eb"),
        ("Feedback changes", summary["feedback_changes_robust_class"], "#7c3aed"),
        ("Static false majority", summary["static_majority_but_dte_not_robust"], "#b91c1c"),
    ]
    max_v = max(v for _, v, _ in bars)
    lines = _base_svg(
        width,
        height,
        "Semiconductor model benchmark",
        "Static expected flow overstates robust transitions relative to DTE.",
    )
    lines.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#9ca3af"/>'
    )
    bar_w = plot_w / len(bars) * 0.58
    for idx, (label, value, color) in enumerate(bars):
        cx = left + (idx + 0.5) * plot_w / len(bars)
        h = value / max_v * plot_h
        x = cx - bar_w / 2
        y = top + plot_h - h
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
            f'fill="{color}" rx="3"/>'
        )
        lines.append(
            f'<text x="{cx:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
            f'class="label">{value}</text>'
        )
        lines.append(
            f'<text x="{cx:.1f}" y="{top + plot_h + 20}" text-anchor="middle" '
            f'class="axis">{_esc(label)}</text>'
        )
    lines.append(
        f'<text x="{left - 10}" y="{top + plot_h + 4}" text-anchor="end" class="axis">0</text>'
    )
    lines.append(
        f'<text x="{left - 10}" y="{top + 4}" text-anchor="end" class="axis">{max_v}</text>'
    )
    return _close_svg(lines)


def _topology_null_scatter(rows: list[dict[str, Any]]) -> str:
    width, height = 810, 430
    left, right, top, bottom = 80, 175, 76, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = 0.68, 0.92
    y_min, y_max = 0.68, 1.02

    def x_pos(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def y_pos(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    lines = _base_svg(
        width,
        height,
        "Topology nulls: share is not feasibility",
        "Some randomized topologies show high onshore share without robust completion.",
    )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
        'fill="#f9fafb" stroke="#d1d5db"/>'
    )
    for x_tick in [0.70, 0.80, 0.90]:
        x = x_pos(x_tick)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle" class="axis">{x_tick:.2f}</text>')
    for y_tick in [0.70, 0.80, 0.90, 1.00]:
        y = y_pos(y_tick)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" class="axis">{y_tick:.2f}</text>')
    for row in rows:
        robust = row["viable_rate"] >= 2 / 3
        pull = float(row["domestic_pull"])
        color = "#047857" if robust else "#b91c1c"
        stroke = "#111827" if pull > 0 else "#ffffff"
        r = 6 if pull > 0 else 4.5
        lines.append(
            f'<circle cx="{x_pos(float(row["mean_onshore_share"])):.1f}" '
            f'cy="{y_pos(float(row["mean_completion"])):.1f}" r="{r}" '
            f'fill="{color}" stroke="{stroke}" stroke-width="1.4" opacity="0.86"/>'
        )
    lines.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" class="subtitle">Mean onshore share</text>',
            f'<text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 {top + plot_h / 2:.1f})" text-anchor="middle" class="subtitle">Mean completion</text>',
            f'<circle cx="{left + plot_w + 34}" cy="{top + 20}" r="5" fill="#047857"/>',
            f'<text x="{left + plot_w + 48}" y="{top + 24}" class="label">Robust</text>',
            f'<circle cx="{left + plot_w + 34}" cy="{top + 44}" r="5" fill="#b91c1c"/>',
            f'<text x="{left + plot_w + 48}" y="{top + 48}" class="label">Not robust</text>',
            f'<circle cx="{left + plot_w + 34}" cy="{top + 68}" r="6" fill="#9ca3af" stroke="#111827" stroke-width="1.4"/>',
            f'<text x="{left + plot_w + 48}" y="{top + 72}" class="label">Pull enabled</text>',
        ]
    )
    return _close_svg(lines)


def _dte_mechanism_schematic() -> str:
    width, height = 940, 470
    lines = _base_svg(
        width,
        height,
        "Dynamic Topology Engine mechanism",
        "Telemetry-aligned softmax routing couples movement, state feedback, and memory updates.",
    )
    lines.append(
        '<defs><marker id="arrow0" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#374151"/></marker></defs>'
    )
    boxes = [
        ("Graph G=(V,E)\nD, node features N", 55, 140, "#e0f2fe"),
        ("Agent state\nX_t, a_t", 235, 140, "#dcfce7"),
        ("Dynamic weights\nW_ij = alpha D_ij\n- beta_ij A_j - S_ij", 405, 118, "#fef3c7"),
        ("Softmax routing\nP_ij proportional to\nexp(-W_ij/tau)", 610, 118, "#ede9fe"),
        ("Move + observe\nX_{t+1}, reward,\ntraffic", 780, 140, "#fee2e2"),
    ]
    for text, x, y, fill in boxes:
        lines.append(
            f'<rect x="{x}" y="{y}" width="130" height="96" rx="8" fill="{fill}" stroke="#374151"/>'
        )
        for idx, part in enumerate(text.split("\n")):
            cls = "node" if idx == 0 else "small"
            lines.append(
                f'<text x="{x + 65}" y="{y + 25 + idx * 16}" text-anchor="middle" class="{cls}">{_esc(part)}</text>'
            )
    for x1, y1, x2, y2 in [
        (185, 188, 235, 188),
        (365, 188, 405, 188),
        (535, 188, 610, 188),
        (740, 188, 780, 188),
    ]:
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#374151" stroke-width="1.8" marker-end="url(#arrow0)"/>'
        )
    lines.append(
        '<rect x="250" y="315" width="190" height="72" rx="8" fill="#f9fafb" stroke="#374151"/>'
    )
    lines.append(
        '<text x="345" y="343" text-anchor="middle" class="node">Telemetry feedback</text>'
    )
    lines.append(
        '<text x="345" y="365" text-anchor="middle" class="small">a_{t+1} = norm((1-lambda)a_t + lambda N_j + eps)</text>'
    )
    lines.append(
        '<rect x="515" y="315" width="190" height="72" rx="8" fill="#f9fafb" stroke="#374151"/>'
    )
    lines.append(
        '<text x="610" y="343" text-anchor="middle" class="node">Preference memory</text>'
    )
    lines.append(
        '<text x="610" y="365" text-anchor="middle" class="small">beta, friction deltas, policy lanes</text>'
    )
    lines.append(
        '<path d="M845 236 C820 304 690 320 610 315" fill="none" stroke="#6b7280" stroke-width="1.8" marker-end="url(#arrow0)"/>'
    )
    lines.append(
        '<path d="M790 236 C700 292 455 300 345 315" fill="none" stroke="#6b7280" stroke-width="1.8" marker-end="url(#arrow0)"/>'
    )
    lines.append(
        '<path d="M610 315 C560 270 520 238 505 218" fill="none" stroke="#6b7280" stroke-width="1.8" marker-end="url(#arrow0)"/>'
    )
    return _close_svg(lines)


def _feasibility_allocation_schematic() -> str:
    width, height = 900, 430
    lines = _base_svg(
        width,
        height,
        "Feasibility, allocation, and adaptation",
        "A route can be attractive, selected, and still fail as a completed viable transition.",
    )
    lines.append(
        '<defs><marker id="arrow3" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#374151"/></marker></defs>'
    )
    stages = [
        ("Attractiveness\nroute looks good", 70, 145, "#e0f2fe"),
        ("Allocation\nagents choose route", 275, 145, "#dcfce7"),
        ("Feasibility\nBOM, capacity,\nservice gates", 480, 126, "#fef3c7"),
        ("Completed transition\nfinished lots,\nrobust share", 685, 126, "#ede9fe"),
    ]
    for text, x, y, fill in stages:
        lines.append(
            f'<rect x="{x}" y="{y}" width="145" height="95" rx="8" fill="{fill}" stroke="#374151"/>'
        )
        for idx, part in enumerate(text.split("\n")):
            cls = "node" if idx == 0 else "small"
            lines.append(
                f'<text x="{x + 72.5}" y="{y + 27 + idx * 16}" text-anchor="middle" class="{cls}">{_esc(part)}</text>'
            )
    for x1, y1, x2, y2 in [(215, 192, 275, 192), (420, 192, 480, 192), (625, 192, 685, 192)]:
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#374151" stroke-width="1.8" marker-end="url(#arrow3)"/>'
        )
    lines.append(
        '<path d="M757 221 C720 320 275 320 168 221" fill="none" stroke="#6b7280" stroke-width="1.8" marker-end="url(#arrow3)"/>'
    )
    lines.append(
        '<text x="460" y="335" text-anchor="middle" class="subtitle">Adaptation: completed or blocked outcomes reshape future telemetry and preference memory.</text>'
    )
    lines.append(
        '<text x="460" y="372" text-anchor="middle" class="subtitle">Paper boundary: route share is not evidence of institutional viability without completion.</text>'
    )
    return _close_svg(lines)


def _layered_memory_schematic() -> str:
    width, height = 920, 460
    lines = _base_svg(
        width,
        height,
        "DTE layered-memory state",
        "The process is Markov only after structural, preference, state, ecology, and delayed buffers are included.",
    )
    boxes = [
        ("Position / occupancy\nX_t", 50, 150, "#e0f2fe"),
        ("State memory\nA_t telemetry", 220, 150, "#dcfce7"),
        ("Structural memory\nM_s topology, gates,\ncontracts, capacity", 390, 132, "#fef3c7"),
        ("Preference memory\nM_p beta, S,\npolicy lanes", 590, 132, "#ede9fe"),
        ("Delayed buffers\nB_t pending rewards,\nattribution records", 745, 150, "#fee2e2"),
    ]
    for text, x, y, fill in boxes:
        lines.append(
            f'<rect x="{x}" y="{y}" width="135" height="86" rx="8" fill="{fill}" stroke="#374151"/>'
        )
        for idx, part in enumerate(text.split("\n")):
            cls = "node" if idx == 0 else "small"
            lines.append(
                f'<text x="{x + 67.5}" y="{y + 26 + idx * 16}" text-anchor="middle" class="{cls}">{_esc(part)}</text>'
            )
    arrows = [
        (185, 193, 220, 193),
        (355, 193, 390, 193),
        (525, 193, 590, 193),
        (725, 193, 745, 193),
    ]
    lines.append(
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#374151"/></marker></defs>'
    )
    for x1, y1, x2, y2 in arrows:
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#374151" stroke-width="1.8" marker-end="url(#arrow)"/>'
        )
    lines.append(
        '<rect x="275" y="302" width="370" height="72" rx="8" fill="#f9fafb" stroke="#374151"/>'
    )
    lines.append(
        '<text x="460" y="330" text-anchor="middle" class="node">Ecology R_t</text>'
    )
    lines.append(
        '<text x="460" y="352" text-anchor="middle" class="small">reward, stock, demand, hazard, service capacity</text>'
    )
    lines.append(
        '<line x1="460" y1="302" x2="460" y2="242" stroke="#374151" stroke-width="1.8" marker-end="url(#arrow)"/>'
    )
    lines.append(
        '<path d="M812 236 C812 296 108 296 108 236" fill="none" stroke="#6b7280" stroke-width="1.8" marker-end="url(#arrow)"/>'
    )
    lines.append(
        '<text x="460" y="408" text-anchor="middle" class="subtitle">Z_t = (X_t, A_t, M_s(t), M_p(t), B_t, R_t)</text>'
    )
    return _close_svg(lines)


def _semiconductor_topology_schematic() -> str:
    width, height = 980, 520
    lines = _base_svg(
        width,
        height,
        "Semiconductor topology abstraction",
        "Interventions only reroute flow where the graph still has a real choice point.",
    )
    lines.append(
        '<defs><marker id="arrow2" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#374151"/></marker></defs>'
    )
    nodes = {
        "Allocation": (70, 220, "#e0f2fe"),
        "US demand": (215, 110, "#dbeafe"),
        "China demand": (215, 330, "#fee2e2"),
        "US fabs": (390, 95, "#dcfce7"),
        "US gates": (555, 95, "#fef3c7"),
        "US finished": (725, 110, "#dcfce7"),
        "Taiwan fabs": (390, 275, "#ede9fe"),
        "Taiwan OSAT": (555, 275, "#ede9fe"),
        "Export review": (720, 275, "#fee2e2"),
        "Shipping/port": (835, 360, "#f3f4f6"),
        "Inputs": (390, 430, "#fef3c7"),
        "Reserve": (725, 430, "#fef3c7"),
    }
    edges = [
        ("Allocation", "US demand"),
        ("Allocation", "China demand"),
        ("US demand", "US fabs"),
        ("US demand", "Taiwan fabs"),
        ("China demand", "Taiwan fabs"),
        ("US fabs", "US gates"),
        ("Inputs", "US gates"),
        ("US gates", "US finished"),
        ("Taiwan fabs", "Taiwan OSAT"),
        ("Inputs", "Taiwan OSAT"),
        ("Taiwan OSAT", "Export review"),
        ("Export review", "Shipping/port"),
        ("Shipping/port", "US demand"),
        ("US finished", "US demand"),
        ("Reserve", "US demand"),
    ]
    for a, b in edges:
        x1, y1, _ = nodes[a]
        x2, y2, _ = nodes[b]
        lines.append(
            f'<line x1="{x1 + 55}" y1="{y1 + 28}" x2="{x2 + 5}" y2="{y2 + 28}" '
            'stroke="#6b7280" stroke-width="1.4" marker-end="url(#arrow2)"/>'
        )
    for label, (x, y, fill) in nodes.items():
        lines.append(
            f'<rect x="{x}" y="{y}" width="110" height="56" rx="8" fill="{fill}" stroke="#374151"/>'
        )
        for idx, part in enumerate(label.split(" ")):
            lines.append(
                f'<text x="{x + 55}" y="{y + 24 + idx * 15}" text-anchor="middle" class="node">{_esc(part)}</text>'
            )
    lines.append(
        '<text x="356" y="75" class="small">Upstream route choices: interventions can change allocation here.</text>'
    )
    lines.append(
        '<text x="612" y="258" class="small">Serial offshore corridor: downstream friction is inert unless exits exist.</text>'
    )
    return _close_svg(lines)


def generate_paper_figures() -> dict[str, str]:
    FIGURE_DIR.mkdir(exist_ok=True)
    falsification = _load_json(FALSIFICATION_PATH)
    model = _load_json(MODEL_BENCHMARK_PATH)
    outputs = {
        "dte_mechanism": FIGURE_DIR / "dte_mechanism_schematic.svg",
        "layered_memory": FIGURE_DIR / "dte_layered_memory_schematic.svg",
        "feasibility_allocation": FIGURE_DIR / "feasibility_allocation_adaptation.svg",
        "semiconductor_topology": FIGURE_DIR / "semiconductor_topology_schematic.svg",
        "choice_relocation": FIGURE_DIR / "semiconductor_choice_point_relocation.svg",
        "topology_surgery": FIGURE_DIR / "semiconductor_topology_surgery.svg",
        "feedback_continuum": FIGURE_DIR / "semiconductor_feedback_continuum.svg",
        "model_benchmark": FIGURE_DIR / "semiconductor_model_benchmark.svg",
        "topology_nulls": FIGURE_DIR / "semiconductor_topology_nulls.svg",
    }
    outputs["dte_mechanism"].write_text(_dte_mechanism_schematic(), encoding="utf-8")
    outputs["layered_memory"].write_text(_layered_memory_schematic(), encoding="utf-8")
    outputs["feasibility_allocation"].write_text(
        _feasibility_allocation_schematic(),
        encoding="utf-8",
    )
    outputs["semiconductor_topology"].write_text(_semiconductor_topology_schematic(), encoding="utf-8")
    outputs["choice_relocation"].write_text(
        _line_plot(
            falsification["choice_point_relocation"]["grouped"],
            "location",
            "budget",
            "mean_onshore_share",
            "Choice-point relocation",
            "Equal budgets only move share when applied before route commitment.",
            "Mean onshore share",
            {
                "upstream_choice": "#047857",
                "route_commitment": "#b91c1c",
                "downstream_serial": "#7c3aed",
            },
        ),
        encoding="utf-8",
    )
    outputs["topology_surgery"].write_text(
        _line_plot(
            falsification["topology_surgery"]["grouped"],
            "topology",
            "budget",
            "mean_onshore_share",
            "Topology surgery activates downstream penalties",
            "Adding reconsideration exits creates a choice where the serial corridor had none.",
            "Mean onshore share",
            {"serial": "#b91c1c", "reconsideration_exits": "#047857"},
        ),
        encoding="utf-8",
    )
    outputs["feedback_continuum"].write_text(
        _line_plot(
            [
                {**row, "domestic_pull_label": f"pull={row['domestic_pull']:.1f}"}
                for row in falsification["feedback_continuum"]["grouped"]
            ],
            "domestic_pull_label",
            "feedback_rate",
            "viable_rate",
            "Feedback shapes rather than monotonically improves robustness",
            "Viable-transition rate across feedback rates.",
            "Viable rate",
            {"pull=0.0": "#b91c1c", "pull=1.0": "#047857"},
        ),
        encoding="utf-8",
    )
    outputs["model_benchmark"].write_text(
        _bar_chart_model(model["summary"]),
        encoding="utf-8",
    )
    outputs["topology_nulls"].write_text(
        _topology_null_scatter(falsification["randomized_topology_nulls"]["grouped"]),
        encoding="utf-8",
    )
    report = render_report(outputs, falsification, model)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return {name: str(path) for name, path in outputs.items()}


def render_report(
    outputs: dict[str, Path],
    falsification: dict[str, Any],
    model: dict[str, Any],
) -> str:
    summary = falsification["summary"]
    model_summary = model["summary"]
    relocation = summary["relocation_max_share_lift"]
    surgery = summary["surgery_max_share_lift"]
    best_relocation = max(relocation, key=relocation.get)
    best_surgery = max(surgery, key=surgery.get)
    lines = [
        "# Paper Figure Generation Report",
        "",
        "## Generated Figures",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- `{name}`: `figures/{path.name}`")
    lines.extend(
        [
            "",
            "## Key Readings",
            "",
            (
                "Choice relocation max share lift: "
                f"`{relocation[best_relocation]:.3f}` at `{best_relocation}`."
            ),
            (
                "Topology surgery max share lift: "
                f"`{surgery[best_surgery]:.3f}` for `{best_surgery}`."
            ),
            (
                "Randomized nulls with high share but no robust transition: "
                f"`{summary['nulls_with_high_share_but_no_robust_transition']}`."
            ),
            (
                "Static majority predictions not robust under DTE: "
                f"`{model_summary['static_majority_but_dte_not_robust']}` of "
                f"`{model_summary['static_majority_cells']}`."
            ),
            "",
            "## Manuscript Use",
            "",
            "Use these as draft figures for the DTE mechanism, layered-memory",
            "definition, semiconductor choice-point results, topology surgery,",
            "feedback continuum, model benchmark, and topology-null falsification.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    result = generate_paper_figures()
    print(json.dumps(result, indent=2))

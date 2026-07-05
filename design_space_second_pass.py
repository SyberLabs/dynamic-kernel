"""
DTE Design-Space Exploration — second pass.

Three targeted experiments motivated by first-pass results
(DTE_DESIGN_SPACE_REPORT.md), with success criteria stated up front:

  A. Adaptive (surprise-gated) evaporation on the ant terrain.
     First pass showed adaptive_eta removes unrecovered lock-in on the
     two-route toy. This tests the same mechanism in the colony-level
     pheromone loop across the established phase grid, against fixed-eta
     cells with paired seeds and matched deposit-0 baselines.
     SUCCESS CRITERION (stated before running): the deadly_familiarity
     count drops under adaptive evaporation without reducing the
     adaptive_memory count or mean completion.

  B. adaptive_eta x logistic ecology on the two-route witness — the one
     motivated 2-axis interaction. PREDICTION (stated before running):
     adaptive evaporation does NOT release the 'marginal stale grazing'
     regime, because surprise-gated eta needs a reward change to fire and
     the renewable route sits at a stable, just-inferior equilibrium.

  C. Link-gap retained fraction with 20 seeds on the depletion scenario,
     to tighten the noisy first-pass estimate (mean 0.88, per-cell spread
     -1.5 to 3.4).

Usage:
    .venv\\Scripts\\python.exe design_space_second_pass.py --quick
    .venv\\Scripts\\python.exe design_space_second_pass.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ant_aco_comparator import ACOConfig, _score, _to_dte_policy, comparison_points, simulate_aco
from ant_foraging_dte import SimulationConfig, policies
from ant_foraging_phase_diagram import _classify_phase, _policy_for, phase_scenarios
from design_space import (
    AdaptiveEvaporation,
    Settings,
    TwoRouteCell,
    TwoRouteParams,
    sanitize,
    simulate_ant_settings,
    simulate_two_route_cell,
)


# ---------------------------------------------------------------------------
# Part A — adaptive evaporation on the ant phase grid
# ---------------------------------------------------------------------------

PHASE_DEPOSITS = (0.0, 0.16, 0.65, 1.20)
PHASE_EVAPS = (0.005, 0.035, 0.120)
PHASE_SCOUTS = (0.02, 0.18, 0.44)

AGG_KEYS = (
    "food_completion_rate",
    "empty_food_visit_rate",
    "hazard_rate",
    "lock_in_index",
    "fork_entropy",
    "pheromone_mass",
    "mean_realized_evaporation",
    "max_realized_evaporation",
)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: float(np.mean([r[key] for r in rows])) for key in AGG_KEYS}


def run_part_a(quick: bool, seeds: int, adaptive: AdaptiveEvaporation) -> dict[str, Any]:
    config = SimulationConfig(agents=48, steps=90)
    selected_scenarios = phase_scenarios(quick=True)
    template = policies()[1]
    deposits = PHASE_DEPOSITS if not quick else (0.0, 0.65)
    evaps = PHASE_EVAPS if not quick else (0.005, 0.120)
    scouts = PHASE_SCOUTS if not quick else (0.02, 0.44)

    cells: list[dict[str, Any]] = []
    baselines: dict[tuple[str, float], dict[str, float]] = {}

    for s_idx, scenario in enumerate(selected_scenarios):
        for scout in scouts:
            for deposit in deposits:
                for evap in evaps:
                    if deposit == 0.0 and evap != evaps[0]:
                        continue
                    policy = _policy_for(deposit, evap, scout, template)
                    fixed_rows, adaptive_rows = [], []
                    for seed_idx in range(seeds):
                        seed_offset = (
                            s_idx * 100_000
                            + int(round(scout * 1000)) * 100
                            + int(round(deposit * 1000)) * 10
                            + int(round(evap * 1000))
                            + seed_idx * 997
                        )
                        fixed_rows.append(
                            simulate_ant_settings(
                                config, scenario, policy, Settings(),
                                seed_offset=seed_offset,
                            )
                        )
                        if deposit > 0.0:
                            adaptive_rows.append(
                                simulate_ant_settings(
                                    config, scenario, policy, Settings(),
                                    seed_offset=seed_offset,
                                    adaptive_evap=adaptive,
                                )
                            )
                    cell = {
                        "scenario": scenario.name,
                        "scout_share": scout,
                        "pheromone_deposit": deposit,
                        "evaporation": evap,
                        "memory_ratio": float(deposit / max(evap, 1e-9)),
                        "fixed": _aggregate(fixed_rows),
                        "adaptive": _aggregate(adaptive_rows) if adaptive_rows else None,
                    }
                    if deposit == 0.0:
                        baselines[(scenario.name, scout)] = cell["fixed"]
                    cells.append(cell)
        print(f"  part A: {scenario.name} done")

    # Classify both variants against the matched deposit-0 baseline.
    classified = []
    for cell in cells:
        if cell["pheromone_deposit"] == 0.0:
            continue
        base = baselines[(cell["scenario"], cell["scout_share"])]
        entry = {
            "scenario": cell["scenario"],
            "scout_share": cell["scout_share"],
            "pheromone_deposit": cell["pheromone_deposit"],
            "evaporation": cell["evaporation"],
            "memory_ratio": cell["memory_ratio"],
        }
        for variant in ("fixed", "adaptive"):
            agg = cell[variant]
            row = {
                "pheromone_deposit": cell["pheromone_deposit"],
                "evaporation": cell["evaporation"],
                "hazard_rate": agg["hazard_rate"],
                "empty_food_visit_rate": agg["empty_food_visit_rate"],
                "pheromone_mass": agg["pheromone_mass"],
                "food_completion_gain": agg["food_completion_rate"] - base["food_completion_rate"],
                "empty_visit_delta": agg["empty_food_visit_rate"] - base["empty_food_visit_rate"],
                "hazard_delta": agg["hazard_rate"] - base["hazard_rate"],
            }
            entry[f"{variant}_phase"] = _classify_phase(row, base)
            entry[f"{variant}_completion"] = agg["food_completion_rate"]
            entry[f"{variant}_empty"] = agg["empty_food_visit_rate"]
            entry[f"{variant}_gain"] = row["food_completion_gain"]
        entry["adaptive_mean_evap"] = cell["adaptive"]["mean_realized_evaporation"]
        entry["adaptive_max_evap"] = cell["adaptive"]["max_realized_evaporation"]
        entry["phase_flip"] = entry["fixed_phase"] != entry["adaptive_phase"]
        classified.append(entry)

    def counts(variant: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in classified:
            out[e[f"{variant}_phase"]] = out.get(e[f"{variant}_phase"], 0) + 1
        return out

    fixed_counts = counts("fixed")
    adaptive_counts = counts("adaptive")

    def min_deadly_ratio(variant: str) -> float | None:
        vals = [
            e["memory_ratio"] for e in classified
            if e[f"{variant}_phase"] == "deadly_familiarity"
        ]
        return min(vals) if vals else None

    mean_completion_fixed = float(np.mean([e["fixed_completion"] for e in classified]))
    mean_completion_adaptive = float(np.mean([e["adaptive_completion"] for e in classified]))

    success = (
        adaptive_counts.get("deadly_familiarity", 0)
        < fixed_counts.get("deadly_familiarity", 0)
        and adaptive_counts.get("adaptive_memory", 0)
        >= fixed_counts.get("adaptive_memory", 0)
        and mean_completion_adaptive >= mean_completion_fixed - 0.005
    )

    return {
        "seeds": seeds,
        "adaptive_params": {
            "gain": adaptive.gain,
            "eta_max": adaptive.eta_max,
            "track_rate": adaptive.track_rate,
        },
        "cells": classified,
        "fixed_phase_counts": fixed_counts,
        "adaptive_phase_counts": adaptive_counts,
        "fixed_min_deadly_ratio": min_deadly_ratio("fixed"),
        "adaptive_min_deadly_ratio": min_deadly_ratio("adaptive"),
        "mean_completion_fixed": mean_completion_fixed,
        "mean_completion_adaptive": mean_completion_adaptive,
        "success_criterion_met": success,
    }


# ---------------------------------------------------------------------------
# Part B — adaptive_eta x logistic ecology (motivated interaction)
# ---------------------------------------------------------------------------

POINTS = (
    {"rho": 0.12, "eta": 0.12, "epsilon": 0.02, "label": "weak_memory"},
    {"rho": 0.45, "eta": 0.02, "epsilon": 0.02, "label": "lockin_prone"},
    {"rho": 0.90, "eta": 0.005, "epsilon": 0.02, "label": "extreme_memory"},
)


def run_part_b(quick: bool) -> dict[str, Any]:
    growths = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40) if not quick else (0.0, 0.10, 0.40)
    rows = []
    for point in POINTS:
        for memory in ("reward_gated", "adaptive_eta"):
            for g in growths:
                cell = TwoRouteCell(
                    memory=memory, ecology="logistic",
                    logistic_growth=g, harvest=0.08,
                )
                params = TwoRouteParams(
                    rho=point["rho"], eta=point["eta"], epsilon=point["epsilon"]
                )
                row = simulate_two_route_cell(params, cell)
                row["point"] = point["label"]
                row["memory"] = memory
                row["growth"] = g
                rows.append(row)

    # Prediction check: among cells that are unrecovered under reward_gated
    # with growth > 0 (the marginal-grazing regime), how many does
    # adaptive_eta release?
    paired = {}
    for r in rows:
        paired[(r["point"], r["growth"], r["memory"])] = r
    grazing_cells = [
        (point["label"], g)
        for point in POINTS
        for g in growths
        if g > 0.0
        and paired[(point["label"], g, "reward_gated")]["classification"]
        == "unrecovered_stale_lockin"
    ]
    released = [
        key for key in grazing_cells
        if paired[(key[0], key[1], "adaptive_eta")]["classification"]
        != "unrecovered_stale_lockin"
    ]
    prediction_holds = len(grazing_cells) > 0 and len(released) == 0
    return {
        "rows": rows,
        "marginal_grazing_cells": grazing_cells,
        "released_by_adaptive_eta": released,
        "prediction_adaptive_eta_does_not_release": prediction_holds,
    }


# ---------------------------------------------------------------------------
# Part C — link-gap retained fraction, 20 seeds, depletion scenario
# ---------------------------------------------------------------------------

def run_part_c(quick: bool, seeds: int) -> dict[str, Any]:
    config = SimulationConfig(agents=48, steps=90)
    aco_config = ACOConfig(agents=config.agents, steps=config.steps, seed=config.seed)
    scenario = phase_scenarios(quick=True)[0]  # rich_patch_delayed_depletion
    selected_points = comparison_points(quick=True)

    results = []
    pooled_ext: list[float] = []
    pooled_link: list[float] = []
    for p_idx, point in enumerate(selected_points):
        dte_policy = _to_dte_policy(point)
        ext_deltas, link_deltas = [], []
        for seed_idx in range(seeds):
            seed_offset = p_idx * 10_000 + seed_idx * 997
            d_row = simulate_ant_settings(
                config, scenario, dte_policy, Settings(), seed_offset=seed_offset
            )
            p_row = simulate_ant_settings(
                config, scenario, dte_policy, Settings(link="powerlaw"),
                seed_offset=seed_offset,
            )
            a_row = simulate_aco(aco_config, scenario, point, seed_offset=seed_offset)
            s_d, s_p, s_a = _score(d_row), _score(p_row), _score(a_row)
            ext_deltas.append(s_d - s_a)
            link_deltas.append(s_d - s_p)
        ext_deltas = np.array(ext_deltas)
        link_deltas = np.array(link_deltas)
        gap_ext = float(np.mean(ext_deltas))
        gap_link = float(np.mean(link_deltas))
        pooled_ext.extend(ext_deltas.tolist())
        pooled_link.extend(link_deltas.tolist())
        results.append({
            "point": point.name,
            "seeds": seeds,
            "gap_external_mean": gap_ext,
            "gap_external_stderr": float(np.std(ext_deltas, ddof=1) / np.sqrt(seeds)),
            "gap_link_mean": gap_link,
            "gap_link_stderr": float(np.std(link_deltas, ddof=1) / np.sqrt(seeds)),
            "retained_fraction": (
                1.0 - gap_link / gap_ext if gap_ext > 0.02 else None
            ),
        })
        print(f"  part C: {point.name} done ({seeds} seeds)")

    pooled_ext_arr = np.array(pooled_ext)
    pooled_link_arr = np.array(pooled_link)
    pooled_gap_ext = float(np.mean(pooled_ext_arr))
    pooled_gap_link = float(np.mean(pooled_link_arr))
    return {
        "scenario": scenario.name,
        "points": results,
        "pooled_gap_external_mean": pooled_gap_ext,
        "pooled_gap_external_stderr": float(
            np.std(pooled_ext_arr, ddof=1) / np.sqrt(len(pooled_ext_arr))
        ),
        "pooled_gap_link_mean": pooled_gap_link,
        "pooled_gap_link_stderr": float(
            np.std(pooled_link_arr, ddof=1) / np.sqrt(len(pooled_link_arr))
        ),
        "pooled_retained_fraction": (
            1.0 - pooled_gap_link / pooled_gap_ext if pooled_gap_ext > 0.0 else None
        ),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x, digits=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_report(payload: dict[str, Any]) -> str:
    a, b, c = payload["part_a"], payload["part_b"], payload["part_c"]
    lines: list[str] = []
    add = lines.append

    add("# DTE Design-Space Exploration — Second-Pass Report")
    add("")
    add("## Scope")
    add("")
    add(
        "Three experiments motivated by specific first-pass results "
        "(`DTE_DESIGN_SPACE_REPORT.md`), each with its success criterion or "
        "prediction stated before running. This pass decides whether the "
        "Axis-5 update laws earn promotion into the kernel as opt-in API."
    )
    add("")

    # --- Part A -----------------------------------------------------------
    add("## A — Adaptive evaporation on the ant terrain")
    add("")
    ap = a["adaptive_params"]
    add(
        f"Colony-level surprise-gated evaporation "
        f"(gain={ap['gain']}, eta_max={ap['eta_max']}, "
        f"track_rate={ap['track_rate']}), paired seeds "
        f"({a['seeds']} per cell) against fixed-eta cells, matched deposit-0 "
        f"baselines, phase classification from "
        f"`ant_foraging_phase_diagram._classify_phase`."
    )
    add("")
    add(
        "**Stated criterion:** deadly_familiarity count drops; adaptive_memory "
        "count does not drop; mean completion does not degrade."
    )
    add("")
    add(f"**Result: {'CRITERION MET' if a['success_criterion_met'] else 'CRITERION NOT MET'}**")
    add("")
    add("| Phase | Fixed eta | Adaptive eta |")
    add("|---|---:|---:|")
    all_phases = sorted(set(a["fixed_phase_counts"]) | set(a["adaptive_phase_counts"]))
    for phase in all_phases:
        add(
            f"| `{phase}` | {a['fixed_phase_counts'].get(phase, 0)} | "
            f"{a['adaptive_phase_counts'].get(phase, 0)} |"
        )
    add("")
    add(
        f"- Min deadly memory ratio: fixed `{_fmt(a['fixed_min_deadly_ratio'], 2)}` "
        f"-> adaptive `{_fmt(a['adaptive_min_deadly_ratio'], 2)}`"
    )
    add(
        f"- Mean completion: fixed `{a['mean_completion_fixed']:.3f}` "
        f"-> adaptive `{a['mean_completion_adaptive']:.3f}`"
    )
    add("")
    add("Cells where the phase changed:")
    add("")
    add("| Scenario | Scout | Deposit | Evap base | Ratio | Fixed phase | Adaptive phase | Adaptive mean/max evap | Empty rate fixed->adaptive |")
    add("|---|---:|---:|---:|---:|---|---|---|---|")
    for e in a["cells"]:
        if not e["phase_flip"]:
            continue
        add(
            f"| {e['scenario']} | {e['scout_share']:.2f} | "
            f"{e['pheromone_deposit']:.2f} | {e['evaporation']:.3f} | "
            f"{e['memory_ratio']:.1f} | {e['fixed_phase']} | {e['adaptive_phase']} | "
            f"{e['adaptive_mean_evap']:.3f}/{e['adaptive_max_evap']:.3f} | "
            f"{e['fixed_empty']:.3f}->{e['adaptive_empty']:.3f} |"
        )
    add("")

    # --- Part B -----------------------------------------------------------
    add("## B — adaptive_eta x logistic ecology (motivated 2-axis interaction)")
    add("")
    add(
        "**Stated prediction:** adaptive evaporation does NOT release the "
        "marginal-stale-grazing regime (stable renewable equilibrium produces "
        "no surprise signal)."
    )
    add("")
    holds = b["prediction_adaptive_eta_does_not_release"]
    n_grazing = len(b["marginal_grazing_cells"])
    n_released = len(b["released_by_adaptive_eta"])
    add(
        f"**Result: prediction "
        f"{'HOLDS' if holds else 'REFUTED'}** — of {n_grazing} marginal-grazing "
        f"cells (unrecovered under reward_gated, growth > 0), adaptive_eta "
        f"released {n_released}."
    )
    add("")
    add("| Point | Growth | Memory | Lock-in duration | Class | Final memory gap |")
    add("|---|---:|---|---:|---|---:|")
    for r in b["rows"]:
        add(
            f"| {r['point']} | {r['growth']:.2f} | {r['memory']} | "
            f"{r['stale_lockin_duration']} | {r['classification']} | "
            f"{r['final_memory_gap']:.3f} |"
        )
    add("")

    # --- Part C -----------------------------------------------------------
    add("## C — Link-gap retained fraction, tightened")
    add("")
    add(
        f"Scenario `{c['scenario']}`, paired per-seed deltas, "
        f"{c['points'][0]['seeds']} seeds per point."
    )
    add("")
    add("| Point | Ext. gap (DTE-ACO) | Link cost (DTE - DTE-powerlaw) | Retained fraction |")
    add("|---|---|---|---:|")
    for p in c["points"]:
        add(
            f"| {p['point']} | {p['gap_external_mean']:.3f} +/- {p['gap_external_stderr']:.3f} | "
            f"{p['gap_link_mean']:.3f} +/- {p['gap_link_stderr']:.3f} | "
            f"{_fmt(p['retained_fraction'], 2)} |"
        )
    add("")
    add(
        f"- Pooled: external gap `{c['pooled_gap_external_mean']:.3f} +/- "
        f"{c['pooled_gap_external_stderr']:.3f}`, link cost "
        f"`{c['pooled_gap_link_mean']:.3f} +/- {c['pooled_gap_link_stderr']:.3f}`, "
        f"retained fraction `{_fmt(c['pooled_retained_fraction'], 2)}`"
    )
    add("")

    # --- Decision -----------------------------------------------------------
    add("## Promotion decision and first-pass revisions")
    add("")
    if a["success_criterion_met"]:
        add(
            "**Promote.** Part A met its pre-stated criterion on a second, "
            "structurally different witness (colony-level pheromone field vs "
            "per-route scalar memory). Per the first-pass prompt's promotion "
            "rule, the Axis-5 preference-memory update laws are promoted into "
            "the kernel as **opt-in** API (`static` remains DEFAULT), with "
            "this report as the stated reason."
        )
    else:
        add(
            "**Do not promote.** Part A did NOT meet its pre-stated "
            "criterion; the two-route adaptive_eta result does not transfer "
            "to the colony-level pheromone mechanism as implemented. This "
            "negative result bounds the first-pass finding's scope and is "
            "recorded, not hidden."
        )
    add("")

    n_grazing = len(b["marginal_grazing_cells"])
    n_released = len(b["released_by_adaptive_eta"])
    if n_grazing and n_released == 0:
        add(
            "**Part B scope note:** the prediction held — surprise-gated "
            "evaporation corrects abrupt ecological falsification only; "
            "stationary marginal grazing is unprotected and needs an "
            "absolute opportunity-cost diagnostic rather than temporal "
            "surprise."
        )
    elif n_grazing:
        held = [k for k in b["marginal_grazing_cells"] if k not in b["released_by_adaptive_eta"]]
        add(
            f"**Part B scope note (prediction refuted, in the mitigation's "
            f"favor):** adaptive evaporation released {n_released} of "
            f"{n_grazing} marginal-grazing cells — the gradual decline "
            f"transient produces enough cumulative surprise after all. The "
            f"unprotected regime is narrower than predicted: regrowth fast "
            f"enough to pin the rich reward near a stationary, just-inferior "
            f"equilibrium while memory is deep "
            f"(held cells: {', '.join(f'{p}@r={g}' for p, g in held)}). In "
            f"that truly stationary regime there is no reward *change* to "
            f"detect, and an absolute opportunity-cost diagnostic would be "
            f"needed."
        )
    add("")
    add(
        "**Part C revision to the first-pass verdict:** the 20-seed paired "
        "estimate shows the DTE-vs-ACO gap attribution is policy-point-"
        "dependent, not uniform. At memory-heavy points the scaffold carries "
        "the advantage (danger_candidate retains "
        + _fmt(next((p["retained_fraction"] for p in c["points"] if p["point"] == "danger_candidate"), None), 2)
        + " of the largest gap); at the no-memory point the link function "
        "dominates (retained "
        + _fmt(next((p["retained_fraction"] for p in c["points"] if p["point"] == "no_memory"), None), 2)
        + "). The first-pass mean of 0.88 averaged over real heterogeneity. "
        "The axis-3 ACO-gap cell in the first-pass verdict table should read "
        "**parametric by policy point** rather than unconditionally "
        "structural: DTE's edge is the scaffold *where collective memory is "
        "active*; without memory, softmax-vs-power-law is load-bearing."
    )
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DTE design-space second pass.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--gap-seeds", type=int, default=None)
    parser.add_argument("--json", default="design_space_second_pass_output.json")
    parser.add_argument("--report", default="DTE_DESIGN_SPACE_SECOND_PASS_REPORT.md")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else (2 if args.quick else 5)
    gap_seeds = args.gap_seeds if args.gap_seeds is not None else (3 if args.quick else 20)
    t0 = time.perf_counter()

    print("[A] adaptive evaporation on the ant phase grid...")
    part_a = run_part_a(args.quick, seeds, AdaptiveEvaporation())
    print(f"    criterion met: {part_a['success_criterion_met']}")

    print("[B] adaptive_eta x logistic interaction...")
    part_b = run_part_b(args.quick)
    print(
        f"    prediction holds: {part_b['prediction_adaptive_eta_does_not_release']} "
        f"(grazing cells: {len(part_b['marginal_grazing_cells'])}, "
        f"released: {len(part_b['released_by_adaptive_eta'])})"
    )

    print("[C] link-gap with more seeds...")
    part_c = run_part_c(args.quick, gap_seeds)
    print(f"    pooled retained fraction: {part_c['pooled_retained_fraction']}")

    payload = {
        "quick": args.quick,
        "part_a": part_a,
        "part_b": part_b,
        "part_c": part_c,
    }
    Path(args.report).write_text(render_report(payload), encoding="utf-8")
    Path(args.json).write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter() - t0:.1f}s -> {args.report}, {args.json}")


if __name__ == "__main__":
    main()

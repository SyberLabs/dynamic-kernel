"""
Reduce per-cell Social Media DTE robustness outputs into one report.

Usage:
    .venv\\Scripts\\python.exe social_media_reduce.py slurm/results/social_robust_cell_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from social_media_robustness import summarize_replicates, write_outputs


def _cell_key(row: dict) -> tuple:
    cell = row["cell"]
    return (
        cell["intent"],
        float(cell["lambda"]),
        float(cell["tau"]),
        float(cell["sigma"]),
    )


def reduce_payloads(paths: list[Path]) -> dict:
    replicates = []
    source_files = []
    agents = 0
    steps = 0

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_files.append(str(path))
        replicates.extend(payload.get("replicates", []))
        summary = payload.get("summary", {})
        agents = agents or int(summary.get("agents", 0))
        steps = steps or int(summary.get("steps", 0))

    cells = len({_cell_key(row) for row in replicates})
    seeds = int(round(len(replicates) / cells)) if cells else 0
    summary = summarize_replicates(
        replicates=replicates,
        cells=cells,
        seeds=seeds,
        agents=agents,
        steps=steps,
    )
    summary["source_files"] = source_files
    return {"summary": summary, "replicates": replicates}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduce social-media robustness JSON outputs.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("social_media_robustness_reduced.json"))
    parser.add_argument("--output-md", type=Path, default=Path("SOCIAL_MEDIA_ROBUSTNESS_REDUCED.md"))
    args = parser.parse_args()

    payload = reduce_payloads(args.inputs)
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

"""Summarize AlphaPoker experiment metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentMetric:
    name: str
    iterations: int | None
    game_value_p0: float | None
    exploitability: float | None
    infosets: int | None


def load_metrics(experiments_dir: Path) -> list[ExperimentMetric]:
    metrics: list[ExperimentMetric] = []
    for path in sorted(experiments_dir.glob("*/metrics.json")):
        payload: dict[str, Any] = json.loads(path.read_text())
        metrics.append(
            ExperimentMetric(
                name=path.parent.name,
                iterations=payload.get("iterations"),
                game_value_p0=payload.get("game_value_p0"),
                exploitability=payload.get("exploitability"),
                infosets=payload.get("infosets"),
            )
        )
    return metrics


def markdown_table(metrics: list[ExperimentMetric]) -> str:
    lines = [
        "| experiment | iterations | game_value_p0 | exploitability | infosets |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        lines.append(
            "| {name} | {iterations} | {game_value_p0} | {exploitability} | {infosets} |".format(
                name=metric.name,
                iterations="" if metric.iterations is None else metric.iterations,
                game_value_p0=(
                    ""
                    if metric.game_value_p0 is None
                    else f"{metric.game_value_p0:.6f}"
                ),
                exploitability=(
                    ""
                    if metric.exploitability is None
                    else f"{metric.exploitability:.6f}"
                ),
                infosets="" if metric.infosets is None else metric.infosets,
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(markdown_table(load_metrics(args.experiments_dir)))


if __name__ == "__main__":
    main()


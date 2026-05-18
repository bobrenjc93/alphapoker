"""Distill an existing Leduc strategy JSON into a policy/value checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alphapoker.train import write_json
from alphapoker.train_leduc import train_network


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(Path(args.strategy_json).read_text())
    if payload.get("game") != "leduc_poker":
        raise ValueError("Expected a Leduc strategy JSON")

    strategy = payload["strategy"]
    out_dir = Path(args.out)
    metrics: dict[str, Any] = {
        "source_strategy_json": str(args.strategy_json),
        "source_metrics": payload.get("metrics", {}),
    }
    metrics.update(train_network(strategy, out_dir, args.epochs))
    write_json(out_dir / "distill_metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    return parser


def main() -> None:
    metrics = run(build_parser().parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


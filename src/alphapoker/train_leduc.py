"""Train the tabular Leduc CFR baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alphapoker.leduc_cfr import LeducCFRTrainer, LeducTrainingResult
from alphapoker.train import write_json


def result_to_dict(result: LeducTrainingResult) -> dict[str, float | int]:
    payload: dict[str, float | int] = {
        "iterations": result.iterations,
        "game_value_p0": result.game_value_p0,
        "infosets": result.infosets,
    }
    if result.exploitability is not None:
        payload["exploitability"] = result.exploitability
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    trainer = LeducCFRTrainer(cfr_plus=not args.vanilla_cfr)
    result = trainer.train(args.iterations, compute_exploitability=args.best_response)
    strategy = trainer.average_strategy()

    out_dir = Path(args.out)
    metrics: dict[str, Any] = result_to_dict(result)
    write_json(
        out_dir / "strategy.json",
        {
            "game": "leduc_poker",
            "algorithm": "cfr" if args.vanilla_cfr else "cfr_plus",
            "metrics": metrics,
            "strategy": strategy,
        },
    )
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("experiments/leduc_cfr_smoke"))
    parser.add_argument("--vanilla-cfr", action="store_true")
    parser.add_argument("--best-response", action="store_true")
    return parser


def main() -> None:
    metrics = run(build_parser().parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Sweep threshold policies for a Hold'em equity model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alphapoker.evaluate_holdem_equity_model import run as run_evaluation
from alphapoker.train import write_json


def parse_threshold_configs(configs: str) -> list[tuple[float, float, float]]:
    parsed = []
    for item in configs.split(";"):
        if not item.strip():
            continue
        parts = [float(part) for part in item.split(",")]
        if len(parts) != 3:
            raise ValueError("Each config must be bet,raise,call")
        parsed.append((parts[0], parts[1], parts[2]))
    if not parsed:
        raise ValueError("At least one threshold config is required")
    return parsed


def run(args: argparse.Namespace) -> dict[str, Any]:
    results = []
    for index, (bet_threshold, raise_threshold, call_threshold) in enumerate(
        parse_threshold_configs(args.configs)
    ):
        eval_args = argparse.Namespace(
            checkpoint=args.checkpoint,
            hands=args.hands,
            seed=args.seed,
            opponent_policy=args.opponent_policy,
            equity_sims=args.equity_sims,
            model_player=args.model_player,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_threshold=call_threshold,
            out=None,
        )
        metrics = run_evaluation(eval_args)
        metrics["config_index"] = index
        results.append(metrics)

    best = max(results, key=lambda item: item["avg_utility_model"])
    payload = {
        "checkpoint": str(args.checkpoint),
        "hands": args.hands,
        "seed": args.seed,
        "opponent_policy": args.opponent_policy,
        "equity_sims": args.equity_sims,
        "model_player": args.model_player,
        "best": best,
        "results": results,
    }
    if args.out is not None:
        write_json(Path(args.out), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", choices=["random", "equity"], default="equity")
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--model-player", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--configs",
        default="0.58,0.72,0.36;0.65,0.82,0.42",
        help="Semicolon-separated bet,raise,call threshold triples.",
    )
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

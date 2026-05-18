"""Sweep threshold policies for a Hold'em equity model."""

from __future__ import annotations

import argparse
import json
import math
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


def parse_model_players(value: str) -> tuple[int, ...]:
    if value == "both":
        return (0, 1)
    try:
        model_player = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("model-player must be 0, 1, or both") from error
    if model_player not in (0, 1):
        raise argparse.ArgumentTypeError("model-player must be 0, 1, or both")
    return (model_player,)


def model_player_label(model_players: tuple[int, ...]) -> int | str:
    return "both" if model_players == (0, 1) else model_players[0]


def normalize_model_players(value: int | str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    return parse_model_players(str(value))


def _weighted_mean(metrics: list[dict[str, Any]], key: str) -> float:
    total_hands = sum(int(item["hands"]) for item in metrics)
    return (
        sum(float(item[key]) * int(item["hands"]) for item in metrics) / total_hands
        if total_hands
        else 0.0
    )


def _pooled_stdev(metrics: list[dict[str, Any]], mean_key: str, stdev_key: str) -> float:
    total_hands = sum(int(item["hands"]) for item in metrics)
    if total_hands <= 1:
        return 0.0
    mean = _weighted_mean(metrics, mean_key)
    sum_squares = 0.0
    for item in metrics:
        hands = int(item["hands"])
        stdev = float(item[stdev_key])
        item_mean = float(item[mean_key])
        sum_squares += max(0, hands - 1) * stdev * stdev
        sum_squares += hands * (item_mean - mean) * (item_mean - mean)
    return math.sqrt(sum_squares / (total_hands - 1))


def aggregate_model_player_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if len(metrics) == 1:
        return metrics[0]

    total_hands = sum(int(item["hands"]) for item in metrics)
    model_stdev = _pooled_stdev(metrics, "avg_utility_model", "utility_stdev_model")
    p0_stdev = _pooled_stdev(metrics, "avg_utility_p0", "utility_stdev_p0")
    first = metrics[0]
    return {
        "checkpoint": first["checkpoint"],
        "hands": total_hands,
        "hands_per_model_player": first["hands"],
        "model_player": "both",
        "avg_utility_model": _weighted_mean(metrics, "avg_utility_model"),
        "utility_stdev_model": model_stdev,
        "utility_stderr_model": model_stdev / (total_hands**0.5) if total_hands else 0.0,
        "avg_utility_p0": _weighted_mean(metrics, "avg_utility_p0"),
        "utility_stdev_p0": p0_stdev,
        "utility_stderr_p0": p0_stdev / (total_hands**0.5) if total_hands else 0.0,
        "avg_actions": _weighted_mean(metrics, "avg_actions"),
        "folds": sum(int(item["folds"]) for item in metrics),
        "showdowns": sum(int(item["showdowns"]) for item in metrics),
        "opponent_policy": first["opponent_policy"],
        "equity_sims": first["equity_sims"],
        "bet_threshold": first["bet_threshold"],
        "raise_threshold": first["raise_threshold"],
        "call_threshold": first["call_threshold"],
        "seed": first["seed"],
        "seat_metrics": metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_players = normalize_model_players(args.model_player)
    results = []
    for index, (bet_threshold, raise_threshold, call_threshold) in enumerate(
        parse_threshold_configs(args.configs)
    ):
        seat_results = []
        for model_player in model_players:
            eval_args = argparse.Namespace(
                checkpoint=args.checkpoint,
                hands=args.hands,
                seed=args.seed,
                opponent_policy=args.opponent_policy,
                equity_sims=args.equity_sims,
                model_player=model_player,
                bet_threshold=bet_threshold,
                raise_threshold=raise_threshold,
                call_threshold=call_threshold,
                out=None,
            )
            seat_results.append(run_evaluation(eval_args))
        metrics = aggregate_model_player_metrics(seat_results)
        metrics["config_index"] = index
        results.append(metrics)

    best = max(results, key=lambda item: item["avg_utility_model"])
    payload = {
        "checkpoint": str(args.checkpoint),
        "hands": args.hands,
        "seed": args.seed,
        "opponent_policy": args.opponent_policy,
        "equity_sims": args.equity_sims,
        "model_player": model_player_label(model_players),
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
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
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

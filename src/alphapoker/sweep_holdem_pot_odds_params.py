"""Sweep fixed-limit Hold'em pot-odds policy parameters."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from alphapoker.holdem import pot_odds_equity_policy
from alphapoker.holdem_evaluation import evaluate_policy_match
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json
from alphapoker.train_holdem_policy_gradient import (
    model_player_label,
    normalize_model_players,
    parse_model_players,
)


def parse_param_configs(configs: str) -> list[tuple[float, float, float]]:
    parsed = []
    for item in configs.split(";"):
        if not item.strip():
            continue
        parts = [float(part) for part in item.split(",")]
        if len(parts) != 3:
            raise ValueError("Each config must be bet,raise,call_margin")
        parsed.append((parts[0], parts[1], parts[2]))
    if not parsed:
        raise ValueError("At least one parameter config is required")
    return parsed


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


def aggregate_seat_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if len(metrics) == 1:
        return metrics[0]
    total_hands = sum(int(item["hands"]) for item in metrics)
    model_stdev = _pooled_stdev(metrics, "avg_utility_model", "utility_stdev_model")
    p0_stdev = _pooled_stdev(metrics, "avg_utility_p0", "utility_stdev_p0")
    first = metrics[0]
    return {
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
        "call_margin": first["call_margin"],
        "seed": first["seed"],
        "seat_metrics": metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_players = normalize_model_players(args.model_player)
    results = []
    for index, (bet_threshold, raise_threshold, call_margin) in enumerate(
        parse_param_configs(args.configs)
    ):
        seat_metrics = []
        for model_player in model_players:
            model_rng = random.Random(args.seed + index * 10_003 + model_player)
            opponent_rng = random.Random(args.seed + 1_000_003 + index * 10_003 + model_player)
            metrics = {
                **evaluate_policy_match(
                    model_policy=pot_odds_equity_policy(
                        model_rng,
                        simulations=args.equity_sims,
                        bet_threshold=bet_threshold,
                        raise_threshold=raise_threshold,
                        call_margin=call_margin,
                    ),
                    opponent_policy=make_policy(
                        args.opponent_policy,
                        opponent_rng,
                        args.equity_sims,
                        args.rollout_sims,
                    ),
                    hands=args.hands,
                    seed=args.seed + index * 100_003 + model_player * 1_000_003,
                    model_player=model_player,
                ),
                "opponent_policy": args.opponent_policy,
                "equity_sims": args.equity_sims,
                "bet_threshold": bet_threshold,
                "raise_threshold": raise_threshold,
                "call_margin": call_margin,
            }
            seat_metrics.append(metrics)
        result = aggregate_seat_metrics(seat_metrics)
        result["config_index"] = index
        results.append(result)
        if args.progress:
            print(
                f"config {index}: avg_utility_model={result['avg_utility_model']:.3f} "
                f"stderr={result['utility_stderr_model']:.3f}",
                file=sys.stderr,
                flush=True,
            )

    best = max(results, key=lambda item: item["avg_utility_model"])
    payload = {
        "hands": args.hands,
        "seed": args.seed,
        "opponent_policy": args.opponent_policy,
        "equity_sims": args.equity_sims,
        "rollout_sims": args.rollout_sims,
        "model_player": model_player_label(model_players),
        "best": best,
        "results": results,
    }
    if args.out is not None:
        write_json(Path(args.out), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--configs",
        default="0.50,0.65,-0.05;0.58,0.72,0.0;0.66,0.82,0.05",
        help="Semicolon-separated bet,raise,call_margin triples.",
    )
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

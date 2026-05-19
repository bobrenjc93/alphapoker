"""Evaluate a sampled abstract CFR Hold'em checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from alphapoker.evaluate_holdem_model import (
    aggregate_model_player_metrics,
    make_opponent_policy,
    parse_model_players,
)
from alphapoker.holdem_evaluation import evaluate_policy_match
from alphapoker.holdem_mccfr import HoldemAbstractionCFRTrainer, holdem_policy_from_trainer
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json


def normalize_model_players(value: int | str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    return parse_model_players(str(value))


def run(args: argparse.Namespace) -> dict[str, Any]:
    trainer = HoldemAbstractionCFRTrainer.load_checkpoint(args.checkpoint)
    model_players = normalize_model_players(args.model_player)
    seat_metrics = []
    for model_player in model_players:
        eval_seed = args.seed + model_player
        model_rng = random.Random(eval_seed)
        fallback_rng = random.Random(eval_seed + 1)
        opponent_rng = random.Random(eval_seed + 2)
        fallback_policy = make_policy(
            args.fallback_policy,
            fallback_rng,
            args.equity_sims,
            args.rollout_sims,
        )
        seat_metrics.append(
            {
                "checkpoint": str(args.checkpoint),
                **evaluate_policy_match(
                    model_policy=holdem_policy_from_trainer(
                        trainer,
                        model_rng,
                        fallback_policy=fallback_policy,
                        min_strategy_weight=args.min_strategy_weight,
                    ),
                    opponent_policy=make_opponent_policy(
                        args.opponent_policy,
                        opponent_rng,
                        args.equity_sims,
                        args.rollout_sims,
                    ),
                    hands=args.hands,
                    seed=eval_seed,
                    model_player=model_player,
                ),
                "opponent_policy": args.opponent_policy,
                "equity_sims": args.equity_sims,
                "rollout_sims": args.rollout_sims,
                "fallback_policy": args.fallback_policy,
                "min_strategy_weight": args.min_strategy_weight,
            }
        )
    metrics: dict[str, Any] = aggregate_model_player_metrics(seat_metrics)
    metrics["fallback_policy"] = args.fallback_policy
    metrics["min_strategy_weight"] = args.min_strategy_weight
    metrics["traversal"] = trainer.traversal
    metrics["abstraction"] = trainer.abstraction
    metrics["iterations"] = trainer.iterations
    metrics["infosets"] = len(trainer.infosets)
    if args.out is not None:
        write_json(args.out, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--fallback-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--min-strategy-weight", type=float, default=0.0)
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

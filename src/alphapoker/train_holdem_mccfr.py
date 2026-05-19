"""Train sampled abstract CFR for fixed-limit Hold'em."""

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
from alphapoker.holdem_mccfr import (
    HoldemAbstractionCFRTrainer,
    holdem_policy_from_abstract_strategy,
)
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json


def normalize_model_players(value: int | str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    return parse_model_players(str(value))


def run(args: argparse.Namespace) -> dict[str, Any]:
    trainer = HoldemAbstractionCFRTrainer(
        seed=args.seed,
        cfr_plus=not args.vanilla_cfr,
        linear_averaging=not args.uniform_averaging,
        max_bets_per_round=args.max_bets_per_round,
    )
    result = trainer.train(args.iterations)
    strategy = trainer.average_strategy()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "holdem_mccfr.json"
    trainer.save_checkpoint(checkpoint)

    metrics: dict[str, Any] = {
        "iterations": result.iterations,
        "infosets": result.infosets,
        "sampled_game_value_p0": result.sampled_game_value_p0,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "algorithm": "cfr" if args.vanilla_cfr else "cfr_plus",
        "average_weighting": "uniform" if args.uniform_averaging else "linear",
        "max_bets_per_round": args.max_bets_per_round,
    }

    if args.eval_hands > 0:
        model_players = normalize_model_players(args.model_player)
        seat_metrics = []
        for model_player in model_players:
            eval_seed = args.seed + 10_000_019 + model_player
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
                    "checkpoint": str(checkpoint),
                    **evaluate_policy_match(
                        model_policy=holdem_policy_from_abstract_strategy(
                            strategy,
                            model_rng,
                            fallback_policy=fallback_policy,
                        ),
                        opponent_policy=make_opponent_policy(
                            args.opponent_policy,
                            opponent_rng,
                            args.equity_sims,
                            args.rollout_sims,
                        ),
                        hands=args.eval_hands,
                        seed=eval_seed,
                        model_player=model_player,
                    ),
                    "opponent_policy": args.opponent_policy,
                    "equity_sims": args.equity_sims,
                    "rollout_sims": args.rollout_sims,
                    "fallback_policy": args.fallback_policy,
                }
            )
        eval_metrics = aggregate_model_player_metrics(seat_metrics)
        eval_metrics["fallback_policy"] = args.fallback_policy
        metrics["evaluation"] = eval_metrics

    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vanilla-cfr", action="store_true")
    parser.add_argument("--uniform-averaging", action="store_true")
    parser.add_argument("--max-bets-per-round", type=int, default=2)
    parser.add_argument("--eval-hands", type=int, default=0)
    parser.add_argument("--opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--fallback-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

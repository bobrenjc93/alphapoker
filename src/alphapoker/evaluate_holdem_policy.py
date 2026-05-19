"""Evaluate a named fixed-limit Hold'em policy against another named policy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from alphapoker.evaluate_holdem_model import (
    aggregate_model_player_metrics,
    normalize_model_players,
    parse_model_players,
)
from alphapoker.holdem_evaluation import evaluate_policy_match
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_players = normalize_model_players(args.model_player)
    seat_metrics = []
    for model_player in model_players:
        policy_rng = random.Random(args.seed + 1)
        opponent_rng = random.Random(args.seed + 2)
        seat_metrics.append(
            {
                "policy": args.policy,
                **evaluate_policy_match(
                    model_policy=make_policy(
                        args.policy,
                        policy_rng,
                        args.equity_sims,
                        args.rollout_sims,
                    ),
                    opponent_policy=make_policy(
                        args.opponent_policy,
                        opponent_rng,
                        args.equity_sims,
                        args.rollout_sims,
                    ),
                    hands=args.hands,
                    seed=args.seed,
                    model_player=model_player,
                ),
                "opponent_policy": args.opponent_policy,
                "equity_sims": args.equity_sims,
                "rollout_sims": args.rollout_sims,
            }
        )
    metrics: dict[str, Any] = aggregate_model_player_metrics(seat_metrics)
    if args.out is not None:
        write_json(args.out, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=HOLDEM_SELF_PLAY_POLICIES, required=True)
    parser.add_argument("--opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--equity-sims", type=int, default=128)
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

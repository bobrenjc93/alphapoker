"""Random fixed-limit Hold'em self-play baseline."""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from alphapoker.holdem import (
    deal_fixed_limit_holdem,
    equity_threshold_policy,
    play_fixed_limit_holdem_hand,
    random_holdem_policy,
)


def make_policy(name: str, rng: random.Random, equity_sims: int):
    if name == "random":
        return random_holdem_policy(rng)
    if name == "equity":
        return equity_threshold_policy(rng, simulations=equity_sims)
    raise ValueError(f"Unknown policy: {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    deal_rng = random.Random(args.seed)
    policy_rng = random.Random(args.seed + 1)
    policies = (
        make_policy(args.player0_policy, policy_rng, args.equity_sims),
        make_policy(args.player1_policy, policy_rng, args.equity_sims),
    )

    total_utility_p0 = 0.0
    showdowns = 0
    folds = 0
    total_actions = 0
    for _ in range(args.hands):
        terminal, actions = play_fixed_limit_holdem_hand(
            deal_fixed_limit_holdem(deal_rng),
            policies,
        )
        total_utility_p0 += terminal.utility(0)
        total_actions += len(actions)
        if terminal.showdown:
            showdowns += 1
        else:
            folds += 1

    return {
        "hands": args.hands,
        "avg_utility_p0": total_utility_p0 / args.hands if args.hands else 0.0,
        "avg_actions": total_actions / args.hands if args.hands else 0.0,
        "showdowns": showdowns,
        "folds": folds,
        "seed": args.seed,
        "player0_policy": args.player0_policy,
        "player1_policy": args.player1_policy,
        "equity_sims": args.equity_sims,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--player0-policy", choices=["random", "equity"], default="random")
    parser.add_argument("--player1-policy", choices=["random", "equity"], default="random")
    parser.add_argument("--equity-sims", type=int, default=128)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

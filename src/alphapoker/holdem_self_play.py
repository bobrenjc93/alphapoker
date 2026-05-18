"""Random fixed-limit Hold'em self-play baseline."""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from alphapoker.holdem import (
    deal_fixed_limit_holdem,
    play_fixed_limit_holdem_hand,
    random_holdem_policy,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    deal_rng = random.Random(args.seed)
    policy_rng = random.Random(args.seed + 1)
    policies = (random_holdem_policy(policy_rng), random_holdem_policy(policy_rng))

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
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

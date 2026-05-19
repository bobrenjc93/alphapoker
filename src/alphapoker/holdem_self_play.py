"""Random fixed-limit Hold'em self-play baseline."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from alphapoker.holdem import (
    cached_pot_odds_equity_policy,
    cached_pot_odds_rollout_policy,
    deal_fixed_limit_holdem,
    equity_threshold_policy,
    hybrid_pot_odds_equity_policy,
    play_fixed_limit_holdem_hand,
    pot_odds_equity_policy,
    pot_odds_rollout_policy,
    random_holdem_policy,
    river_exact_pot_odds_equity_policy,
    turn_river_exact_pot_odds_equity_policy,
)
from alphapoker.train import write_json

HOLDEM_SELF_PLAY_POLICIES = (
    "random",
    "equity",
    "pot-odds",
    "cached-pot-odds",
    "tuned-pot-odds",
    "cached-tuned-pot-odds",
    "river-exact-tuned-pot-odds",
    "turn-river-exact-tuned-pot-odds",
    "hybrid-pot-odds",
    "rollout-pot-odds",
    "cached-rollout-pot-odds",
)


def make_policy(
    name: str,
    rng: random.Random,
    equity_sims: int,
    rollout_sims: int | None = None,
):
    if name == "random":
        return random_holdem_policy(rng)
    if name == "equity":
        return equity_threshold_policy(rng, simulations=equity_sims)
    if name == "pot-odds":
        return pot_odds_equity_policy(rng, simulations=equity_sims)
    if name == "cached-pot-odds":
        return cached_pot_odds_equity_policy(simulations=equity_sims)
    if name == "tuned-pot-odds":
        return pot_odds_equity_policy(
            rng,
            simulations=equity_sims,
            bet_threshold=0.54,
            raise_threshold=0.76,
            call_margin=0.05,
        )
    if name == "cached-tuned-pot-odds":
        return cached_pot_odds_equity_policy(
            simulations=equity_sims,
            bet_threshold=0.54,
            raise_threshold=0.76,
            call_margin=0.05,
        )
    if name == "river-exact-tuned-pot-odds":
        return river_exact_pot_odds_equity_policy(simulations=equity_sims)
    if name == "turn-river-exact-tuned-pot-odds":
        return turn_river_exact_pot_odds_equity_policy(simulations=equity_sims)
    if name == "hybrid-pot-odds":
        return hybrid_pot_odds_equity_policy(rng, simulations=equity_sims)
    if name == "rollout-pot-odds":
        return pot_odds_rollout_policy(
            rng,
            simulations=rollout_sims if rollout_sims is not None else equity_sims,
            equity_sims=equity_sims,
        )
    if name == "cached-rollout-pot-odds":
        return cached_pot_odds_rollout_policy(
            rng,
            simulations=rollout_sims if rollout_sims is not None else equity_sims,
            equity_sims=equity_sims,
        )
    raise ValueError(f"Unknown policy: {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    deal_rng = random.Random(args.seed)
    policy_rng = random.Random(args.seed + 1)
    policies = (
        make_policy(args.player0_policy, policy_rng, args.equity_sims, args.rollout_sims),
        make_policy(args.player1_policy, policy_rng, args.equity_sims, args.rollout_sims),
    )

    total_utility_p0 = 0.0
    utilities_p0: list[float] = []
    showdowns = 0
    folds = 0
    total_actions = 0
    for _ in range(args.hands):
        terminal, actions = play_fixed_limit_holdem_hand(
            deal_fixed_limit_holdem(deal_rng),
            policies,
        )
        utility_p0 = terminal.utility(0)
        total_utility_p0 += utility_p0
        utilities_p0.append(utility_p0)
        total_actions += len(actions)
        if terminal.showdown:
            showdowns += 1
        else:
            folds += 1

    utility_stdev = statistics.stdev(utilities_p0) if len(utilities_p0) > 1 else 0.0
    metrics = {
        "hands": args.hands,
        "avg_utility_p0": total_utility_p0 / args.hands if args.hands else 0.0,
        "utility_stdev_p0": utility_stdev,
        "utility_stderr_p0": utility_stdev / (args.hands**0.5) if args.hands else 0.0,
        "avg_actions": total_actions / args.hands if args.hands else 0.0,
        "showdowns": showdowns,
        "folds": folds,
        "seed": args.seed,
        "player0_policy": args.player0_policy,
        "player1_policy": args.player1_policy,
        "equity_sims": args.equity_sims,
        "rollout_sims": args.rollout_sims,
    }
    if args.out is not None:
        write_json(Path(args.out), metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--player0-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="random")
    parser.add_argument("--player1-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="random")
    parser.add_argument("--equity-sims", type=int, default=128)
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

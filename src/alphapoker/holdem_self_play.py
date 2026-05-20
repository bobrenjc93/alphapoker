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
    opponent_range_pot_odds_equity_policy,
    play_fixed_limit_holdem_hand,
    pot_odds_equity_policy,
    pot_odds_rollout_policy,
    policy_rollout_action_values,
    policy_rollout_policy,
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
    "tight-turn-river-exact-pot-odds",
    "balanced-turn-river-exact-pot-odds",
    "tight-range-pot-odds",
    "tight-range-default-safe-rollout-pot-odds",
    "tight-fast-range-default-safe-rollout-pot-odds",
    "tight-range-rollout-pot-odds",
    "tight-range-safe-rollout-pot-odds",
    "hybrid-pot-odds",
    "rollout-pot-odds",
    "cached-rollout-pot-odds",
    "tuned-rollout-pot-odds",
    "cached-tuned-rollout-pot-odds",
    "tight-rollout-pot-odds",
    "balanced-rollout-pot-odds",
    "tight-safe-rollout-pot-odds",
    "balanced-safe-rollout-pot-odds",
)


def make_policy_action_value_fn(
    name: str,
    rng: random.Random,
    equity_sims: int,
    rollout_sims: int | None = None,
    rollout_margin: float = 1.0,
):
    if name in (
        "tight-rollout-pot-odds",
        "tight-safe-rollout-pot-odds",
        "tight-range-default-safe-rollout-pot-odds",
        "tight-fast-range-default-safe-rollout-pot-odds",
        "tight-range-rollout-pot-odds",
        "tight-range-safe-rollout-pot-odds",
    ):
        bet_threshold = 0.62
        raise_threshold = 0.84
    elif name in ("balanced-rollout-pot-odds", "balanced-safe-rollout-pot-odds"):
        bet_threshold = 0.58
        raise_threshold = 0.82
    else:
        return None

    def baseline(_: random.Random):
        return turn_river_exact_pot_odds_equity_policy(
            simulations=equity_sims,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=0.08,
        )

    def range_baseline(policy_rng: random.Random):
        return opponent_range_pot_odds_equity_policy(
            policy_rng,
            simulations=equity_sims,
            opponent_policy_factory=baseline,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=0.08,
            cache_policy_matches=True,
        )

    simulations = rollout_sims if rollout_sims is not None else equity_sims
    safe_default = "safe-rollout" in name
    range_rollout = name in (
        "tight-range-rollout-pot-odds",
        "tight-range-safe-rollout-pot-odds",
    )
    range_default = name in (
        "tight-range-default-safe-rollout-pot-odds",
        "tight-fast-range-default-safe-rollout-pot-odds",
        "tight-range-safe-rollout-pot-odds",
    )
    fast_range_default = name == "tight-fast-range-default-safe-rollout-pot-odds"
    range_default_sims = max(1, min(equity_sims, 2)) if fast_range_default else equity_sims
    range_max_attempts = 4 if fast_range_default else 32
    policy_factory = range_baseline if range_rollout else baseline

    def default_range_baseline(policy_rng: random.Random):
        return opponent_range_pot_odds_equity_policy(
            policy_rng,
            simulations=range_default_sims,
            opponent_policy_factory=baseline,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=0.08,
            max_attempts_per_sample=range_max_attempts,
            cache_policy_matches=True,
        )

    default_factory = default_range_baseline if range_default else baseline

    def action_and_values(state):
        action_values = policy_rollout_action_values(
            state,
            rng,
            simulations=simulations,
            continuation_policy_factory=policy_factory,
            opponent_policy_factory=policy_factory,
        )
        legal_actions = state.legal_actions()
        best_action = max(legal_actions, key=lambda action: action_values[action])
        if not safe_default:
            return best_action, action_values
        default_policy = default_factory(random.Random(rng.randrange(2**63)))
        default_action = default_policy(state)
        if default_action not in legal_actions:
            return best_action, action_values
        if action_values[best_action] >= action_values[default_action] + rollout_margin:
            return best_action, action_values
        return default_action, action_values

    return action_and_values


def make_policy(
    name: str,
    rng: random.Random,
    equity_sims: int,
    rollout_sims: int | None = None,
    rollout_margin: float = 1.0,
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
    if name == "tight-turn-river-exact-pot-odds":
        return turn_river_exact_pot_odds_equity_policy(
            simulations=equity_sims,
            bet_threshold=0.62,
            raise_threshold=0.84,
            call_margin=0.08,
        )
    if name == "balanced-turn-river-exact-pot-odds":
        return turn_river_exact_pot_odds_equity_policy(
            simulations=equity_sims,
            bet_threshold=0.58,
            raise_threshold=0.82,
            call_margin=0.08,
        )
    if name == "tight-range-pot-odds":
        def tight_baseline(_: random.Random):
            return turn_river_exact_pot_odds_equity_policy(
                simulations=equity_sims,
                bet_threshold=0.62,
                raise_threshold=0.84,
                call_margin=0.08,
            )

        return opponent_range_pot_odds_equity_policy(
            rng,
            simulations=equity_sims,
            opponent_policy_factory=tight_baseline,
            bet_threshold=0.62,
            raise_threshold=0.84,
            call_margin=0.08,
            cache_policy_matches=True,
        )
    if name in (
        "tight-range-default-safe-rollout-pot-odds",
        "tight-fast-range-default-safe-rollout-pot-odds",
    ):
        fast_range_default = name == "tight-fast-range-default-safe-rollout-pot-odds"
        range_default_sims = max(1, min(equity_sims, 2)) if fast_range_default else equity_sims
        range_max_attempts = 4 if fast_range_default else 32

        def tight_baseline(_: random.Random):
            return turn_river_exact_pot_odds_equity_policy(
                simulations=equity_sims,
                bet_threshold=0.62,
                raise_threshold=0.84,
                call_margin=0.08,
            )

        def range_baseline(policy_rng: random.Random):
            return opponent_range_pot_odds_equity_policy(
                policy_rng,
                simulations=range_default_sims,
                opponent_policy_factory=tight_baseline,
                bet_threshold=0.62,
                raise_threshold=0.84,
                call_margin=0.08,
                max_attempts_per_sample=range_max_attempts,
                cache_policy_matches=True,
            )

        return policy_rollout_policy(
            rng,
            simulations=rollout_sims if rollout_sims is not None else equity_sims,
            continuation_policy_factory=tight_baseline,
            opponent_policy_factory=tight_baseline,
            default_policy_factory=range_baseline,
            improvement_margin=rollout_margin,
        )
    if name in ("tight-range-rollout-pot-odds", "tight-range-safe-rollout-pot-odds"):
        def tight_baseline(_: random.Random):
            return turn_river_exact_pot_odds_equity_policy(
                simulations=equity_sims,
                bet_threshold=0.62,
                raise_threshold=0.84,
                call_margin=0.08,
            )

        def range_baseline(policy_rng: random.Random):
            return opponent_range_pot_odds_equity_policy(
                policy_rng,
                simulations=equity_sims,
                opponent_policy_factory=tight_baseline,
                bet_threshold=0.62,
                raise_threshold=0.84,
                call_margin=0.08,
                cache_policy_matches=True,
            )

        rollout_kwargs: dict[str, Any] = {}
        if name == "tight-range-safe-rollout-pot-odds":
            rollout_kwargs = {
                "default_policy_factory": range_baseline,
                "improvement_margin": rollout_margin,
            }
        return policy_rollout_policy(
            rng,
            simulations=rollout_sims if rollout_sims is not None else equity_sims,
            continuation_policy_factory=range_baseline,
            opponent_policy_factory=range_baseline,
            **rollout_kwargs,
        )
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
    if name == "tuned-rollout-pot-odds":
        return pot_odds_rollout_policy(
            rng,
            simulations=rollout_sims if rollout_sims is not None else equity_sims,
            equity_sims=equity_sims,
            bet_threshold=0.54,
            raise_threshold=0.76,
            call_margin=0.05,
        )
    if name == "cached-tuned-rollout-pot-odds":
        return cached_pot_odds_rollout_policy(
            rng,
            simulations=rollout_sims if rollout_sims is not None else equity_sims,
            equity_sims=equity_sims,
            bet_threshold=0.54,
            raise_threshold=0.76,
            call_margin=0.05,
        )
    if name == "tight-rollout-pot-odds":
        def baseline(_: random.Random):
            return turn_river_exact_pot_odds_equity_policy(
                simulations=equity_sims,
                bet_threshold=0.62,
                raise_threshold=0.84,
                call_margin=0.08,
            )

        return policy_rollout_policy(
            rng,
            simulations=rollout_sims if rollout_sims is not None else equity_sims,
            continuation_policy_factory=baseline,
            opponent_policy_factory=baseline,
        )
    if name == "balanced-rollout-pot-odds":
        def baseline(_: random.Random):
            return turn_river_exact_pot_odds_equity_policy(
                simulations=equity_sims,
                bet_threshold=0.58,
                raise_threshold=0.82,
                call_margin=0.08,
            )

        return policy_rollout_policy(
            rng,
            simulations=rollout_sims if rollout_sims is not None else equity_sims,
            continuation_policy_factory=baseline,
            opponent_policy_factory=baseline,
        )
    if name == "tight-safe-rollout-pot-odds":
        def baseline(_: random.Random):
            return turn_river_exact_pot_odds_equity_policy(
                simulations=equity_sims,
                bet_threshold=0.62,
                raise_threshold=0.84,
                call_margin=0.08,
            )

        return policy_rollout_policy(
            rng,
            simulations=rollout_sims if rollout_sims is not None else equity_sims,
            continuation_policy_factory=baseline,
            opponent_policy_factory=baseline,
            default_policy_factory=baseline,
            improvement_margin=rollout_margin,
        )
    if name == "balanced-safe-rollout-pot-odds":
        def baseline(_: random.Random):
            return turn_river_exact_pot_odds_equity_policy(
                simulations=equity_sims,
                bet_threshold=0.58,
                raise_threshold=0.82,
                call_margin=0.08,
            )

        return policy_rollout_policy(
            rng,
            simulations=rollout_sims if rollout_sims is not None else equity_sims,
            continuation_policy_factory=baseline,
            opponent_policy_factory=baseline,
            default_policy_factory=baseline,
            improvement_margin=rollout_margin,
        )
    raise ValueError(f"Unknown policy: {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    deal_rng = random.Random(args.seed)
    policy_rng = random.Random(args.seed + 1)
    policies = (
        make_policy(
            args.player0_policy,
            policy_rng,
            args.equity_sims,
            args.rollout_sims,
            args.rollout_margin,
        ),
        make_policy(
            args.player1_policy,
            policy_rng,
            args.equity_sims,
            args.rollout_sims,
            args.rollout_margin,
        ),
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
        "rollout_margin": args.rollout_margin,
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
    parser.add_argument("--rollout-margin", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

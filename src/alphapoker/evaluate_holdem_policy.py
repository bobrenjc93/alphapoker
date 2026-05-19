"""Evaluate a named fixed-limit Hold'em policy against another named policy."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from alphapoker.evaluate_holdem_model import (
    aggregate_model_player_metrics,
    normalize_model_players,
    parse_model_players,
)
from alphapoker.holdem import (
    HoldemPolicy,
    cached_pot_odds_equity_policy,
    hybrid_pot_odds_equity_policy,
    opponent_range_pot_odds_equity_policy,
    pot_odds_equity_policy,
    river_exact_pot_odds_equity_policy,
    turn_river_exact_pot_odds_equity_policy,
)
from alphapoker.holdem_evaluation import (
    aggregate_policy_match_shards,
    evaluate_policy_match,
    evaluate_policy_match_paired_seats,
)
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json

POLICY_THRESHOLD_DEFAULTS = {
    "pot-odds": (0.58, 0.72, 0.0),
    "cached-pot-odds": (0.58, 0.72, 0.0),
    "tuned-pot-odds": (0.54, 0.76, 0.05),
    "cached-tuned-pot-odds": (0.54, 0.76, 0.05),
    "river-exact-tuned-pot-odds": (0.54, 0.76, 0.05),
    "turn-river-exact-tuned-pot-odds": (0.54, 0.76, 0.05),
    "tight-turn-river-exact-pot-odds": (0.62, 0.84, 0.08),
    "tight-range-pot-odds": (0.62, 0.84, 0.08),
    "hybrid-pot-odds": (0.54, 0.76, 0.05),
}


def split_hands(hands: int, jobs: int) -> list[int]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if hands < 0:
        raise ValueError("hands must be non-negative")
    if hands == 0:
        return [0]
    shard_count = min(hands, jobs)
    base_hands, extra_hands = divmod(hands, shard_count)
    return [base_hands + (1 if shard < extra_hands else 0) for shard in range(shard_count)]


def resolve_policy_thresholds(
    policy: str,
    *,
    bet_threshold: float | None,
    raise_threshold: float | None,
    call_margin: float | None,
) -> tuple[float, float, float] | None:
    if bet_threshold is None and raise_threshold is None and call_margin is None:
        return None
    if policy not in POLICY_THRESHOLD_DEFAULTS:
        raise ValueError("threshold overrides require a pot-odds policy")
    default_bet, default_raise, default_call_margin = POLICY_THRESHOLD_DEFAULTS[policy]
    return (
        default_bet if bet_threshold is None else bet_threshold,
        default_raise if raise_threshold is None else raise_threshold,
        default_call_margin if call_margin is None else call_margin,
    )


def make_evaluation_policy(
    name: str,
    rng: random.Random,
    *,
    equity_sims: int,
    rollout_sims: int | None,
    rollout_margin: float,
    thresholds: tuple[float, float, float] | None,
) -> HoldemPolicy:
    if thresholds is None:
        return make_policy(name, rng, equity_sims, rollout_sims, rollout_margin)
    bet_threshold, raise_threshold, call_margin = thresholds
    if name in ("pot-odds", "tuned-pot-odds"):
        return pot_odds_equity_policy(
            rng,
            simulations=equity_sims,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
        )
    if name in ("cached-pot-odds", "cached-tuned-pot-odds"):
        return cached_pot_odds_equity_policy(
            simulations=equity_sims,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
        )
    if name == "river-exact-tuned-pot-odds":
        return river_exact_pot_odds_equity_policy(
            simulations=equity_sims,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
        )
    if name in ("turn-river-exact-tuned-pot-odds", "tight-turn-river-exact-pot-odds"):
        return turn_river_exact_pot_odds_equity_policy(
            simulations=equity_sims,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
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
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
        )
    if name == "hybrid-pot-odds":
        return hybrid_pot_odds_equity_policy(
            rng,
            simulations=equity_sims,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
        )
    raise ValueError("threshold overrides require a pot-odds policy")


def evaluate_policy_shard(
    *,
    policy: str,
    opponent_policy: str,
    hands: int,
    seed: int,
    equity_sims: int,
    rollout_sims: int | None,
    rollout_margin: float,
    opponent_equity_sims: int,
    opponent_rollout_sims: int | None,
    model_player: int,
    shard_index: int,
    policy_thresholds: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    shard_seed = seed + shard_index * 1_000_003
    policy_rng = random.Random(shard_seed + 1)
    opponent_rng = random.Random(shard_seed + 2)
    metrics = {
        "policy": policy,
        **evaluate_policy_match(
            model_policy=make_evaluation_policy(
                policy,
                policy_rng,
                equity_sims=equity_sims,
                rollout_sims=rollout_sims,
                rollout_margin=rollout_margin,
                thresholds=policy_thresholds,
            ),
            opponent_policy=make_policy(
                opponent_policy,
                opponent_rng,
                opponent_equity_sims,
                opponent_rollout_sims,
                rollout_margin,
            ),
            hands=hands,
            seed=shard_seed,
            model_player=model_player,
        ),
        "opponent_policy": opponent_policy,
        "equity_sims": equity_sims,
        "rollout_sims": rollout_sims,
        "opponent_equity_sims": opponent_equity_sims,
        "opponent_rollout_sims": opponent_rollout_sims,
        "rollout_margin": rollout_margin,
        "shard_index": shard_index,
    }
    if policy_thresholds is not None:
        metrics["bet_threshold"] = policy_thresholds[0]
        metrics["raise_threshold"] = policy_thresholds[1]
        metrics["call_margin"] = policy_thresholds[2]
    return metrics


def evaluate_policy_paired_shard(
    *,
    policy: str,
    opponent_policy: str,
    hands: int,
    seed: int,
    equity_sims: int,
    rollout_sims: int | None,
    rollout_margin: float,
    opponent_equity_sims: int,
    opponent_rollout_sims: int | None,
    shard_index: int,
    policy_thresholds: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    shard_seed = seed + shard_index * 1_000_003
    model_policies = tuple(
        make_evaluation_policy(
            policy,
            random.Random(shard_seed + 10 + model_player),
            equity_sims=equity_sims,
            rollout_sims=rollout_sims,
            rollout_margin=rollout_margin,
            thresholds=policy_thresholds,
        )
        for model_player in (0, 1)
    )
    opponent_policies = tuple(
        make_policy(
            opponent_policy,
            random.Random(shard_seed + 20 + model_player),
            opponent_equity_sims,
            opponent_rollout_sims,
            rollout_margin,
        )
        for model_player in (0, 1)
    )
    metrics = {
        "policy": policy,
        **evaluate_policy_match_paired_seats(
            model_policies=model_policies,
            opponent_policies=opponent_policies,
            hands=hands,
            seed=shard_seed,
        ),
        "opponent_policy": opponent_policy,
        "equity_sims": equity_sims,
        "rollout_sims": rollout_sims,
        "opponent_equity_sims": opponent_equity_sims,
        "opponent_rollout_sims": opponent_rollout_sims,
        "rollout_margin": rollout_margin,
        "shard_index": shard_index,
    }
    if policy_thresholds is not None:
        metrics["bet_threshold"] = policy_thresholds[0]
        metrics["raise_threshold"] = policy_thresholds[1]
        metrics["call_margin"] = policy_thresholds[2]
    return metrics


def report_progress(enabled: bool, result: dict[str, Any]) -> None:
    if not enabled:
        return
    print(
        f"player {result['model_player']} shard {result['shard_index']}: "
        f"hands={result['hands']} "
        f"avg_utility_model={result['avg_utility_model']:.3f} "
        f"stderr={result['utility_stderr_model']:.3f}",
        file=sys.stderr,
        flush=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    policy_thresholds = resolve_policy_thresholds(
        args.policy,
        bet_threshold=args.bet_threshold,
        raise_threshold=args.raise_threshold,
        call_margin=args.call_margin,
    )
    model_players = normalize_model_players(args.model_player)
    shard_hands = split_hands(args.hands, args.jobs)
    opponent_equity_sims = (
        args.equity_sims if args.opponent_equity_sims is None else args.opponent_equity_sims
    )
    opponent_rollout_sims = (
        args.rollout_sims if args.opponent_rollout_sims is None else args.opponent_rollout_sims
    )
    if args.paired_seats and model_players != (0, 1):
        raise ValueError("--paired-seats requires --model-player both")
    if args.paired_seats:
        shard_kwargs = [
            {
                "policy": args.policy,
                "opponent_policy": args.opponent_policy,
                "hands": hands,
                "seed": args.seed,
                "equity_sims": args.equity_sims,
                "rollout_sims": args.rollout_sims,
                "rollout_margin": args.rollout_margin,
                "opponent_equity_sims": opponent_equity_sims,
                "opponent_rollout_sims": opponent_rollout_sims,
                "shard_index": shard_index,
                "policy_thresholds": policy_thresholds,
            }
            for shard_index, hands in enumerate(shard_hands)
        ]
        if args.jobs == 1:
            shard_metrics = []
            for kwargs in shard_kwargs:
                result = evaluate_policy_paired_shard(**kwargs)
                shard_metrics.append(result)
                report_progress(args.progress, result)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = [
                    executor.submit(evaluate_policy_paired_shard, **kwargs)
                    for kwargs in shard_kwargs
                ]
                shard_metrics = []
                for future in as_completed(futures):
                    result = future.result()
                    shard_metrics.append(result)
                    report_progress(args.progress, result)
            shard_metrics.sort(key=lambda item: item["shard_index"])
        metrics = aggregate_policy_match_shards(shard_metrics)
        metrics["jobs"] = args.jobs
        metrics["shard_hands"] = shard_hands
        metrics["paired_seats"] = True
        metrics["opponent_equity_sims"] = opponent_equity_sims
        metrics["opponent_rollout_sims"] = opponent_rollout_sims
        if policy_thresholds is not None:
            metrics["bet_threshold"] = policy_thresholds[0]
            metrics["raise_threshold"] = policy_thresholds[1]
            metrics["call_margin"] = policy_thresholds[2]
        metrics["rollout_margin"] = args.rollout_margin
        if args.out is not None:
            write_json(args.out, metrics)
        return metrics

    shard_metrics_by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    shard_kwargs = [
        {
            "policy": args.policy,
            "opponent_policy": args.opponent_policy,
            "hands": hands,
            "seed": args.seed,
            "equity_sims": args.equity_sims,
            "rollout_sims": args.rollout_sims,
            "rollout_margin": args.rollout_margin,
            "opponent_equity_sims": opponent_equity_sims,
            "opponent_rollout_sims": opponent_rollout_sims,
            "model_player": model_player,
            "shard_index": shard_index,
            "policy_thresholds": policy_thresholds,
        }
        for model_player in model_players
        for shard_index, hands in enumerate(shard_hands)
    ]

    if args.jobs == 1:
        for kwargs in shard_kwargs:
            result = evaluate_policy_shard(**kwargs)
            shard_metrics_by_player[int(result["model_player"])].append(result)
            report_progress(args.progress, result)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(evaluate_policy_shard, **kwargs) for kwargs in shard_kwargs]
            for future in as_completed(futures):
                result = future.result()
                shard_metrics_by_player[int(result["model_player"])].append(result)
                report_progress(args.progress, result)

    seat_metrics = []
    for model_player in model_players:
        player_shards = sorted(
            shard_metrics_by_player[model_player],
            key=lambda item: item["shard_index"],
        )
        seat_metrics.append(aggregate_policy_match_shards(player_shards))
    metrics: dict[str, Any] = aggregate_model_player_metrics(seat_metrics)
    metrics["jobs"] = args.jobs
    metrics["shard_hands"] = shard_hands
    metrics["paired_seats"] = False
    metrics["opponent_equity_sims"] = opponent_equity_sims
    metrics["opponent_rollout_sims"] = opponent_rollout_sims
    if policy_thresholds is not None:
        metrics["bet_threshold"] = policy_thresholds[0]
        metrics["raise_threshold"] = policy_thresholds[1]
        metrics["call_margin"] = policy_thresholds[2]
    metrics["rollout_margin"] = args.rollout_margin
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
    parser.add_argument("--rollout-margin", type=float, default=1.0)
    parser.add_argument("--opponent-equity-sims", type=int)
    parser.add_argument("--opponent-rollout-sims", type=int)
    parser.add_argument("--bet-threshold", type=float)
    parser.add_argument("--raise-threshold", type=float)
    parser.add_argument("--call-margin", type=float)
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--paired-seats", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

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
from alphapoker.holdem_evaluation import aggregate_policy_match_shards, evaluate_policy_match
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json


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


def evaluate_policy_shard(
    *,
    policy: str,
    opponent_policy: str,
    hands: int,
    seed: int,
    equity_sims: int,
    rollout_sims: int | None,
    model_player: int,
    shard_index: int,
) -> dict[str, Any]:
    shard_seed = seed + shard_index * 1_000_003
    policy_rng = random.Random(shard_seed + 1)
    opponent_rng = random.Random(shard_seed + 2)
    return {
        "policy": policy,
        **evaluate_policy_match(
            model_policy=make_policy(
                policy,
                policy_rng,
                equity_sims,
                rollout_sims,
            ),
            opponent_policy=make_policy(
                opponent_policy,
                opponent_rng,
                equity_sims,
                rollout_sims,
            ),
            hands=hands,
            seed=shard_seed,
            model_player=model_player,
        ),
        "opponent_policy": opponent_policy,
        "equity_sims": equity_sims,
        "rollout_sims": rollout_sims,
        "shard_index": shard_index,
    }


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
    model_players = normalize_model_players(args.model_player)
    shard_hands = split_hands(args.hands, args.jobs)
    shard_metrics_by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    shard_kwargs = [
        {
            "policy": args.policy,
            "opponent_policy": args.opponent_policy,
            "hands": hands,
            "seed": args.seed,
            "equity_sims": args.equity_sims,
            "rollout_sims": args.rollout_sims,
            "model_player": model_player,
            "shard_index": shard_index,
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
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Evaluate a sampled abstract CFR Hold'em checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from alphapoker.evaluate_holdem_model import (
    aggregate_model_player_metrics,
    make_opponent_policy,
    parse_model_players,
)
from alphapoker.holdem_evaluation import (
    aggregate_policy_match_shards,
    evaluate_policy_match,
    evaluate_policy_match_paired_seats,
)
from alphapoker.holdem_mccfr import (
    HOLDEM_MCCFR_STRATEGY_MODES,
    HoldemAbstractionCFRTrainer,
    holdem_policy_from_trainer,
)
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json


def normalize_model_players(value: int | str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    return parse_model_players(str(value))


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


def evaluate_mccfr_shard(
    *,
    checkpoint: Path,
    hands: int,
    seed: int,
    opponent_policy: str,
    fallback_policy: str,
    min_strategy_weight: float,
    strategy_mode: str,
    equity_sims: int,
    rollout_sims: int | None,
    model_player: int,
    shard_index: int,
) -> dict[str, Any]:
    trainer = HoldemAbstractionCFRTrainer.load_checkpoint(checkpoint)
    eval_seed = seed + shard_index * 1_000_003
    model_rng = random.Random(eval_seed)
    fallback_rng = random.Random(eval_seed + 1)
    opponent_rng = random.Random(eval_seed + 2)
    fallback = make_policy(
        fallback_policy,
        fallback_rng,
        equity_sims,
        rollout_sims,
    )
    return {
        "checkpoint": str(checkpoint),
        **evaluate_policy_match(
            model_policy=holdem_policy_from_trainer(
                trainer,
                model_rng,
                fallback_policy=fallback,
                min_strategy_weight=min_strategy_weight,
                strategy_mode=strategy_mode,
            ),
            opponent_policy=make_opponent_policy(
                opponent_policy,
                opponent_rng,
                equity_sims,
                rollout_sims,
            ),
            hands=hands,
            seed=eval_seed,
            model_player=model_player,
        ),
        "opponent_policy": opponent_policy,
        "equity_sims": equity_sims,
        "rollout_sims": rollout_sims,
        "fallback_policy": fallback_policy,
        "min_strategy_weight": min_strategy_weight,
        "strategy_mode": strategy_mode,
        "shard_index": shard_index,
    }


def evaluate_mccfr_paired_shard(
    *,
    checkpoint: Path,
    hands: int,
    seed: int,
    opponent_policy: str,
    fallback_policy: str,
    min_strategy_weight: float,
    strategy_mode: str,
    equity_sims: int,
    rollout_sims: int | None,
    shard_index: int,
) -> dict[str, Any]:
    trainer = HoldemAbstractionCFRTrainer.load_checkpoint(checkpoint)
    eval_seed = seed + shard_index * 1_000_003
    model_policies = []
    opponent_policies = []
    for model_player in (0, 1):
        fallback = make_policy(
            fallback_policy,
            random.Random(eval_seed + 10 + model_player),
            equity_sims,
            rollout_sims,
        )
        model_policies.append(
            holdem_policy_from_trainer(
                trainer,
                random.Random(eval_seed + 20 + model_player),
                fallback_policy=fallback,
                min_strategy_weight=min_strategy_weight,
                strategy_mode=strategy_mode,
            )
        )
        opponent_policies.append(
            make_opponent_policy(
                opponent_policy,
                random.Random(eval_seed + 30 + model_player),
                equity_sims,
                rollout_sims,
            )
        )

    return {
        "checkpoint": str(checkpoint),
        **evaluate_policy_match_paired_seats(
            model_policies=(model_policies[0], model_policies[1]),
            opponent_policies=(opponent_policies[0], opponent_policies[1]),
            hands=hands,
            seed=eval_seed,
        ),
        "opponent_policy": opponent_policy,
        "equity_sims": equity_sims,
        "rollout_sims": rollout_sims,
        "fallback_policy": fallback_policy,
        "min_strategy_weight": min_strategy_weight,
        "strategy_mode": strategy_mode,
        "shard_index": shard_index,
    }


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    hands: int,
    seed: int,
    opponent_policy: str,
    fallback_policy: str,
    min_strategy_weight: float,
    strategy_mode: str,
    equity_sims: int,
    rollout_sims: int | None,
    model_players: tuple[int, ...],
    jobs: int,
    paired_seats: bool = False,
) -> dict[str, Any]:
    trainer = HoldemAbstractionCFRTrainer.load_checkpoint(checkpoint)
    shard_hands = split_hands(hands, jobs)
    if paired_seats and model_players != (0, 1):
        raise ValueError("paired_seats requires model_players=(0, 1)")
    if paired_seats:
        shard_kwargs = [
            {
                "checkpoint": checkpoint,
                "hands": shard_size,
                "seed": seed,
                "opponent_policy": opponent_policy,
                "fallback_policy": fallback_policy,
                "min_strategy_weight": min_strategy_weight,
                "strategy_mode": strategy_mode,
                "equity_sims": equity_sims,
                "rollout_sims": rollout_sims,
                "shard_index": shard_index,
            }
            for shard_index, shard_size in enumerate(shard_hands)
        ]
        if jobs == 1:
            shard_metrics = [
                evaluate_mccfr_paired_shard(**kwargs) for kwargs in shard_kwargs
            ]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [
                    executor.submit(evaluate_mccfr_paired_shard, **kwargs)
                    for kwargs in shard_kwargs
                ]
                shard_metrics = [future.result() for future in as_completed(futures)]
            shard_metrics.sort(key=lambda item: item["shard_index"])
        metrics = aggregate_policy_match_shards(shard_metrics)
        metrics["fallback_policy"] = fallback_policy
        metrics["min_strategy_weight"] = min_strategy_weight
        metrics["strategy_mode"] = strategy_mode
        metrics["traversal"] = trainer.traversal
        metrics["abstraction"] = trainer.abstraction
        metrics["iterations"] = trainer.iterations
        metrics["infosets"] = len(trainer.infosets)
        metrics["jobs"] = jobs
        metrics["shard_hands"] = shard_hands
        metrics["paired_seats"] = True
        return metrics

    shard_metrics_by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    shard_kwargs = [
        {
            "checkpoint": checkpoint,
            "hands": shard_size,
            "seed": seed,
            "opponent_policy": opponent_policy,
            "fallback_policy": fallback_policy,
            "min_strategy_weight": min_strategy_weight,
            "strategy_mode": strategy_mode,
            "equity_sims": equity_sims,
            "rollout_sims": rollout_sims,
            "model_player": model_player,
            "shard_index": shard_index,
        }
        for model_player in model_players
        for shard_index, shard_size in enumerate(shard_hands)
    ]

    if jobs == 1:
        for kwargs in shard_kwargs:
            result = evaluate_mccfr_shard(**kwargs)
            shard_metrics_by_player[int(result["model_player"])].append(result)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(evaluate_mccfr_shard, **kwargs) for kwargs in shard_kwargs]
            for future in as_completed(futures):
                result = future.result()
                shard_metrics_by_player[int(result["model_player"])].append(result)

    seat_metrics = []
    for model_player in model_players:
        player_shards = sorted(
            shard_metrics_by_player[model_player],
            key=lambda item: item["shard_index"],
        )
        seat_metrics.append(aggregate_policy_match_shards(player_shards))
    metrics: dict[str, Any] = aggregate_model_player_metrics(seat_metrics)
    metrics["fallback_policy"] = fallback_policy
    metrics["min_strategy_weight"] = min_strategy_weight
    metrics["strategy_mode"] = strategy_mode
    metrics["traversal"] = trainer.traversal
    metrics["abstraction"] = trainer.abstraction
    metrics["iterations"] = trainer.iterations
    metrics["infosets"] = len(trainer.infosets)
    metrics["jobs"] = jobs
    metrics["shard_hands"] = shard_hands
    metrics["paired_seats"] = False
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    metrics = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        hands=args.hands,
        seed=args.seed,
        opponent_policy=args.opponent_policy,
        fallback_policy=args.fallback_policy,
        min_strategy_weight=args.min_strategy_weight,
        strategy_mode=args.strategy_mode,
        equity_sims=args.equity_sims,
        rollout_sims=args.rollout_sims,
        model_players=normalize_model_players(args.model_player),
        jobs=args.jobs,
        paired_seats=args.paired_seats,
    )
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
    parser.add_argument("--strategy-mode", choices=HOLDEM_MCCFR_STRATEGY_MODES, default="average")
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--paired-seats", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

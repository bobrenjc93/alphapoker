"""Evaluate a trained fixed-limit Hold'em policy checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from alphapoker.holdem import FixedLimitHoldemState, HoldemPolicy
from alphapoker.holdem_evaluation import (
    ACTION_COUNT_KEYS,
    add_action_counts,
    aggregate_policy_match_shards,
    empty_action_counts,
    evaluate_policy_match,
    evaluate_policy_match_paired_seats,
)
from alphapoker.holdem_features import holdem_legal_action_mask
from alphapoker.holdem_policy_features import (
    policy_feature_encoder_from_checkpoint_data,
)
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json


def parse_model_players(value: str) -> tuple[int, ...]:
    if value == "both":
        return (0, 1)
    try:
        model_player = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("model-player must be 0, 1, or both") from error
    if model_player not in (0, 1):
        raise argparse.ArgumentTypeError("model-player must be 0, 1, or both")
    return (model_player,)


def normalize_model_players(value: int | str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    return parse_model_players(str(value))


def _weighted_mean(metrics: list[dict[str, Any]], key: str) -> float:
    total_hands = sum(int(item["hands"]) for item in metrics)
    return (
        sum(float(item[key]) * int(item["hands"]) for item in metrics) / total_hands
        if total_hands
        else 0.0
    )


def _pooled_stdev(metrics: list[dict[str, Any]], mean_key: str, stdev_key: str) -> float:
    total_hands = sum(int(item["hands"]) for item in metrics)
    if total_hands <= 1:
        return 0.0
    mean = _weighted_mean(metrics, mean_key)
    sum_squares = 0.0
    for item in metrics:
        hands = int(item["hands"])
        stdev = float(item[stdev_key])
        item_mean = float(item[mean_key])
        sum_squares += max(0, hands - 1) * stdev * stdev
        sum_squares += hands * (item_mean - mean) * (item_mean - mean)
    return math.sqrt(sum_squares / (total_hands - 1))


def _summed_action_counts(metrics: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = empty_action_counts()
    for item in metrics:
        add_action_counts(counts, item[key])
    return counts


def aggregate_model_player_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if len(metrics) == 1:
        return metrics[0]
    total_hands = sum(int(item["hands"]) for item in metrics)
    model_stdev = _pooled_stdev(metrics, "avg_utility_model", "utility_stdev_model")
    p0_stdev = _pooled_stdev(metrics, "avg_utility_p0", "utility_stdev_p0")
    first = metrics[0]
    aggregated = {
        "hands": total_hands,
        "hands_per_model_player": first["hands"],
        "model_player": "both",
        "avg_utility_model": _weighted_mean(metrics, "avg_utility_model"),
        "utility_stdev_model": model_stdev,
        "utility_stderr_model": model_stdev / (total_hands**0.5) if total_hands else 0.0,
        "avg_utility_p0": _weighted_mean(metrics, "avg_utility_p0"),
        "utility_stdev_p0": p0_stdev,
        "utility_stderr_p0": p0_stdev / (total_hands**0.5) if total_hands else 0.0,
        "avg_actions": _weighted_mean(metrics, "avg_actions"),
        "folds": sum(int(item["folds"]) for item in metrics),
        "showdowns": sum(int(item["showdowns"]) for item in metrics),
        "seed": first["seed"],
        "seat_metrics": metrics,
    }
    for key in ACTION_COUNT_KEYS:
        if key in first:
            aggregated[key] = _summed_action_counts(metrics, key)
    for key in (
        "checkpoint",
        "policy",
        "opponent_policy",
        "equity_sims",
        "rollout_sims",
        "rollout_margin",
        "fallback_policy",
        "min_strategy_weight",
        "bet_threshold",
        "raise_threshold",
        "call_margin",
    ):
        if key in first:
            aggregated[key] = first[key]
    return aggregated


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


def evaluation_process_context() -> mp.context.BaseContext:
    start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    return mp.get_context(start_method)


def model_policy_from_checkpoint(checkpoint_path: Path, *, feature_seed: int = 0) -> HoldemPolicy:
    import torch

    from alphapoker.holdem_features import HOLDEM_CANONICAL_ACTIONS
    from alphapoker.holdem_model import HoldemPolicyNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_encoder = policy_feature_encoder_from_checkpoint_data(
        checkpoint,
        checkpoint_path=checkpoint_path,
        feature_seed=feature_seed,
    )
    model = HoldemPolicyNet(input_dim=feature_encoder.input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def select_action(state: FixedLimitHoldemState) -> str:
        features = torch.tensor([feature_encoder.encode(state)], dtype=torch.float32)
        mask = torch.tensor(holdem_legal_action_mask(state), dtype=torch.bool)
        with torch.no_grad():
            logits = model(features).squeeze(0)
            logits = logits.masked_fill(~mask, -1e9)
            action_index = int(logits.argmax().item())
        return HOLDEM_CANONICAL_ACTIONS[action_index]

    return select_action


def make_opponent_policy(
    name: str,
    rng: random.Random,
    equity_sims: int,
    rollout_sims: int | None = None,
    rollout_margin: float = 1.0,
) -> HoldemPolicy:
    return make_policy(name, rng, equity_sims, rollout_sims, rollout_margin)


def evaluate_model_shard(
    *,
    checkpoint: Path,
    hands: int,
    seed: int,
    opponent_policy: str,
    equity_sims: int,
    rollout_sims: int | None,
    rollout_margin: float,
    model_player: int,
    shard_index: int,
) -> dict[str, Any]:
    eval_seed = seed + shard_index * 1_000_003
    opponent_rng = random.Random(eval_seed + 1)
    return {
        "checkpoint": str(checkpoint),
        **evaluate_policy_match(
            model_policy=model_policy_from_checkpoint(checkpoint, feature_seed=eval_seed),
            opponent_policy=make_opponent_policy(
                opponent_policy,
                opponent_rng,
                equity_sims,
                rollout_sims,
                rollout_margin,
            ),
            hands=hands,
            seed=eval_seed,
            model_player=model_player,
        ),
        "opponent_policy": opponent_policy,
        "equity_sims": equity_sims,
        "rollout_sims": rollout_sims,
        "rollout_margin": rollout_margin,
        "shard_index": shard_index,
    }


def evaluate_model_paired_shard(
    *,
    checkpoint: Path,
    hands: int,
    seed: int,
    opponent_policy: str,
    equity_sims: int,
    rollout_sims: int | None,
    rollout_margin: float,
    shard_index: int,
) -> dict[str, Any]:
    eval_seed = seed + shard_index * 1_000_003
    model_policies = tuple(
        model_policy_from_checkpoint(checkpoint, feature_seed=eval_seed + model_player)
        for model_player in (0, 1)
    )
    opponent_policies = tuple(
        make_opponent_policy(
            opponent_policy,
            random.Random(eval_seed + 10 + model_player),
            equity_sims,
            rollout_sims,
            rollout_margin,
        )
        for model_player in (0, 1)
    )
    return {
        "checkpoint": str(checkpoint),
        **evaluate_policy_match_paired_seats(
            model_policies=model_policies,
            opponent_policies=opponent_policies,
            hands=hands,
            seed=eval_seed,
        ),
        "opponent_policy": opponent_policy,
        "equity_sims": equity_sims,
        "rollout_sims": rollout_sims,
        "rollout_margin": rollout_margin,
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
    progress = bool(getattr(args, "progress", False))
    rollout_margin = float(getattr(args, "rollout_margin", 1.0))
    model_players = normalize_model_players(args.model_player)
    shard_hands = split_hands(args.hands, args.jobs)
    if args.paired_seats and model_players != (0, 1):
        raise ValueError("--paired-seats requires --model-player both")
    if args.paired_seats:
        shard_kwargs = [
            {
                "checkpoint": args.checkpoint,
                "hands": shard_size,
                "seed": args.seed,
                "opponent_policy": args.opponent_policy,
                "equity_sims": args.equity_sims,
                "rollout_sims": args.rollout_sims,
                "rollout_margin": rollout_margin,
                "shard_index": shard_index,
            }
            for shard_index, shard_size in enumerate(shard_hands)
        ]
        if args.jobs == 1:
            shard_metrics = []
            for kwargs in shard_kwargs:
                result = evaluate_model_paired_shard(**kwargs)
                shard_metrics.append(result)
                report_progress(progress, result)
        else:
            with ProcessPoolExecutor(
                max_workers=args.jobs,
                mp_context=evaluation_process_context(),
            ) as executor:
                futures = [
                    executor.submit(evaluate_model_paired_shard, **kwargs)
                    for kwargs in shard_kwargs
                ]
                shard_metrics = []
                for future in as_completed(futures):
                    result = future.result()
                    shard_metrics.append(result)
                    report_progress(progress, result)
            shard_metrics.sort(key=lambda item: item["shard_index"])
        metrics = aggregate_policy_match_shards(shard_metrics)
        metrics["jobs"] = args.jobs
        metrics["shard_hands"] = shard_hands
        metrics["paired_seats"] = True
        if args.out is not None:
            write_json(args.out, metrics)
        return metrics

    shard_metrics_by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    shard_kwargs = [
        {
            "checkpoint": args.checkpoint,
            "hands": shard_size,
            "seed": args.seed,
            "opponent_policy": args.opponent_policy,
            "equity_sims": args.equity_sims,
            "rollout_sims": args.rollout_sims,
            "rollout_margin": rollout_margin,
            "model_player": model_player,
            "shard_index": shard_index,
        }
        for model_player in model_players
        for shard_index, shard_size in enumerate(shard_hands)
    ]

    if args.jobs == 1:
        for kwargs in shard_kwargs:
            result = evaluate_model_shard(**kwargs)
            shard_metrics_by_player[int(result["model_player"])].append(result)
            report_progress(progress, result)
    else:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            mp_context=evaluation_process_context(),
        ) as executor:
            futures = [executor.submit(evaluate_model_shard, **kwargs) for kwargs in shard_kwargs]
            for future in as_completed(futures):
                result = future.result()
                shard_metrics_by_player[int(result["model_player"])].append(result)
                report_progress(progress, result)

    seat_metrics = []
    for model_player in model_players:
        player_shards = sorted(
            shard_metrics_by_player[model_player],
            key=lambda item: item["shard_index"],
        )
        seat_metrics.append(aggregate_policy_match_shards(player_shards))
    metrics = aggregate_model_player_metrics(seat_metrics)
    metrics["jobs"] = args.jobs
    metrics["shard_hands"] = shard_hands
    metrics["paired_seats"] = False
    if args.out is not None:
        write_json(args.out, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="random")
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--rollout-margin", type=float, default=1.0)
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

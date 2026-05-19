"""Evaluate a trained fixed-limit Hold'em policy checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

from alphapoker.holdem import FixedLimitHoldemState, HoldemPolicy, estimate_holdem_equity
from alphapoker.holdem_equity_feature import (
    equity_estimator_from_checkpoint,
    resolve_equity_checkpoint_path,
)
from alphapoker.holdem_evaluation import evaluate_policy_match
from alphapoker.holdem_features import (
    adapt_holdem_features,
    encode_holdem_state,
    holdem_legal_action_mask,
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
    for key in (
        "checkpoint",
        "policy",
        "opponent_policy",
        "equity_sims",
        "rollout_sims",
        "fallback_policy",
        "min_strategy_weight",
        "bet_threshold",
        "raise_threshold",
        "call_margin",
    ):
        if key in first:
            aggregated[key] = first[key]
    return aggregated


def model_policy_from_checkpoint(checkpoint_path: Path) -> HoldemPolicy:
    import torch

    from alphapoker.holdem_features import HOLDEM_CANONICAL_ACTIONS
    from alphapoker.holdem_model import HoldemPolicyNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    feature_equity_sims = checkpoint.get("feature_equity_sims")
    feature_equity_checkpoint = checkpoint.get("feature_equity_checkpoint")
    if feature_equity_sims is not None and feature_equity_checkpoint is not None:
        raise ValueError("Policy checkpoint cannot set both equity feature modes")
    feature_equity_fn = None
    if feature_equity_checkpoint is not None:
        feature_equity_path = resolve_equity_checkpoint_path(
            feature_equity_checkpoint,
            relative_to=checkpoint_path,
        )
        feature_equity_fn = equity_estimator_from_checkpoint(feature_equity_path)
    feature_rng = random.Random(0)
    model = HoldemPolicyNet(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def select_action(state: FixedLimitHoldemState) -> str:
        raw_features = encode_holdem_state(state)
        if feature_equity_sims is not None:
            player = state.current_player()
            raw_features.append(
                estimate_holdem_equity(
                    state.private_cards[player],
                    state.visible_board(),
                    simulations=int(feature_equity_sims),
                    rng=feature_rng,
                )
            )
        elif feature_equity_fn is not None:
            raw_features.append(feature_equity_fn(state))
        features = torch.tensor([adapt_holdem_features(raw_features, input_dim)], dtype=torch.float32)
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
) -> HoldemPolicy:
    return make_policy(name, rng, equity_sims, rollout_sims)


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_players = normalize_model_players(args.model_player)
    if len(model_players) > 1:
        seat_metrics = []
        for model_player in model_players:
            seat_args = copy.copy(args)
            seat_args.model_player = model_player
            seat_args.out = None
            seat_metrics.append(run(seat_args))
        metrics = aggregate_model_player_metrics(seat_metrics)
        if args.out is not None:
            write_json(args.out, metrics)
        return metrics

    opponent_rng = random.Random(args.seed + 1)
    metrics = {
        "checkpoint": str(args.checkpoint),
        **evaluate_policy_match(
            model_policy=model_policy_from_checkpoint(args.checkpoint),
            opponent_policy=make_opponent_policy(
                args.opponent_policy,
                opponent_rng,
                args.equity_sims,
                args.rollout_sims,
            ),
            hands=args.hands,
            seed=args.seed,
            model_player=model_players[0],
        ),
        "opponent_policy": args.opponent_policy,
        "equity_sims": args.equity_sims,
        "rollout_sims": args.rollout_sims,
    }
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
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

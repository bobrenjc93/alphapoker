"""Shared evaluation helpers for fixed-limit Hold'em policies."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any

from alphapoker.holdem import (
    HoldemPolicy,
    deal_fixed_limit_holdem,
    play_fixed_limit_holdem_hand,
)


def policies_for_model_player(
    model_policy: HoldemPolicy,
    opponent_policy: HoldemPolicy,
    model_player: int,
) -> tuple[HoldemPolicy, HoldemPolicy]:
    if model_player == 0:
        return (model_policy, opponent_policy)
    if model_player == 1:
        return (opponent_policy, model_policy)
    raise ValueError(f"model_player must be 0 or 1, got {model_player}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stderr(values: list[float], stdev: float) -> float:
    return stdev / (len(values) ** 0.5) if values else 0.0


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


def aggregate_policy_match_shards(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        raise ValueError("At least one metric is required")
    if len(metrics) == 1:
        return metrics[0]
    total_hands = sum(int(item["hands"]) for item in metrics)
    model_stdev = _pooled_stdev(metrics, "avg_utility_model", "utility_stdev_model")
    p0_stdev = _pooled_stdev(metrics, "avg_utility_p0", "utility_stdev_p0")
    first = metrics[0]
    aggregated = {
        "hands": total_hands,
        "model_player": first["model_player"],
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
        "shards": len(metrics),
        "shard_metrics": metrics,
    }
    for key in (
        "policy",
        "opponent_policy",
        "equity_sims",
        "rollout_sims",
        "bet_threshold",
        "raise_threshold",
        "call_margin",
    ):
        if key in first:
            aggregated[key] = first[key]
    return aggregated


def evaluate_policy_match(
    *,
    model_policy: HoldemPolicy,
    opponent_policy: HoldemPolicy,
    hands: int,
    seed: int,
    model_player: int = 0,
) -> dict[str, Any]:
    deal_rng = random.Random(seed)
    policies = policies_for_model_player(model_policy, opponent_policy, model_player)

    model_utilities: list[float] = []
    p0_utilities: list[float] = []
    total_actions = 0
    folds = 0
    showdowns = 0
    for _ in range(hands):
        terminal, actions = play_fixed_limit_holdem_hand(deal_fixed_limit_holdem(deal_rng), policies)
        model_utilities.append(terminal.utility(model_player))
        p0_utilities.append(terminal.utility(0))
        total_actions += len(actions)
        if terminal.showdown:
            showdowns += 1
        else:
            folds += 1

    model_stdev = statistics.stdev(model_utilities) if len(model_utilities) > 1 else 0.0
    p0_stdev = statistics.stdev(p0_utilities) if len(p0_utilities) > 1 else 0.0
    return {
        "hands": hands,
        "model_player": model_player,
        "avg_utility_model": _mean(model_utilities),
        "utility_stdev_model": model_stdev,
        "utility_stderr_model": _stderr(model_utilities, model_stdev),
        "avg_utility_p0": _mean(p0_utilities),
        "utility_stdev_p0": p0_stdev,
        "utility_stderr_p0": _stderr(p0_utilities, p0_stdev),
        "avg_actions": total_actions / hands if hands else 0.0,
        "folds": folds,
        "showdowns": showdowns,
        "seed": seed,
    }

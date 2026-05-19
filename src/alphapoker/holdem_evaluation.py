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
from alphapoker.kuhn import BET, CALL, CHECK, FOLD
from alphapoker.leduc import RAISE

HOLDEM_ACTIONS = (CHECK, BET, CALL, FOLD, RAISE)
ACTION_COUNT_KEYS = (
    "model_action_counts",
    "opponent_action_counts",
    "p0_action_counts",
    "p1_action_counts",
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


def empty_action_counts() -> dict[str, int]:
    return {action: 0 for action in HOLDEM_ACTIONS}


def add_action_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for action in HOLDEM_ACTIONS:
        target[action] += int(source.get(action, 0))


def _summed_action_counts(metrics: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = empty_action_counts()
    for item in metrics:
        add_action_counts(counts, item[key])
    return counts


def _record_action(
    *,
    action: str,
    player: int,
    model_player: int,
    model_action_counts: dict[str, int],
    opponent_action_counts: dict[str, int],
    p0_action_counts: dict[str, int],
    p1_action_counts: dict[str, int],
) -> None:
    if player == model_player:
        model_action_counts[action] += 1
    else:
        opponent_action_counts[action] += 1
    if player == 0:
        p0_action_counts[action] += 1
    else:
        p1_action_counts[action] += 1


def _match_metrics(
    *,
    hands: int,
    model_player: int | str,
    model_utilities: list[float],
    p0_utilities: list[float],
    total_actions: int,
    model_action_counts: dict[str, int],
    opponent_action_counts: dict[str, int],
    p0_action_counts: dict[str, int],
    p1_action_counts: dict[str, int],
    folds: int,
    showdowns: int,
    seed: int,
) -> dict[str, Any]:
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
        "model_action_counts": model_action_counts,
        "opponent_action_counts": opponent_action_counts,
        "p0_action_counts": p0_action_counts,
        "p1_action_counts": p1_action_counts,
        "folds": folds,
        "showdowns": showdowns,
        "seed": seed,
    }


def _utility_sample_count(metric: dict[str, Any]) -> int:
    return int(metric.get("paired_deals", metric["hands"]))


def _utility_count_key(metrics: list[dict[str, Any]]) -> str:
    return "paired_deals" if "paired_deals" in metrics[0] else "hands"


def _weighted_mean(
    metrics: list[dict[str, Any]],
    key: str,
    *,
    count_key: str = "hands",
) -> float:
    total_count = sum(int(item[count_key]) for item in metrics)
    return (
        sum(float(item[key]) * int(item[count_key]) for item in metrics) / total_count
        if total_count
        else 0.0
    )


def _pooled_stdev(metrics: list[dict[str, Any]], mean_key: str, stdev_key: str) -> float:
    total_samples = sum(_utility_sample_count(item) for item in metrics)
    if total_samples <= 1:
        return 0.0
    mean = _weighted_mean(metrics, mean_key, count_key=_utility_count_key(metrics))
    sum_squares = 0.0
    for item in metrics:
        samples = _utility_sample_count(item)
        stdev = float(item[stdev_key])
        item_mean = float(item[mean_key])
        sum_squares += max(0, samples - 1) * stdev * stdev
        sum_squares += samples * (item_mean - mean) * (item_mean - mean)
    return math.sqrt(sum_squares / (total_samples - 1))


def aggregate_policy_match_shards(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        raise ValueError("At least one metric is required")
    if len(metrics) == 1:
        return metrics[0]
    total_hands = sum(int(item["hands"]) for item in metrics)
    utility_samples = sum(_utility_sample_count(item) for item in metrics)
    model_stdev = _pooled_stdev(metrics, "avg_utility_model", "utility_stdev_model")
    p0_stdev = _pooled_stdev(metrics, "avg_utility_p0", "utility_stdev_p0")
    first = metrics[0]
    aggregated = {
        "hands": total_hands,
        "model_player": first["model_player"],
        "avg_utility_model": _weighted_mean(
            metrics,
            "avg_utility_model",
            count_key=_utility_count_key(metrics),
        ),
        "utility_stdev_model": model_stdev,
        "utility_stderr_model": model_stdev / (utility_samples**0.5)
        if utility_samples
        else 0.0,
        "avg_utility_p0": _weighted_mean(
            metrics,
            "avg_utility_p0",
            count_key=_utility_count_key(metrics),
        ),
        "utility_stdev_p0": p0_stdev,
        "utility_stderr_p0": p0_stdev / (utility_samples**0.5)
        if utility_samples
        else 0.0,
        "avg_actions": _weighted_mean(metrics, "avg_actions"),
        "folds": sum(int(item["folds"]) for item in metrics),
        "showdowns": sum(int(item["showdowns"]) for item in metrics),
        "seed": first["seed"],
        "shards": len(metrics),
        "shard_metrics": metrics,
    }
    for key in ACTION_COUNT_KEYS:
        if key in first:
            aggregated[key] = _summed_action_counts(metrics, key)
    if "hands_per_model_player" in first:
        aggregated["hands_per_model_player"] = sum(
            int(item["hands_per_model_player"]) for item in metrics
        )
    if "paired_deals" in first:
        aggregated["paired_deals"] = sum(int(item["paired_deals"]) for item in metrics)
    if "seat_metrics" in first:
        seat_count = len(first["seat_metrics"])
        aggregated["seat_metrics"] = [
            aggregate_policy_match_shards(
                [item["seat_metrics"][seat_index] for item in metrics]
            )
            for seat_index in range(seat_count)
        ]
    for key in (
        "policy",
        "checkpoint",
        "opponent_policy",
        "equity_sims",
        "rollout_sims",
        "rollout_margin",
        "blend_checkpoint",
        "blend_weight",
        "fallback_policy",
        "min_strategy_weight",
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
    model_action_counts = empty_action_counts()
    opponent_action_counts = empty_action_counts()
    p0_action_counts = empty_action_counts()
    p1_action_counts = empty_action_counts()
    folds = 0
    showdowns = 0
    for _ in range(hands):
        terminal, actions = play_fixed_limit_holdem_hand(deal_fixed_limit_holdem(deal_rng), policies)
        model_utilities.append(terminal.utility(model_player))
        p0_utilities.append(terminal.utility(0))
        total_actions += len(actions)
        for player, action in actions:
            _record_action(
                action=action,
                player=player,
                model_player=model_player,
                model_action_counts=model_action_counts,
                opponent_action_counts=opponent_action_counts,
                p0_action_counts=p0_action_counts,
                p1_action_counts=p1_action_counts,
            )
        if terminal.showdown:
            showdowns += 1
        else:
            folds += 1

    return _match_metrics(
        hands=hands,
        model_player=model_player,
        model_utilities=model_utilities,
        p0_utilities=p0_utilities,
        total_actions=total_actions,
        model_action_counts=model_action_counts,
        opponent_action_counts=opponent_action_counts,
        p0_action_counts=p0_action_counts,
        p1_action_counts=p1_action_counts,
        folds=folds,
        showdowns=showdowns,
        seed=seed,
    )


def evaluate_policy_match_paired_seats(
    *,
    model_policies: tuple[HoldemPolicy, HoldemPolicy],
    opponent_policies: tuple[HoldemPolicy, HoldemPolicy],
    hands: int,
    seed: int,
) -> dict[str, Any]:
    deal_rng = random.Random(seed)
    seat_metrics = []
    paired_model_utilities: list[float] = []
    paired_p0_utilities: list[float] = []
    total_actions = 0
    model_action_counts = empty_action_counts()
    opponent_action_counts = empty_action_counts()
    p0_action_counts = empty_action_counts()
    p1_action_counts = empty_action_counts()
    folds = 0
    showdowns = 0

    seat_model_utilities: dict[int, list[float]] = {0: [], 1: []}
    seat_p0_utilities: dict[int, list[float]] = {0: [], 1: []}
    seat_actions: dict[int, int] = {0: 0, 1: 0}
    seat_model_action_counts = {0: empty_action_counts(), 1: empty_action_counts()}
    seat_opponent_action_counts = {0: empty_action_counts(), 1: empty_action_counts()}
    seat_p0_action_counts = {0: empty_action_counts(), 1: empty_action_counts()}
    seat_p1_action_counts = {0: empty_action_counts(), 1: empty_action_counts()}
    seat_folds: dict[int, int] = {0: 0, 1: 0}
    seat_showdowns: dict[int, int] = {0: 0, 1: 0}

    for _ in range(hands):
        state = deal_fixed_limit_holdem(deal_rng)
        hand_model_utilities = []
        hand_p0_utilities = []
        for model_player in (0, 1):
            terminal, actions = play_fixed_limit_holdem_hand(
                state,
                policies_for_model_player(
                    model_policies[model_player],
                    opponent_policies[model_player],
                    model_player,
                ),
            )
            model_utility = terminal.utility(model_player)
            p0_utility = terminal.utility(0)
            action_count = len(actions)
            seat_model_utilities[model_player].append(model_utility)
            seat_p0_utilities[model_player].append(p0_utility)
            seat_actions[model_player] += action_count
            total_actions += action_count
            for player, action in actions:
                _record_action(
                    action=action,
                    player=player,
                    model_player=model_player,
                    model_action_counts=seat_model_action_counts[model_player],
                    opponent_action_counts=seat_opponent_action_counts[model_player],
                    p0_action_counts=seat_p0_action_counts[model_player],
                    p1_action_counts=seat_p1_action_counts[model_player],
                )
                _record_action(
                    action=action,
                    player=player,
                    model_player=model_player,
                    model_action_counts=model_action_counts,
                    opponent_action_counts=opponent_action_counts,
                    p0_action_counts=p0_action_counts,
                    p1_action_counts=p1_action_counts,
                )
            if terminal.showdown:
                seat_showdowns[model_player] += 1
                showdowns += 1
            else:
                seat_folds[model_player] += 1
                folds += 1
            hand_model_utilities.append(model_utility)
            hand_p0_utilities.append(p0_utility)
        paired_model_utilities.append(sum(hand_model_utilities) / 2.0)
        paired_p0_utilities.append(sum(hand_p0_utilities) / 2.0)

    for model_player in (0, 1):
        seat_metrics.append(
            _match_metrics(
                hands=hands,
                model_player=model_player,
                model_utilities=seat_model_utilities[model_player],
                p0_utilities=seat_p0_utilities[model_player],
                total_actions=seat_actions[model_player],
                model_action_counts=seat_model_action_counts[model_player],
                opponent_action_counts=seat_opponent_action_counts[model_player],
                p0_action_counts=seat_p0_action_counts[model_player],
                p1_action_counts=seat_p1_action_counts[model_player],
                folds=seat_folds[model_player],
                showdowns=seat_showdowns[model_player],
                seed=seed,
            )
        )

    metrics = _match_metrics(
        hands=hands * 2,
        model_player="both",
        model_utilities=paired_model_utilities,
        p0_utilities=paired_p0_utilities,
        total_actions=total_actions,
        model_action_counts=model_action_counts,
        opponent_action_counts=opponent_action_counts,
        p0_action_counts=p0_action_counts,
        p1_action_counts=p1_action_counts,
        folds=folds,
        showdowns=showdowns,
        seed=seed,
    )
    metrics["hands_per_model_player"] = hands
    metrics["paired_deals"] = hands
    metrics["seat_metrics"] = seat_metrics
    return metrics

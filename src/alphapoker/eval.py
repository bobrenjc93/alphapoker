"""Exact evaluation utilities for small poker games."""

from __future__ import annotations

from itertools import product
from math import isfinite

from alphapoker.kuhn import (
    all_card_deals,
    all_infoset_keys,
    legal_actions_for_history,
    parse_infoset_key,
    KuhnState,
)

ActionDistribution = dict[str, float]
StrategyProfile = dict[str, ActionDistribution]


def normalize_distribution(actions: tuple[str, ...], dist: ActionDistribution) -> ActionDistribution:
    total = sum(max(0.0, dist.get(action, 0.0)) for action in actions)
    if total <= 0.0:
        prob = 1.0 / len(actions)
        return {action: prob for action in actions}
    return {action: max(0.0, dist.get(action, 0.0)) / total for action in actions}


def policy_for_state(strategy: StrategyProfile, state: KuhnState) -> ActionDistribution:
    actions = state.legal_actions()
    return normalize_distribution(actions, strategy.get(state.infoset_key(), {}))


def expected_utility(strategy: StrategyProfile, player: int = 0) -> float:
    """Return exact expected utility for a player under a mixed strategy profile."""

    if player not in (0, 1):
        raise ValueError(f"Unknown player: {player}")

    def walk(state: KuhnState) -> float:
        if state.is_terminal():
            return state.utility(player)

        dist = policy_for_state(strategy, state)
        return sum(prob * walk(state.apply(action)) for action, prob in dist.items())

    return sum(walk(KuhnState.initial(cards)) for cards in all_card_deals()) / len(all_card_deals())


def deterministic_strategy(keys: tuple[str, ...], choices: tuple[str, ...]) -> StrategyProfile:
    profile: StrategyProfile = {}
    for key, choice in zip(keys, choices):
        _, _, history = parse_infoset_key(key)
        actions = legal_actions_for_history(history)
        profile[key] = {action: 1.0 if action == choice else 0.0 for action in actions}
    return profile


def best_response_value(player: int, opponent_strategy: StrategyProfile) -> float:
    """Compute an exact best-response value by enumerating deterministic policies.

    Kuhn poker has only six information sets per player, each with two actions,
    so exhaustive enumeration is clearer and less error-prone than a specialized
    dynamic-programming best-response implementation.
    """

    if player not in (0, 1):
        raise ValueError(f"Unknown player: {player}")

    keys = all_infoset_keys(player)
    action_space = []
    for key in keys:
        _, _, history = parse_infoset_key(key)
        action_space.append(legal_actions_for_history(history))

    best = float("-inf")
    for choices in product(*action_space):
        candidate: StrategyProfile = {
            key: dict(dist) for key, dist in opponent_strategy.items()
        }
        candidate.update(deterministic_strategy(keys, choices))
        value = expected_utility(candidate, player=player)
        if value > best:
            best = value

    if not isfinite(best):
        raise RuntimeError("Best response search failed")
    return best


def nash_conv(strategy: StrategyProfile) -> float:
    """Return two-player NashConv for Kuhn poker.

    At equilibrium, player 0's best-response value plus player 1's
    best-response value is zero. Positive values measure total improvement
    available to unilateral deviations.
    """

    return max(0.0, best_response_value(0, strategy) + best_response_value(1, strategy))


def exploitability(strategy: StrategyProfile) -> float:
    return 0.5 * nash_conv(strategy)


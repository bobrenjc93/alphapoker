"""Feature encoders for fixed-limit Hold'em policies."""

from __future__ import annotations

from alphapoker.holdem import (
    FixedLimitHoldemState,
    HOLDEM_RANKS,
    HOLDEM_SUITS,
    evaluate_holdem_hand,
    pot_odds_call_threshold,
)
from alphapoker.kuhn import BET, CALL, CHECK, FOLD
from alphapoker.leduc import RAISE

HOLDEM_CANONICAL_ACTIONS = (CHECK, BET, CALL, FOLD, RAISE)
HOLDEM_BASE_FEATURE_DIM = 117
HOLDEM_HAND_STRENGTH_FEATURE_DIM = 11
HOLDEM_ACTION_HISTORY_FEATURE_DIM = 5
HOLDEM_FEATURE_DIM = 140


def holdem_card_index(card: str) -> int:
    rank = HOLDEM_RANKS.index(card[0])
    suit = HOLDEM_SUITS.index(card[1])
    return rank * len(HOLDEM_SUITS) + suit


def encode_holdem_state(state: FixedLimitHoldemState) -> list[float]:
    player = state.current_player()
    features = [0.0 for _ in range(52 * 2)]
    for card in state.private_cards[player]:
        features[holdem_card_index(card)] = 1.0
    for card in state.visible_board():
        features[52 + holdem_card_index(card)] = 1.0

    features.extend(1.0 if state.street == street else 0.0 for street in range(4))
    features.extend((1.0, 0.0) if player == 0 else (0.0, 1.0))
    features.extend(
        [
            state.contributions[0] / 100.0,
            state.contributions[1] / 100.0,
            state.round_contributions[0] / 40.0,
            state.round_contributions[1] / 40.0,
            state.outstanding_call_amount() / 20.0,
            state.bets_this_round / max(1, state.max_bets_per_round),
            sum(state.contributions) / 200.0,
        ]
    )
    rank_indices = sorted(HOLDEM_RANKS.index(card[0]) for card in state.private_cards[player])
    suits = [card[1] for card in state.private_cards[player]]
    features.extend(
        [
            rank_indices[1] / (len(HOLDEM_RANKS) - 1),
            rank_indices[0] / (len(HOLDEM_RANKS) - 1),
            1.0 if rank_indices[0] == rank_indices[1] else 0.0,
            1.0 if suits[0] == suits[1] else 0.0,
            abs(rank_indices[1] - rank_indices[0]) / (len(HOLDEM_RANKS) - 1),
            len(state.visible_board()) / 5.0,
        ]
    )
    legal = set(state.legal_actions())
    features.extend(1.0 if action in legal else 0.0 for action in HOLDEM_CANONICAL_ACTIONS)
    features.append(pot_odds_call_threshold(state))
    features.extend(holdem_made_hand_features(state, player))
    return features


def holdem_made_hand_features(state: FixedLimitHoldemState, player: int) -> list[float]:
    visible_board = state.visible_board()
    if len(visible_board) < 3:
        return [0.0 for _ in range(HOLDEM_HAND_STRENGTH_FEATURE_DIM)]

    result = evaluate_holdem_hand(state.private_cards[player], visible_board)
    rank_strength = 1.0 - ((result.rank_class - 1) / 8.0)
    score_strength = (7463.0 - result.score) / 7462.0
    rank_class_one_hot = [0.0 for _ in range(9)]
    rank_class_one_hot[result.rank_class - 1] = 1.0
    return [rank_strength, score_strength, *rank_class_one_hot]


def iter_holdem_history_actions(state: FixedLimitHoldemState):
    for street, history in enumerate(state.histories):
        actor = 0 if street == 0 else 1
        for action in history:
            yield street, actor, action
            actor = 1 - actor


def encode_holdem_action_history_features(state: FixedLimitHoldemState) -> list[float]:
    player = state.current_player()
    own_aggressions = 0
    opponent_aggressions = 0
    own_street_aggressions = 0
    opponent_street_aggressions = 0
    last_actor = None
    last_action = None
    for street, actor, action in iter_holdem_history_actions(state):
        if action in (BET, RAISE):
            if actor == player:
                own_aggressions += 1
                if street == state.street:
                    own_street_aggressions += 1
            else:
                opponent_aggressions += 1
                if street == state.street:
                    opponent_street_aggressions += 1
        last_actor = actor
        last_action = action

    max_total_aggressions = max(1, state.max_bets_per_round * 4)
    max_street_aggressions = max(1, state.max_bets_per_round)
    last_opponent_aggression = (
        last_actor == 1 - player and last_action in (BET, RAISE)
    )
    return [
        own_aggressions / max_total_aggressions,
        opponent_aggressions / max_total_aggressions,
        own_street_aggressions / max_street_aggressions,
        opponent_street_aggressions / max_street_aggressions,
        1.0 if last_opponent_aggression else 0.0,
    ]


def adapt_holdem_features(features: list[float], input_dim: int) -> list[float]:
    if len(features) == input_dim:
        return features
    if len(features) > input_dim:
        return features[:input_dim]
    return [*features, *([0.0] * (input_dim - len(features)))]


def holdem_legal_action_mask(state: FixedLimitHoldemState) -> list[bool]:
    legal = set(state.legal_actions())
    return [action in legal for action in HOLDEM_CANONICAL_ACTIONS]


def holdem_action_index(action: str) -> int:
    return HOLDEM_CANONICAL_ACTIONS.index(action)

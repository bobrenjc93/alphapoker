"""Feature encoders for fixed-limit Hold'em policies."""

from __future__ import annotations

from alphapoker.holdem import FixedLimitHoldemState, HOLDEM_RANKS, HOLDEM_SUITS
from alphapoker.kuhn import BET, CALL, CHECK, FOLD
from alphapoker.leduc import RAISE

HOLDEM_CANONICAL_ACTIONS = (CHECK, BET, CALL, FOLD, RAISE)


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
    return features


def holdem_legal_action_mask(state: FixedLimitHoldemState) -> list[bool]:
    legal = set(state.legal_actions())
    return [action in legal for action in HOLDEM_CANONICAL_ACTIONS]


def holdem_action_index(action: str) -> int:
    return HOLDEM_CANONICAL_ACTIONS.index(action)


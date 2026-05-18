"""Feature encoders shared by neural models and datasets."""

from __future__ import annotations

from alphapoker.kuhn import (
    BET,
    CANONICAL_ACTIONS,
    CHECK,
    CARDS,
    History,
    legal_actions_for_history,
    parse_infoset_key,
)

HISTORIES: tuple[History, ...] = (
    (),
    (CHECK,),
    (BET,),
    (CHECK, BET),
)


def encode_infoset(key: str) -> list[float]:
    player, card, history = parse_infoset_key(key)
    features: list[float] = []
    features.extend(1.0 if card == candidate else 0.0 for candidate in CARDS)
    features.extend((1.0, 0.0) if player == 0 else (0.0, 1.0))
    features.extend(1.0 if history == candidate else 0.0 for candidate in HISTORIES)
    return features


def action_policy_vector(key: str, distribution: dict[str, float]) -> list[float]:
    _, _, history = parse_infoset_key(key)
    legal = set(legal_actions_for_history(history))
    return [
        float(distribution.get(action, 0.0)) if action in legal else 0.0
        for action in CANONICAL_ACTIONS
    ]


def legal_action_mask(key: str) -> list[bool]:
    _, _, history = parse_infoset_key(key)
    legal = set(legal_actions_for_history(history))
    return [action in legal for action in CANONICAL_ACTIONS]

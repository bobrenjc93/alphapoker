"""Feature encoders for Leduc information sets."""

from __future__ import annotations

from dataclasses import dataclass

from alphapoker.kuhn import BET, CALL, CHECK, FOLD
from alphapoker.leduc import RAISE, RANK_NAMES

LEDUC_CANONICAL_ACTIONS = (CHECK, BET, CALL, FOLD, RAISE)


@dataclass(frozen=True)
class ParsedLeducInfoSet:
    player: int
    private_rank: str
    public_rank: str
    round_histories: tuple[tuple[str, ...], tuple[str, ...]]


def _parse_history_label(label: str) -> tuple[str, ...]:
    return () if label == "root" else tuple(label.split("-"))


def parse_leduc_infoset_key(key: str) -> ParsedLeducInfoSet:
    player_text, rest = key.split(":private=", maxsplit=1)
    private_rank, rest = rest.split(":public=", maxsplit=1)
    public_rank, history_text = rest.split(":r0:", maxsplit=1)
    round0_text, round1_text = history_text.split("|r1:", maxsplit=1)
    return ParsedLeducInfoSet(
        player=int(player_text[1:]),
        private_rank=private_rank,
        public_rank=public_rank,
        round_histories=(
            _parse_history_label(round0_text),
            _parse_history_label(round1_text),
        ),
    )


def encode_leduc_infoset(key: str) -> list[float]:
    parsed = parse_leduc_infoset_key(key)
    features: list[float] = []

    features.extend((1.0, 0.0) if parsed.player == 0 else (0.0, 1.0))
    features.extend(1.0 if parsed.private_rank == rank else 0.0 for rank in RANK_NAMES)
    features.append(1.0 if parsed.public_rank == "-" else 0.0)
    features.extend(1.0 if parsed.public_rank == rank else 0.0 for rank in RANK_NAMES)

    for history in parsed.round_histories:
        for action in LEDUC_CANONICAL_ACTIONS:
            features.append(history.count(action) / 2.0)

    return features


def leduc_action_policy_vector(distribution: dict[str, float]) -> list[float]:
    return [float(distribution.get(action, 0.0)) for action in LEDUC_CANONICAL_ACTIONS]


def leduc_legal_action_mask(distribution: dict[str, float]) -> list[bool]:
    legal = set(distribution)
    return [action in legal for action in LEDUC_CANONICAL_ACTIONS]


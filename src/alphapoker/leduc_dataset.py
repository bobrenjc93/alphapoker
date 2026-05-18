"""Dataset helpers for Leduc policy distillation."""

from __future__ import annotations

from dataclasses import dataclass

from alphapoker.leduc_cfr import LeducStrategyProfile
from alphapoker.leduc_features import (
    encode_leduc_infoset,
    leduc_action_policy_vector,
    leduc_legal_action_mask,
)


@dataclass(frozen=True)
class LeducStrategyExample:
    infoset: str
    features: list[float]
    policy: list[float]
    legal_mask: list[bool]


def leduc_strategy_examples(strategy: LeducStrategyProfile) -> list[LeducStrategyExample]:
    return [
        LeducStrategyExample(
            infoset=key,
            features=encode_leduc_infoset(key),
            policy=leduc_action_policy_vector(distribution),
            legal_mask=leduc_legal_action_mask(distribution),
        )
        for key, distribution in sorted(strategy.items())
    ]


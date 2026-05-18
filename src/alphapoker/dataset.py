"""Dataset helpers for policy distillation."""

from __future__ import annotations

from dataclasses import dataclass

from alphapoker.eval import StrategyProfile
from alphapoker.features import action_policy_vector, encode_infoset, legal_action_mask


@dataclass(frozen=True)
class StrategyExample:
    infoset: str
    features: list[float]
    policy: list[float]
    legal_mask: list[bool]


def strategy_examples(strategy: StrategyProfile) -> list[StrategyExample]:
    return [
        StrategyExample(
            infoset=key,
            features=encode_infoset(key),
            policy=action_policy_vector(key, distribution),
            legal_mask=legal_action_mask(key),
        )
        for key, distribution in sorted(strategy.items())
    ]


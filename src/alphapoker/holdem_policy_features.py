"""Feature encoders for trained Hold'em policy checkpoints."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alphapoker.holdem import (
    FixedLimitHoldemState,
    estimate_holdem_equity,
    sampled_holdem_equity,
    turn_river_exact_holdem_equity,
)
from alphapoker.holdem_equity_feature import (
    HoldemEquityEstimator,
    equity_estimator_from_checkpoint,
    resolve_equity_checkpoint_path,
)
from alphapoker.holdem_features import adapt_holdem_features, encode_holdem_state


@dataclass
class HoldemPolicyFeatureEncoder:
    input_dim: int
    feature_equity_sims: int | None = None
    feature_equity_mode: str | None = None
    feature_equity_checkpoint: str | None = None
    feature_equity_fn: HoldemEquityEstimator | None = None
    feature_rng: random.Random | None = None

    @classmethod
    def base(cls, input_dim: int) -> "HoldemPolicyFeatureEncoder":
        return cls(input_dim=input_dim)

    def encode(self, state: FixedLimitHoldemState) -> list[float]:
        features = encode_holdem_state(state)
        if self.feature_equity_sims is not None:
            mode = self.feature_equity_mode or "random"
            player = state.current_player()
            if mode == "random":
                feature_rng = self.feature_rng
                if feature_rng is None:
                    feature_rng = random.Random()
                    self.feature_rng = feature_rng
                equity = estimate_holdem_equity(
                    state.private_cards[player],
                    state.visible_board(),
                    simulations=self.feature_equity_sims,
                    rng=feature_rng,
                )
            elif mode == "sampled":
                equity = sampled_holdem_equity(
                    state.private_cards[player],
                    state.visible_board(),
                    simulations=self.feature_equity_sims,
                )
            elif mode == "turn-river-exact":
                equity = turn_river_exact_holdem_equity(
                    state.private_cards[player],
                    state.visible_board(),
                    simulations=self.feature_equity_sims,
                )
            else:
                raise ValueError(f"Unknown feature equity mode: {mode}")
            features.append(equity)
        elif self.feature_equity_fn is not None:
            features.append(self.feature_equity_fn(state))
        return adapt_holdem_features(features, self.input_dim)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "feature_equity_sims": self.feature_equity_sims,
            "feature_equity_mode": (
                self.feature_equity_mode if self.feature_equity_sims is not None else None
            ),
            "feature_equity_checkpoint": self.feature_equity_checkpoint,
        }


def policy_feature_encoder_from_checkpoint_data(
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: Path | None = None,
    feature_seed: int = 0,
) -> HoldemPolicyFeatureEncoder:
    input_dim = int(checkpoint["input_dim"])
    feature_equity_sims = checkpoint.get("feature_equity_sims")
    feature_equity_mode = checkpoint.get(
        "feature_equity_mode",
        "random" if feature_equity_sims is not None else None,
    )
    feature_equity_checkpoint = checkpoint.get("feature_equity_checkpoint")
    if feature_equity_sims is not None and feature_equity_checkpoint is not None:
        raise ValueError("Policy checkpoint cannot set both equity feature modes")

    feature_equity_fn = None
    resolved_feature_equity_checkpoint = None
    if feature_equity_checkpoint is not None:
        feature_equity_path = resolve_equity_checkpoint_path(
            feature_equity_checkpoint,
            relative_to=checkpoint_path,
        )
        feature_equity_fn = equity_estimator_from_checkpoint(feature_equity_path)
        resolved_feature_equity_checkpoint = str(feature_equity_path.resolve())

    return HoldemPolicyFeatureEncoder(
        input_dim=input_dim,
        feature_equity_sims=(
            int(feature_equity_sims) if feature_equity_sims is not None else None
        ),
        feature_equity_mode=feature_equity_mode,
        feature_equity_checkpoint=resolved_feature_equity_checkpoint,
        feature_equity_fn=feature_equity_fn,
        feature_rng=random.Random(feature_seed),
    )

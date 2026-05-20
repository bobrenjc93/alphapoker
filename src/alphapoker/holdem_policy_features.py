"""Feature encoders for trained Hold'em policy checkpoints."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alphapoker.holdem import (
    FixedLimitHoldemState,
    HoldemPolicy,
    estimate_holdem_equity,
    policy_filtered_holdem_equity,
    sampled_holdem_equity,
    turn_river_exact_holdem_equity,
    turn_river_exact_pot_odds_equity_policy,
)
from alphapoker.holdem_equity_feature import (
    HoldemEquityEstimator,
    equity_estimator_from_checkpoint,
    resolve_equity_checkpoint_path,
)
from alphapoker.holdem_features import (
    HOLDEM_ACTION_HISTORY_FEATURE_DIM,
    adapt_holdem_features,
    encode_holdem_action_history_features,
    encode_holdem_state,
)

POLICY_FEATURE_EQUITY_MODES = ("random", "sampled", "turn-river-exact", "tight-range")


def _tight_range_opponent_policy_factory(simulations: int):
    def factory(_: random.Random) -> HoldemPolicy:
        return turn_river_exact_pot_odds_equity_policy(
            simulations=simulations,
            bet_threshold=0.62,
            raise_threshold=0.84,
            call_margin=0.08,
        )

    return factory


@dataclass
class HoldemPolicyFeatureEncoder:
    input_dim: int
    feature_equity_sims: int | None = None
    feature_equity_mode: str | None = None
    feature_equity_checkpoint: str | None = None
    feature_equity_fn: HoldemEquityEstimator | None = None
    feature_rng: random.Random | None = None
    action_history_features: bool = False

    @classmethod
    def base(cls, input_dim: int) -> "HoldemPolicyFeatureEncoder":
        return cls(input_dim=input_dim)

    def encode(self, state: FixedLimitHoldemState) -> list[float]:
        features = encode_holdem_state(state)
        if self.feature_equity_sims is not None:
            mode = self.feature_equity_mode or "random"
            if mode not in POLICY_FEATURE_EQUITY_MODES:
                raise ValueError(f"Unknown feature equity mode: {mode}")
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
            elif mode == "tight-range":
                feature_rng = self.feature_rng
                if feature_rng is None:
                    feature_rng = random.Random()
                    self.feature_rng = feature_rng
                equity = policy_filtered_holdem_equity(
                    state,
                    player,
                    feature_rng,
                    simulations=self.feature_equity_sims,
                    opponent_policy_factory=_tight_range_opponent_policy_factory(
                        self.feature_equity_sims
                    ),
                    cache_policy_matches=True,
                )
            features.append(equity)
        elif self.feature_equity_fn is not None:
            features.append(self.feature_equity_fn(state))
        if self.action_history_features:
            features.extend(encode_holdem_action_history_features(state))
        return adapt_holdem_features(features, self.input_dim)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "feature_equity_sims": self.feature_equity_sims,
            "feature_equity_mode": (
                self.feature_equity_mode if self.feature_equity_sims is not None else None
            ),
            "feature_equity_checkpoint": self.feature_equity_checkpoint,
            "action_history_features": self.action_history_features,
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
    action_history_features = bool(checkpoint.get("action_history_features", False))
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
        action_history_features=action_history_features,
    )


def policy_feature_encoder_for_training(
    *,
    base_input_dim: int,
    checkpoint: dict[str, Any] | None = None,
    checkpoint_path: Path | None = None,
    feature_seed: int = 0,
    feature_equity_sims: int | None = None,
    feature_equity_mode: str | None = None,
    action_history_features: bool | None = None,
) -> HoldemPolicyFeatureEncoder:
    if feature_equity_mode is not None and feature_equity_mode not in POLICY_FEATURE_EQUITY_MODES:
        raise ValueError(f"Unknown feature equity mode: {feature_equity_mode}")

    if checkpoint is None:
        if feature_equity_mode is not None and feature_equity_sims is None:
            raise ValueError("--feature-equity-mode requires --feature-equity-sims")
        use_action_history_features = bool(action_history_features)
        input_dim = (
            base_input_dim
            + (1 if feature_equity_sims is not None else 0)
            + (
                HOLDEM_ACTION_HISTORY_FEATURE_DIM
                if use_action_history_features
                else 0
            )
        )
        return HoldemPolicyFeatureEncoder(
            input_dim=input_dim,
            feature_equity_sims=feature_equity_sims,
            feature_equity_mode=feature_equity_mode,
            feature_rng=random.Random(feature_seed),
            action_history_features=use_action_history_features,
        )

    encoder = policy_feature_encoder_from_checkpoint_data(
        checkpoint,
        checkpoint_path=checkpoint_path,
        feature_seed=feature_seed,
    )
    if (
        feature_equity_sims is None
        and feature_equity_mode is None
        and action_history_features is None
    ):
        return encoder

    override_sims = (
        feature_equity_sims
        if feature_equity_sims is not None
        else encoder.feature_equity_sims
    )
    if override_sims is None:
        raise ValueError("feature equity override requires --feature-equity-sims")
    if encoder.input_dim <= base_input_dim:
        raise ValueError("init checkpoint has no equity feature input to override")
    return HoldemPolicyFeatureEncoder(
        input_dim=encoder.input_dim
        + (
            HOLDEM_ACTION_HISTORY_FEATURE_DIM
            if bool(action_history_features) and not encoder.action_history_features
            else 0
        ),
        feature_equity_sims=override_sims,
        feature_equity_mode=feature_equity_mode or encoder.feature_equity_mode or "random",
        feature_rng=random.Random(feature_seed),
        action_history_features=(
            encoder.action_history_features
            if action_history_features is None
            else action_history_features
        ),
    )

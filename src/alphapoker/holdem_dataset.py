"""Generate supervised Hold'em policy-distillation examples."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from alphapoker.holdem import (
    FixedLimitHoldemState,
    HoldemPolicy,
    deal_fixed_limit_holdem,
    equity_threshold_policy,
    estimate_holdem_equity,
    sampled_holdem_equity,
    turn_river_exact_holdem_equity,
)
from alphapoker.holdem_equity_feature import HoldemEquityEstimator
from alphapoker.holdem_features import (
    encode_holdem_state,
    holdem_action_index,
    holdem_legal_action_mask,
)
from alphapoker.holdem_self_play import make_policy

HOLDEM_EXPERT_POLICIES = (
    "equity",
    "pot-odds",
    "cached-pot-odds",
    "tuned-pot-odds",
    "cached-tuned-pot-odds",
    "river-exact-tuned-pot-odds",
    "turn-river-exact-tuned-pot-odds",
    "tight-turn-river-exact-pot-odds",
    "balanced-turn-river-exact-pot-odds",
    "hybrid-pot-odds",
    "rollout-pot-odds",
    "cached-rollout-pot-odds",
    "tuned-rollout-pot-odds",
    "cached-tuned-rollout-pot-odds",
    "tight-rollout-pot-odds",
    "balanced-rollout-pot-odds",
    "tight-safe-rollout-pot-odds",
    "balanced-safe-rollout-pot-odds",
)
HOLDEM_DATASET_OPPONENT_POLICIES = (
    "equity",
    "pot-odds",
    "cached-pot-odds",
    "tuned-pot-odds",
    "cached-tuned-pot-odds",
    "river-exact-tuned-pot-odds",
    "turn-river-exact-tuned-pot-odds",
    "tight-turn-river-exact-pot-odds",
    "balanced-turn-river-exact-pot-odds",
    "hybrid-pot-odds",
    "random",
    "rollout-pot-odds",
    "cached-rollout-pot-odds",
    "tuned-rollout-pot-odds",
    "cached-tuned-rollout-pot-odds",
    "tight-rollout-pot-odds",
    "balanced-rollout-pot-odds",
    "tight-safe-rollout-pot-odds",
    "balanced-safe-rollout-pot-odds",
)
HOLDEM_EQUITY_VALUE_OPPONENT_POLICIES = (
    "equity",
    "pot-odds",
    "cached-pot-odds",
    "tuned-pot-odds",
    "cached-tuned-pot-odds",
    "river-exact-tuned-pot-odds",
    "turn-river-exact-tuned-pot-odds",
    "tight-turn-river-exact-pot-odds",
    "balanced-turn-river-exact-pot-odds",
    "hybrid-pot-odds",
    "random",
)
HOLDEM_FEATURE_EQUITY_MODES = ("random", "sampled", "turn-river-exact")


@dataclass(frozen=True)
class HoldemPolicyExample:
    features: list[float]
    action_index: int
    legal_mask: list[bool]


@dataclass(frozen=True)
class HoldemEquityExample:
    features: list[float]
    equity: float


def encode_policy_example_features(
    state: FixedLimitHoldemState,
    *,
    feature_equity_sims: int | None = None,
    feature_equity_mode: str = "random",
    feature_rng: random.Random | None = None,
    feature_equity_fn: HoldemEquityEstimator | None = None,
) -> list[float]:
    if feature_equity_mode not in HOLDEM_FEATURE_EQUITY_MODES:
        raise ValueError(f"Unknown feature equity mode: {feature_equity_mode}")
    if feature_equity_sims is not None and feature_equity_fn is not None:
        raise ValueError("Set only one of feature_equity_sims or feature_equity_fn")
    if feature_equity_fn is not None and feature_equity_mode != "random":
        raise ValueError("feature_equity_mode is only used with feature_equity_sims")
    features = encode_holdem_state(state)
    if feature_equity_fn is not None:
        features.append(float(feature_equity_fn(state)))
        return features
    if feature_equity_sims is None:
        return features
    if feature_rng is None:
        feature_rng = random.Random()
    player = state.current_player()
    if feature_equity_mode == "random":
        equity = estimate_holdem_equity(
            state.private_cards[player],
            state.visible_board(),
            simulations=feature_equity_sims,
            rng=feature_rng,
        )
    elif feature_equity_mode == "sampled":
        equity = sampled_holdem_equity(
            state.private_cards[player],
            state.visible_board(),
            simulations=feature_equity_sims,
        )
    elif feature_equity_mode == "turn-river-exact":
        equity = turn_river_exact_holdem_equity(
            state.private_cards[player],
            state.visible_board(),
            simulations=feature_equity_sims,
        )
    else:
        raise AssertionError(f"Unhandled feature equity mode: {feature_equity_mode}")
    features.append(equity)
    return features


def write_policy_examples(path: Path, examples: list[HoldemPolicyExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "features": example.features,
            "action_index": example.action_index,
            "legal_mask": example.legal_mask,
        }
        for example in examples
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_policy_examples(path: Path) -> list[HoldemPolicyExample]:
    payload = json.loads(path.read_text())
    return [
        HoldemPolicyExample(
            features=[float(value) for value in item["features"]],
            action_index=int(item["action_index"]),
            legal_mask=[bool(value) for value in item["legal_mask"]],
        )
        for item in payload
    ]


def write_equity_value_examples(path: Path, examples: list[HoldemEquityExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"features": example.features, "equity": example.equity} for example in examples]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_equity_value_examples(path: Path) -> list[HoldemEquityExample]:
    payload = json.loads(path.read_text())
    return [
        HoldemEquityExample(
            features=[float(value) for value in item["features"]],
            equity=float(item["equity"]),
        )
        for item in payload
    ]


def generate_equity_policy_examples(
    *,
    hands: int,
    seed: int,
    equity_sims: int,
    expert_player: int | None = None,
    expert_policy: str = "equity",
    opponent_policy: str = "equity",
    rollout_sims: int | None = None,
    feature_equity_sims: int | None = None,
    feature_equity_mode: str = "random",
    feature_equity_fn: HoldemEquityEstimator | None = None,
    expert_behavior_policy: HoldemPolicy | None = None,
) -> list[HoldemPolicyExample]:
    if feature_equity_mode not in HOLDEM_FEATURE_EQUITY_MODES:
        raise ValueError(f"Unknown feature equity mode: {feature_equity_mode}")
    if feature_equity_sims is not None and feature_equity_fn is not None:
        raise ValueError("Set only one of feature_equity_sims or feature_equity_fn")
    if feature_equity_fn is not None and feature_equity_mode != "random":
        raise ValueError("feature_equity_mode is only used with feature_equity_sims")
    deal_rng = random.Random(seed)
    policy_rng = random.Random(seed + 1)
    feature_rng = random.Random(seed + 3)
    if expert_policy not in HOLDEM_EXPERT_POLICIES:
        raise ValueError(f"Unknown expert policy: {expert_policy}")
    expert_action_policy = make_policy(expert_policy, policy_rng, equity_sims, rollout_sims)

    if opponent_policy not in HOLDEM_DATASET_OPPONENT_POLICIES:
        raise ValueError(f"Unknown opponent policy: {opponent_policy}")
    non_expert_policy = make_policy(opponent_policy, policy_rng, equity_sims, rollout_sims)

    examples: list[HoldemPolicyExample] = []
    for _ in range(hands):
        state = deal_fixed_limit_holdem(deal_rng)
        while not state.is_terminal():
            player = state.current_player()
            use_expert = expert_player is None or player == expert_player
            expert_action = expert_action_policy(state) if use_expert else non_expert_policy(state)
            if use_expert:
                examples.append(
                    HoldemPolicyExample(
                        features=encode_policy_example_features(
                            state,
                            feature_equity_sims=feature_equity_sims,
                            feature_equity_mode=feature_equity_mode,
                            feature_rng=feature_rng,
                            feature_equity_fn=feature_equity_fn,
                        ),
                        action_index=holdem_action_index(expert_action),
                        legal_mask=holdem_legal_action_mask(state),
                    )
                )
            if use_expert and expert_behavior_policy is not None:
                action = expert_behavior_policy(state)
                if action not in state.legal_actions():
                    raise ValueError(f"Behavior policy selected illegal action {action!r}")
            else:
                action = expert_action
            state = state.apply(action)
    return examples


def generate_equity_value_examples(
    *,
    hands: int,
    seed: int,
    equity_sims: int,
    player: int | None = 0,
    opponent_policy: str = "random",
) -> list[HoldemEquityExample]:
    if player is None:
        examples: list[HoldemEquityExample] = []
        for seat in (0, 1):
            examples.extend(
                generate_equity_value_examples(
                    hands=hands,
                    seed=seed + seat * 1_000_003,
                    equity_sims=equity_sims,
                    player=seat,
                    opponent_policy=opponent_policy,
                )
            )
        return examples
    if player not in (0, 1):
        raise ValueError(f"player must be 0, 1, or None, got {player}")

    deal_rng = random.Random(seed)
    policy_rng = random.Random(seed + 1)
    label_rng = random.Random(seed + 2)
    player_policy = equity_threshold_policy(policy_rng, simulations=equity_sims)
    if opponent_policy not in HOLDEM_EQUITY_VALUE_OPPONENT_POLICIES:
        raise ValueError(f"Unknown opponent policy: {opponent_policy}")
    other_policy = make_policy(opponent_policy, policy_rng, equity_sims)

    examples: list[HoldemEquityExample] = []
    for _ in range(hands):
        state = deal_fixed_limit_holdem(deal_rng)
        while not state.is_terminal():
            current = state.current_player()
            if current == player:
                examples.append(
                    HoldemEquityExample(
                        features=encode_holdem_state(state),
                        equity=estimate_holdem_equity(
                            state.private_cards[current],
                            state.visible_board(),
                            simulations=equity_sims,
                            rng=label_rng,
                        ),
                    )
                )
                action = player_policy(state)
            else:
                action = other_policy(state)
            state = state.apply(action)
    return examples

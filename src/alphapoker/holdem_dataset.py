"""Generate supervised Hold'em policy-distillation examples."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from alphapoker.holdem import (
    FixedLimitHoldemState,
    HoldemPolicy,
    deal_fixed_limit_holdem,
    equity_threshold_policy,
    estimate_holdem_equity,
    policy_filtered_holdem_equity,
    sampled_holdem_equity,
    turn_river_exact_holdem_equity,
    turn_river_exact_pot_odds_equity_policy,
)
from alphapoker.holdem_equity_feature import HoldemEquityEstimator
from alphapoker.holdem_features import (
    HOLDEM_CANONICAL_ACTIONS,
    encode_holdem_action_history_features,
    encode_holdem_state,
    holdem_action_index,
    holdem_legal_action_mask,
    opponent_aggressions_before_current_decision,
)
from alphapoker.holdem_self_play import make_policy, make_policy_action_value_fn
from alphapoker.kuhn import CALL, FOLD

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
    "tight-range-pot-odds",
    "tight-range-default-safe-rollout-pot-odds",
    "tight-fast-range-default-safe-rollout-pot-odds",
    "tight-range-rollout-pot-odds",
    "tight-range-safe-rollout-pot-odds",
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
    "tight-range-pot-odds",
    "tight-range-default-safe-rollout-pot-odds",
    "tight-fast-range-default-safe-rollout-pot-odds",
    "tight-range-rollout-pot-odds",
    "tight-range-safe-rollout-pot-odds",
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
    "tight-range-pot-odds",
    "hybrid-pot-odds",
    "random",
)
HOLDEM_FEATURE_EQUITY_MODES = ("random", "sampled", "turn-river-exact", "tight-range")


@dataclass(frozen=True)
class HoldemPolicyExample:
    features: list[float]
    action_index: int
    legal_mask: list[bool]
    action_probs: list[float] | None = None
    action_values: list[float] | None = None


@dataclass(frozen=True)
class HoldemEquityExample:
    features: list[float]
    equity: float


def _tight_range_opponent_policy_factory(simulations: int):
    def factory(_: random.Random) -> HoldemPolicy:
        return turn_river_exact_pot_odds_equity_policy(
            simulations=simulations,
            bet_threshold=0.62,
            raise_threshold=0.84,
            call_margin=0.08,
        )

    return factory


def encode_policy_example_features(
    state: FixedLimitHoldemState,
    *,
    feature_equity_sims: int | None = None,
    feature_equity_mode: str = "random",
    feature_rng: random.Random | None = None,
    feature_equity_fn: HoldemEquityEstimator | None = None,
    action_history_features: bool = False,
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
        if action_history_features:
            features.extend(encode_holdem_action_history_features(state))
        return features
    if feature_equity_sims is None:
        if action_history_features:
            features.extend(encode_holdem_action_history_features(state))
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
    elif feature_equity_mode == "tight-range":
        equity = policy_filtered_holdem_equity(
            state,
            player,
            feature_rng,
            simulations=feature_equity_sims,
            opponent_policy_factory=_tight_range_opponent_policy_factory(feature_equity_sims),
            cache_policy_matches=True,
        )
    else:
        raise AssertionError(f"Unhandled feature equity mode: {feature_equity_mode}")
    features.append(equity)
    if action_history_features:
        features.extend(encode_holdem_action_history_features(state))
    return features


def write_policy_examples(path: Path, examples: list[HoldemPolicyExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for example in examples:
        item = {
            "features": example.features,
            "action_index": example.action_index,
            "legal_mask": example.legal_mask,
        }
        if example.action_probs is not None:
            item["action_probs"] = example.action_probs
        if example.action_values is not None:
            item["action_values"] = example.action_values
        payload.append(item)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_policy_examples(path: Path) -> list[HoldemPolicyExample]:
    payload = json.loads(path.read_text())
    return [
        HoldemPolicyExample(
            features=[float(value) for value in item["features"]],
            action_index=int(item["action_index"]),
            legal_mask=[bool(value) for value in item["legal_mask"]],
            action_probs=(
                [float(value) for value in item["action_probs"]]
                if item.get("action_probs") is not None
                else None
            ),
            action_values=(
                [float(value) for value in item["action_values"]]
                if item.get("action_values") is not None
                else None
            ),
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


def soft_action_probs_from_values(
    action_values: dict[str, float],
    legal_mask: list[bool],
    temperature: float,
) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("soft target temperature must be positive")
    legal_indices = [
        index
        for index, legal in enumerate(legal_mask)
        if legal and HOLDEM_CANONICAL_ACTIONS[index] in action_values
    ]
    if not legal_indices:
        raise ValueError("soft target values require at least one legal action")
    max_value = max(action_values[HOLDEM_CANONICAL_ACTIONS[index]] for index in legal_indices)
    weights = [0.0 for _ in HOLDEM_CANONICAL_ACTIONS]
    total = 0.0
    for index in legal_indices:
        action = HOLDEM_CANONICAL_ACTIONS[index]
        weight = math.exp((action_values[action] - max_value) / temperature)
        weights[index] = weight
        total += weight
    return [weight / total for weight in weights]


def should_record_policy_example(
    state: FixedLimitHoldemState,
    *,
    record_facing_bet_only: bool = False,
    record_min_opponent_aggressions: int | None = None,
) -> bool:
    if record_facing_bet_only:
        legal_actions = set(state.legal_actions())
        if CALL not in legal_actions or FOLD not in legal_actions:
            return False
    if record_min_opponent_aggressions is not None:
        if record_min_opponent_aggressions < 1:
            raise ValueError("record_min_opponent_aggressions must be positive")
        if (
            opponent_aggressions_before_current_decision(state)
            < record_min_opponent_aggressions
        ):
            return False
    return True


def generate_equity_policy_examples(
    *,
    hands: int,
    seed: int,
    equity_sims: int,
    expert_player: int | None = None,
    expert_policy: str = "equity",
    opponent_policy: str = "equity",
    rollout_sims: int | None = None,
    rollout_margin: float = 1.0,
    feature_equity_sims: int | None = None,
    feature_equity_mode: str = "random",
    feature_equity_fn: HoldemEquityEstimator | None = None,
    expert_behavior_policy: HoldemPolicy | None = None,
    action_history_features: bool = False,
    soft_target_temperature: float | None = None,
    record_facing_bet_only: bool = False,
    record_min_opponent_aggressions: int | None = None,
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
    expert_action_policy = make_policy(
        expert_policy,
        policy_rng,
        equity_sims,
        rollout_sims,
        rollout_margin,
    )
    expert_action_value_fn = None
    if soft_target_temperature is not None:
        if soft_target_temperature <= 0.0:
            raise ValueError("soft target temperature must be positive")
        expert_action_value_fn = make_policy_action_value_fn(
            expert_policy,
            policy_rng,
            equity_sims,
            rollout_sims,
            rollout_margin,
        )
        if expert_action_value_fn is None:
            raise ValueError(f"soft targets are not available for expert policy {expert_policy}")

    if opponent_policy not in HOLDEM_DATASET_OPPONENT_POLICIES:
        raise ValueError(f"Unknown opponent policy: {opponent_policy}")
    non_expert_policy = make_policy(
        opponent_policy,
        policy_rng,
        equity_sims,
        rollout_sims,
        rollout_margin,
    )

    examples: list[HoldemPolicyExample] = []
    for _ in range(hands):
        state = deal_fixed_limit_holdem(deal_rng)
        while not state.is_terminal():
            player = state.current_player()
            use_expert = expert_player is None or player == expert_player
            record_example = use_expert and should_record_policy_example(
                state,
                record_facing_bet_only=record_facing_bet_only,
                record_min_opponent_aggressions=record_min_opponent_aggressions,
            )
            action_probs = None
            dense_action_values = None
            expert_action = None
            if record_example and expert_action_value_fn is not None:
                expert_action, action_values = expert_action_value_fn(state)
                legal_mask = holdem_legal_action_mask(state)
                action_probs = soft_action_probs_from_values(
                    action_values,
                    legal_mask,
                    soft_target_temperature,
                )
                dense_action_values = [
                    float(action_values.get(action, 0.0))
                    for action in HOLDEM_CANONICAL_ACTIONS
                ]
            else:
                if use_expert and (record_example or expert_behavior_policy is None):
                    expert_action = expert_action_policy(state)
                elif not use_expert:
                    expert_action = non_expert_policy(state)
                legal_mask = holdem_legal_action_mask(state) if record_example else None
            if record_example:
                if expert_action is None:
                    raise AssertionError("recorded examples require an expert action")
                if legal_mask is None:
                    raise AssertionError("recorded examples require a legal mask")
                examples.append(
                    HoldemPolicyExample(
                        features=encode_policy_example_features(
                            state,
                            feature_equity_sims=feature_equity_sims,
                            feature_equity_mode=feature_equity_mode,
                            feature_rng=feature_rng,
                            feature_equity_fn=feature_equity_fn,
                            action_history_features=action_history_features,
                        ),
                        action_index=holdem_action_index(expert_action),
                        legal_mask=legal_mask,
                        action_probs=action_probs,
                        action_values=dense_action_values,
                    )
                )
            if use_expert and expert_behavior_policy is not None:
                action = expert_behavior_policy(state)
                if action not in state.legal_actions():
                    raise ValueError(f"Behavior policy selected illegal action {action!r}")
            else:
                if expert_action is None:
                    raise AssertionError("expert policy did not select an action")
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

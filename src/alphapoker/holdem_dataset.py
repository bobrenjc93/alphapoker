"""Generate supervised Hold'em policy-distillation examples."""

from __future__ import annotations

import random
from dataclasses import dataclass

from alphapoker.holdem import (
    HoldemPolicy,
    deal_fixed_limit_holdem,
    equity_threshold_policy,
    estimate_holdem_equity,
    random_holdem_policy,
)
from alphapoker.holdem_features import (
    encode_holdem_state,
    holdem_action_index,
    holdem_legal_action_mask,
)


@dataclass(frozen=True)
class HoldemPolicyExample:
    features: list[float]
    action_index: int
    legal_mask: list[bool]


@dataclass(frozen=True)
class HoldemEquityExample:
    features: list[float]
    equity: float


def generate_equity_policy_examples(
    *,
    hands: int,
    seed: int,
    equity_sims: int,
    expert_player: int | None = None,
    opponent_policy: str = "equity",
    expert_behavior_policy: HoldemPolicy | None = None,
) -> list[HoldemPolicyExample]:
    deal_rng = random.Random(seed)
    policy_rng = random.Random(seed + 1)
    equity_policy = equity_threshold_policy(policy_rng, simulations=equity_sims)
    if opponent_policy == "equity":
        non_expert_policy = equity_policy
    elif opponent_policy == "random":
        non_expert_policy = random_holdem_policy(policy_rng)
    else:
        raise ValueError(f"Unknown opponent policy: {opponent_policy}")

    examples: list[HoldemPolicyExample] = []
    for _ in range(hands):
        state = deal_fixed_limit_holdem(deal_rng)
        while not state.is_terminal():
            player = state.current_player()
            use_expert = expert_player is None or player == expert_player
            expert_action = equity_policy(state) if use_expert else non_expert_policy(state)
            if use_expert:
                examples.append(
                    HoldemPolicyExample(
                        features=encode_holdem_state(state),
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
    player: int = 0,
    opponent_policy: str = "random",
) -> list[HoldemEquityExample]:
    deal_rng = random.Random(seed)
    policy_rng = random.Random(seed + 1)
    label_rng = random.Random(seed + 2)
    player_policy = equity_threshold_policy(policy_rng, simulations=equity_sims)
    if opponent_policy == "random":
        other_policy = random_holdem_policy(policy_rng)
    elif opponent_policy == "equity":
        other_policy = equity_threshold_policy(policy_rng, simulations=equity_sims)
    else:
        raise ValueError(f"Unknown opponent policy: {opponent_policy}")

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

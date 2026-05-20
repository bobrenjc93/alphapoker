import random

import pytest


pytest.importorskip("treys")

from alphapoker.holdem import RAISE, FixedLimitHoldemState, deal_fixed_limit_holdem  # noqa: E402
from alphapoker.holdem_dataset import (  # noqa: E402
    HoldemEquityExample,
    HoldemPolicyExample,
    generate_equity_policy_examples,
    generate_equity_value_examples,
    encode_policy_example_features,
    read_policy_examples,
    read_equity_value_examples,
    soft_action_probs_from_values,
    write_policy_examples,
    write_equity_value_examples,
)
from alphapoker.holdem_features import (  # noqa: E402
    HOLDEM_BASE_FEATURE_DIM,
    HOLDEM_ACTION_HISTORY_FEATURE_DIM,
    HOLDEM_FEATURE_DIM,
    HOLDEM_HAND_STRENGTH_FEATURE_DIM,
    adapt_holdem_features,
    encode_holdem_action_history_features,
    encode_holdem_state,
    holdem_legal_action_mask,
)
from alphapoker.kuhn import CALL  # noqa: E402


def test_holdem_features_have_fixed_shape() -> None:
    state = deal_fixed_limit_holdem()

    assert len(encode_holdem_state(state)) == HOLDEM_FEATURE_DIM
    assert len(holdem_legal_action_mask(state)) == 5
    assert len(adapt_holdem_features(encode_holdem_state(state), HOLDEM_BASE_FEATURE_DIM)) == HOLDEM_BASE_FEATURE_DIM


def test_holdem_made_hand_features_activate_on_flop() -> None:
    preflop = FixedLimitHoldemState.initial(
        (("As", "Qs"), ("Ah", "Ad")),
        ("2s", "7s", "9s", "Kd", "3c"),
    )
    flop = preflop.apply(CALL)

    assert encode_holdem_state(preflop)[-HOLDEM_HAND_STRENGTH_FEATURE_DIM:] == [
        0.0 for _ in range(HOLDEM_HAND_STRENGTH_FEATURE_DIM)
    ]
    flop_strength_features = encode_holdem_state(flop)[-HOLDEM_HAND_STRENGTH_FEATURE_DIM:]
    assert flop_strength_features[0] > 0.0
    assert sum(flop_strength_features[2:]) == 1.0


def test_holdem_policy_features_can_include_equity_estimate() -> None:
    state = deal_fixed_limit_holdem()
    features = encode_policy_example_features(state, feature_equity_sims=2)

    assert len(features) == HOLDEM_FEATURE_DIM + 1
    assert 0.0 <= features[-1] <= 1.0


def test_holdem_policy_features_can_include_deterministic_equity_estimate() -> None:
    state = deal_fixed_limit_holdem()
    first = encode_policy_example_features(
        state,
        feature_equity_sims=2,
        feature_equity_mode="turn-river-exact",
    )
    second = encode_policy_example_features(
        state,
        feature_equity_sims=2,
        feature_equity_mode="turn-river-exact",
    )

    assert len(first) == HOLDEM_FEATURE_DIM + 1
    assert first[-1] == second[-1]
    assert 0.0 <= first[-1] <= 1.0


def test_holdem_policy_features_can_include_tight_range_equity_estimate() -> None:
    state = deal_fixed_limit_holdem()
    features = encode_policy_example_features(
        state,
        feature_equity_sims=2,
        feature_equity_mode="tight-range",
        feature_rng=random.Random(19),
    )

    assert len(features) == HOLDEM_FEATURE_DIM + 1
    assert 0.0 <= features[-1] <= 1.0


def test_holdem_policy_features_can_include_learned_equity_estimate() -> None:
    state = deal_fixed_limit_holdem()
    features = encode_policy_example_features(state, feature_equity_fn=lambda _: 0.42)

    assert len(features) == HOLDEM_FEATURE_DIM + 1
    assert features[-1] == 0.42


def test_holdem_policy_features_can_include_action_history() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Qs"), ("Ah", "Ad")),
        ("2s", "7s", "9s", "Kd", "3c"),
    ).apply(RAISE)

    history_features = encode_holdem_action_history_features(state)
    features = encode_policy_example_features(state, action_history_features=True)

    assert len(history_features) == HOLDEM_ACTION_HISTORY_FEATURE_DIM
    assert history_features == [0.0, 1.0 / 16.0, 0.0, 1.0 / 4.0, 1.0]
    assert len(features) == HOLDEM_FEATURE_DIM + HOLDEM_ACTION_HISTORY_FEATURE_DIM
    assert features[-HOLDEM_ACTION_HISTORY_FEATURE_DIM:] == history_features


def test_holdem_policy_features_reject_two_equity_feature_modes() -> None:
    state = deal_fixed_limit_holdem()

    with pytest.raises(ValueError, match="only one"):
        encode_policy_example_features(
            state,
            feature_equity_sims=2,
            feature_equity_fn=lambda _: 0.42,
        )


def test_holdem_policy_features_reject_bad_equity_feature_mode() -> None:
    state = deal_fixed_limit_holdem()

    with pytest.raises(ValueError, match="Unknown feature equity mode"):
        encode_policy_example_features(
            state,
            feature_equity_sims=2,
            feature_equity_mode="bad",
        )


def test_generate_equity_policy_examples_smoke() -> None:
    examples = generate_equity_policy_examples(hands=2, seed=5, equity_sims=4)

    assert examples
    assert {len(example.features) for example in examples} == {HOLDEM_FEATURE_DIM}
    assert {len(example.legal_mask) for example in examples} == {5}


def test_generate_equity_policy_examples_for_one_player_vs_random() -> None:
    examples = generate_equity_policy_examples(
        hands=3,
        seed=6,
        equity_sims=4,
        expert_player=0,
        opponent_policy="random",
    )

    assert examples


def test_generate_equity_policy_examples_vs_pot_odds() -> None:
    examples = generate_equity_policy_examples(
        hands=2,
        seed=7,
        equity_sims=4,
        expert_player=0,
        opponent_policy="pot-odds",
    )

    assert examples


def test_generate_equity_policy_examples_from_pot_odds_expert() -> None:
    examples = generate_equity_policy_examples(
        hands=2,
        seed=12,
        equity_sims=4,
        expert_policy="pot-odds",
        opponent_policy="random",
    )

    assert examples


def test_generate_equity_policy_examples_from_tuned_pot_odds_expert() -> None:
    examples = generate_equity_policy_examples(
        hands=2,
        seed=14,
        equity_sims=4,
        expert_policy="tuned-pot-odds",
        opponent_policy="pot-odds",
    )

    assert examples


def test_generate_equity_policy_examples_from_hybrid_pot_odds_expert() -> None:
    examples = generate_equity_policy_examples(
        hands=2,
        seed=16,
        equity_sims=4,
        expert_policy="hybrid-pot-odds",
        opponent_policy="pot-odds",
    )

    assert examples


def test_generate_equity_policy_examples_from_tight_exact_expert() -> None:
    examples = generate_equity_policy_examples(
        hands=1,
        seed=17,
        equity_sims=2,
        expert_policy="tight-turn-river-exact-pot-odds",
        opponent_policy="tight-turn-river-exact-pot-odds",
    )

    assert examples


def test_generate_equity_policy_examples_from_balanced_exact_expert() -> None:
    examples = generate_equity_policy_examples(
        hands=1,
        seed=18,
        equity_sims=2,
        expert_policy="balanced-turn-river-exact-pot-odds",
        opponent_policy="balanced-turn-river-exact-pot-odds",
    )

    assert examples


def test_generate_equity_policy_examples_from_rollout_expert() -> None:
    examples = generate_equity_policy_examples(
        hands=1,
        seed=13,
        equity_sims=2,
        rollout_sims=2,
        expert_policy="rollout-pot-odds",
        opponent_policy="pot-odds",
    )

    assert examples


def test_generate_equity_policy_examples_from_tuned_rollout_expert() -> None:
    examples = generate_equity_policy_examples(
        hands=1,
        seed=14,
        equity_sims=2,
        rollout_sims=2,
        expert_policy="cached-tuned-rollout-pot-odds",
        opponent_policy="cached-tuned-pot-odds",
    )

    assert examples


def test_generate_equity_policy_examples_with_soft_rollout_targets() -> None:
    examples = generate_equity_policy_examples(
        hands=1,
        seed=21,
        equity_sims=2,
        rollout_sims=1,
        expert_policy="tight-safe-rollout-pot-odds",
        opponent_policy="tight-safe-rollout-pot-odds",
        soft_target_temperature=1.0,
    )

    assert examples
    assert any(example.action_probs is not None for example in examples)
    for example in examples:
        assert example.action_probs is not None
        assert example.action_values is not None
        assert len(example.action_values) == len(example.legal_mask)
        assert sum(example.action_probs) == pytest.approx(1.0)
        for probability, action_value, legal in zip(
            example.action_probs,
            example.action_values,
            example.legal_mask,
        ):
            if not legal:
                assert probability == 0.0
                assert action_value == 0.0


def test_soft_action_probs_from_values_rejects_bad_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        soft_action_probs_from_values({"check": 0.0}, [True, False, False, False, False], 0.0)


def test_generate_equity_policy_examples_with_behavior_policy() -> None:
    examples = generate_equity_policy_examples(
        hands=2,
        seed=9,
        equity_sims=4,
        expert_player=0,
        opponent_policy="random",
        expert_behavior_policy=lambda state: state.legal_actions()[0],
    )

    assert examples


def test_generate_equity_value_examples_smoke() -> None:
    examples = generate_equity_value_examples(hands=2, seed=10, equity_sims=4)

    assert examples
    assert {len(example.features) for example in examples} == {HOLDEM_FEATURE_DIM}
    assert all(0.0 <= example.equity <= 1.0 for example in examples)


def test_generate_equity_value_examples_for_both_seats() -> None:
    examples = generate_equity_value_examples(hands=3, seed=11, equity_sims=4, player=None)

    assert examples
    assert any(example.features[108] == 1.0 for example in examples)
    assert any(example.features[109] == 1.0 for example in examples)


def test_generate_equity_value_examples_vs_tuned_pot_odds() -> None:
    examples = generate_equity_value_examples(
        hands=2,
        seed=15,
        equity_sims=4,
        opponent_policy="tuned-pot-odds",
    )

    assert examples


def test_generate_equity_value_examples_vs_tight_exact_policy() -> None:
    examples = generate_equity_value_examples(
        hands=1,
        seed=18,
        equity_sims=2,
        opponent_policy="tight-turn-river-exact-pot-odds",
    )

    assert examples


def test_equity_value_example_cache_round_trip(tmp_path) -> None:
    path = tmp_path / "examples.json"
    examples = [HoldemEquityExample(features=[0.0, 1.0], equity=0.25)]

    write_equity_value_examples(path, examples)

    assert read_equity_value_examples(path) == examples


def test_policy_example_cache_round_trip(tmp_path) -> None:
    path = tmp_path / "policy_examples.json"
    examples = [
        HoldemPolicyExample(
            features=[0.0, 1.0],
            action_index=2,
            legal_mask=[True, False],
            action_probs=[0.25, 0.75],
            action_values=[1.0, 0.0],
        )
    ]

    write_policy_examples(path, examples)

    assert read_policy_examples(path) == examples

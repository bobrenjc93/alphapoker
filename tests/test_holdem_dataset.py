import pytest


pytest.importorskip("treys")

from alphapoker.holdem import deal_fixed_limit_holdem  # noqa: E402
from alphapoker.holdem_dataset import (  # noqa: E402
    generate_equity_policy_examples,
    generate_equity_value_examples,
)
from alphapoker.holdem_features import encode_holdem_state, holdem_legal_action_mask  # noqa: E402


def test_holdem_features_have_fixed_shape() -> None:
    state = deal_fixed_limit_holdem()

    assert len(encode_holdem_state(state)) == 117
    assert len(holdem_legal_action_mask(state)) == 5


def test_generate_equity_policy_examples_smoke() -> None:
    examples = generate_equity_policy_examples(hands=2, seed=5, equity_sims=4)

    assert examples
    assert {len(example.features) for example in examples} == {117}
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
    assert {len(example.features) for example in examples} == {117}
    assert all(0.0 <= example.equity <= 1.0 for example in examples)


def test_generate_equity_value_examples_for_both_seats() -> None:
    examples = generate_equity_value_examples(hands=3, seed=11, equity_sims=4, player=None)

    assert examples
    assert any(example.features[108] == 1.0 for example in examples)
    assert any(example.features[109] == 1.0 for example in examples)

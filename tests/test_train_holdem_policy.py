import pytest


pytest.importorskip("treys")

from alphapoker.holdem_dataset import HoldemPolicyExample, write_policy_examples  # noqa: E402
from alphapoker.train_holdem_policy import (  # noqa: E402
    _shard_hands,
    _split_train_validation_indices,
    build_parser,
    run,
)


def test_train_holdem_policy_parser_accepts_pot_odds_expert() -> None:
    args = build_parser().parse_args(
        [
            "--expert-policy",
            "pot-odds",
            "--opponent-policy",
            "pot-odds",
            "--rollout-sims",
            "2",
            "--feature-equity-sims",
            "3",
            "--feature-equity-mode",
            "sampled",
            "--examples-in",
            "examples.json",
            "--examples-out",
            "cached.json",
            "--class-weighting",
            "balanced",
            "--jobs",
            "4",
            "--validation-fraction",
            "0.2",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "pot-odds"
    assert args.opponent_policy == "pot-odds"
    assert args.rollout_sims == 2
    assert args.feature_equity_sims == 3
    assert args.feature_equity_mode == "sampled"
    assert args.class_weighting == "balanced"
    assert args.jobs == 4
    assert args.validation_fraction == 0.2
    assert str(args.examples_in) == "examples.json"
    assert str(args.examples_out) == "cached.json"


def test_train_holdem_policy_parser_accepts_feature_equity_checkpoint() -> None:
    args = build_parser().parse_args(
        [
            "--feature-equity-checkpoint",
            "equity.pt",
            "--out",
            "out",
        ]
    )

    assert str(args.feature_equity_checkpoint) == "equity.pt"


def test_train_holdem_policy_parser_accepts_rollout_expert() -> None:
    args = build_parser().parse_args(
        [
            "--expert-policy",
            "rollout-pot-odds",
            "--opponent-policy",
            "pot-odds",
            "--rollout-sims",
            "2",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "rollout-pot-odds"
    assert args.rollout_sims == 2


def test_train_holdem_policy_parser_accepts_tuned_rollout_expert() -> None:
    args = build_parser().parse_args(
        [
            "--expert-policy",
            "cached-tuned-rollout-pot-odds",
            "--opponent-policy",
            "cached-tuned-pot-odds",
            "--rollout-sims",
            "2",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "cached-tuned-rollout-pot-odds"
    assert args.rollout_sims == 2


def test_train_holdem_policy_parser_accepts_tuned_expert() -> None:
    args = build_parser().parse_args(
        [
            "--expert-policy",
            "tuned-pot-odds",
            "--opponent-policy",
            "tuned-pot-odds",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "tuned-pot-odds"
    assert args.opponent_policy == "tuned-pot-odds"


def test_train_holdem_policy_parser_accepts_tight_exact_expert() -> None:
    args = build_parser().parse_args(
        [
            "--expert-policy",
            "tight-turn-river-exact-pot-odds",
            "--opponent-policy",
            "tight-turn-river-exact-pot-odds",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "tight-turn-river-exact-pot-odds"
    assert args.opponent_policy == "tight-turn-river-exact-pot-odds"


def test_train_holdem_policy_parser_accepts_balanced_exact_expert() -> None:
    args = build_parser().parse_args(
        [
            "--expert-policy",
            "balanced-turn-river-exact-pot-odds",
            "--opponent-policy",
            "balanced-turn-river-exact-pot-odds",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "balanced-turn-river-exact-pot-odds"
    assert args.opponent_policy == "balanced-turn-river-exact-pot-odds"


def test_train_holdem_policy_parser_accepts_tight_range_expert() -> None:
    args = build_parser().parse_args(
        [
            "--expert-policy",
            "tight-range-pot-odds",
            "--opponent-policy",
            "tight-turn-river-exact-pot-odds",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "tight-range-pot-odds"
    assert args.opponent_policy == "tight-turn-river-exact-pot-odds"


def test_train_holdem_policy_parser_accepts_turn_river_exact_feature() -> None:
    args = build_parser().parse_args(
        [
            "--feature-equity-sims",
            "8",
            "--feature-equity-mode",
            "turn-river-exact",
            "--out",
            "out",
        ]
    )

    assert args.feature_equity_sims == 8
    assert args.feature_equity_mode == "turn-river-exact"


def test_shard_hands_for_parallel_policy_training() -> None:
    assert _shard_hands(11, 4) == [3, 3, 3, 2]
    assert _shard_hands(2, 4) == [1, 1]


def test_split_train_validation_indices_is_deterministic() -> None:
    first = _split_train_validation_indices(10, 0.2, 5)
    second = _split_train_validation_indices(10, 0.2, 5)

    assert first == second
    train_indices, validation_indices = first
    assert len(train_indices) == 8
    assert len(validation_indices) == 2
    assert sorted(train_indices + validation_indices) == list(range(10))


def test_train_holdem_policy_records_validation_metrics(tmp_path) -> None:
    examples_path = tmp_path / "examples.json"
    examples = [
        HoldemPolicyExample(
            features=[float(index % 2), 1.0],
            action_index=index % 2,
            legal_mask=[True, True, False, False, False],
        )
        for index in range(8)
    ]
    write_policy_examples(examples_path, examples)

    args = build_parser().parse_args(
        [
            "--examples-in",
            str(examples_path),
            "--epochs",
            "2",
            "--validation-fraction",
            "0.25",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    metrics = run(args)

    assert metrics["train_examples"] == 6
    assert metrics["validation_examples"] == 2
    assert metrics["selection_metric"] == "validation_loss"
    assert metrics["best_validation_loss"] is not None
    assert metrics["validation_accuracy"] is not None

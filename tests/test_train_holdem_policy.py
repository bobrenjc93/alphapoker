import pytest


pytest.importorskip("treys")

from alphapoker.holdem_dataset import HoldemPolicyExample, write_policy_examples  # noqa: E402
from alphapoker.holdem_model import HoldemPolicyNet  # noqa: E402
from alphapoker.train_holdem_policy import (  # noqa: E402
    _shard_hands,
    _split_train_validation_indices,
    build_parser,
    class_weight_exponent_for_mode,
    class_weights_from_targets,
    example_weights_from_masks,
    load_policy_checkpoint_state,
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
            "--rollout-margin",
            "1.5",
            "--feature-equity-sims",
            "3",
            "--feature-equity-mode",
            "sampled",
            "--action-history-features",
            "--init-checkpoint",
            "policy.pt",
            "--init-kl-weight",
            "0.5",
            "--init-allow-input-expansion",
            "--examples-in",
            "examples.json",
            "--examples-out",
            "cached.json",
            "--class-weighting",
            "balanced",
            "--class-weight-exponent",
            "0.75",
            "--facing-bet-weight",
            "3.0",
            "--jobs",
            "4",
            "--progress",
            "--validation-fraction",
            "0.2",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "pot-odds"
    assert args.opponent_policy == "pot-odds"
    assert args.rollout_sims == 2
    assert args.rollout_margin == 1.5
    assert args.feature_equity_sims == 3
    assert args.feature_equity_mode == "sampled"
    assert args.action_history_features
    assert str(args.init_checkpoint) == "policy.pt"
    assert args.init_kl_weight == 0.5
    assert args.init_allow_input_expansion
    assert args.class_weighting == "balanced"
    assert args.class_weight_exponent == 0.75
    assert args.facing_bet_weight == 3.0
    assert args.jobs == 4
    assert args.progress
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


def test_train_holdem_policy_parser_accepts_sqrt_balanced_weighting() -> None:
    args = build_parser().parse_args(
        [
            "--class-weighting",
            "sqrt-balanced",
            "--out",
            "out",
        ]
    )

    assert args.class_weighting == "sqrt-balanced"


def test_sqrt_balanced_class_weights_dampen_rare_action_weight() -> None:
    torch = pytest.importorskip("torch")
    targets = torch.tensor([0, 0, 0, 1])

    balanced = class_weights_from_targets(targets, 2, "balanced")
    sqrt_balanced = class_weights_from_targets(targets, 2, "sqrt-balanced")

    assert sqrt_balanced[1] > sqrt_balanced[0]
    assert sqrt_balanced[1] / sqrt_balanced[0] < balanced[1] / balanced[0]


def test_class_weight_exponent_interpolates_rare_action_weight() -> None:
    torch = pytest.importorskip("torch")
    targets = torch.tensor([0, 0, 0, 1])

    sqrt_balanced = class_weights_from_targets(targets, 2, "sqrt-balanced")
    power_balanced = class_weights_from_targets(targets, 2, "balanced", 0.75)
    balanced = class_weights_from_targets(targets, 2, "balanced")

    sqrt_ratio = sqrt_balanced[1] / sqrt_balanced[0]
    power_ratio = power_balanced[1] / power_balanced[0]
    balanced_ratio = balanced[1] / balanced[0]
    assert sqrt_ratio < power_ratio < balanced_ratio
    assert class_weight_exponent_for_mode("balanced", 0.75) == 0.75
    assert class_weight_exponent_for_mode("sqrt-balanced") == 0.5


def test_class_weight_exponent_requires_weighting() -> None:
    with pytest.raises(ValueError, match="requires class weighting"):
        class_weight_exponent_for_mode("none", 0.75)
    with pytest.raises(ValueError, match="positive"):
        class_weight_exponent_for_mode("balanced", 0.0)


def test_facing_bet_weights_upweight_call_fold_states() -> None:
    torch = pytest.importorskip("torch")
    masks = torch.tensor(
        [
            [False, False, True, True, False],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )

    weights = example_weights_from_masks(masks, 3.0)

    assert weights.tolist() == [3.0, 1.0]


def test_facing_bet_weight_must_be_positive() -> None:
    torch = pytest.importorskip("torch")
    masks = torch.tensor([[False, False, True, True, False]], dtype=torch.bool)

    with pytest.raises(ValueError, match="must be positive"):
        example_weights_from_masks(masks, 0.0)


def test_policy_checkpoint_input_expansion_preserves_old_columns() -> None:
    torch = pytest.importorskip("torch")
    source_model = HoldemPolicyNet(input_dim=1)
    target_model = HoldemPolicyNet(input_dim=2)
    for parameter in source_model.parameters():
        parameter.data.fill_(0.5)

    init_input_dim, expanded = load_policy_checkpoint_state(
        target_model,
        {
            "model_state_dict": source_model.state_dict(),
            "input_dim": 1,
        },
        target_input_dim=2,
        allow_input_expansion=True,
    )

    first_layer = target_model.state_dict()["net.0.weight"]
    assert init_input_dim == 1
    assert expanded
    assert torch.allclose(first_layer[:, :1], torch.full_like(first_layer[:, :1], 0.5))
    assert torch.allclose(first_layer[:, 1:], torch.zeros_like(first_layer[:, 1:]))


def test_train_holdem_policy_parser_accepts_tight_range_feature() -> None:
    args = build_parser().parse_args(
        [
            "--feature-equity-sims",
            "4",
            "--feature-equity-mode",
            "tight-range",
            "--out",
            "out",
        ]
    )

    assert args.feature_equity_sims == 4
    assert args.feature_equity_mode == "tight-range"


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
    assert metrics["rollout_margin"] == 1.0
    assert metrics["init_kl_weight"] == 0.0
    assert metrics["facing_bet_weight"] == 1.0
    assert metrics["facing_bet_examples"] == 0


def test_init_kl_weight_requires_init_checkpoint(tmp_path) -> None:
    examples_path = tmp_path / "examples.json"
    write_policy_examples(
        examples_path,
        [
            HoldemPolicyExample(
                features=[1.0],
                action_index=0,
                legal_mask=[True, False, False, False, False],
            )
        ],
    )

    args = build_parser().parse_args(
        [
            "--examples-in",
            str(examples_path),
            "--init-kl-weight",
            "0.5",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(ValueError, match="requires --init-checkpoint"):
        run(args)

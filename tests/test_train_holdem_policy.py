import pytest


pytest.importorskip("treys")

from alphapoker.holdem_dataset import HoldemPolicyExample, write_policy_examples  # noqa: E402
from alphapoker.holdem_features import (  # noqa: E402
    HOLDEM_ACTION_HISTORY_FEATURE_DIM,
    HOLDEM_CANONICAL_ACTIONS,
    HOLDEM_FEATURE_DIM,
    HOLDEM_PLAYER_FEATURE_DIM,
    HOLDEM_PLAYER_FEATURE_OFFSET,
)
from alphapoker.holdem_model import HoldemPolicyNet  # noqa: E402
from alphapoker.train_holdem_policy import (  # noqa: E402
    _shard_hands,
    _split_train_validation_indices,
    action_weight_overrides_from_specs,
    action_value_example_weights_from_mask,
    apply_action_weight_overrides,
    build_parser,
    class_weight_exponent_for_mode,
    class_weights_from_targets,
    example_weights_from_masks,
    facing_bet_action_weights_from_masks_targets,
    load_policy_checkpoint_state,
    opponent_aggression_count_mask_from_features,
    player_action_weight_overrides_from_specs,
    player_action_weights_from_features_targets,
    player_action_value_weights_from_features_mask,
    player_facing_bet_action_weights_from_features_masks_targets,
    player_value_weight_overrides_from_specs,
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
            "--soft-target-temperature",
            "0.75",
            "--record-facing-bet-only",
            "--record-min-opponent-aggressions",
            "2",
            "--action-value-loss-weight",
            "0.25",
            "--action-value-target-scale",
            "2.0",
            "--action-value-example-weight",
            "4.0",
            "--init-checkpoint",
            "policy.pt",
            "--init-kl-weight",
            "0.5",
            "--init-kl-example-weighting",
            "uniform",
            "--init-allow-input-expansion",
            "--examples-in",
            "examples.json",
            "--extra-examples-in",
            "focused.json",
            "--examples-out",
            "cached.json",
            "--class-weighting",
            "balanced",
            "--class-weight-exponent",
            "0.75",
            "--action-weight",
            "raise=2.0",
            "--action-weight",
            "fold=0.5",
            "--player-action-weight",
            "1:raise=3.0",
            "--facing-bet-action-weight",
            "call=2.0",
            "--facing-bet-action-weight-after-opponent-aggressions",
            "2",
            "--player-facing-bet-action-weight",
            "1:fold=0.5",
            "--player-facing-bet-action-weight-after-opponent-aggressions",
            "3",
            "--player-action-value-weight",
            "1=2.5",
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
    assert args.soft_target_temperature == 0.75
    assert args.record_facing_bet_only
    assert args.record_min_opponent_aggressions == 2
    assert args.action_value_loss_weight == 0.25
    assert args.action_value_target_scale == 2.0
    assert args.action_value_example_weight == 4.0
    assert str(args.init_checkpoint) == "policy.pt"
    assert args.init_kl_weight == 0.5
    assert args.init_kl_example_weighting == "uniform"
    assert args.init_allow_input_expansion
    assert args.class_weighting == "balanced"
    assert args.class_weight_exponent == 0.75
    assert args.action_weight == ["raise=2.0", "fold=0.5"]
    assert args.player_action_weight == ["1:raise=3.0"]
    assert args.facing_bet_action_weight == ["call=2.0"]
    assert args.facing_bet_action_weight_after_opponent_aggressions == 2
    assert args.player_facing_bet_action_weight == ["1:fold=0.5"]
    assert args.player_facing_bet_action_weight_after_opponent_aggressions == 3
    assert args.player_action_value_weight == ["1=2.5"]
    assert args.facing_bet_weight == 3.0
    assert args.jobs == 4
    assert args.progress
    assert args.validation_fraction == 0.2
    assert str(args.examples_in) == "examples.json"
    assert [str(path) for path in args.extra_examples_in] == ["focused.json"]
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


def test_action_weight_overrides_adjust_effective_class_weights() -> None:
    torch = pytest.importorskip("torch")
    base_weights = torch.ones(5)

    overrides = action_weight_overrides_from_specs(["raise=2.0", "fold=0.5"])
    weights = apply_action_weight_overrides(base_weights, overrides)

    raise_index = HOLDEM_CANONICAL_ACTIONS.index("raise")
    fold_index = HOLDEM_CANONICAL_ACTIONS.index("fold")
    assert weights[raise_index] / weights[fold_index] == pytest.approx(4.0)
    assert float(weights.mean()) == pytest.approx(1.0)


def test_action_weight_overrides_validate_specs() -> None:
    with pytest.raises(ValueError, match="ACTION=WEIGHT"):
        action_weight_overrides_from_specs(["raise"])
    with pytest.raises(ValueError, match="unknown"):
        action_weight_overrides_from_specs(["jam=2.0"])
    with pytest.raises(ValueError, match="positive"):
        action_weight_overrides_from_specs(["raise=0.0"])


def test_player_action_weight_overrides_adjust_selected_examples() -> None:
    torch = pytest.importorskip("torch")
    feature_dim = HOLDEM_PLAYER_FEATURE_OFFSET + HOLDEM_PLAYER_FEATURE_DIM
    features = torch.zeros((3, feature_dim))
    features[0, HOLDEM_PLAYER_FEATURE_OFFSET] = 1.0
    features[1, HOLDEM_PLAYER_FEATURE_OFFSET + 1] = 1.0
    features[2, HOLDEM_PLAYER_FEATURE_OFFSET + 1] = 1.0
    targets = torch.tensor(
        [
            HOLDEM_CANONICAL_ACTIONS.index("raise"),
            HOLDEM_CANONICAL_ACTIONS.index("raise"),
            HOLDEM_CANONICAL_ACTIONS.index("call"),
        ]
    )

    overrides = player_action_weight_overrides_from_specs(["1:raise=4.0"])
    weights = player_action_weights_from_features_targets(features, targets, overrides)

    assert weights.tolist() == [1.0, 4.0, 1.0]


def test_player_action_weight_overrides_validate_specs() -> None:
    with pytest.raises(ValueError, match="PLAYER:ACTION=WEIGHT"):
        player_action_weight_overrides_from_specs(["1-raise=2.0"])
    with pytest.raises(ValueError, match="0 or 1"):
        player_action_weight_overrides_from_specs(["2:raise=2.0"])
    with pytest.raises(ValueError, match="unknown"):
        player_action_weight_overrides_from_specs(["1:jam=2.0"])
    with pytest.raises(ValueError, match="positive"):
        player_action_weight_overrides_from_specs(["1:raise=0.0"])


def test_action_value_example_weights_apply_to_value_rows() -> None:
    torch = pytest.importorskip("torch")
    action_value_mask = torch.tensor([True, False, True], dtype=torch.bool)

    weights = action_value_example_weights_from_mask(action_value_mask, 2.5)

    assert weights.tolist() == [2.5, 1.0, 2.5]
    with pytest.raises(ValueError, match="positive"):
        action_value_example_weights_from_mask(action_value_mask, 0.0)


def test_player_action_value_weights_apply_to_value_rows_for_player() -> None:
    torch = pytest.importorskip("torch")
    feature_dim = HOLDEM_PLAYER_FEATURE_OFFSET + HOLDEM_PLAYER_FEATURE_DIM
    features = torch.zeros((4, feature_dim))
    features[0, HOLDEM_PLAYER_FEATURE_OFFSET] = 1.0
    features[1, HOLDEM_PLAYER_FEATURE_OFFSET + 1] = 1.0
    features[2, HOLDEM_PLAYER_FEATURE_OFFSET + 1] = 1.0
    features[3, HOLDEM_PLAYER_FEATURE_OFFSET] = 1.0
    action_value_mask = torch.tensor([True, True, False, True], dtype=torch.bool)

    overrides = player_value_weight_overrides_from_specs(["1=3.0"])
    weights = player_action_value_weights_from_features_mask(
        features,
        action_value_mask,
        overrides,
    )

    assert weights.tolist() == [1.0, 3.0, 1.0, 1.0]


def test_player_action_value_weight_overrides_validate_specs() -> None:
    with pytest.raises(ValueError, match="PLAYER=WEIGHT"):
        player_value_weight_overrides_from_specs(["1"])
    with pytest.raises(ValueError, match="invalid player"):
        player_value_weight_overrides_from_specs(["1:raise=2.0"])
    with pytest.raises(ValueError, match="0 or 1"):
        player_value_weight_overrides_from_specs(["2=2.0"])
    with pytest.raises(ValueError, match="positive"):
        player_value_weight_overrides_from_specs(["1=0.0"])


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


def test_facing_bet_action_weights_apply_only_to_response_targets() -> None:
    torch = pytest.importorskip("torch")
    masks = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, True, True, False],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    targets = torch.tensor(
        [
            HOLDEM_CANONICAL_ACTIONS.index("call"),
            HOLDEM_CANONICAL_ACTIONS.index("fold"),
            HOLDEM_CANONICAL_ACTIONS.index("call"),
        ]
    )

    overrides = action_weight_overrides_from_specs(["call=2.5"])
    weights = facing_bet_action_weights_from_masks_targets(masks, targets, overrides)

    assert weights.tolist() == [2.5, 1.0, 1.0]


def test_opponent_aggression_count_mask_uses_action_history_features() -> None:
    torch = pytest.importorskip("torch")
    features = torch.zeros(
        (3, HOLDEM_FEATURE_DIM + HOLDEM_ACTION_HISTORY_FEATURE_DIM)
    )
    features[0, -4] = 1.0 / 16.0
    features[1, -4] = 2.0 / 16.0
    features[2, -4] = 3.0 / 16.0

    mask = opponent_aggression_count_mask_from_features(features, 2)

    assert mask.tolist() == [False, True, True]
    with pytest.raises(ValueError, match="requires action-history features"):
        opponent_aggression_count_mask_from_features(torch.zeros((1, HOLDEM_FEATURE_DIM)), 2)
    with pytest.raises(ValueError, match="threshold must be positive"):
        opponent_aggression_count_mask_from_features(features, 0)


def test_facing_bet_action_weights_can_gate_on_opponent_aggressions() -> None:
    torch = pytest.importorskip("torch")
    features = torch.zeros(
        (3, HOLDEM_FEATURE_DIM + HOLDEM_ACTION_HISTORY_FEATURE_DIM)
    )
    features[0, -4] = 1.0 / 16.0
    features[1, -4] = 2.0 / 16.0
    features[2, -4] = 3.0 / 16.0
    masks = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, True, True, False],
            [False, False, True, True, False],
        ],
        dtype=torch.bool,
    )
    targets = torch.tensor(
        [
            HOLDEM_CANONICAL_ACTIONS.index("call"),
            HOLDEM_CANONICAL_ACTIONS.index("call"),
            HOLDEM_CANONICAL_ACTIONS.index("fold"),
        ]
    )

    overrides = action_weight_overrides_from_specs(["call=2.5"])
    weights = facing_bet_action_weights_from_masks_targets(
        masks,
        targets,
        overrides,
        features=features,
        after_opponent_aggressions=2,
    )

    assert weights.tolist() == [1.0, 2.5, 1.0]


def test_player_facing_bet_action_weights_apply_only_to_selected_player_response() -> None:
    torch = pytest.importorskip("torch")
    feature_dim = HOLDEM_PLAYER_FEATURE_OFFSET + HOLDEM_PLAYER_FEATURE_DIM
    features = torch.zeros((4, feature_dim))
    features[0, HOLDEM_PLAYER_FEATURE_OFFSET] = 1.0
    features[1, HOLDEM_PLAYER_FEATURE_OFFSET + 1] = 1.0
    features[2, HOLDEM_PLAYER_FEATURE_OFFSET + 1] = 1.0
    features[3, HOLDEM_PLAYER_FEATURE_OFFSET + 1] = 1.0
    masks = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, True, True, False],
            [False, False, True, True, False],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    targets = torch.tensor(
        [
            HOLDEM_CANONICAL_ACTIONS.index("call"),
            HOLDEM_CANONICAL_ACTIONS.index("call"),
            HOLDEM_CANONICAL_ACTIONS.index("fold"),
            HOLDEM_CANONICAL_ACTIONS.index("call"),
        ]
    )

    overrides = player_action_weight_overrides_from_specs(["1:call=3.0"])
    weights = player_facing_bet_action_weights_from_features_masks_targets(
        features,
        masks,
        targets,
        overrides,
    )

    assert weights.tolist() == [1.0, 3.0, 1.0, 1.0]


def test_player_facing_bet_action_weights_can_gate_on_opponent_aggressions() -> None:
    torch = pytest.importorskip("torch")
    feature_dim = HOLDEM_FEATURE_DIM + HOLDEM_ACTION_HISTORY_FEATURE_DIM
    features = torch.zeros((4, feature_dim))
    features[0, HOLDEM_PLAYER_FEATURE_OFFSET + 1] = 1.0
    features[1, HOLDEM_PLAYER_FEATURE_OFFSET + 1] = 1.0
    features[2, HOLDEM_PLAYER_FEATURE_OFFSET] = 1.0
    features[3, HOLDEM_PLAYER_FEATURE_OFFSET + 1] = 1.0
    features[0, -4] = 1.0 / 16.0
    features[1, -4] = 2.0 / 16.0
    features[2, -4] = 2.0 / 16.0
    features[3, -4] = 3.0 / 16.0
    masks = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, True, True, False],
            [False, False, True, True, False],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    targets = torch.tensor(
        [
            HOLDEM_CANONICAL_ACTIONS.index("call"),
            HOLDEM_CANONICAL_ACTIONS.index("call"),
            HOLDEM_CANONICAL_ACTIONS.index("call"),
            HOLDEM_CANONICAL_ACTIONS.index("call"),
        ]
    )

    overrides = player_action_weight_overrides_from_specs(["1:call=3.0"])
    weights = player_facing_bet_action_weights_from_features_masks_targets(
        features,
        masks,
        targets,
        overrides,
        after_opponent_aggressions=2,
    )

    assert weights.tolist() == [1.0, 3.0, 1.0, 1.0]


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
    assert metrics["init_kl_example_weighting"] == "example"
    assert metrics["facing_bet_weight"] == 1.0
    assert metrics["facing_bet_examples"] == 0
    empty_action_counts = {action: 0 for action in HOLDEM_CANONICAL_ACTIONS}
    assert metrics["facing_bet_target_action_counts"] == empty_action_counts
    assert metrics["facing_bet_predicted_action_counts"] == empty_action_counts
    assert metrics["action_weight_overrides"] == {}
    assert metrics["player_action_weight_overrides"] == {}
    assert metrics["facing_bet_action_weight_overrides"] == {}
    assert metrics["facing_bet_action_weighted_examples"] == 0
    assert metrics["player_facing_bet_action_weight_overrides"] == {}
    assert metrics["player_facing_bet_action_weighted_examples"] == 0
    assert metrics["player_target_action_counts"] is None
    assert metrics["player_facing_bet_target_action_counts"] is None
    assert metrics["player_facing_bet_predicted_action_counts"] is None
    assert metrics["soft_target_temperature"] is None
    assert metrics["soft_target_examples"] == 0
    assert metrics["action_value_target_examples"] == 0
    assert metrics["action_value_loss_weight"] == 0.0
    assert metrics["action_value_example_weight"] == 1.0
    assert metrics["action_value_weighted_examples"] == 0
    assert metrics["player_action_value_weight_overrides"] == {}
    assert metrics["player_action_value_weighted_examples"] == 0


def test_train_holdem_policy_appends_extra_cached_examples(tmp_path) -> None:
    base_examples_path = tmp_path / "base_examples.json"
    extra_examples_path = tmp_path / "extra_examples.json"
    write_policy_examples(
        base_examples_path,
        [
            HoldemPolicyExample(
                features=[1.0],
                action_index=0,
                legal_mask=[True, False, False, False, False],
            )
        ],
    )
    write_policy_examples(
        extra_examples_path,
        [
            HoldemPolicyExample(
                features=[0.0],
                action_index=1,
                legal_mask=[True, True, False, False, False],
            )
        ],
    )

    args = build_parser().parse_args(
        [
            "--examples-in",
            str(base_examples_path),
            "--extra-examples-in",
            str(extra_examples_path),
            "--epochs",
            "1",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    metrics = run(args)

    assert metrics["examples"] == 2
    assert metrics["extra_examples"] == 1
    assert metrics["extra_examples_in"] == [str(extra_examples_path)]


def test_train_holdem_policy_records_facing_bet_action_counts(tmp_path) -> None:
    examples_path = tmp_path / "examples.json"
    feature_dim = HOLDEM_PLAYER_FEATURE_OFFSET + HOLDEM_PLAYER_FEATURE_DIM

    def features_for_player(player: int) -> list[float]:
        features = [0.0 for _ in range(feature_dim)]
        features[HOLDEM_PLAYER_FEATURE_OFFSET + player] = 1.0
        return features

    examples = [
        HoldemPolicyExample(
            features=features_for_player(0),
            action_index=HOLDEM_CANONICAL_ACTIONS.index("call"),
            legal_mask=[False, False, True, True, False],
        ),
        HoldemPolicyExample(
            features=features_for_player(0),
            action_index=HOLDEM_CANONICAL_ACTIONS.index("fold"),
            legal_mask=[False, False, True, True, True],
        ),
        HoldemPolicyExample(
            features=features_for_player(1),
            action_index=HOLDEM_CANONICAL_ACTIONS.index("call"),
            legal_mask=[False, False, True, True, False],
        ),
        HoldemPolicyExample(
            features=features_for_player(1),
            action_index=HOLDEM_CANONICAL_ACTIONS.index("raise"),
            legal_mask=[False, False, True, True, True],
        ),
        HoldemPolicyExample(
            features=features_for_player(1),
            action_index=HOLDEM_CANONICAL_ACTIONS.index("bet"),
            legal_mask=[True, True, False, False, False],
        ),
    ]
    write_policy_examples(examples_path, examples)

    args = build_parser().parse_args(
        [
            "--examples-in",
            str(examples_path),
            "--epochs",
            "2",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    metrics = run(args)

    assert metrics["facing_bet_examples"] == 4
    assert metrics["facing_bet_target_action_counts"] == {
        "check": 0,
        "bet": 0,
        "call": 2,
        "fold": 1,
        "raise": 1,
    }
    assert sum(metrics["facing_bet_predicted_action_counts"].values()) == 4
    assert metrics["player_facing_bet_target_action_counts"] == {
        "0": {
            "check": 0,
            "bet": 0,
            "call": 1,
            "fold": 1,
            "raise": 0,
        },
        "1": {
            "check": 0,
            "bet": 0,
            "call": 1,
            "fold": 0,
            "raise": 1,
        },
    }
    assert sum(metrics["player_facing_bet_predicted_action_counts"]["0"].values()) == 2
    assert sum(metrics["player_facing_bet_predicted_action_counts"]["1"].values()) == 2


def test_train_holdem_policy_accepts_soft_targets(tmp_path) -> None:
    examples_path = tmp_path / "examples.json"
    examples = [
        HoldemPolicyExample(
            features=[float(index % 2), 1.0],
            action_index=index % 2,
            legal_mask=[True, True, False, False, False],
            action_probs=[0.25, 0.75, 0.0, 0.0, 0.0],
        )
        for index in range(6)
    ]
    write_policy_examples(examples_path, examples)

    args = build_parser().parse_args(
        [
            "--examples-in",
            str(examples_path),
            "--epochs",
            "2",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    metrics = run(args)

    assert metrics["soft_target_examples"] == 6
    assert metrics["soft_target_action_mass"]["check"] == pytest.approx(1.5)
    assert metrics["soft_target_action_mass"]["bet"] == pytest.approx(4.5)


def test_train_holdem_policy_accepts_action_value_targets(tmp_path) -> None:
    examples_path = tmp_path / "examples.json"
    feature_dim = HOLDEM_PLAYER_FEATURE_OFFSET + HOLDEM_PLAYER_FEATURE_DIM
    examples = [
        HoldemPolicyExample(
            features=[
                1.0 if feature_index == HOLDEM_PLAYER_FEATURE_OFFSET + index % 2 else 0.0
                for feature_index in range(feature_dim)
            ],
            action_index=index % 2,
            legal_mask=[True, True, False, False, False],
            action_probs=[0.25, 0.75, 0.0, 0.0, 0.0],
            action_values=[0.0, 2.0, 0.0, 0.0, 0.0],
        )
        for index in range(6)
    ]
    write_policy_examples(examples_path, examples)

    args = build_parser().parse_args(
        [
            "--examples-in",
            str(examples_path),
            "--action-value-loss-weight",
            "0.1",
            "--action-value-target-scale",
            "2.0",
            "--action-value-example-weight",
            "2.0",
            "--player-action-value-weight",
            "1=3.0",
            "--epochs",
            "2",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    metrics = run(args)

    assert metrics["action_value_target_examples"] == 6
    assert metrics["action_value_loss_weight"] == 0.1
    assert metrics["action_value_target_scale"] == 2.0
    assert metrics["action_value_example_weight"] == 2.0
    assert metrics["action_value_weighted_examples"] == 6
    assert metrics["player_action_value_weight_overrides"] == {"1": 3.0}
    assert metrics["player_action_value_weighted_examples"] == 3


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

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.holdem_features import HOLDEM_FEATURE_DIM  # noqa: E402
from alphapoker.holdem_model import HoldemPolicyNet  # noqa: E402
from alphapoker.train_holdem_actor_critic import build_parser, run  # noqa: E402


def assert_same_state_dict(left, right) -> None:
    assert left.keys() == right.keys()
    for key, left_value in left.items():
        assert torch.equal(left_value, right[key])


def test_actor_critic_parser_accepts_weighted_mix() -> None:
    args = build_parser().parse_args(
        [
            "--opponent-policies",
            "random,pot-odds",
            "--opponent-policy-weights",
            "0.1,0.9",
            "--rollout-sims",
            "4",
            "--rollout-margin",
            "1.5",
            "--model-player",
            "both",
            "--model-player-weights",
            "0.4,0.6",
            "--value-loss-coef",
            "0.25",
            "--feature-equity-sims",
            "4",
            "--feature-equity-mode",
            "sampled",
            "--checkpoint-selection",
            "evaluation",
            "--selection-eval-hands",
            "3",
            "--selection-eval-interval-hands",
            "2",
            "--selection-eval-opponent-policy",
            "random",
            "--selection-eval-rollout-sims",
            "1",
            "--selection-eval-rollout-margin",
            "1.25",
            "--selection-eval-model-player",
            "both",
            "--selection-eval-jobs",
            "1",
            "--selection-eval-paired-seats",
            "--selection-eval-seed",
            "222",
            "--out",
            "out",
        ]
    )

    assert args.model_player == (0, 1)
    assert args.model_player_weights == (0.4, 0.6)
    assert args.opponent_policies == ("random", "pot-odds")
    assert args.opponent_policy_weights == (0.1, 0.9)
    assert args.rollout_sims == 4
    assert args.rollout_margin == 1.5
    assert args.value_loss_coef == 0.25
    assert args.feature_equity_sims == 4
    assert args.feature_equity_mode == "sampled"
    assert args.checkpoint_selection == "evaluation"
    assert args.selection_eval_hands == 3
    assert args.selection_eval_interval_hands == 2
    assert args.selection_eval_opponent_policy == "random"
    assert args.selection_eval_rollout_sims == 1
    assert args.selection_eval_rollout_margin == 1.25
    assert args.selection_eval_model_player == (0, 1)
    assert args.selection_eval_jobs == 1
    assert args.selection_eval_paired_seats
    assert args.selection_eval_seed == 222


def test_actor_critic_parser_accepts_evaluation_options() -> None:
    args = build_parser().parse_args(
        [
            "--out",
            "out",
            "--eval-hands",
            "5",
            "--eval-opponent-policy",
            "tuned-pot-odds",
            "--eval-model-player",
            "both",
            "--eval-jobs",
            "2",
            "--eval-paired-seats",
            "--eval-seed",
            "123",
            "--eval-rollout-sims",
            "3",
            "--eval-rollout-margin",
            "1.75",
        ]
    )

    assert args.eval_hands == 5
    assert args.eval_opponent_policy == "tuned-pot-odds"
    assert args.eval_model_player == (0, 1)
    assert args.eval_jobs == 2
    assert args.eval_paired_seats
    assert args.eval_seed == 123
    assert args.eval_rollout_sims == 3
    assert args.eval_rollout_margin == 1.75


def test_actor_critic_records_rollout_training_options(tmp_path) -> None:
    metrics = run(
        build_parser().parse_args(
            [
                "--hands",
                "1",
                "--batch-hands",
                "1",
                "--opponent-policy",
                "tight-safe-rollout-pot-odds",
                "--rollout-sims",
                "1",
                "--rollout-margin",
                "1.5",
                "--out",
                str(tmp_path),
            ]
        )
    )

    assert metrics["opponent_policy"] == "tight-safe-rollout-pot-odds"
    assert metrics["rollout_sims"] == 1
    assert metrics["rollout_margin"] == 1.5


def test_actor_critic_run_records_evaluation(tmp_path) -> None:
    metrics = run(
        build_parser().parse_args(
            [
                "--hands",
                "2",
                "--batch-hands",
                "1",
                "--opponent-policy",
                "random",
                "--out",
                str(tmp_path),
                "--eval-hands",
                "1",
                "--eval-opponent-policy",
                "random",
            ]
        )
    )

    assert metrics["evaluation"]["hands"] == 1
    assert metrics["evaluation"]["opponent_policy"] == "random"


def test_actor_critic_can_select_final_checkpoint(tmp_path) -> None:
    metrics = run(
        build_parser().parse_args(
            [
                "--hands",
                "2",
                "--batch-hands",
                "1",
                "--opponent-policy",
                "random",
                "--checkpoint-selection",
                "final",
                "--out",
                str(tmp_path),
            ]
        )
    )
    policy_checkpoint = torch.load(tmp_path / "holdem_policy.pt", map_location="cpu", weights_only=False)
    final_policy_checkpoint = torch.load(
        tmp_path / "holdem_policy_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    value_checkpoint = torch.load(tmp_path / "holdem_value.pt", map_location="cpu", weights_only=False)
    final_value_checkpoint = torch.load(
        tmp_path / "holdem_value_final.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert metrics["checkpoint_selection"] == "final"
    assert metrics["final_checkpoint"] == str(tmp_path / "holdem_policy_final.pt")
    assert metrics["final_value_checkpoint"] == str(tmp_path / "holdem_value_final.pt")
    assert_same_state_dict(
        policy_checkpoint["model_state_dict"],
        final_policy_checkpoint["model_state_dict"],
    )
    assert_same_state_dict(
        value_checkpoint["model_state_dict"],
        final_value_checkpoint["model_state_dict"],
    )


def test_actor_critic_evaluation_selection_requires_eval_hands(tmp_path) -> None:
    with pytest.raises(ValueError, match="selection-eval-hands"):
        run(
            build_parser().parse_args(
                [
                    "--hands",
                    "1",
                    "--checkpoint-selection",
                    "evaluation",
                    "--out",
                    str(tmp_path),
                ]
            )
        )


def test_actor_critic_can_select_evaluation_checkpoint(tmp_path) -> None:
    metrics = run(
        build_parser().parse_args(
            [
                "--hands",
                "2",
                "--batch-hands",
                "1",
                "--opponent-policy",
                "random",
                "--checkpoint-selection",
                "evaluation",
                "--selection-eval-hands",
                "1",
                "--selection-eval-interval-hands",
                "1",
                "--selection-eval-opponent-policy",
                "random",
                "--selection-eval-rollout-sims",
                "2",
                "--selection-eval-rollout-margin",
                "1.5",
                "--selection-eval-seed",
                "50",
                "--out",
                str(tmp_path),
            ]
        )
    )

    assert metrics["checkpoint_selection"] == "evaluation"
    assert metrics["selection_eval_hands"] == 1
    assert metrics["selection_eval_interval_hands"] == 1
    assert metrics["selection_eval_rollout_sims"] == 2
    assert metrics["selection_eval_rollout_margin"] == 1.5
    assert [item["hands_played"] for item in metrics["selection_evaluations"]] == [0, 1, 2]
    assert {item["rollout_margin"] for item in metrics["selection_evaluations"]} == {1.5}
    assert metrics["best_selection_eval_hands_played"] in (0, 1, 2)
    assert not (tmp_path / "holdem_policy_selection_candidate.pt").exists()
    assert (tmp_path / "holdem_policy.pt").exists()
    assert (tmp_path / "holdem_value.pt").exists()


def test_actor_critic_preserves_exact_feature_checkpoint(tmp_path) -> None:
    init_checkpoint = tmp_path / "init_policy.pt"
    out_dir = tmp_path / "out"
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM + 1).state_dict(),
            "canonical_actions": ["bet", "call", "check", "fold", "raise"],
            "input_dim": HOLDEM_FEATURE_DIM + 1,
            "feature_equity_sims": 2,
            "feature_equity_mode": "turn-river-exact",
            "feature_equity_checkpoint": None,
        },
        init_checkpoint,
    )

    metrics = run(
        build_parser().parse_args(
            [
                "--hands",
                "2",
                "--batch-hands",
                "1",
                "--opponent-policy",
                "random",
                "--init-checkpoint",
                str(init_checkpoint),
                "--out",
                str(out_dir),
            ]
        )
    )
    checkpoint = torch.load(out_dir / "holdem_policy.pt", map_location="cpu", weights_only=False)

    assert checkpoint["input_dim"] == HOLDEM_FEATURE_DIM + 1
    assert checkpoint["feature_equity_sims"] == 2
    assert checkpoint["feature_equity_mode"] == "turn-river-exact"
    assert metrics["feature_equity_sims"] == 2
    assert metrics["feature_equity_mode"] == "turn-river-exact"


def test_actor_critic_can_override_checkpoint_feature_mode(tmp_path) -> None:
    init_checkpoint = tmp_path / "init_policy.pt"
    out_dir = tmp_path / "out"
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM + 1).state_dict(),
            "canonical_actions": ["bet", "call", "check", "fold", "raise"],
            "input_dim": HOLDEM_FEATURE_DIM + 1,
            "feature_equity_sims": 2,
            "feature_equity_mode": "turn-river-exact",
            "feature_equity_checkpoint": None,
        },
        init_checkpoint,
    )

    metrics = run(
        build_parser().parse_args(
            [
                "--hands",
                "2",
                "--batch-hands",
                "1",
                "--opponent-policy",
                "random",
                "--init-checkpoint",
                str(init_checkpoint),
                "--feature-equity-sims",
                "4",
                "--feature-equity-mode",
                "sampled",
                "--out",
                str(out_dir),
            ]
        )
    )
    checkpoint = torch.load(out_dir / "holdem_policy.pt", map_location="cpu", weights_only=False)

    assert checkpoint["input_dim"] == HOLDEM_FEATURE_DIM + 1
    assert checkpoint["feature_equity_sims"] == 4
    assert checkpoint["feature_equity_mode"] == "sampled"
    assert metrics["feature_equity_sims"] == 4
    assert metrics["feature_equity_mode"] == "sampled"

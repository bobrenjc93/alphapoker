import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.holdem_features import HOLDEM_FEATURE_DIM  # noqa: E402
from alphapoker.holdem_model import HoldemPolicyNet  # noqa: E402
from alphapoker.train_holdem_actor_critic import build_parser, run  # noqa: E402


def test_actor_critic_parser_accepts_weighted_mix() -> None:
    args = build_parser().parse_args(
        [
            "--opponent-policies",
            "random,pot-odds",
            "--opponent-policy-weights",
            "0.1,0.9",
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
            "--out",
            "out",
        ]
    )

    assert args.model_player == (0, 1)
    assert args.model_player_weights == (0.4, 0.6)
    assert args.opponent_policies == ("random", "pot-odds")
    assert args.opponent_policy_weights == (0.1, 0.9)
    assert args.value_loss_coef == 0.25
    assert args.feature_equity_sims == 4
    assert args.feature_equity_mode == "sampled"


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
        ]
    )

    assert args.eval_hands == 5
    assert args.eval_opponent_policy == "tuned-pot-odds"
    assert args.eval_model_player == (0, 1)
    assert args.eval_jobs == 2
    assert args.eval_paired_seats
    assert args.eval_seed == 123


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

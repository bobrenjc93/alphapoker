import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.holdem_features import HOLDEM_FEATURE_DIM  # noqa: E402
from alphapoker.holdem_model import HoldemPolicyNet  # noqa: E402
from alphapoker.train_holdem_policy_gradient import (  # noqa: E402
    build_parser,
    parse_model_players,
    parse_policy_mix,
    parse_policy_weights,
    run,
)


def test_policy_gradient_parser_accepts_pot_odds() -> None:
    args = build_parser().parse_args(
        [
            "--model-player",
            "1",
            "--model-player-weights",
            "1.0",
            "--opponent-policy",
            "pot-odds",
            "--opponent-policies",
            "random,pot-odds",
            "--opponent-policy-weights",
            "0.25,0.75",
            "--init-checkpoint",
            "model.pt",
            "--feature-equity-sims",
            "4",
            "--feature-equity-mode",
            "sampled",
            "--out",
            "out",
        ]
    )

    assert args.model_player == (1,)
    assert args.model_player_weights == (1.0,)
    assert args.opponent_policy == "pot-odds"
    assert args.opponent_policies == ("random", "pot-odds")
    assert args.opponent_policy_weights == (0.25, 0.75)
    assert str(args.init_checkpoint) == "model.pt"
    assert args.feature_equity_sims == 4
    assert args.feature_equity_mode == "sampled"


def test_parse_policy_mix_rejects_unknown_policy() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--opponent-policies", "random,bad", "--out", "out"])
    assert parse_policy_mix("random,pot-odds") == ("random", "pot-odds")
    assert parse_policy_weights("0.25,0.75") == (0.25, 0.75)


def test_policy_gradient_parser_accepts_both_model_players() -> None:
    args = build_parser().parse_args(
        ["--model-player", "both", "--model-player-weights", "0.25,0.75", "--out", "out"]
    )

    assert args.model_player == (0, 1)
    assert args.model_player_weights == (0.25, 0.75)
    assert parse_model_players("both") == (0, 1)


def test_policy_gradient_parser_accepts_evaluation_options() -> None:
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


def test_policy_gradient_run_records_evaluation(tmp_path) -> None:
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
                "--eval-model-player",
                "both",
                "--eval-paired-seats",
            ]
        )
    )

    assert metrics["evaluation"]["hands"] == 2
    assert metrics["evaluation"]["paired_deals"] == 1
    assert metrics["evaluation"]["paired_seats"]
    assert metrics["evaluation"]["opponent_policy"] == "random"


def test_policy_gradient_preserves_exact_feature_checkpoint(tmp_path) -> None:
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


def test_policy_gradient_can_override_checkpoint_feature_mode(tmp_path) -> None:
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

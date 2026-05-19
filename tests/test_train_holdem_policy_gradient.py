import pytest


pytest.importorskip("torch")
pytest.importorskip("treys")

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
            "--eval-seed",
            "123",
        ]
    )

    assert args.eval_hands == 5
    assert args.eval_opponent_policy == "tuned-pot-odds"
    assert args.eval_model_player == (0, 1)
    assert args.eval_jobs == 2
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
            ]
        )
    )

    assert metrics["evaluation"]["hands"] == 1
    assert metrics["evaluation"]["opponent_policy"] == "random"

import pytest


pytest.importorskip("treys")

from alphapoker.train_holdem_mccfr import build_parser, run  # noqa: E402


def test_train_holdem_mccfr_parser_accepts_eval_options() -> None:
    args = build_parser().parse_args(
        [
            "--iterations",
            "3",
            "--checkpoint-in",
            "checkpoint.json",
            "--eval-hands",
            "2",
            "--max-bets-per-round",
            "4",
            "--traversal",
            "external",
            "--abstraction",
            "medium",
            "--model-player",
            "both",
            "--eval-jobs",
            "2",
            "--eval-paired-seats",
            "--opponent-policy",
            "tuned-pot-odds",
            "--fallback-policy",
            "tuned-pot-odds",
            "--min-strategy-weight",
            "10",
            "--discard-checkpoint",
            "--out",
            "out",
        ]
    )

    assert args.iterations == 3
    assert args.checkpoint_in.name == "checkpoint.json"
    assert args.eval_hands == 2
    assert args.max_bets_per_round == 4
    assert args.traversal == "external"
    assert args.abstraction == "medium"
    assert args.model_player == (0, 1)
    assert args.eval_jobs == 2
    assert args.eval_paired_seats
    assert args.opponent_policy == "tuned-pot-odds"
    assert args.fallback_policy == "tuned-pot-odds"
    assert args.min_strategy_weight == 10
    assert args.discard_checkpoint


def test_train_holdem_mccfr_parser_accepts_equity_abstraction() -> None:
    args = build_parser().parse_args(["--abstraction", "equity", "--out", "out"])

    assert args.abstraction == "equity"


def test_train_holdem_mccfr_run_smoke(tmp_path) -> None:
    metrics = run(
        build_parser().parse_args(
            [
                "--iterations",
                "2",
                "--eval-hands",
                "1",
                "--max-bets-per-round",
                "4",
                "--traversal",
                "external",
                "--abstraction",
                "coarse",
                "--equity-sims",
                "2",
                "--discard-checkpoint",
                "--out",
                str(tmp_path / "run"),
            ]
        )
    )

    assert metrics["iterations"] == 2
    assert metrics["start_iterations"] == 0
    assert metrics["train_iterations"] == 2
    assert metrics["infosets"] > 0
    assert metrics["max_bets_per_round"] == 4
    assert metrics["traversal"] == "external"
    assert metrics["abstraction"] == "coarse"
    assert metrics["min_strategy_weight"] == 0.0
    assert not metrics["checkpoint_saved"]
    assert not (tmp_path / "run" / "holdem_mccfr.json").exists()
    assert metrics["evaluation"]["hands"] == 1
    assert metrics["evaluation"]["jobs"] == 1


def test_train_holdem_mccfr_resumes_from_checkpoint(tmp_path) -> None:
    first = run(
        build_parser().parse_args(
            [
                "--iterations",
                "1",
                "--max-bets-per-round",
                "4",
                "--traversal",
                "external",
                "--abstraction",
                "coarse",
                "--out",
                str(tmp_path / "first"),
            ]
        )
    )
    checkpoint = tmp_path / "first" / "holdem_mccfr.json"

    resumed = run(
        build_parser().parse_args(
            [
                "--iterations",
                "1",
                "--checkpoint-in",
                str(checkpoint),
                "--discard-checkpoint",
                "--out",
                str(tmp_path / "resumed"),
            ]
        )
    )

    assert checkpoint.exists()
    assert first["iterations"] == 1
    assert resumed["start_iterations"] == 1
    assert resumed["train_iterations"] == 1
    assert resumed["iterations"] == 2
    assert resumed["checkpoint_in"] == str(checkpoint)
    assert not (tmp_path / "resumed" / "holdem_mccfr.json").exists()

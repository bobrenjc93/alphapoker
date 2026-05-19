import pytest


pytest.importorskip("treys")

from alphapoker.train_holdem_mccfr import build_parser, run  # noqa: E402


def test_train_holdem_mccfr_parser_accepts_eval_options() -> None:
    args = build_parser().parse_args(
        [
            "--iterations",
            "3",
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
            "--opponent-policy",
            "tuned-pot-odds",
            "--fallback-policy",
            "tuned-pot-odds",
            "--min-strategy-weight",
            "10",
            "--out",
            "out",
        ]
    )

    assert args.iterations == 3
    assert args.eval_hands == 2
    assert args.max_bets_per_round == 4
    assert args.traversal == "external"
    assert args.abstraction == "medium"
    assert args.model_player == (0, 1)
    assert args.opponent_policy == "tuned-pot-odds"
    assert args.fallback_policy == "tuned-pot-odds"
    assert args.min_strategy_weight == 10


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
                "--out",
                str(tmp_path / "run"),
            ]
        )
    )

    assert metrics["iterations"] == 2
    assert metrics["infosets"] > 0
    assert metrics["max_bets_per_round"] == 4
    assert metrics["traversal"] == "external"
    assert metrics["abstraction"] == "coarse"
    assert metrics["min_strategy_weight"] == 0.0
    assert metrics["evaluation"]["hands"] == 1

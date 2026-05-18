import pytest


pytest.importorskip("treys")

from alphapoker.holdem_self_play import build_parser, run  # noqa: E402


def test_holdem_self_play_smoke() -> None:
    args = build_parser().parse_args(["--hands", "5", "--seed", "3"])
    metrics = run(args)

    assert metrics["hands"] == 5
    assert metrics["showdowns"] + metrics["folds"] == 5
    assert metrics["avg_actions"] > 0


def test_holdem_self_play_equity_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "3",
            "--seed",
            "4",
            "--player0-policy",
            "equity",
            "--player1-policy",
            "random",
            "--equity-sims",
            "8",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 3
    assert metrics["player0_policy"] == "equity"

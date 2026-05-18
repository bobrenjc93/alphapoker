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


def test_holdem_self_play_writes_metrics(tmp_path) -> None:
    out = tmp_path / "metrics.json"
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "8",
            "--out",
            str(out),
        ]
    )
    metrics = run(args)

    assert out.exists()
    assert metrics["utility_stderr_p0"] >= 0.0

import argparse

import pytest


pytest.importorskip("treys")

from alphapoker.train_holdem_equity import (  # noqa: E402
    _shard_hands,
    build_parser,
    parse_player,
    player_label,
)


def test_parse_player_accepts_both_seats() -> None:
    assert parse_player("0") == 0
    assert parse_player("1") == 1
    assert parse_player("both") is None
    assert player_label(None) == "both"


def test_parse_player_rejects_bad_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_player("2")


def test_train_holdem_equity_parser_accepts_both() -> None:
    args = build_parser().parse_args(
        [
            "--player",
            "both",
            "--examples-in",
            "examples.json",
            "--examples-out",
            "cached.json",
            "--jobs",
            "3",
            "--out",
            "out",
        ]
    )

    assert args.player is None
    assert args.jobs == 3
    assert str(args.examples_in) == "examples.json"
    assert str(args.examples_out) == "cached.json"


def test_train_holdem_equity_parser_accepts_tuned_pot_odds() -> None:
    args = build_parser().parse_args(
        [
            "--opponent-policy",
            "tuned-pot-odds",
            "--out",
            "out",
        ]
    )

    assert args.opponent_policy == "tuned-pot-odds"


def test_shard_hands_for_parallel_training() -> None:
    assert _shard_hands(10, 3) == [4, 3, 3]
    assert _shard_hands(2, 4) == [1, 1]

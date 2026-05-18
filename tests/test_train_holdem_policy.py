import pytest


pytest.importorskip("treys")

from alphapoker.train_holdem_policy import build_parser  # noqa: E402


def test_train_holdem_policy_parser_accepts_pot_odds_expert() -> None:
    args = build_parser().parse_args(
        [
            "--expert-policy",
            "pot-odds",
            "--opponent-policy",
            "pot-odds",
            "--examples-in",
            "examples.json",
            "--examples-out",
            "cached.json",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "pot-odds"
    assert args.opponent_policy == "pot-odds"
    assert str(args.examples_in) == "examples.json"
    assert str(args.examples_out) == "cached.json"

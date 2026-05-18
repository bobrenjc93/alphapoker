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
            "--rollout-sims",
            "2",
            "--feature-equity-sims",
            "3",
            "--examples-in",
            "examples.json",
            "--examples-out",
            "cached.json",
            "--class-weighting",
            "balanced",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "pot-odds"
    assert args.opponent_policy == "pot-odds"
    assert args.rollout_sims == 2
    assert args.feature_equity_sims == 3
    assert args.class_weighting == "balanced"
    assert str(args.examples_in) == "examples.json"
    assert str(args.examples_out) == "cached.json"


def test_train_holdem_policy_parser_accepts_rollout_expert() -> None:
    args = build_parser().parse_args(
        [
            "--expert-policy",
            "rollout-pot-odds",
            "--opponent-policy",
            "pot-odds",
            "--rollout-sims",
            "2",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "rollout-pot-odds"
    assert args.rollout_sims == 2


def test_train_holdem_policy_parser_accepts_tuned_expert() -> None:
    args = build_parser().parse_args(
        [
            "--expert-policy",
            "tuned-pot-odds",
            "--opponent-policy",
            "tuned-pot-odds",
            "--out",
            "out",
        ]
    )

    assert args.expert_policy == "tuned-pot-odds"
    assert args.opponent_policy == "tuned-pot-odds"

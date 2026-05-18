import pytest


pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.train_holdem_policy_gradient import build_parser  # noqa: E402


def test_policy_gradient_parser_accepts_pot_odds() -> None:
    args = build_parser().parse_args(
        [
            "--model-player",
            "1",
            "--opponent-policy",
            "pot-odds",
            "--out",
            "out",
        ]
    )

    assert args.model_player == 1
    assert args.opponent_policy == "pot-odds"

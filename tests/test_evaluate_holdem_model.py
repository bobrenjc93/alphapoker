import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.evaluate_holdem_model import build_parser, make_opponent_policy  # noqa: E402


def test_make_opponent_policy_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        make_opponent_policy("bad", __import__("random").Random(0), 8)


def test_holdem_model_eval_parser_accepts_model_player() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--model-player",
            "1",
        ]
    )

    assert args.model_player == 1


def test_holdem_model_eval_parser_accepts_pot_odds_opponent() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--opponent-policy",
            "pot-odds",
        ]
    )

    assert args.opponent_policy == "pot-odds"

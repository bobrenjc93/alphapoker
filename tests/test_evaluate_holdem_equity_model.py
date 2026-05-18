import pytest


pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.evaluate_holdem_equity_model import make_opponent_policy  # noqa: E402
from alphapoker.evaluate_holdem_equity_model import build_parser  # noqa: E402


def test_make_equity_model_opponent_policy_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        make_opponent_policy("bad", __import__("random").Random(0), 8)


def test_equity_model_eval_parser_accepts_thresholds() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--bet-threshold",
            "0.6",
            "--raise-threshold",
            "0.8",
            "--call-threshold",
            "0.4",
        ]
    )

    assert args.bet_threshold == 0.6
    assert args.raise_threshold == 0.8
    assert args.call_threshold == 0.4

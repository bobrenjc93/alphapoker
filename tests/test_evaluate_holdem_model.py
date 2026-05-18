import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.evaluate_holdem_model import make_opponent_policy  # noqa: E402


def test_make_opponent_policy_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        make_opponent_policy("bad", __import__("random").Random(0), 8)

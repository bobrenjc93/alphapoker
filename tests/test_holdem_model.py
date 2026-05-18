import pytest


torch = pytest.importorskip("torch")

from alphapoker.holdem_model import HoldemEquityNet, HoldemPolicyNet, HoldemValueNet  # noqa: E402
from alphapoker.holdem_features import HOLDEM_FEATURE_DIM  # noqa: E402


def test_holdem_policy_net_shapes() -> None:
    model = HoldemPolicyNet()
    logits = model(torch.zeros(11, HOLDEM_FEATURE_DIM))

    assert tuple(logits.shape) == (11, 5)


def test_holdem_equity_net_shapes() -> None:
    model = HoldemEquityNet()
    logits = model(torch.zeros(11, HOLDEM_FEATURE_DIM))

    assert tuple(logits.shape) == (11,)


def test_holdem_value_net_shapes() -> None:
    model = HoldemValueNet()
    values = model(torch.zeros(11, HOLDEM_FEATURE_DIM))

    assert tuple(values.shape) == (11,)

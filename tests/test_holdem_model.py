import pytest


torch = pytest.importorskip("torch")

from alphapoker.holdem_model import HoldemEquityNet, HoldemPolicyNet  # noqa: E402


def test_holdem_policy_net_shapes() -> None:
    model = HoldemPolicyNet()
    logits = model(torch.zeros(11, 117))

    assert tuple(logits.shape) == (11, 5)


def test_holdem_equity_net_shapes() -> None:
    model = HoldemEquityNet()
    logits = model(torch.zeros(11, 117))

    assert tuple(logits.shape) == (11,)

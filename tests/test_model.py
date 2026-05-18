import pytest


torch = pytest.importorskip("torch")

from alphapoker.model import KuhnPolicyValueNet  # noqa: E402


def test_kuhn_policy_value_model_shapes() -> None:
    model = KuhnPolicyValueNet()
    policy, value = model(torch.zeros(3, 9))
    assert tuple(policy.shape) == (3, 4)
    assert tuple(value.shape) == (3,)

import pytest


torch = pytest.importorskip("torch")

from alphapoker.leduc_model import LeducPolicyValueNet  # noqa: E402


def test_leduc_policy_value_model_shapes() -> None:
    model = LeducPolicyValueNet()
    policy, value = model(torch.zeros(7, 19))
    assert tuple(policy.shape) == (7, 5)
    assert tuple(value.shape) == (7,)

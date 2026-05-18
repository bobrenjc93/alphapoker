import pytest


pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.train_holdem_actor_critic import build_parser  # noqa: E402


def test_actor_critic_parser_accepts_weighted_mix() -> None:
    args = build_parser().parse_args(
        [
            "--opponent-policies",
            "random,pot-odds",
            "--opponent-policy-weights",
            "0.1,0.9",
            "--value-loss-coef",
            "0.25",
            "--out",
            "out",
        ]
    )

    assert args.opponent_policies == ("random", "pot-odds")
    assert args.opponent_policy_weights == (0.1, 0.9)
    assert args.value_loss_coef == 0.25

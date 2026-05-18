import pytest


pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.evaluate_holdem_policy_imitation import build_parser  # noqa: E402


def test_holdem_policy_imitation_parser_accepts_pot_odds() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--expert-policy",
            "pot-odds",
            "--opponent-policy",
            "pot-odds",
        ]
    )

    assert args.expert_policy == "pot-odds"
    assert args.opponent_policy == "pot-odds"

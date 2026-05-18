import pytest


pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.train_holdem_policy_gradient import build_parser, parse_policy_mix  # noqa: E402


def test_policy_gradient_parser_accepts_pot_odds() -> None:
    args = build_parser().parse_args(
        [
            "--model-player",
            "1",
            "--opponent-policy",
            "pot-odds",
            "--opponent-policies",
            "random,pot-odds",
            "--init-checkpoint",
            "model.pt",
            "--out",
            "out",
        ]
    )

    assert args.model_player == 1
    assert args.opponent_policy == "pot-odds"
    assert args.opponent_policies == ("random", "pot-odds")
    assert str(args.init_checkpoint) == "model.pt"


def test_parse_policy_mix_rejects_unknown_policy() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--opponent-policies", "random,bad", "--out", "out"])
    assert parse_policy_mix("random,pot-odds") == ("random", "pot-odds")

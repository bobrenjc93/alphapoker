import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("treys")

from alphapoker.evaluate_holdem_model import (  # noqa: E402
    build_parser,
    make_opponent_policy,
    model_policy_from_checkpoint,
    parse_model_players,
)
from alphapoker.holdem import deal_fixed_limit_holdem  # noqa: E402
from alphapoker.holdem_features import HOLDEM_FEATURE_DIM  # noqa: E402
from alphapoker.holdem_model import HoldemEquityNet, HoldemPolicyNet  # noqa: E402


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

    assert args.model_player == (1,)


def test_holdem_model_eval_parser_accepts_both_model_players() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--model-player",
            "both",
        ]
    )

    assert args.model_player == (0, 1)
    assert parse_model_players("both") == (0, 1)


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


def test_holdem_model_eval_parser_accepts_rollout_opponent() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--opponent-policy",
            "rollout-pot-odds",
            "--rollout-sims",
            "2",
        ]
    )

    assert args.opponent_policy == "rollout-pot-odds"
    assert args.rollout_sims == 2


def test_holdem_model_eval_parser_accepts_tuned_pot_odds_opponent() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--opponent-policy",
            "tuned-pot-odds",
        ]
    )

    assert args.opponent_policy == "tuned-pot-odds"


def test_model_policy_loads_relative_feature_equity_checkpoint(tmp_path) -> None:
    equity_checkpoint = tmp_path / "equity.pt"
    policy_checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "model_state_dict": HoldemEquityNet().state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM,
        },
        equity_checkpoint,
    )
    torch.save(
        {
            "model_state_dict": HoldemPolicyNet(input_dim=HOLDEM_FEATURE_DIM + 1).state_dict(),
            "input_dim": HOLDEM_FEATURE_DIM + 1,
            "feature_equity_sims": None,
            "feature_equity_checkpoint": "equity.pt",
        },
        policy_checkpoint,
    )

    state = deal_fixed_limit_holdem()
    action = model_policy_from_checkpoint(policy_checkpoint)(state)

    assert action in state.legal_actions()

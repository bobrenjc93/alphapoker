import pytest


pytest.importorskip("treys")

from alphapoker.holdem import CALL, CHECK, FOLD  # noqa: E402
from alphapoker.holdem_evaluation import evaluate_policy_match  # noqa: E402
from alphapoker.holdem_evaluation import policies_for_model_player  # noqa: E402


def passive_policy(state):
    legal = state.legal_actions()
    return CHECK if CHECK in legal else CALL


def folding_policy(state):
    legal = state.legal_actions()
    return FOLD if FOLD in legal else CHECK


def test_policies_for_model_player_rejects_bad_seat() -> None:
    with pytest.raises(ValueError, match="model_player"):
        policies_for_model_player(passive_policy, folding_policy, 2)


def test_evaluate_policy_match_reports_model_utility_for_player_one() -> None:
    metrics = evaluate_policy_match(
        model_policy=passive_policy,
        opponent_policy=folding_policy,
        hands=5,
        seed=3,
        model_player=1,
    )

    assert metrics["model_player"] == 1
    assert metrics["avg_utility_model"] == pytest.approx(-metrics["avg_utility_p0"])

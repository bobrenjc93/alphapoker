import pytest


pytest.importorskip("treys")

from alphapoker.holdem import BET, CALL, CHECK, FOLD  # noqa: E402
from alphapoker.holdem_evaluation import aggregate_policy_match_shards  # noqa: E402
from alphapoker.holdem_evaluation import evaluate_policy_match  # noqa: E402
from alphapoker.holdem_evaluation import evaluate_policy_match_paired_seats  # noqa: E402
from alphapoker.holdem_evaluation import policies_for_model_player  # noqa: E402


def action_total(counts: dict[str, int]) -> int:
    return sum(counts.values())


def passive_policy(state):
    legal = state.legal_actions()
    return CHECK if CHECK in legal else CALL


def folding_policy(state):
    legal = state.legal_actions()
    return FOLD if FOLD in legal else CHECK


def betting_policy(state):
    legal = state.legal_actions()
    return BET if BET in legal else CALL


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


def test_evaluate_policy_match_reports_action_counts_by_role() -> None:
    metrics = evaluate_policy_match(
        model_policy=passive_policy,
        opponent_policy=folding_policy,
        hands=5,
        seed=3,
        model_player=0,
    )

    total_actions = int(metrics["avg_actions"] * metrics["hands"])
    assert action_total(metrics["model_action_counts"]) + action_total(
        metrics["opponent_action_counts"]
    ) == total_actions
    assert action_total(metrics["p0_action_counts"]) + action_total(
        metrics["p1_action_counts"]
    ) == total_actions
    assert metrics["model_action_counts"][FOLD] == 0
    assert metrics["model_action_counts"] == metrics["p0_action_counts"]


def test_evaluate_policy_match_reports_facing_bet_action_counts() -> None:
    metrics = evaluate_policy_match(
        model_policy=betting_policy,
        opponent_policy=folding_policy,
        hands=5,
        seed=3,
        model_player=0,
    )

    assert metrics["model_facing_bet_action_counts"][CALL] > 0
    assert metrics["opponent_facing_bet_action_counts"][FOLD] > 0
    assert metrics["p0_facing_bet_action_counts"][CALL] > 0
    assert metrics["p1_facing_bet_action_counts"][FOLD] > 0


def test_evaluate_policy_match_paired_seats_reports_pair_stats() -> None:
    metrics = evaluate_policy_match_paired_seats(
        model_policies=(passive_policy, passive_policy),
        opponent_policies=(folding_policy, folding_policy),
        hands=5,
        seed=3,
    )

    assert metrics["model_player"] == "both"
    assert metrics["hands"] == 10
    assert metrics["hands_per_model_player"] == 5
    assert metrics["paired_deals"] == 5
    assert len(metrics["seat_metrics"]) == 2
    assert action_total(metrics["model_action_counts"]) == sum(
        action_total(seat["model_action_counts"]) for seat in metrics["seat_metrics"]
    )
    assert action_total(metrics["opponent_action_counts"]) == sum(
        action_total(seat["opponent_action_counts"]) for seat in metrics["seat_metrics"]
    )
    assert action_total(metrics["model_facing_bet_action_counts"]) == sum(
        action_total(seat["model_facing_bet_action_counts"])
        for seat in metrics["seat_metrics"]
    )
    assert action_total(metrics["opponent_facing_bet_action_counts"]) == sum(
        action_total(seat["opponent_facing_bet_action_counts"])
        for seat in metrics["seat_metrics"]
    )


def test_aggregate_policy_match_shards_uses_paired_deals_for_stderr() -> None:
    shard = {
        "hands": 20,
        "hands_per_model_player": 10,
        "paired_deals": 10,
        "model_player": "both",
        "avg_utility_model": 1.0,
        "utility_stdev_model": 2.0,
        "avg_utility_p0": 1.0,
        "utility_stdev_p0": 2.0,
        "avg_actions": 6.0,
        "model_action_counts": {CHECK: 40, CALL: 20},
        "opponent_action_counts": {CHECK: 30, FOLD: 30},
        "p0_action_counts": {CHECK: 35, CALL: 10, FOLD: 15},
        "p1_action_counts": {CHECK: 35, CALL: 10, FOLD: 15},
        "folds": 8,
        "showdowns": 12,
        "seed": 3,
    }

    metrics = aggregate_policy_match_shards([shard, {**shard, "seed": 4}])

    assert metrics["hands"] == 40
    assert metrics["paired_deals"] == 20
    assert metrics["utility_stderr_model"] == pytest.approx(
        metrics["utility_stdev_model"] / (metrics["paired_deals"] ** 0.5)
    )
    assert metrics["model_action_counts"][CHECK] == 80
    assert metrics["opponent_action_counts"][FOLD] == 60

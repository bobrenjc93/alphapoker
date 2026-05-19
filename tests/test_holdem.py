import pytest


pytest.importorskip("treys")

import random  # noqa: E402

from alphapoker.holdem import (  # noqa: E402
    FixedLimitHoldemState,
    cached_pot_odds_equity_policy,
    cached_pot_odds_rollout_policy,
    deal_fixed_limit_holdem,
    compare_holdem_hands,
    equity_threshold_policy,
    estimate_holdem_equity,
    evaluate_holdem_hand,
    exact_river_holdem_equity,
    exact_turn_holdem_equity,
    holdem_belief_state_matches_opponent_policy,
    hybrid_pot_odds_equity_policy,
    opponent_range_pot_odds_equity_policy,
    play_fixed_limit_holdem_hand,
    policy_filtered_holdem_equity,
    pot_odds_call_threshold,
    pot_odds_equity_policy,
    preflop_holdem_equity_heuristic,
    pot_odds_rollout_action_values,
    pot_odds_rollout_policy,
    policy_rollout_action_values,
    policy_rollout_policy,
    random_holdem_policy,
    river_exact_pot_odds_equity_policy,
    sample_holdem_belief_state,
    sample_holdem_belief_state_matching_opponent_policy,
    sampled_holdem_equity,
    turn_river_exact_holdem_equity,
    turn_river_exact_pot_odds_equity_policy,
)
from alphapoker.kuhn import BET, CALL, CHECK, FOLD  # noqa: E402
from alphapoker.leduc import RAISE  # noqa: E402


def test_holdem_evaluator_ranks_flush_over_pair() -> None:
    flush = evaluate_holdem_hand(("As", "Qs"), ("2s", "7s", "9s", "Kd", "3c"))
    pair = evaluate_holdem_hand(("Ah", "Qd"), ("2s", "7s", "9s", "Kd", "3c"))

    assert flush.score < pair.score
    assert flush.class_name == "Flush"


def test_holdem_compare_detects_tie_on_board_straight() -> None:
    assert compare_holdem_hands(
        ("As", "Ad"),
        ("Kc", "Kd"),
        ("2h", "3d", "4s", "5c", "6h"),
    ) == 0


def test_holdem_evaluator_validates_card_counts() -> None:
    with pytest.raises(ValueError, match="two private"):
        evaluate_holdem_hand(("As",), ("2h", "3d", "4s"))

    with pytest.raises(ValueError, match="three to five"):
        evaluate_holdem_hand(("As", "Ad"), ("2h", "3d"))


def test_fixed_limit_holdem_initial_blind_state() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    )

    assert state.current_player() == 0
    assert state.visible_board() == ()
    assert state.contributions == (1, 2)
    assert state.legal_actions() == (CALL, FOLD, RAISE)


def test_fixed_limit_holdem_call_advances_to_flop() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    ).apply(CALL)

    assert state.street == 1
    assert state.visible_board() == ("2h", "3d", "4s")
    assert state.current_player() == 1
    assert state.contributions == (2, 2)
    assert state.legal_actions() == (CHECK, BET)


def test_fixed_limit_holdem_bet_call_uses_street_bet_sizes() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    )
    state = state.apply(CALL)
    state = state.apply(BET).apply(CALL)
    assert state.street == 2
    assert state.contributions == (4, 4)

    state = state.apply(BET).apply(CALL)
    assert state.street == 3
    assert state.contributions == (8, 8)


def test_fixed_limit_holdem_preflop_fold_awards_blinds() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    ).apply(FOLD)

    assert state.is_terminal()
    assert state.winner() == 1
    assert state.utility(0) == -1.0
    assert state.utility(1) == 1.0


def test_fixed_limit_holdem_showdown_uses_holdem_evaluator() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Qs"), ("Ah", "Ad")),
        ("2s", "7s", "9s", "Kd", "3c"),
    )
    state = state.apply(CALL)
    state = state.apply(CHECK).apply(CHECK)
    state = state.apply(CHECK).apply(CHECK)
    state = state.apply(CHECK).apply(CHECK)

    assert state.is_terminal()
    assert state.winner() == 0
    assert state.utility(0) == 2.0
    assert state.utility(1) == -2.0


def test_deal_fixed_limit_holdem_is_unique_and_seeded() -> None:
    state0 = deal_fixed_limit_holdem(random.Random(7))
    state1 = deal_fixed_limit_holdem(random.Random(7))
    all_cards = [*state0.private_cards[0], *state0.private_cards[1], *state0.board_cards]

    assert state0 == state1
    assert len(all_cards) == len(set(all_cards))


def test_random_fixed_limit_holdem_play_reaches_terminal() -> None:
    rng = random.Random(11)
    state, actions = play_fixed_limit_holdem_hand(
        deal_fixed_limit_holdem(rng),
        (random_holdem_policy(rng), random_holdem_policy(rng)),
    )

    assert state.is_terminal()
    assert actions
    assert state.utility(0) + state.utility(1) == 0.0


def test_holdem_equity_estimator_orders_premium_and_weak_hands() -> None:
    rng = random.Random(13)
    premium = estimate_holdem_equity(("As", "Ad"), (), simulations=200, rng=rng)
    weak = estimate_holdem_equity(("7c", "2d"), (), simulations=200, rng=rng)

    assert premium > 0.75
    assert weak < 0.45


def test_sampled_holdem_equity_is_deterministic() -> None:
    first = sampled_holdem_equity(("As", "Ad"), (), simulations=32)
    second = sampled_holdem_equity(("Ad", "As"), (), simulations=32)

    assert first == second
    assert 0.0 <= first <= 1.0


def test_exact_river_holdem_equity_scores_unbeatable_hand() -> None:
    equity = exact_river_holdem_equity(
        ("As", "Ks"),
        ("Qs", "Js", "Ts", "2c", "3d"),
    )

    assert equity == 1.0


def test_exact_river_holdem_equity_validates_board_complete() -> None:
    with pytest.raises(ValueError, match="five board"):
        exact_river_holdem_equity(("As", "Ks"), ("Qs", "Js", "Ts", "2c"))


def test_exact_turn_holdem_equity_scores_unbeatable_hand() -> None:
    equity = exact_turn_holdem_equity(
        ("As", "Ks"),
        ("Qs", "Js", "Ts", "2c"),
    )

    assert equity == 1.0


def test_exact_turn_holdem_equity_validates_turn_board() -> None:
    with pytest.raises(ValueError, match="four board"):
        exact_turn_holdem_equity(("As", "Ks"), ("Qs", "Js", "Ts"))


def test_turn_river_exact_holdem_equity_uses_exact_turn_and_river() -> None:
    private_cards = ("As", "Ks")
    turn_board = ("Qs", "Js", "Ts", "2c")
    river_board = ("Qs", "Js", "Ts", "2c", "3d")

    assert turn_river_exact_holdem_equity(private_cards, turn_board, simulations=2) == 1.0
    assert turn_river_exact_holdem_equity(private_cards, river_board, simulations=2) == 1.0


def test_preflop_holdem_equity_heuristic_orders_hands() -> None:
    premium = preflop_holdem_equity_heuristic(("As", "Ad"))
    suited_broadway = preflop_holdem_equity_heuristic(("As", "Ks"))
    weak = preflop_holdem_equity_heuristic(("7c", "2d"))

    assert premium > suited_broadway > weak
    assert 0.0 <= weak <= 1.0


def test_equity_threshold_policy_selects_legal_actions() -> None:
    rng = random.Random(17)
    state = deal_fixed_limit_holdem(rng)
    policy = equity_threshold_policy(rng, simulations=16)
    action = policy(state)

    assert action in state.legal_actions()


def test_pot_odds_call_threshold_uses_current_pot() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    )

    assert pot_odds_call_threshold(state) == pytest.approx(0.25)


def test_pot_odds_equity_policy_selects_legal_actions() -> None:
    rng = random.Random(19)
    state = deal_fixed_limit_holdem(rng)
    policy = pot_odds_equity_policy(rng, simulations=16)
    action = policy(state)

    assert action in state.legal_actions()


def test_cached_pot_odds_equity_policy_selects_legal_actions() -> None:
    rng = random.Random(20)
    state = deal_fixed_limit_holdem(rng)
    policy = cached_pot_odds_equity_policy(simulations=16)
    action = policy(state)

    assert action in state.legal_actions()


def test_river_exact_pot_odds_equity_policy_selects_legal_actions() -> None:
    rng = random.Random(21)
    state = deal_fixed_limit_holdem(rng)
    policy = river_exact_pot_odds_equity_policy(simulations=16)
    action = policy(state)

    assert action in state.legal_actions()


def test_turn_river_exact_pot_odds_equity_policy_selects_legal_turn_action() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ks"), ("Ah", "Ad")),
        ("Qs", "Js", "Ts", "2c", "3d"),
    )
    state = state.apply(CALL)
    state = state.apply(CHECK).apply(CHECK)
    policy = turn_river_exact_pot_odds_equity_policy(simulations=4)
    action = policy(state)

    assert state.visible_board() == ("Qs", "Js", "Ts", "2c")
    assert action in state.legal_actions()


def test_hybrid_pot_odds_equity_policy_selects_legal_actions() -> None:
    rng = random.Random(31)
    state = deal_fixed_limit_holdem(rng)
    policy = hybrid_pot_odds_equity_policy(rng, simulations=4)
    action = policy(state)

    assert action in state.legal_actions()


def test_holdem_belief_state_preserves_public_information() -> None:
    rng = random.Random(23)
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    ).apply(CALL)

    sampled = sample_holdem_belief_state(state, 0, rng)
    all_cards = [*sampled.private_cards[0], *sampled.private_cards[1], *sampled.board_cards]

    assert sampled.private_cards[0] == ("As", "Ad")
    assert sampled.visible_board() == ("2h", "3d", "4s")
    assert len(all_cards) == len(set(all_cards))


def test_holdem_belief_state_matches_observed_opponent_policy() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    )
    state = state.apply(CALL).apply(CHECK)

    def checking_policy(state: FixedLimitHoldemState) -> str:
        legal = state.legal_actions()
        return CHECK if CHECK in legal else CALL

    def betting_policy(state: FixedLimitHoldemState) -> str:
        legal = state.legal_actions()
        return BET if BET in legal else RAISE

    assert holdem_belief_state_matches_opponent_policy(state, 0, checking_policy)
    assert not holdem_belief_state_matches_opponent_policy(state, 0, betting_policy)


def test_holdem_belief_state_policy_filter_rejects_inconsistent_samples() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    )
    state = state.apply(CALL).apply(CHECK)

    def checking_policy(state: FixedLimitHoldemState) -> str:
        legal = state.legal_actions()
        return CHECK if CHECK in legal else CALL

    def betting_policy(state: FixedLimitHoldemState) -> str:
        legal = state.legal_actions()
        return BET if BET in legal else RAISE

    accepted = sample_holdem_belief_state_matching_opponent_policy(
        state,
        0,
        random.Random(24),
        opponent_policy_factory=lambda _: checking_policy,
        max_attempts=1,
    )
    rejected = sample_holdem_belief_state_matching_opponent_policy(
        state,
        0,
        random.Random(24),
        opponent_policy_factory=lambda _: betting_policy,
        max_attempts=3,
    )

    assert accepted is not None
    assert rejected is None


def test_holdem_belief_state_policy_filter_can_cache_matches() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    )
    state = state.apply(CALL).apply(CHECK)
    policy_calls = 0

    def checking_policy(state: FixedLimitHoldemState) -> str:
        legal = state.legal_actions()
        return CHECK if CHECK in legal else CALL

    def policy_factory(_: random.Random):
        nonlocal policy_calls
        policy_calls += 1
        return checking_policy

    match_cache: dict[tuple[str, str], bool] = {}
    first = sample_holdem_belief_state_matching_opponent_policy(
        state,
        0,
        random.Random(24),
        opponent_policy_factory=policy_factory,
        max_attempts=1,
        match_cache=match_cache,
    )
    second = sample_holdem_belief_state_matching_opponent_policy(
        state,
        0,
        random.Random(24),
        opponent_policy_factory=policy_factory,
        max_attempts=1,
        match_cache=match_cache,
    )

    assert first is not None
    assert second is not None
    assert policy_calls == 1


def test_policy_filtered_holdem_equity_cache_preserves_deterministic_result() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    )
    state = state.apply(CALL).apply(CHECK)

    def checking_policy(state: FixedLimitHoldemState) -> str:
        legal = state.legal_actions()
        return CHECK if CHECK in legal else CALL

    no_cache = policy_filtered_holdem_equity(
        state,
        0,
        random.Random(26),
        simulations=8,
        opponent_policy_factory=lambda _: checking_policy,
        max_attempts_per_sample=4,
    )
    cached = policy_filtered_holdem_equity(
        state,
        0,
        random.Random(26),
        simulations=8,
        opponent_policy_factory=lambda _: checking_policy,
        max_attempts_per_sample=4,
        cache_policy_matches=True,
    )

    assert cached == no_cache


def test_opponent_range_pot_odds_equity_policy_selects_legal_actions() -> None:
    rng = random.Random(25)
    state = deal_fixed_limit_holdem(rng).apply(CALL)

    def baseline(_: random.Random):
        return turn_river_exact_pot_odds_equity_policy(simulations=2)

    policy = opponent_range_pot_odds_equity_policy(
        rng,
        simulations=2,
        opponent_policy_factory=baseline,
        max_attempts_per_sample=4,
    )
    action = policy(state)

    assert action in state.legal_actions()


def test_pot_odds_rollout_policy_selects_legal_actions() -> None:
    rng = random.Random(29)
    state = deal_fixed_limit_holdem(rng)
    policy = pot_odds_rollout_policy(rng, simulations=2, equity_sims=2)
    action_values = pot_odds_rollout_action_values(state, rng, simulations=2, equity_sims=2)
    action = policy(state)

    assert set(action_values) == set(state.legal_actions())
    assert action in state.legal_actions()


def test_cached_pot_odds_rollout_policy_selects_legal_actions() -> None:
    rng = random.Random(30)
    state = deal_fixed_limit_holdem(rng)
    policy = cached_pot_odds_rollout_policy(rng, simulations=2, equity_sims=2)
    action_values = pot_odds_rollout_action_values(
        state,
        rng,
        simulations=2,
        equity_sims=2,
        cached_equity=True,
    )
    action = policy(state)

    assert set(action_values) == set(state.legal_actions())
    assert action in state.legal_actions()


def test_tuned_pot_odds_rollout_policy_selects_legal_actions() -> None:
    rng = random.Random(31)
    state = deal_fixed_limit_holdem(rng)
    policy = cached_pot_odds_rollout_policy(
        rng,
        simulations=2,
        equity_sims=2,
        bet_threshold=0.54,
        raise_threshold=0.76,
        call_margin=0.05,
    )
    action_values = pot_odds_rollout_action_values(
        state,
        rng,
        simulations=2,
        equity_sims=2,
        cached_equity=True,
        bet_threshold=0.54,
        raise_threshold=0.76,
        call_margin=0.05,
    )
    action = policy(state)

    assert set(action_values) == set(state.legal_actions())
    assert action in state.legal_actions()


def test_policy_rollout_policy_selects_legal_actions() -> None:
    rng = random.Random(32)
    state = deal_fixed_limit_holdem(rng)

    def baseline(_: random.Random):
        return turn_river_exact_pot_odds_equity_policy(simulations=2)

    policy = policy_rollout_policy(
        rng,
        simulations=2,
        continuation_policy_factory=baseline,
        opponent_policy_factory=baseline,
        default_policy_factory=baseline,
        improvement_margin=1.0,
    )
    action_values = policy_rollout_action_values(
        state,
        rng,
        simulations=2,
        continuation_policy_factory=baseline,
        opponent_policy_factory=baseline,
    )
    action = policy(state)

    assert set(action_values) == set(state.legal_actions())
    assert action in state.legal_actions()

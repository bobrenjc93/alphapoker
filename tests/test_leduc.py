import pytest

from alphapoker.kuhn import BET, CALL, CHECK, FOLD
from alphapoker.leduc import RAISE, LeducState, all_leduc_deals


def test_all_deals_have_identity_multiplicity_and_valid_rank_counts() -> None:
    deals = all_leduc_deals()
    assert len(deals) == 120
    for deal in deals:
        assert all(deal.count(rank) <= 2 for rank in (0, 1, 2))


def test_checks_advance_to_public_round() -> None:
    state = LeducState.initial((0, 1), public_card=2)
    state = state.apply(CHECK).apply(CHECK)

    assert state.round_index == 1
    assert state.visible_public_card() == 2
    assert state.current_player() == 0
    assert state.contributions == (1, 1)
    assert state.legal_actions() == (CHECK, BET)


def test_bet_call_then_showdown_pair_beats_high_card() -> None:
    state = LeducState.initial((0, 2), public_card=0)
    state = state.apply(BET).apply(CALL)
    assert state.round_index == 1
    assert state.contributions == (3, 3)

    state = state.apply(CHECK).apply(CHECK)
    assert state.is_terminal()
    assert state.winner() == 0
    assert state.utility(0) == 3.0
    assert state.utility(1) == -3.0


def test_raise_cap_removes_raise_action() -> None:
    state = LeducState.initial((0, 1), public_card=2)
    state = state.apply(BET).apply(RAISE)

    assert state.bets_this_round == 2
    assert state.legal_actions() == (CALL, FOLD)


def test_fold_after_raise_awards_current_pot() -> None:
    state = LeducState.initial((0, 1), public_card=2)
    state = state.apply(BET).apply(RAISE).apply(FOLD)

    assert state.is_terminal()
    assert state.winner() == 1
    assert state.contributions == (3, 5)
    assert state.utility(0) == -3.0
    assert state.utility(1) == 3.0


def test_equal_showdown_strength_splits_pot() -> None:
    state = LeducState.initial((0, 0), public_card=1)
    state = state.apply(CHECK).apply(CHECK)
    state = state.apply(CHECK).apply(CHECK)

    assert state.winner() is None
    assert state.utility(0) == 0.0
    assert state.utility(1) == 0.0


def test_invalid_rank_multiplicity_is_rejected() -> None:
    with pytest.raises(ValueError):
        LeducState.initial((2, 2), public_card=2)

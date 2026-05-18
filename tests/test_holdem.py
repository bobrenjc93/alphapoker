import pytest


pytest.importorskip("treys")

from alphapoker.holdem import compare_holdem_hands, evaluate_holdem_hand  # noqa: E402
from alphapoker.holdem import FixedLimitHoldemState  # noqa: E402
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

import pytest


pytest.importorskip("treys")

from alphapoker.holdem import compare_holdem_hands, evaluate_holdem_hand  # noqa: E402


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

"""Texas Hold'em hand evaluation utilities.

The ranking implementation delegates to `treys`, a proven poker hand evaluator.
This module keeps AlphaPoker's public interface small and testable while avoiding
custom hand-ranking logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from treys import Card, Evaluator


@dataclass(frozen=True)
class HoldemHandResult:
    score: int
    rank_class: int
    class_name: str


def parse_cards(cards: list[str] | tuple[str, ...]) -> list[int]:
    return [Card.new(card) for card in cards]


def evaluate_holdem_hand(
    private_cards: list[str] | tuple[str, str],
    board_cards: list[str] | tuple[str, ...],
) -> HoldemHandResult:
    """Evaluate a Hold'em hand.

    Args:
        private_cards: Two card strings such as ("As", "Kd").
        board_cards: Three to five public cards.

    Returns:
        Treys-compatible score metadata. Lower scores are stronger.
    """

    if len(private_cards) != 2:
        raise ValueError("Hold'em evaluation requires exactly two private cards")
    if not 3 <= len(board_cards) <= 5:
        raise ValueError("Hold'em evaluation requires three to five board cards")

    evaluator = Evaluator()
    score = evaluator.evaluate(parse_cards(list(board_cards)), parse_cards(list(private_cards)))
    rank_class = evaluator.get_rank_class(score)
    return HoldemHandResult(
        score=score,
        rank_class=rank_class,
        class_name=evaluator.class_to_string(rank_class),
    )


def compare_holdem_hands(
    player0_private: list[str] | tuple[str, str],
    player1_private: list[str] | tuple[str, str],
    board_cards: list[str] | tuple[str, ...],
) -> int:
    """Compare two Hold'em hands.

    Returns:
        1 if player 0 wins, -1 if player 1 wins, 0 for a tie.
    """

    score0 = evaluate_holdem_hand(player0_private, board_cards).score
    score1 = evaluate_holdem_hand(player1_private, board_cards).score
    if score0 < score1:
        return 1
    if score1 < score0:
        return -1
    return 0


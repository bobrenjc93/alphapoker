"""Kuhn poker rules.

Kuhn poker is the smallest useful imperfect-information poker benchmark:
three cards, two players, one private card each, one betting round, and one
optional unit bet. It is small enough for exact tests while still requiring
reasoning over hidden information and mixed strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

CHECK = "check"
BET = "bet"
CALL = "call"
FOLD = "fold"

CANONICAL_ACTIONS = (CHECK, BET, CALL, FOLD)
CARDS = (0, 1, 2)
CARD_NAMES = ("J", "Q", "K")

History = tuple[str, ...]

TERMINAL_HISTORIES: set[History] = {
    (CHECK, CHECK),
    (BET, CALL),
    (BET, FOLD),
    (CHECK, BET, CALL),
    (CHECK, BET, FOLD),
}


def card_name(card: int) -> str:
    return CARD_NAMES[card]


def parse_card(name: str) -> int:
    return CARD_NAMES.index(name)


def history_label(history: History) -> str:
    return "root" if not history else "-".join(history)


def parse_history(label: str) -> History:
    return () if label == "root" else tuple(label.split("-"))


def infoset_key(player: int, card: int, history: History) -> str:
    return f"P{player}:{card_name(card)}:{history_label(history)}"


def parse_infoset_key(key: str) -> tuple[int, int, History]:
    player_text, card_text, history_text = key.split(":", maxsplit=2)
    return int(player_text[1:]), parse_card(card_text), parse_history(history_text)


def all_card_deals() -> tuple[tuple[int, int], ...]:
    """Return all ordered private-card deals."""

    return tuple(permutations(CARDS, 2))


def legal_actions_for_history(history: History) -> tuple[str, ...]:
    if history == ():
        return (CHECK, BET)
    if history == (CHECK,):
        return (CHECK, BET)
    if history == (BET,):
        return (CALL, FOLD)
    if history == (CHECK, BET):
        return (CALL, FOLD)
    return ()


def player_to_act_for_history(history: History) -> int:
    if history == () or history == (CHECK, BET):
        return 0
    if history in {(CHECK,), (BET,)}:
        return 1
    raise ValueError(f"Terminal or invalid history has no player to act: {history}")


def player_histories(player: int) -> tuple[History, ...]:
    if player == 0:
        return ((), (CHECK, BET))
    if player == 1:
        return ((CHECK,), (BET,))
    raise ValueError(f"Unknown player: {player}")


def all_infoset_keys(player: int | None = None) -> tuple[str, ...]:
    players = (0, 1) if player is None else (player,)
    keys: list[str] = []
    for p in players:
        for card in CARDS:
            for history in player_histories(p):
                keys.append(infoset_key(p, card, history))
    return tuple(keys)


@dataclass(frozen=True)
class KuhnState:
    """Immutable Kuhn poker state."""

    cards: tuple[int, int]
    history: History = ()
    ante: int = 1
    bet_size: int = 1

    @classmethod
    def initial(cls, cards: tuple[int, int]) -> "KuhnState":
        if len(cards) != 2:
            raise ValueError("Kuhn poker needs exactly two private cards")
        if cards[0] == cards[1]:
            raise ValueError("Private cards must be distinct")
        if any(card not in CARDS for card in cards):
            raise ValueError(f"Cards must be in {CARDS}")
        return cls(cards=cards)

    def is_terminal(self) -> bool:
        return self.history in TERMINAL_HISTORIES

    def current_player(self) -> int:
        return player_to_act_for_history(self.history)

    def legal_actions(self) -> tuple[str, ...]:
        return legal_actions_for_history(self.history)

    def apply(self, action: str) -> "KuhnState":
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action!r} for history {self.history!r}")
        return KuhnState(
            cards=self.cards,
            history=(*self.history, action),
            ante=self.ante,
            bet_size=self.bet_size,
        )

    def infoset_key(self) -> str:
        player = self.current_player()
        return infoset_key(player, self.cards[player], self.history)

    def contributions(self) -> tuple[int, int]:
        contrib = [self.ante, self.ante]
        prefix: History = ()
        for action in self.history:
            actor = player_to_act_for_history(prefix)
            if action in {BET, CALL}:
                contrib[actor] += self.bet_size
            prefix = (*prefix, action)
        return contrib[0], contrib[1]

    def winner(self) -> int:
        if not self.is_terminal():
            raise ValueError("Non-terminal states do not have a winner")
        if self.history == (BET, FOLD):
            return 0
        if self.history == (CHECK, BET, FOLD):
            return 1
        return 0 if self.cards[0] > self.cards[1] else 1

    def utility(self, player: int) -> float:
        if player not in (0, 1):
            raise ValueError(f"Unknown player: {player}")
        if not self.is_terminal():
            raise ValueError("Utility is only defined for terminal states")

        contrib = self.contributions()
        pot = contrib[0] + contrib[1]
        if self.winner() == player:
            return float(pot - contrib[player])
        return float(-contrib[player])


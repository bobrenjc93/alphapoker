"""Two-player limit Leduc poker rules.

Leduc is the standard next benchmark after Kuhn poker. The deck has two copies
of each rank J/Q/K. Each player antes, receives one private card, plays a limit
betting round, sees one public card, then plays a second betting round.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from alphapoker.kuhn import BET, CALL, CHECK, FOLD

RAISE = "raise"

RANKS = (0, 1, 2)
RANK_NAMES = ("J", "Q", "K")
LEDUC_DECK = tuple((rank, copy) for rank in RANKS for copy in range(2))

History = tuple[str, ...]
RoundHistories = tuple[History, History]


def rank_name(rank: int) -> str:
    return RANK_NAMES[rank]


def all_leduc_deals() -> tuple[tuple[int, int, int], ...]:
    """Return equally likely ordered private/public rank deals.

    Duplicate rank triples are intentional: they represent distinct underlying
    card identities from the six-card Leduc deck.
    """

    deals: list[tuple[int, int, int]] = []
    for private0, private1, public in permutations(LEDUC_DECK, 3):
        deals.append((private0[0], private1[0], public[0]))
    return tuple(deals)


def _validate_rank_counts(private_cards: tuple[int, int], public_card: int) -> None:
    for rank in (*private_cards, public_card):
        if rank not in RANKS:
            raise ValueError(f"Rank must be one of {RANKS}: {rank}")
    for rank in RANKS:
        if (*private_cards, public_card).count(rank) > 2:
            raise ValueError("A Leduc deck only has two copies of each rank")


def _history_label(histories: RoundHistories) -> str:
    labels = []
    for index, history in enumerate(histories):
        labels.append(f"r{index}:{'root' if not history else '-'.join(history)}")
    return "|".join(labels)


@dataclass(frozen=True)
class LeducState:
    """Immutable limit Leduc poker state with chance fixed at construction."""

    private_cards: tuple[int, int]
    public_card: int
    round_index: int = 0
    histories: RoundHistories = ((), ())
    contributions: tuple[int, int] = (1, 1)
    round_contributions: tuple[int, int] = (0, 0)
    to_act: int = 0
    bets_this_round: int = 0
    folded_player: int | None = None
    showdown: bool = False
    ante: int = 1
    bet_sizes: tuple[int, int] = (2, 4)
    max_bets_per_round: int = 2

    @classmethod
    def initial(
        cls,
        private_cards: tuple[int, int],
        public_card: int,
        *,
        ante: int = 1,
        bet_sizes: tuple[int, int] = (2, 4),
        max_bets_per_round: int = 2,
    ) -> "LeducState":
        _validate_rank_counts(private_cards, public_card)
        return cls(
            private_cards=private_cards,
            public_card=public_card,
            contributions=(ante, ante),
            ante=ante,
            bet_sizes=bet_sizes,
            max_bets_per_round=max_bets_per_round,
        )

    def is_terminal(self) -> bool:
        return self.folded_player is not None or self.showdown

    def current_player(self) -> int:
        if self.is_terminal():
            raise ValueError("Terminal states have no current player")
        return self.to_act

    def visible_public_card(self) -> int | None:
        return self.public_card if self.round_index == 1 else None

    def current_round_history(self) -> History:
        return self.histories[self.round_index]

    def outstanding_call_amount(self) -> int:
        other = 1 - self.to_act
        return max(0, self.round_contributions[other] - self.round_contributions[self.to_act])

    def legal_actions(self) -> tuple[str, ...]:
        if self.is_terminal():
            return ()
        if self.outstanding_call_amount() > 0:
            actions = [CALL, FOLD]
            if self.bets_this_round < self.max_bets_per_round:
                actions.append(RAISE)
            return tuple(actions)

        actions = [CHECK]
        if self.bets_this_round < self.max_bets_per_round:
            actions.append(BET)
        return tuple(actions)

    def apply(self, action: str) -> "LeducState":
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action!r} for state {self}")

        histories = self._append_history(action)
        contributions = list(self.contributions)
        round_contributions = list(self.round_contributions)
        bets_this_round = self.bets_this_round

        if action == CHECK:
            if self.current_round_history() == (CHECK,):
                return self._finish_betting_round(histories, tuple(contributions))
            return self._next_player(
                histories=histories,
                contributions=tuple(contributions),
                round_contributions=tuple(round_contributions),
                bets_this_round=bets_this_round,
            )

        if action == BET:
            amount = self.bet_sizes[self.round_index]
            contributions[self.to_act] += amount
            round_contributions[self.to_act] += amount
            bets_this_round += 1
            return self._next_player(
                histories=histories,
                contributions=tuple(contributions),
                round_contributions=tuple(round_contributions),
                bets_this_round=bets_this_round,
            )

        if action == RAISE:
            amount = self.outstanding_call_amount() + self.bet_sizes[self.round_index]
            contributions[self.to_act] += amount
            round_contributions[self.to_act] += amount
            bets_this_round += 1
            return self._next_player(
                histories=histories,
                contributions=tuple(contributions),
                round_contributions=tuple(round_contributions),
                bets_this_round=bets_this_round,
            )

        if action == CALL:
            amount = self.outstanding_call_amount()
            contributions[self.to_act] += amount
            round_contributions[self.to_act] += amount
            return self._finish_betting_round(histories, tuple(contributions))

        if action == FOLD:
            return self._replace(
                histories=histories,
                contributions=tuple(contributions),
                folded_player=self.to_act,
            )

        raise AssertionError(f"Unhandled action: {action}")

    def information_key(self) -> str:
        player = self.current_player()
        public = "-" if self.visible_public_card() is None else rank_name(self.public_card)
        return (
            f"P{player}:private={rank_name(self.private_cards[player])}:"
            f"public={public}:{_history_label(self.histories)}"
        )

    def winner(self) -> int | None:
        if not self.is_terminal():
            raise ValueError("Non-terminal states do not have a winner")
        if self.folded_player is not None:
            return 1 - self.folded_player

        strength0 = self._hand_strength(0)
        strength1 = self._hand_strength(1)
        if strength0 == strength1:
            return None
        return 0 if strength0 > strength1 else 1

    def utility(self, player: int) -> float:
        if player not in (0, 1):
            raise ValueError(f"Unknown player: {player}")
        if not self.is_terminal():
            raise ValueError("Utility is only defined for terminal states")

        pot = sum(self.contributions)
        winner = self.winner()
        if winner is None:
            return pot / 2.0 - self.contributions[player]
        if winner == player:
            return float(pot - self.contributions[player])
        return float(-self.contributions[player])

    def _hand_strength(self, player: int) -> tuple[int, int]:
        private = self.private_cards[player]
        has_pair = int(private == self.public_card)
        return has_pair, private

    def _append_history(self, action: str) -> RoundHistories:
        histories = [list(self.histories[0]), list(self.histories[1])]
        histories[self.round_index].append(action)
        return tuple(tuple(history) for history in histories)  # type: ignore[return-value]

    def _finish_betting_round(
        self,
        histories: RoundHistories,
        contributions: tuple[int, int],
    ) -> "LeducState":
        if self.round_index == 0:
            return self._replace(
                round_index=1,
                histories=histories,
                contributions=contributions,
                round_contributions=(0, 0),
                to_act=0,
                bets_this_round=0,
            )
        return self._replace(
            histories=histories,
            contributions=contributions,
            showdown=True,
        )

    def _next_player(
        self,
        *,
        histories: RoundHistories,
        contributions: tuple[int, int],
        round_contributions: tuple[int, int],
        bets_this_round: int,
    ) -> "LeducState":
        return self._replace(
            histories=histories,
            contributions=contributions,
            round_contributions=round_contributions,
            bets_this_round=bets_this_round,
            to_act=1 - self.to_act,
        )

    def _replace(self, **kwargs: object) -> "LeducState":
        values = {
            "private_cards": self.private_cards,
            "public_card": self.public_card,
            "round_index": self.round_index,
            "histories": self.histories,
            "contributions": self.contributions,
            "round_contributions": self.round_contributions,
            "to_act": self.to_act,
            "bets_this_round": self.bets_this_round,
            "folded_player": self.folded_player,
            "showdown": self.showdown,
            "ante": self.ante,
            "bet_sizes": self.bet_sizes,
            "max_bets_per_round": self.max_bets_per_round,
        }
        values.update(kwargs)
        return LeducState(**values)  # type: ignore[arg-type]


"""Texas Hold'em hand evaluation utilities.

The ranking implementation delegates to `treys`, a proven poker hand evaluator.
This module keeps AlphaPoker's public interface small and testable while avoiding
custom hand-ranking logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from alphapoker.kuhn import BET, CALL, CHECK, FOLD
from alphapoker.leduc import RAISE
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


History = tuple[str, ...]
StreetHistories = tuple[History, History, History, History]


@dataclass(frozen=True)
class FixedLimitHoldemState:
    """Heads-up fixed-limit Texas Hold'em state with chance fixed up front."""

    private_cards: tuple[tuple[str, str], tuple[str, str]]
    board_cards: tuple[str, str, str, str, str]
    street: int = 0
    histories: StreetHistories = ((), (), (), ())
    contributions: tuple[int, int] = (1, 2)
    round_contributions: tuple[int, int] = (1, 2)
    to_act: int = 0
    bets_this_round: int = 1
    folded_player: int | None = None
    showdown: bool = False
    small_blind: int = 1
    big_blind: int = 2
    small_bet: int = 2
    big_bet: int = 4
    max_bets_per_round: int = 4

    @classmethod
    def initial(
        cls,
        private_cards: tuple[tuple[str, str], tuple[str, str]],
        board_cards: tuple[str, str, str, str, str],
        *,
        small_blind: int = 1,
        big_blind: int = 2,
        small_bet: int = 2,
        big_bet: int = 4,
        max_bets_per_round: int = 4,
    ) -> "FixedLimitHoldemState":
        cards = [*private_cards[0], *private_cards[1], *board_cards]
        if len(private_cards[0]) != 2 or len(private_cards[1]) != 2:
            raise ValueError("Each Hold'em player needs exactly two private cards")
        if len(board_cards) != 5:
            raise ValueError("Hold'em state needs exactly five predetermined board cards")
        if len(set(cards)) != len(cards):
            raise ValueError("Cards must be unique")

        return cls(
            private_cards=private_cards,
            board_cards=board_cards,
            contributions=(small_blind, big_blind),
            round_contributions=(small_blind, big_blind),
            small_blind=small_blind,
            big_blind=big_blind,
            small_bet=small_bet,
            big_bet=big_bet,
            max_bets_per_round=max_bets_per_round,
        )

    def is_terminal(self) -> bool:
        return self.folded_player is not None or self.showdown

    def current_player(self) -> int:
        if self.is_terminal():
            raise ValueError("Terminal states have no current player")
        return self.to_act

    def visible_board(self) -> tuple[str, ...]:
        board_count = (0, 3, 4, 5)[self.street]
        return self.board_cards[:board_count]

    def current_round_history(self) -> History:
        return self.histories[self.street]

    def outstanding_call_amount(self) -> int:
        other = 1 - self.to_act
        return max(0, self.round_contributions[other] - self.round_contributions[self.to_act])

    def current_bet_size(self) -> int:
        return self.small_bet if self.street < 2 else self.big_bet

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

    def apply(self, action: str) -> "FixedLimitHoldemState":
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action!r} for state {self}")

        histories = self._append_history(action)
        contributions = list(self.contributions)
        round_contributions = list(self.round_contributions)
        bets_this_round = self.bets_this_round

        if action == FOLD:
            return self._replace(
                histories=histories,
                contributions=tuple(contributions),
                folded_player=self.to_act,
            )

        if action == CALL:
            amount = self.outstanding_call_amount()
            contributions[self.to_act] += amount
            round_contributions[self.to_act] += amount
            return self._finish_betting_round(histories, tuple(contributions))

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
            amount = self.current_bet_size()
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
            amount = self.outstanding_call_amount() + self.current_bet_size()
            contributions[self.to_act] += amount
            round_contributions[self.to_act] += amount
            bets_this_round += 1
            return self._next_player(
                histories=histories,
                contributions=tuple(contributions),
                round_contributions=tuple(round_contributions),
                bets_this_round=bets_this_round,
            )

        raise AssertionError(f"Unhandled action: {action}")

    def winner(self) -> int | None:
        if not self.is_terminal():
            raise ValueError("Non-terminal states do not have a winner")
        if self.folded_player is not None:
            return 1 - self.folded_player

        comparison = compare_holdem_hands(
            self.private_cards[0],
            self.private_cards[1],
            self.board_cards,
        )
        if comparison == 0:
            return None
        return 0 if comparison == 1 else 1

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

    def _append_history(self, action: str) -> StreetHistories:
        histories = [list(history) for history in self.histories]
        histories[self.street].append(action)
        return tuple(tuple(history) for history in histories)  # type: ignore[return-value]

    def _finish_betting_round(
        self,
        histories: StreetHistories,
        contributions: tuple[int, int],
    ) -> "FixedLimitHoldemState":
        if self.street == 3:
            return self._replace(
                histories=histories,
                contributions=contributions,
                showdown=True,
            )

        return self._replace(
            street=self.street + 1,
            histories=histories,
            contributions=contributions,
            round_contributions=(0, 0),
            to_act=1,
            bets_this_round=0,
        )

    def _next_player(
        self,
        *,
        histories: StreetHistories,
        contributions: tuple[int, int],
        round_contributions: tuple[int, int],
        bets_this_round: int,
    ) -> "FixedLimitHoldemState":
        return self._replace(
            histories=histories,
            contributions=contributions,
            round_contributions=round_contributions,
            bets_this_round=bets_this_round,
            to_act=1 - self.to_act,
        )

    def _replace(self, **kwargs: object) -> "FixedLimitHoldemState":
        values = {
            "private_cards": self.private_cards,
            "board_cards": self.board_cards,
            "street": self.street,
            "histories": self.histories,
            "contributions": self.contributions,
            "round_contributions": self.round_contributions,
            "to_act": self.to_act,
            "bets_this_round": self.bets_this_round,
            "folded_player": self.folded_player,
            "showdown": self.showdown,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "small_bet": self.small_bet,
            "big_bet": self.big_bet,
            "max_bets_per_round": self.max_bets_per_round,
        }
        values.update(kwargs)
        return FixedLimitHoldemState(**values)  # type: ignore[arg-type]


HOLDEM_RANKS = "23456789TJQKA"
HOLDEM_SUITS = "shdc"
STANDARD_HOLDEM_DECK = tuple(f"{rank}{suit}" for rank in HOLDEM_RANKS for suit in HOLDEM_SUITS)

HoldemPolicy = Callable[[FixedLimitHoldemState], str]


def deal_fixed_limit_holdem(rng: random.Random | None = None) -> FixedLimitHoldemState:
    rng = rng or random.Random()
    deck = list(STANDARD_HOLDEM_DECK)
    rng.shuffle(deck)
    return FixedLimitHoldemState.initial(
        ((deck[0], deck[1]), (deck[2], deck[3])),
        (deck[4], deck[5], deck[6], deck[7], deck[8]),
    )


def random_holdem_policy(rng: random.Random) -> HoldemPolicy:
    def select_action(state: FixedLimitHoldemState) -> str:
        return rng.choice(state.legal_actions())

    return select_action


def estimate_holdem_equity(
    private_cards: tuple[str, str],
    visible_board: tuple[str, ...],
    *,
    simulations: int = 256,
    rng: random.Random | None = None,
) -> float:
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if len(private_cards) != 2:
        raise ValueError("equity estimation requires exactly two private cards")
    if len(visible_board) > 5:
        raise ValueError("visible board cannot exceed five cards")

    rng = rng or random.Random()
    known = set(private_cards) | set(visible_board)
    if len(known) != len(private_cards) + len(visible_board):
        raise ValueError("Known cards must be unique")

    deck = [card for card in STANDARD_HOLDEM_DECK if card not in known]
    wins = 0.0
    for _ in range(simulations):
        sample = rng.sample(deck, 2 + (5 - len(visible_board)))
        opponent_private = (sample[0], sample[1])
        board = (*visible_board, *sample[2:])
        comparison = compare_holdem_hands(private_cards, opponent_private, board)
        if comparison == 1:
            wins += 1.0
        elif comparison == 0:
            wins += 0.5
    return wins / simulations


def pot_odds_call_threshold(state: FixedLimitHoldemState, *, margin: float = 0.0) -> float:
    call_amount = state.outstanding_call_amount()
    if call_amount <= 0:
        return 0.0
    pot_after_call = sum(state.contributions) + call_amount
    return min(1.0, max(0.0, call_amount / pot_after_call + margin))


def equity_threshold_policy(
    rng: random.Random,
    *,
    simulations: int = 128,
    bet_threshold: float = 0.58,
    raise_threshold: float = 0.72,
    call_threshold: float = 0.36,
) -> HoldemPolicy:
    def select_action(state: FixedLimitHoldemState) -> str:
        player = state.current_player()
        equity = estimate_holdem_equity(
            state.private_cards[player],
            state.visible_board(),
            simulations=simulations,
            rng=rng,
        )
        legal = state.legal_actions()
        if state.outstanding_call_amount() > 0:
            if RAISE in legal and equity >= raise_threshold:
                return RAISE
            if equity >= call_threshold:
                return CALL
            return FOLD

        if BET in legal and equity >= bet_threshold:
            return BET
        return CHECK

    return select_action


def pot_odds_equity_policy(
    rng: random.Random,
    *,
    simulations: int = 128,
    bet_threshold: float = 0.58,
    raise_threshold: float = 0.72,
    call_margin: float = 0.0,
) -> HoldemPolicy:
    def select_action(state: FixedLimitHoldemState) -> str:
        player = state.current_player()
        equity = estimate_holdem_equity(
            state.private_cards[player],
            state.visible_board(),
            simulations=simulations,
            rng=rng,
        )
        legal = state.legal_actions()
        if state.outstanding_call_amount() > 0:
            if RAISE in legal and equity >= raise_threshold:
                return RAISE
            if equity >= pot_odds_call_threshold(state, margin=call_margin):
                return CALL
            return FOLD

        if BET in legal and equity >= bet_threshold:
            return BET
        return CHECK

    return select_action


def sample_holdem_belief_state(
    state: FixedLimitHoldemState,
    player: int,
    rng: random.Random,
) -> FixedLimitHoldemState:
    if player not in (0, 1):
        raise ValueError(f"Unknown player: {player}")

    visible_board = state.visible_board()
    known_cards = set(state.private_cards[player]) | set(visible_board)
    deck = [card for card in STANDARD_HOLDEM_DECK if card not in known_cards]
    sampled_cards = rng.sample(deck, 2 + (5 - len(visible_board)))
    sampled_opponent_private = (sampled_cards[0], sampled_cards[1])
    sampled_board = (*visible_board, *sampled_cards[2:])
    sampled_private = [state.private_cards[0], state.private_cards[1]]
    sampled_private[1 - player] = sampled_opponent_private
    return state._replace(
        private_cards=(sampled_private[0], sampled_private[1]),
        board_cards=sampled_board,
    )


def pot_odds_rollout_action_values(
    state: FixedLimitHoldemState,
    rng: random.Random,
    *,
    simulations: int = 16,
    equity_sims: int = 16,
) -> dict[str, float]:
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    player = state.current_player()
    legal_actions = state.legal_actions()
    seed = rng.randrange(2**63)
    values: dict[str, float] = {}
    for action_index, action in enumerate(legal_actions):
        total_utility = 0.0
        for simulation_index in range(simulations):
            simulation_seed = seed + action_index * 1_000_003 + simulation_index
            simulation_rng = random.Random(simulation_seed)
            sampled_state = sample_holdem_belief_state(state, player, simulation_rng)
            rollout_state = sampled_state.apply(action)
            policy_rng = random.Random(simulation_seed + 50_000_003)
            continuation_policy = pot_odds_equity_policy(policy_rng, simulations=equity_sims)
            opponent_policy = pot_odds_equity_policy(policy_rng, simulations=equity_sims)
            policies = [opponent_policy, opponent_policy]
            policies[player] = continuation_policy
            terminal, _ = play_fixed_limit_holdem_hand(
                rollout_state,
                (policies[0], policies[1]),
            )
            total_utility += terminal.utility(player)
        values[action] = total_utility / simulations
    return values


def pot_odds_rollout_policy(
    rng: random.Random,
    *,
    simulations: int = 16,
    equity_sims: int = 16,
) -> HoldemPolicy:
    def select_action(state: FixedLimitHoldemState) -> str:
        action_values = pot_odds_rollout_action_values(
            state,
            rng,
            simulations=simulations,
            equity_sims=equity_sims,
        )
        return max(state.legal_actions(), key=lambda action: action_values[action])

    return select_action


def play_fixed_limit_holdem_hand(
    initial_state: FixedLimitHoldemState,
    policies: tuple[HoldemPolicy, HoldemPolicy],
    *,
    max_actions: int = 128,
) -> tuple[FixedLimitHoldemState, list[tuple[int, str]]]:
    state = initial_state
    actions: list[tuple[int, str]] = []
    while not state.is_terminal():
        if len(actions) >= max_actions:
            raise RuntimeError("Hold'em hand exceeded max_actions")
        player = state.current_player()
        action = policies[player](state)
        if action not in state.legal_actions():
            raise ValueError(f"Policy selected illegal action {action!r}")
        actions.append((player, action))
        state = state.apply(action)
    return state, actions

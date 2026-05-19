"""Sampled abstract CFR for fixed-limit heads-up Hold'em."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from hashlib import blake2b
from pathlib import Path
from typing import Any

from alphapoker.cfr import InfoSet
from alphapoker.eval import normalize_distribution
from alphapoker.holdem import (
    FixedLimitHoldemState,
    HoldemPolicy,
    deal_fixed_limit_holdem,
    evaluate_holdem_hand,
    estimate_holdem_equity,
    preflop_holdem_equity_heuristic,
)
from alphapoker.holdem_features import HOLDEM_RANKS
from alphapoker.kuhn import BET, CALL, CHECK, FOLD
from alphapoker.leduc import RAISE

HoldemAbstractStrategy = dict[str, dict[str, float]]
HOLDEM_ABSTRACTIONS = ("fine", "medium", "equity", "coarse")
_EQUITY_ABSTRACTION_SIMS = 8


@dataclass(frozen=True)
class HoldemMCCFRTrainingResult:
    iterations: int
    infosets: int
    sampled_game_value_p0: float


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _rank_index(card: str) -> int:
    return HOLDEM_RANKS.index(card[0])


def _preflop_bucket(private_cards: tuple[str, str]) -> str:
    ranks = sorted((_rank_index(card) for card in private_cards), reverse=True)
    suited = private_cards[0][1] == private_cards[1][1]
    pair = ranks[0] == ranks[1]
    gap = ranks[0] - ranks[1]
    if pair:
        return f"pair{ranks[0] // 2}"
    gap_bucket = min(4, gap)
    suited_label = "s" if suited else "o"
    return f"hi{ranks[0] // 2}:lo{ranks[1] // 3}:g{gap_bucket}:{suited_label}"


def _coarse_preflop_bucket(private_cards: tuple[str, str]) -> str:
    ranks = sorted((_rank_index(card) for card in private_cards), reverse=True)
    suited = private_cards[0][1] == private_cards[1][1]
    gap = ranks[0] - ranks[1]
    if ranks[0] == ranks[1]:
        return f"pp{ranks[0] // 4}"
    return f"h{ranks[0] // 4}:l{ranks[1] // 4}:c{int(gap <= 2)}:s{int(suited)}"


def _medium_preflop_bucket(private_cards: tuple[str, str]) -> str:
    ranks = sorted((_rank_index(card) for card in private_cards), reverse=True)
    suited = private_cards[0][1] == private_cards[1][1]
    gap = ranks[0] - ranks[1]
    equity_bucket = int(min(0.999, preflop_holdem_equity_heuristic(private_cards)) * 10)
    if ranks[0] == ranks[1]:
        return f"e{equity_bucket}:pp{ranks[0] // 3}"
    return f"e{equity_bucket}:h{ranks[0] // 3}:g{min(4, gap)}:s{int(suited)}"


def _equity_preflop_bucket(private_cards: tuple[str, str]) -> str:
    ranks = sorted((_rank_index(card) for card in private_cards), reverse=True)
    suited = private_cards[0][1] == private_cards[1][1]
    gap = ranks[0] - ranks[1]
    equity_bucket = int(min(0.999, preflop_holdem_equity_heuristic(private_cards)) * 12)
    pair = ranks[0] == ranks[1]
    return f"e{equity_bucket}:p{int(pair)}:h{ranks[0] // 2}:g{min(5, gap)}:s{int(suited)}"


def _board_texture_bucket(board: tuple[str, ...]) -> str:
    ranks = [_rank_index(card) for card in board]
    suits = [card[1] for card in board]
    paired = len(set(ranks)) < len(ranks)
    max_suit_count = max((suits.count(suit) for suit in set(suits)), default=0)
    high_rank = max(ranks) if ranks else 0
    return f"b{len(board)}:p{int(paired)}:f{min(3, max_suit_count)}:h{high_rank // 3}"


def _coarse_board_texture_bucket(board: tuple[str, ...]) -> str:
    ranks = [_rank_index(card) for card in board]
    suits = [card[1] for card in board]
    paired = len(set(ranks)) < len(ranks)
    max_suit_count = max((suits.count(suit) for suit in set(suits)), default=0)
    high_rank = max(ranks) if ranks else 0
    return f"b{len(board)}:p{int(paired)}:f{int(max_suit_count >= 3)}:h{high_rank // 5}"


def _postflop_bucket(state: FixedLimitHoldemState, player: int) -> str:
    board = state.visible_board()
    result = evaluate_holdem_hand(state.private_cards[player], board)
    rank_strength = 1.0 - ((result.rank_class - 1) / 8.0)
    score_strength = (7463.0 - result.score) / 7462.0
    strength_bucket = int(max(0.0, min(0.999, (rank_strength + score_strength) / 2.0)) * 6)
    return f"hc{result.rank_class}:sb{strength_bucket}:{_board_texture_bucket(board)}"


def _coarse_postflop_bucket(state: FixedLimitHoldemState, player: int) -> str:
    board = state.visible_board()
    result = evaluate_holdem_hand(state.private_cards[player], board)
    if result.rank_class <= 5:
        made_group = "strong"
    elif result.rank_class <= 7:
        made_group = "made"
    elif result.rank_class == 8:
        made_group = "pair"
    else:
        made_group = "high"
    rank_strength = 1.0 - ((result.rank_class - 1) / 8.0)
    score_strength = (7463.0 - result.score) / 7462.0
    strength_bucket = int(max(0.0, min(0.999, (rank_strength + score_strength) / 2.0)) * 3)
    return f"{made_group}:s{strength_bucket}:{_coarse_board_texture_bucket(board)}"


def _flush_draw_bucket(cards: tuple[str, ...]) -> str:
    suit_counts = [sum(1 for card in cards if card[1] == suit) for suit in "shdc"]
    max_count = max(suit_counts, default=0)
    if max_count >= 5:
        return "made"
    if max_count == 4:
        return "draw"
    return "none"


def _straight_draw_bucket(cards: tuple[str, ...]) -> str:
    ranks = {_rank_index(card) + 2 for card in cards}
    if 14 in ranks:
        ranks.add(1)
    for start in range(1, 11):
        if all(start + offset in ranks for offset in range(5)):
            return "made"
    for start in range(1, 12):
        if all(start + offset in ranks for offset in range(4)):
            return "open"
    for start in range(1, 11):
        if sum(1 for offset in range(5) if start + offset in ranks) == 4:
            return "gutshot"
    return "none"


def _medium_postflop_bucket(state: FixedLimitHoldemState, player: int) -> str:
    board = state.visible_board()
    result = evaluate_holdem_hand(state.private_cards[player], board)
    rank_strength = 1.0 - ((result.rank_class - 1) / 8.0)
    score_strength = (7463.0 - result.score) / 7462.0
    strength_bucket = int(max(0.0, min(0.999, (rank_strength + score_strength) / 2.0)) * 5)
    cards = (*state.private_cards[player], *board)
    return (
        f"hc{result.rank_class}:s{strength_bucket}:"
        f"fd{_flush_draw_bucket(cards)}:sd{_straight_draw_bucket(cards)}:"
        f"{_coarse_board_texture_bucket(board)}"
    )


def _stable_card_seed(private_cards: tuple[str, str], board: tuple[str, ...]) -> int:
    payload = ",".join((*sorted(private_cards), "|", *sorted(board))).encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")


@lru_cache(maxsize=200_000)
def _sampled_equity_bucket(private_cards: tuple[str, str], board: tuple[str, ...]) -> int:
    equity = estimate_holdem_equity(
        private_cards,
        board,
        simulations=_EQUITY_ABSTRACTION_SIMS,
        rng=random.Random(_stable_card_seed(private_cards, board)),
    )
    return int(min(0.999, equity) * 12)


def _equity_postflop_bucket(state: FixedLimitHoldemState, player: int) -> str:
    board = state.visible_board()
    sorted_private = tuple(sorted(state.private_cards[player]))
    private_cards = (sorted_private[0], sorted_private[1])
    result = evaluate_holdem_hand(private_cards, board)
    cards = (*private_cards, *board)
    return (
        f"e{_sampled_equity_bucket(private_cards, tuple(sorted(board)))}:"
        f"hc{result.rank_class}:"
        f"fd{_flush_draw_bucket(cards)}:sd{_straight_draw_bucket(cards)}:"
        f"{_coarse_board_texture_bucket(board)}"
    )


def _exact_history_key(state: FixedLimitHoldemState) -> str:
    return "|".join(
        ",".join(history) if history else "-"
        for history in state.histories[: state.street + 1]
    )


def _coarse_history_key(state: FixedLimitHoldemState) -> str:
    street_summaries = []
    for history in state.histories[: state.street + 1]:
        aggressive = sum(1 for action in history if action in (BET, RAISE))
        calls = sum(1 for action in history if action == CALL)
        checks = sum(1 for action in history if action == CHECK)
        folds = sum(1 for action in history if action == FOLD)
        street_summaries.append(
            f"a{min(4, aggressive)}c{min(2, calls)}x{min(2, checks)}f{folds}"
        )
    state_summary = (
        f"to{state.to_act}:call{int(state.outstanding_call_amount() > 0)}:"
        f"rb{min(4, state.bets_this_round)}:pot{min(10, sum(state.contributions) // 4)}"
    )
    return "|".join([*street_summaries, state_summary])


def abstract_holdem_information_key(
    state: FixedLimitHoldemState,
    *,
    abstraction: str = "fine",
) -> str:
    if abstraction not in HOLDEM_ABSTRACTIONS:
        raise ValueError(f"abstraction must be one of {', '.join(HOLDEM_ABSTRACTIONS)}")
    player = state.current_player()
    if state.street == 0:
        if abstraction == "coarse":
            hand_bucket = _coarse_preflop_bucket(state.private_cards[player])
        elif abstraction == "equity":
            hand_bucket = _equity_preflop_bucket(state.private_cards[player])
        elif abstraction == "medium":
            hand_bucket = _medium_preflop_bucket(state.private_cards[player])
        else:
            hand_bucket = _preflop_bucket(state.private_cards[player])
    else:
        if abstraction == "coarse":
            hand_bucket = _coarse_postflop_bucket(state, player)
        elif abstraction == "equity":
            hand_bucket = _equity_postflop_bucket(state, player)
        elif abstraction == "medium":
            hand_bucket = _medium_postflop_bucket(state, player)
        else:
            hand_bucket = _postflop_bucket(state, player)
    histories = (
        _coarse_history_key(state)
        if abstraction in ("coarse", "medium", "equity")
        else _exact_history_key(state)
    )
    return f"p{player}:s{state.street}:{hand_bucket}:{histories}"


class HoldemAbstractionCFRTrainer:
    """Sampled CFR over a compact Hold'em hand-strength abstraction."""

    def __init__(
        self,
        *,
        seed: int = 0,
        cfr_plus: bool = True,
        linear_averaging: bool = True,
        max_bets_per_round: int = 4,
        traversal: str = "external",
        abstraction: str = "coarse",
    ) -> None:
        if max_bets_per_round < 1:
            raise ValueError("max_bets_per_round must be positive")
        if traversal not in ("external", "full"):
            raise ValueError("traversal must be external or full")
        if abstraction not in HOLDEM_ABSTRACTIONS:
            raise ValueError(f"abstraction must be one of {', '.join(HOLDEM_ABSTRACTIONS)}")
        self.rng = random.Random(seed)
        self.seed = seed
        self.cfr_plus = cfr_plus
        self.linear_averaging = linear_averaging
        self.max_bets_per_round = max_bets_per_round
        self.traversal = traversal
        self.abstraction = abstraction
        self.iterations = 0
        self.infosets: dict[str, InfoSet] = {}
        self.sampled_utility_sum = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "game": "fixed_limit_holdem",
            "algorithm": "sampled_abstract_cfr_plus" if self.cfr_plus else "sampled_abstract_cfr",
            "average_weighting": "linear" if self.linear_averaging else "uniform",
            "iterations": self.iterations,
            "seed": self.seed,
            "max_bets_per_round": self.max_bets_per_round,
            "traversal": self.traversal,
            "abstraction": self.abstraction,
            "rng_state": self.rng.getstate(),
            "sampled_utility_sum": self.sampled_utility_sum,
            "infosets": {
                key: infoset.to_dict()
                for key, infoset in sorted(self.infosets.items())
            },
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "HoldemAbstractionCFRTrainer":
        if payload.get("game") != "fixed_limit_holdem":
            raise ValueError("Checkpoint is not for fixed-limit Hold'em")
        trainer = cls(
            seed=int(payload.get("seed", 0)),
            cfr_plus=str(payload.get("algorithm", "")).endswith("cfr_plus"),
            linear_averaging=payload.get("average_weighting", "linear") == "linear",
            max_bets_per_round=int(payload.get("max_bets_per_round", 2)),
            traversal=str(payload.get("traversal", "full")),
            abstraction=str(payload.get("abstraction", "fine")),
        )
        trainer.iterations = int(payload["iterations"])
        trainer.sampled_utility_sum = float(payload.get("sampled_utility_sum", 0.0))
        if "rng_state" in payload:
            trainer.rng.setstate(_nested_tuple(payload["rng_state"]))
        trainer.infosets = {
            key: InfoSet.from_dict(infoset_payload)
            for key, infoset_payload in payload["infosets"].items()
        }
        return trainer

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(self.state_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "HoldemAbstractionCFRTrainer":
        return cls.from_state_dict(json.loads(Path(path).read_text()))

    def _infoset_for_state(self, state: FixedLimitHoldemState) -> InfoSet:
        key = abstract_holdem_information_key(state, abstraction=self.abstraction)
        actions = state.legal_actions()
        if key not in self.infosets:
            self.infosets[key] = InfoSet(actions=actions)
        return self.infosets[key]

    def _cfr(self, state: FixedLimitHoldemState, reach_p0: float, reach_p1: float) -> float:
        if state.is_terminal():
            return state.utility(0)

        player = state.current_player()
        infoset = self._infoset_for_state(state)
        strategy = infoset.current_strategy()
        average_weight = float(self.iterations + 1) if self.linear_averaging else 1.0
        infoset.accumulate_strategy(
            reach_p0 if player == 0 else reach_p1,
            strategy,
            weight=average_weight,
        )

        action_utilities: list[float] = []
        node_utility = 0.0
        for action, probability in zip(infoset.actions, strategy):
            child = state.apply(action)
            if player == 0:
                action_utility = self._cfr(child, reach_p0 * probability, reach_p1)
            else:
                action_utility = self._cfr(child, reach_p0, reach_p1 * probability)
            action_utilities.append(action_utility)
            node_utility += probability * action_utility

        opponent_reach = reach_p1 if player == 0 else reach_p0
        for index, action_utility in enumerate(action_utilities):
            regret = (
                action_utility - node_utility
                if player == 0
                else node_utility - action_utility
            )
            infoset.regret_sum[index] += opponent_reach * regret
            if self.cfr_plus and infoset.regret_sum[index] < 0.0:
                infoset.regret_sum[index] = 0.0

        return node_utility

    def _external_cfr(self, state: FixedLimitHoldemState, updating_player: int) -> float:
        if state.is_terminal():
            return state.utility(updating_player)

        player = state.current_player()
        infoset = self._infoset_for_state(state)
        strategy = infoset.current_strategy()
        average_weight = float(self.iterations + 1) if self.linear_averaging else 1.0
        infoset.accumulate_strategy(1.0, strategy, weight=average_weight)

        if player != updating_player:
            sampled_action = self._sample_action(infoset.actions, strategy)
            return self._external_cfr(state.apply(sampled_action), updating_player)

        action_utilities = [
            self._external_cfr(state.apply(action), updating_player)
            for action in infoset.actions
        ]
        node_utility = sum(
            probability * action_utility
            for probability, action_utility in zip(strategy, action_utilities)
        )
        for index, action_utility in enumerate(action_utilities):
            infoset.regret_sum[index] += action_utility - node_utility
            if self.cfr_plus and infoset.regret_sum[index] < 0.0:
                infoset.regret_sum[index] = 0.0
        return node_utility

    def _sample_action(self, actions: tuple[str, ...], strategy: list[float]) -> str:
        sample = self.rng.random()
        cumulative = 0.0
        for action, probability in zip(actions, strategy):
            cumulative += probability
            if sample <= cumulative:
                return action
        return actions[-1]

    def _deal_state(self) -> FixedLimitHoldemState:
        state = deal_fixed_limit_holdem(self.rng)
        if state.max_bets_per_round == self.max_bets_per_round:
            return state
        return state._replace(max_bets_per_round=self.max_bets_per_round)

    def train(self, iterations: int) -> HoldemMCCFRTrainingResult:
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        for _ in range(iterations):
            if self.traversal == "full":
                utility = self._cfr(self._deal_state(), 1.0, 1.0)
            else:
                utility = self._external_cfr(self._deal_state(), 0)
                self._external_cfr(self._deal_state(), 1)
            self.sampled_utility_sum += utility
            self.iterations += 1

        return HoldemMCCFRTrainingResult(
            iterations=self.iterations,
            infosets=len(self.infosets),
            sampled_game_value_p0=self.sampled_utility_sum / self.iterations,
        )

    def average_strategy(self) -> HoldemAbstractStrategy:
        return {
            key: infoset.average_strategy()
            for key, infoset in sorted(self.infosets.items())
        }


def holdem_policy_from_abstract_strategy(
    strategy: HoldemAbstractStrategy,
    rng: random.Random,
    *,
    fallback_policy: HoldemPolicy | None = None,
    abstraction: str = "fine",
) -> HoldemPolicy:
    def select_action(state: FixedLimitHoldemState) -> str:
        actions = state.legal_actions()
        key = abstract_holdem_information_key(state, abstraction=abstraction)
        if key not in strategy and fallback_policy is not None:
            return fallback_policy(state)
        distribution = normalize_distribution(actions, strategy.get(key, {}))
        sample = rng.random()
        cumulative = 0.0
        for action in actions:
            cumulative += distribution[action]
            if sample <= cumulative:
                return action
        return actions[-1]

    return select_action


def holdem_policy_from_trainer(
    trainer: HoldemAbstractionCFRTrainer,
    rng: random.Random,
    *,
    fallback_policy: HoldemPolicy | None = None,
    min_strategy_weight: float = 0.0,
) -> HoldemPolicy:
    def strategy_support(infoset: InfoSet) -> float:
        return (
            float(infoset.strategy_update_count)
            if infoset.strategy_update_count > 0
            else sum(infoset.strategy_sum)
        )

    def select_action(state: FixedLimitHoldemState) -> str:
        actions = state.legal_actions()
        key = abstract_holdem_information_key(state, abstraction=trainer.abstraction)
        infoset = trainer.infosets.get(key)
        if infoset is None:
            if fallback_policy is not None:
                return fallback_policy(state)
            distribution = normalize_distribution(actions, {})
        elif strategy_support(infoset) < min_strategy_weight and fallback_policy is not None:
            return fallback_policy(state)
        else:
            distribution = normalize_distribution(actions, infoset.average_strategy())

        sample = rng.random()
        cumulative = 0.0
        for action in actions:
            cumulative += distribution[action]
            if sample <= cumulative:
                return action
        return actions[-1]

    return select_action

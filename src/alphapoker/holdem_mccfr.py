"""Sampled abstract CFR for fixed-limit heads-up Hold'em."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alphapoker.cfr import InfoSet
from alphapoker.eval import normalize_distribution
from alphapoker.holdem import (
    FixedLimitHoldemState,
    HoldemPolicy,
    deal_fixed_limit_holdem,
    evaluate_holdem_hand,
)
from alphapoker.holdem_features import HOLDEM_RANKS

HoldemAbstractStrategy = dict[str, dict[str, float]]


@dataclass(frozen=True)
class HoldemMCCFRTrainingResult:
    iterations: int
    infosets: int
    sampled_game_value_p0: float


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


def _board_texture_bucket(board: tuple[str, ...]) -> str:
    ranks = [_rank_index(card) for card in board]
    suits = [card[1] for card in board]
    paired = len(set(ranks)) < len(ranks)
    max_suit_count = max((suits.count(suit) for suit in set(suits)), default=0)
    high_rank = max(ranks) if ranks else 0
    return f"b{len(board)}:p{int(paired)}:f{min(3, max_suit_count)}:h{high_rank // 3}"


def _postflop_bucket(state: FixedLimitHoldemState, player: int) -> str:
    board = state.visible_board()
    result = evaluate_holdem_hand(state.private_cards[player], board)
    rank_strength = 1.0 - ((result.rank_class - 1) / 8.0)
    score_strength = (7463.0 - result.score) / 7462.0
    strength_bucket = int(max(0.0, min(0.999, (rank_strength + score_strength) / 2.0)) * 6)
    return f"hc{result.rank_class}:sb{strength_bucket}:{_board_texture_bucket(board)}"


def abstract_holdem_information_key(state: FixedLimitHoldemState) -> str:
    player = state.current_player()
    if state.street == 0:
        hand_bucket = _preflop_bucket(state.private_cards[player])
    else:
        hand_bucket = _postflop_bucket(state, player)
    histories = "|".join(
        ",".join(history) if history else "-"
        for history in state.histories[: state.street + 1]
    )
    return f"p{player}:s{state.street}:{hand_bucket}:{histories}"


class HoldemAbstractionCFRTrainer:
    """Chance-sampled CFR over a compact Hold'em hand-strength abstraction."""

    def __init__(
        self,
        *,
        seed: int = 0,
        cfr_plus: bool = True,
        linear_averaging: bool = True,
        max_bets_per_round: int = 2,
    ) -> None:
        if max_bets_per_round < 1:
            raise ValueError("max_bets_per_round must be positive")
        self.rng = random.Random(seed)
        self.seed = seed
        self.cfr_plus = cfr_plus
        self.linear_averaging = linear_averaging
        self.max_bets_per_round = max_bets_per_round
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
        )
        trainer.iterations = int(payload["iterations"])
        trainer.sampled_utility_sum = float(payload.get("sampled_utility_sum", 0.0))
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
        key = abstract_holdem_information_key(state)
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

    def _deal_state(self) -> FixedLimitHoldemState:
        state = deal_fixed_limit_holdem(self.rng)
        if state.max_bets_per_round == self.max_bets_per_round:
            return state
        return state._replace(max_bets_per_round=self.max_bets_per_round)

    def train(self, iterations: int) -> HoldemMCCFRTrainingResult:
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        for _ in range(iterations):
            utility = self._cfr(self._deal_state(), 1.0, 1.0)
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
) -> HoldemPolicy:
    def select_action(state: FixedLimitHoldemState) -> str:
        actions = state.legal_actions()
        key = abstract_holdem_information_key(state)
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

"""Counterfactual regret minimization for Kuhn poker."""

from __future__ import annotations

from dataclasses import dataclass, field

from alphapoker.eval import StrategyProfile, best_response_value, expected_utility, exploitability
from alphapoker.kuhn import KuhnState, all_card_deals


@dataclass
class InfoSet:
    actions: tuple[str, ...]
    regret_sum: list[float] = field(default_factory=list)
    strategy_sum: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.regret_sum:
            self.regret_sum = [0.0 for _ in self.actions]
        if not self.strategy_sum:
            self.strategy_sum = [0.0 for _ in self.actions]

    def current_strategy(self) -> list[float]:
        positives = [max(0.0, regret) for regret in self.regret_sum]
        normalizer = sum(positives)
        if normalizer > 0.0:
            return [value / normalizer for value in positives]
        return [1.0 / len(self.actions) for _ in self.actions]

    def accumulate_strategy(self, reach_probability: float, strategy: list[float]) -> None:
        for index, probability in enumerate(strategy):
            self.strategy_sum[index] += reach_probability * probability

    def average_strategy(self) -> dict[str, float]:
        normalizer = sum(self.strategy_sum)
        if normalizer > 0.0:
            return {
                action: self.strategy_sum[index] / normalizer
                for index, action in enumerate(self.actions)
            }
        strategy = self.current_strategy()
        return {action: strategy[index] for index, action in enumerate(self.actions)}

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": list(self.actions),
            "regret_sum": self.regret_sum,
            "strategy_sum": self.strategy_sum,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "InfoSet":
        return cls(
            actions=tuple(str(action) for action in payload["actions"]),  # type: ignore[index]
            regret_sum=[float(value) for value in payload["regret_sum"]],  # type: ignore[index]
            strategy_sum=[float(value) for value in payload["strategy_sum"]],  # type: ignore[index]
        )


@dataclass(frozen=True)
class TrainingResult:
    iterations: int
    game_value_p0: float
    best_response_p0: float
    best_response_p1: float
    exploitability: float


class KuhnCFRTrainer:
    """Tabular CFR/CFR+ trainer for Kuhn poker."""

    def __init__(self, *, cfr_plus: bool = True) -> None:
        self.cfr_plus = cfr_plus
        self.iterations = 0
        self.infosets: dict[str, InfoSet] = {}

    def _infoset_for_state(self, state: KuhnState) -> InfoSet:
        key = state.infoset_key()
        actions = state.legal_actions()
        if key not in self.infosets:
            self.infosets[key] = InfoSet(actions=actions)
        return self.infosets[key]

    def _cfr(self, state: KuhnState, reach_p0: float, reach_p1: float) -> float:
        if state.is_terminal():
            return state.utility(0)

        player = state.current_player()
        infoset = self._infoset_for_state(state)
        strategy = infoset.current_strategy()
        infoset.accumulate_strategy(reach_p0 if player == 0 else reach_p1, strategy)

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

    def train(self, iterations: int) -> TrainingResult:
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        deals = all_card_deals()
        for _ in range(iterations):
            for cards in deals:
                self._cfr(KuhnState.initial(cards), 1.0, 1.0)
            self.iterations += 1

        strategy = self.average_strategy()
        return TrainingResult(
            iterations=self.iterations,
            game_value_p0=expected_utility(strategy, player=0),
            best_response_p0=best_response_value(0, strategy),
            best_response_p1=best_response_value(1, strategy),
            exploitability=exploitability(strategy),
        )

    def average_strategy(self) -> StrategyProfile:
        return {
            key: infoset.average_strategy()
            for key, infoset in sorted(self.infosets.items())
        }

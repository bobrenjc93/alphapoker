"""Tabular CFR for limit Leduc poker."""

from __future__ import annotations

from dataclasses import dataclass

from alphapoker.cfr import InfoSet
from alphapoker.eval import normalize_distribution
from alphapoker.leduc import LeducState, all_leduc_deals

LeducStrategyProfile = dict[str, dict[str, float]]


@dataclass(frozen=True)
class LeducTrainingResult:
    iterations: int
    game_value_p0: float
    infosets: int


def leduc_policy_for_state(
    strategy: LeducStrategyProfile,
    state: LeducState,
) -> dict[str, float]:
    return normalize_distribution(state.legal_actions(), strategy.get(state.information_key(), {}))


def expected_leduc_utility(strategy: LeducStrategyProfile, player: int = 0) -> float:
    """Return exact expected utility under a Leduc strategy profile."""

    if player not in (0, 1):
        raise ValueError(f"Unknown player: {player}")

    def walk(state: LeducState) -> float:
        if state.is_terminal():
            return state.utility(player)
        dist = leduc_policy_for_state(strategy, state)
        return sum(prob * walk(state.apply(action)) for action, prob in dist.items())

    total = 0.0
    deals = all_leduc_deals()
    for private0, private1, public in deals:
        total += walk(LeducState.initial((private0, private1), public))
    return total / len(deals)


class LeducCFRTrainer:
    """Full-tree tabular CFR/CFR+ for two-player limit Leduc poker."""

    def __init__(self, *, cfr_plus: bool = True) -> None:
        self.cfr_plus = cfr_plus
        self.iterations = 0
        self.infosets: dict[str, InfoSet] = {}

    def _infoset_for_state(self, state: LeducState) -> InfoSet:
        key = state.information_key()
        actions = state.legal_actions()
        if key not in self.infosets:
            self.infosets[key] = InfoSet(actions=actions)
        return self.infosets[key]

    def _cfr(self, state: LeducState, reach_p0: float, reach_p1: float) -> float:
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

    def train(self, iterations: int) -> LeducTrainingResult:
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        deals = all_leduc_deals()
        for _ in range(iterations):
            for private0, private1, public in deals:
                self._cfr(LeducState.initial((private0, private1), public), 1.0, 1.0)
            self.iterations += 1

        strategy = self.average_strategy()
        return LeducTrainingResult(
            iterations=self.iterations,
            game_value_p0=expected_leduc_utility(strategy, player=0),
            infosets=len(strategy),
        )

    def average_strategy(self) -> LeducStrategyProfile:
        return {
            key: infoset.average_strategy()
            for key, infoset in sorted(self.infosets.items())
        }


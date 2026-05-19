import random

import pytest


pytest.importorskip("treys")

from alphapoker.holdem import FixedLimitHoldemState, deal_fixed_limit_holdem  # noqa: E402
from alphapoker.holdem_mccfr import (  # noqa: E402
    HoldemAbstractionCFRTrainer,
    abstract_holdem_information_key,
    holdem_policy_from_abstract_strategy,
    holdem_policy_from_trainer,
)


def test_holdem_abstraction_key_includes_player_street_and_history() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    )

    key = abstract_holdem_information_key(state)

    assert key.startswith("p0:s0:")
    assert key.endswith(":-")


def test_holdem_mccfr_trainer_smoke_and_checkpoint(tmp_path) -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=3, max_bets_per_round=4, traversal="external")
    result = trainer.train(2)
    checkpoint = tmp_path / "holdem_mccfr.json"

    trainer.save_checkpoint(checkpoint)
    loaded = HoldemAbstractionCFRTrainer.load_checkpoint(checkpoint)

    assert result.iterations == 2
    assert result.infosets > 0
    assert loaded.iterations == trainer.iterations
    assert loaded.traversal == "external"
    assert loaded.average_strategy().keys() == trainer.average_strategy().keys()


def test_holdem_mccfr_policy_selects_legal_action() -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=4, max_bets_per_round=4, traversal="external")
    trainer.train(2)
    state = deal_fixed_limit_holdem(random.Random(5))
    policy = holdem_policy_from_abstract_strategy(
        trainer.average_strategy(),
        random.Random(6),
    )

    assert policy(state) in state.legal_actions()


def test_holdem_mccfr_trainer_policy_can_fallback_on_low_weight() -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=8, max_bets_per_round=4, traversal="external")
    trainer.train(1)
    state = deal_fixed_limit_holdem(random.Random(9))
    policy = holdem_policy_from_trainer(
        trainer,
        random.Random(10),
        fallback_policy=lambda fallback_state: fallback_state.legal_actions()[0],
        min_strategy_weight=1_000_000.0,
    )

    assert policy(state) == state.legal_actions()[0]


def test_holdem_mccfr_full_traversal_smoke() -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=7, max_bets_per_round=1, traversal="full")

    result = trainer.train(1)

    assert result.iterations == 1
    assert result.infosets > 0

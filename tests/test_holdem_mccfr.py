import random

import pytest


pytest.importorskip("treys")

from alphapoker.holdem import FixedLimitHoldemState, deal_fixed_limit_holdem  # noqa: E402
from alphapoker.holdem_mccfr import (  # noqa: E402
    HoldemAbstractionCFRTrainer,
    abstract_holdem_information_key,
    holdem_policy_from_abstract_strategy,
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
    trainer = HoldemAbstractionCFRTrainer(seed=3, max_bets_per_round=1)
    result = trainer.train(2)
    checkpoint = tmp_path / "holdem_mccfr.json"

    trainer.save_checkpoint(checkpoint)
    loaded = HoldemAbstractionCFRTrainer.load_checkpoint(checkpoint)

    assert result.iterations == 2
    assert result.infosets > 0
    assert loaded.iterations == trainer.iterations
    assert loaded.average_strategy().keys() == trainer.average_strategy().keys()


def test_holdem_mccfr_policy_selects_legal_action() -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=4, max_bets_per_round=1)
    trainer.train(2)
    state = deal_fixed_limit_holdem(random.Random(5))
    policy = holdem_policy_from_abstract_strategy(
        trainer.average_strategy(),
        random.Random(6),
    )

    assert policy(state) in state.legal_actions()

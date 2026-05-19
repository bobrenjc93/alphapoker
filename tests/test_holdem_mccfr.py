import random

import pytest


pytest.importorskip("treys")

from alphapoker.holdem import FixedLimitHoldemState, deal_fixed_limit_holdem  # noqa: E402
from alphapoker.kuhn import CALL  # noqa: E402
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


def test_holdem_coarse_abstraction_summarizes_history() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ad"), ("Kc", "Kd")),
        ("2h", "3d", "4s", "5c", "6h"),
    ).apply(CALL)

    key = abstract_holdem_information_key(state, abstraction="coarse")

    assert key.startswith("p1:s1:")
    assert "to1:call0" in key


def test_holdem_medium_abstraction_tracks_draws() -> None:
    state = FixedLimitHoldemState.initial(
        (("9s", "Ts"), ("2h", "3d")),
        ("Js", "Qs", "2c", "4d", "5h"),
    )._replace(street=1, to_act=0, round_contributions=(0, 0), bets_this_round=0)

    key = abstract_holdem_information_key(state, abstraction="medium")

    assert key.startswith("p0:s1:")
    assert ":fddraw:" in key
    assert ":sdopen:" in key
    assert "to0:call0" in key


def test_holdem_equity_abstraction_is_stable_and_tracks_equity() -> None:
    state = FixedLimitHoldemState.initial(
        (("As", "Ks"), ("2h", "3d")),
        ("Qs", "Js", "2c", "4d", "5h"),
    )._replace(street=1, to_act=0, round_contributions=(0, 0), bets_this_round=0)

    first_key = abstract_holdem_information_key(state, abstraction="equity")
    second_key = abstract_holdem_information_key(state, abstraction="equity")

    assert first_key == second_key
    assert first_key.startswith("p0:s1:e")
    assert ":fddraw:" in first_key
    assert ":sdopen:" in first_key


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
    assert loaded.abstraction == "coarse"
    assert loaded.average_strategy().keys() == trainer.average_strategy().keys()


def test_holdem_mccfr_policy_selects_legal_action() -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=4, max_bets_per_round=4, traversal="external")
    trainer.train(2)
    state = deal_fixed_limit_holdem(random.Random(5))
    policy = holdem_policy_from_abstract_strategy(
        trainer.average_strategy(),
        random.Random(6),
        abstraction=trainer.abstraction,
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


def test_holdem_mccfr_medium_abstraction_smoke() -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=11, max_bets_per_round=4, abstraction="medium")

    result = trainer.train(2)

    assert result.iterations == 2
    assert result.infosets > 0
    assert trainer.abstraction == "medium"


def test_holdem_mccfr_equity_abstraction_smoke() -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=12, max_bets_per_round=4, abstraction="equity")

    result = trainer.train(2)

    assert result.iterations == 2
    assert result.infosets > 0
    assert trainer.abstraction == "equity"

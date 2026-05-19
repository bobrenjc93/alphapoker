from alphapoker.cfr import InfoSet, KuhnCFRTrainer
from alphapoker.eval import expected_utility, exploitability


def test_cfr_converges_toward_kuhn_value() -> None:
    trainer = KuhnCFRTrainer()
    result = trainer.train(20_000)
    strategy = trainer.average_strategy()

    assert abs(result.game_value_p0 - (-1.0 / 18.0)) < 0.02
    assert abs(expected_utility(strategy, player=0) - result.game_value_p0) < 1e-12
    assert exploitability(strategy) < 0.03


def test_infoset_tracks_strategy_update_count() -> None:
    infoset = InfoSet(actions=("check", "bet"))
    infoset.accumulate_strategy(1.0, [0.75, 0.25], weight=10.0)
    infoset.accumulate_strategy(0.0, [0.5, 0.5], weight=10.0)

    payload = infoset.to_dict()
    loaded = InfoSet.from_dict(payload)
    legacy_loaded = InfoSet.from_dict({key: value for key, value in payload.items() if key != "strategy_update_count"})

    assert infoset.strategy_update_count == 1
    assert loaded.strategy_update_count == 1
    assert legacy_loaded.strategy_update_count == 0

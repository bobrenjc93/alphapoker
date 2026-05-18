from alphapoker.cfr import KuhnCFRTrainer
from alphapoker.eval import expected_utility, exploitability


def test_cfr_converges_toward_kuhn_value() -> None:
    trainer = KuhnCFRTrainer()
    result = trainer.train(20_000)
    strategy = trainer.average_strategy()

    assert abs(result.game_value_p0 - (-1.0 / 18.0)) < 0.02
    assert abs(expected_utility(strategy, player=0) - result.game_value_p0) < 1e-12
    assert exploitability(strategy) < 0.03


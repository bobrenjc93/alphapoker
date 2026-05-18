import math

from alphapoker.leduc_cfr import LeducCFRTrainer, expected_leduc_utility


def test_leduc_cfr_smoke_trains_public_and_private_infosets() -> None:
    trainer = LeducCFRTrainer()
    result = trainer.train(2)
    strategy = trainer.average_strategy()

    assert result.infosets == len(strategy)
    assert result.infosets > 50
    assert math.isfinite(result.game_value_p0)
    assert abs(expected_leduc_utility(strategy, player=0) - result.game_value_p0) < 1e-12
    assert any("public=-" in key for key in strategy)
    assert any("public=J" in key or "public=Q" in key or "public=K" in key for key in strategy)


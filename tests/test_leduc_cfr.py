import math

from alphapoker.leduc_cfr import (
    LeducCFRTrainer,
    best_response_leduc,
    expected_leduc_utility,
    leduc_exploitability,
)


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


def test_leduc_best_response_improves_against_uniform_policy() -> None:
    uniform_strategy = {}
    value_p0 = expected_leduc_utility(uniform_strategy, player=0)
    value_p1 = expected_leduc_utility(uniform_strategy, player=1)

    br0 = best_response_leduc(0, uniform_strategy)
    br1 = best_response_leduc(1, uniform_strategy)

    assert br0.value >= value_p0
    assert br1.value >= value_p1
    assert br0.policy
    assert br1.policy
    assert leduc_exploitability(uniform_strategy) > 0.0


def test_leduc_cfr_checkpoint_round_trip(tmp_path) -> None:
    checkpoint = tmp_path / "leduc_checkpoint.json"
    trainer = LeducCFRTrainer()
    trainer.train(1)
    strategy_before = trainer.average_strategy()
    trainer.save_checkpoint(checkpoint)

    loaded = LeducCFRTrainer.load_checkpoint(checkpoint)
    assert loaded.iterations == 1
    assert loaded.average_strategy() == strategy_before

    result = loaded.train(1)
    assert result.iterations == 2
    assert result.infosets == 288


def test_leduc_cfr_linear_averaging_smoke() -> None:
    trainer = LeducCFRTrainer(linear_averaging=True)
    result = trainer.train(2)

    assert trainer.linear_averaging
    assert result.iterations == 2
    assert result.infosets == 288

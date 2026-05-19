import pytest


pytest.importorskip("treys")

from alphapoker.holdem_mccfr import HoldemAbstractionCFRTrainer  # noqa: E402
from alphapoker.sweep_holdem_mccfr_min_weight import (  # noqa: E402
    build_parser,
    parse_min_strategy_weights,
    run,
)


def test_parse_min_strategy_weights() -> None:
    assert parse_min_strategy_weights("0,100,500.5") == (0.0, 100.0, 500.5)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--checkpoint", "c.json", "--weights", "-1"])


def test_mccfr_min_weight_sweep_smoke(tmp_path) -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=3, max_bets_per_round=4, abstraction="coarse")
    trainer.train(1)
    checkpoint = tmp_path / "holdem_mccfr.json"
    trainer.save_checkpoint(checkpoint)

    metrics = run(
        build_parser().parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--hands",
                "1",
                "--model-player",
                "both",
                "--opponent-policy",
                "random",
                "--fallback-policy",
                "random",
                "--strategy-mode",
                "current",
                "--weights",
                "0,100",
                "--equity-sims",
                "2",
                "--paired-seats",
            ]
        )
    )

    assert metrics["checkpoint"] == str(checkpoint)
    assert metrics["model_player"] == "both"
    assert metrics["paired_seats"]
    assert metrics["strategy_mode"] == "current"
    assert metrics["weights"] == [0.0, 100.0]
    assert len(metrics["results"]) == 2
    assert metrics["best"]["min_strategy_weight"] in {0.0, 100.0}
    assert metrics["best"]["paired_seats"]

import pytest


pytest.importorskip("treys")

from alphapoker.evaluate_holdem_mccfr import build_parser, run  # noqa: E402
from alphapoker.holdem_mccfr import HoldemAbstractionCFRTrainer  # noqa: E402


def test_evaluate_holdem_mccfr_parser_accepts_options() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "holdem_mccfr.json",
            "--hands",
            "2",
            "--model-player",
            "both",
            "--jobs",
            "2",
            "--fallback-policy",
            "tuned-pot-odds",
            "--min-strategy-weight",
            "5",
        ]
    )

    assert args.checkpoint.name == "holdem_mccfr.json"
    assert args.hands == 2
    assert args.model_player == (0, 1)
    assert args.jobs == 2
    assert args.fallback_policy == "tuned-pot-odds"
    assert args.min_strategy_weight == 5


def test_evaluate_holdem_mccfr_run_smoke(tmp_path) -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=3, traversal="external")
    trainer.train(2)
    checkpoint = tmp_path / "holdem_mccfr.json"
    trainer.save_checkpoint(checkpoint)

    metrics = run(
        build_parser().parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--hands",
                "1",
                "--equity-sims",
                "2",
            ]
        )
    )

    assert metrics["hands"] == 1
    assert metrics["checkpoint"] == str(checkpoint)
    assert metrics["abstraction"] == "coarse"
    assert metrics["jobs"] == 1
    assert metrics["shard_hands"] == [1]

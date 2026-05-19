import pytest


pytest.importorskip("treys")

from alphapoker.evaluate_holdem_mccfr import build_parser, evaluate_mccfr_shard, run  # noqa: E402
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
            "--paired-seats",
        ]
    )

    assert args.checkpoint.name == "holdem_mccfr.json"
    assert args.hands == 2
    assert args.model_player == (0, 1)
    assert args.jobs == 2
    assert args.fallback_policy == "tuned-pot-odds"
    assert args.min_strategy_weight == 5
    assert args.paired_seats


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
    assert not metrics["paired_seats"]


def test_evaluate_holdem_mccfr_run_paired_seats_smoke(tmp_path) -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=5, traversal="external")
    trainer.train(2)
    checkpoint = tmp_path / "holdem_mccfr.json"
    trainer.save_checkpoint(checkpoint)

    metrics = run(
        build_parser().parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--hands",
                "2",
                "--equity-sims",
                "2",
                "--model-player",
                "both",
                "--paired-seats",
                "--jobs",
                "2",
            ]
        )
    )

    assert metrics["model_player"] == "both"
    assert metrics["hands"] == 4
    assert metrics["hands_per_model_player"] == 2
    assert metrics["paired_deals"] == 2
    assert metrics["paired_seats"]
    assert metrics["jobs"] == 2
    assert metrics["shard_hands"] == [1, 1]
    assert len(metrics["seat_metrics"]) == 2


def test_evaluate_holdem_mccfr_paired_seats_requires_both(tmp_path) -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=6, traversal="external")
    trainer.train(1)
    checkpoint = tmp_path / "holdem_mccfr.json"
    trainer.save_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="paired_seats"):
        run(
            build_parser().parse_args(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--hands",
                    "1",
                    "--paired-seats",
                ]
            )
        )


def test_evaluate_holdem_mccfr_reuses_deals_across_seats(tmp_path) -> None:
    trainer = HoldemAbstractionCFRTrainer(seed=4, traversal="external")
    trainer.train(1)
    checkpoint = tmp_path / "holdem_mccfr.json"
    trainer.save_checkpoint(checkpoint)

    p0_metrics = evaluate_mccfr_shard(
        checkpoint=checkpoint,
        hands=0,
        seed=100,
        opponent_policy="random",
        fallback_policy="random",
        min_strategy_weight=0.0,
        equity_sims=2,
        rollout_sims=None,
        model_player=0,
        shard_index=3,
    )
    p1_metrics = evaluate_mccfr_shard(
        checkpoint=checkpoint,
        hands=0,
        seed=100,
        opponent_policy="random",
        fallback_policy="random",
        min_strategy_weight=0.0,
        equity_sims=2,
        rollout_sims=None,
        model_player=1,
        shard_index=3,
    )

    assert p0_metrics["seed"] == p1_metrics["seed"]
    assert p0_metrics["model_player"] == 0
    assert p1_metrics["model_player"] == 1

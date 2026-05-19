import pytest


pytest.importorskip("treys")

from alphapoker.evaluate_holdem_policy import build_parser, run, split_hands  # noqa: E402


def test_split_hands_balances_jobs() -> None:
    assert split_hands(10, 3) == [4, 3, 3]
    assert split_hands(2, 8) == [1, 1]


def test_split_hands_rejects_invalid_jobs() -> None:
    with pytest.raises(ValueError, match="jobs"):
        split_hands(10, 0)


def test_evaluate_holdem_policy_parser_accepts_hybrid_and_both() -> None:
    args = build_parser().parse_args(
        [
            "--policy",
            "hybrid-pot-odds",
            "--opponent-policy",
            "tuned-pot-odds",
            "--model-player",
            "both",
            "--jobs",
            "2",
            "--progress",
        ]
    )

    assert args.policy == "hybrid-pot-odds"
    assert args.opponent_policy == "tuned-pot-odds"
    assert args.model_player == (0, 1)
    assert args.jobs == 2
    assert args.progress


def test_evaluate_holdem_policy_run_smoke(tmp_path) -> None:
    out = tmp_path / "metrics.json"
    metrics = run(
        build_parser().parse_args(
            [
                "--policy",
                "hybrid-pot-odds",
                "--opponent-policy",
                "pot-odds",
                "--hands",
                "1",
                "--seed",
                "3",
                "--equity-sims",
                "2",
                "--model-player",
                "both",
                "--out",
                str(out),
            ]
        )
    )

    assert out.exists()
    assert metrics["policy"] == "hybrid-pot-odds"
    assert metrics["opponent_policy"] == "pot-odds"
    assert metrics["hands"] == 2
    assert metrics["hands_per_model_player"] == 1
    assert metrics["jobs"] == 1
    assert metrics["shard_hands"] == [1]
    assert len(metrics["seat_metrics"]) == 2

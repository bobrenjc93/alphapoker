import pytest


pytest.importorskip("treys")

from alphapoker.evaluate_holdem_policy import (  # noqa: E402
    build_parser,
    resolve_policy_thresholds,
    run,
    split_hands,
)


def test_split_hands_balances_jobs() -> None:
    assert split_hands(10, 3) == [4, 3, 3]
    assert split_hands(2, 8) == [1, 1]


def test_split_hands_rejects_invalid_jobs() -> None:
    with pytest.raises(ValueError, match="jobs"):
        split_hands(10, 0)


def test_resolve_policy_thresholds_uses_family_defaults() -> None:
    assert resolve_policy_thresholds(
        "hybrid-pot-odds",
        bet_threshold=0.58,
        raise_threshold=None,
        call_margin=None,
    ) == (0.58, 0.76, 0.05)


def test_resolve_policy_thresholds_uses_exact_policy_defaults() -> None:
    assert resolve_policy_thresholds(
        "turn-river-exact-tuned-pot-odds",
        bet_threshold=0.62,
        raise_threshold=0.84,
        call_margin=None,
    ) == (0.62, 0.84, 0.05)


def test_resolve_policy_thresholds_uses_tight_exact_policy_defaults() -> None:
    assert resolve_policy_thresholds(
        "tight-turn-river-exact-pot-odds",
        bet_threshold=None,
        raise_threshold=0.86,
        call_margin=None,
    ) == (0.62, 0.86, 0.08)


def test_resolve_policy_thresholds_uses_tight_range_policy_defaults() -> None:
    assert resolve_policy_thresholds(
        "tight-range-pot-odds",
        bet_threshold=None,
        raise_threshold=0.86,
        call_margin=None,
    ) == (0.62, 0.86, 0.08)


def test_resolve_policy_thresholds_rejects_unsupported_policy() -> None:
    with pytest.raises(ValueError, match="threshold overrides"):
        resolve_policy_thresholds(
            "random",
            bet_threshold=0.58,
            raise_threshold=None,
            call_margin=None,
        )


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
            "--bet-threshold",
            "0.58",
            "--raise-threshold",
            "0.78",
            "--call-margin",
            "0.05",
            "--paired-seats",
            "--progress",
        ]
    )

    assert args.policy == "hybrid-pot-odds"
    assert args.opponent_policy == "tuned-pot-odds"
    assert args.model_player == (0, 1)
    assert args.jobs == 2
    assert args.bet_threshold == 0.58
    assert args.raise_threshold == 0.78
    assert args.call_margin == 0.05
    assert args.paired_seats
    assert args.progress


def test_evaluate_holdem_policy_parser_accepts_safe_rollout_margin() -> None:
    args = build_parser().parse_args(
        [
            "--policy",
            "tight-safe-rollout-pot-odds",
            "--opponent-policy",
            "tight-turn-river-exact-pot-odds",
            "--rollout-sims",
            "2",
            "--rollout-margin",
            "1.5",
        ]
    )

    assert args.policy == "tight-safe-rollout-pot-odds"
    assert args.rollout_sims == 2
    assert args.rollout_margin == 1.5


def test_evaluate_holdem_policy_parser_accepts_opponent_sim_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--policy",
            "tight-turn-river-exact-pot-odds",
            "--opponent-policy",
            "tight-turn-river-exact-pot-odds",
            "--equity-sims",
            "64",
            "--opponent-equity-sims",
            "8",
            "--rollout-sims",
            "4",
            "--opponent-rollout-sims",
            "2",
        ]
    )

    assert args.equity_sims == 64
    assert args.opponent_equity_sims == 8
    assert args.rollout_sims == 4
    assert args.opponent_rollout_sims == 2


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
                "--bet-threshold",
                "0.58",
                "--raise-threshold",
                "0.78",
                "--call-margin",
                "0.05",
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
    assert metrics["opponent_equity_sims"] == 2
    assert metrics["opponent_rollout_sims"] is None
    assert metrics["bet_threshold"] == 0.58
    assert metrics["raise_threshold"] == 0.78
    assert metrics["call_margin"] == 0.05
    assert len(metrics["seat_metrics"]) == 2


def test_evaluate_holdem_policy_run_paired_seats_smoke(tmp_path) -> None:
    out = tmp_path / "metrics.json"
    metrics = run(
        build_parser().parse_args(
            [
                "--policy",
                "pot-odds",
                "--opponent-policy",
                "tuned-pot-odds",
                "--hands",
                "2",
                "--seed",
                "4",
                "--equity-sims",
                "2",
                "--model-player",
                "both",
                "--paired-seats",
                "--jobs",
                "2",
                "--out",
                str(out),
            ]
        )
    )

    assert out.exists()
    assert metrics["model_player"] == "both"
    assert metrics["hands"] == 4
    assert metrics["hands_per_model_player"] == 2
    assert metrics["paired_deals"] == 2
    assert metrics["paired_seats"]
    assert metrics["jobs"] == 2
    assert metrics["shard_hands"] == [1, 1]
    assert metrics["opponent_equity_sims"] == 2


def test_evaluate_holdem_policy_run_exact_threshold_override_smoke(tmp_path) -> None:
    out = tmp_path / "metrics.json"
    metrics = run(
        build_parser().parse_args(
            [
                "--policy",
                "turn-river-exact-tuned-pot-odds",
                "--opponent-policy",
                "tuned-pot-odds",
                "--hands",
                "1",
                "--seed",
                "14",
                "--equity-sims",
                "2",
                "--model-player",
                "both",
                "--paired-seats",
                "--bet-threshold",
                "0.62",
                "--raise-threshold",
                "0.84",
                "--call-margin",
                "0.08",
                "--out",
                str(out),
            ]
        )
    )

    assert out.exists()
    assert metrics["policy"] == "turn-river-exact-tuned-pot-odds"
    assert metrics["bet_threshold"] == 0.62
    assert metrics["raise_threshold"] == 0.84
    assert metrics["call_margin"] == 0.08
    assert metrics["shard_hands"] == [1]
    assert len(metrics["seat_metrics"]) == 2

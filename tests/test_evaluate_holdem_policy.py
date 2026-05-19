import pytest


pytest.importorskip("treys")

from alphapoker.evaluate_holdem_policy import build_parser, run  # noqa: E402


def test_evaluate_holdem_policy_parser_accepts_hybrid_and_both() -> None:
    args = build_parser().parse_args(
        [
            "--policy",
            "hybrid-pot-odds",
            "--opponent-policy",
            "tuned-pot-odds",
            "--model-player",
            "both",
        ]
    )

    assert args.policy == "hybrid-pot-odds"
    assert args.opponent_policy == "tuned-pot-odds"
    assert args.model_player == (0, 1)


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
    assert len(metrics["seat_metrics"]) == 2

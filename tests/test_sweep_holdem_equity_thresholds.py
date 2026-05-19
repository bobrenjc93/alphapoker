import pytest

from alphapoker import sweep_holdem_equity_thresholds
from alphapoker.sweep_holdem_equity_thresholds import (
    build_parser,
    normalize_model_players,
    parse_model_players,
    parse_threshold_configs,
)


def test_parse_threshold_configs() -> None:
    assert parse_threshold_configs("0.1,0.2,0.3;0.4,0.5,0.6") == [
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    ]


def test_parse_threshold_configs_rejects_bad_items() -> None:
    with pytest.raises(ValueError):
        parse_threshold_configs("0.1,0.2")


def test_sweep_parser_accepts_model_player() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--model-player",
            "1",
        ]
    )

    assert args.model_player == (1,)


def test_parse_model_players_accepts_both() -> None:
    assert parse_model_players("both") == (0, 1)
    assert normalize_model_players(1) == (1,)


def test_sweep_aggregates_both_model_players(monkeypatch) -> None:
    calls = []

    def fake_run_evaluation(args):
        calls.append(args.model_player)
        avg_utility_model = 1.0 if args.model_player == 0 else -0.5
        return {
            "checkpoint": str(args.checkpoint),
            "hands": args.hands,
            "model_player": args.model_player,
            "avg_utility_model": avg_utility_model,
            "utility_stdev_model": 2.0,
            "utility_stderr_model": 2.0 / (args.hands**0.5),
            "avg_utility_p0": avg_utility_model,
            "utility_stdev_p0": 2.0,
            "utility_stderr_p0": 2.0 / (args.hands**0.5),
            "avg_actions": 4.0,
            "folds": 2,
            "showdowns": 3,
            "opponent_policy": args.opponent_policy,
            "equity_sims": args.equity_sims,
            "rollout_sims": args.rollout_sims,
            "bet_threshold": args.bet_threshold,
            "raise_threshold": args.raise_threshold,
            "call_threshold": args.call_threshold,
            "seed": args.seed,
        }

    monkeypatch.setattr(sweep_holdem_equity_thresholds, "run_evaluation", fake_run_evaluation)
    payload = sweep_holdem_equity_thresholds.run(
        build_parser().parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--hands",
                "10",
                "--model-player",
                "both",
                "--configs",
                "0.1,0.2,0.3",
            ]
        )
    )

    assert calls == [0, 1]
    assert payload["model_player"] == "both"
    assert payload["best"]["avg_utility_model"] == pytest.approx(0.25)
    assert payload["best"]["hands"] == 20

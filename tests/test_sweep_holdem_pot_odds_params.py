import pytest


pytest.importorskip("treys")

from alphapoker import sweep_holdem_pot_odds_params  # noqa: E402
from alphapoker.sweep_holdem_pot_odds_params import (  # noqa: E402
    build_parser,
    parse_param_configs,
    run,
)


def test_parse_pot_odds_param_configs() -> None:
    assert parse_param_configs("0.5,0.7,-0.05;0.6,0.8,0.02") == [
        (0.5, 0.7, -0.05),
        (0.6, 0.8, 0.02),
    ]
    with pytest.raises(ValueError, match="bet,raise,call_margin"):
        parse_param_configs("0.5,0.7")


def test_pot_odds_param_sweep_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "3",
            "--equity-sims",
            "2",
            "--model-player",
            "both",
            "--configs",
            "0.5,0.7,0.0",
        ]
    )
    metrics = run(args)

    assert metrics["model_player"] == "both"
    assert metrics["jobs"] == 1
    assert metrics["best"]["hands"] == 4
    assert metrics["best"]["bet_threshold"] == 0.5


def test_hybrid_pot_odds_param_sweep_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "4",
            "--equity-sims",
            "2",
            "--policy-family",
            "hybrid-pot-odds",
            "--opponent-policy",
            "tuned-pot-odds",
            "--model-player",
            "both",
            "--configs",
            "0.54,0.76,0.05",
        ]
    )
    metrics = run(args)

    assert metrics["policy_family"] == "hybrid-pot-odds"
    assert metrics["best"]["policy_family"] == "hybrid-pot-odds"
    assert metrics["best"]["opponent_policy"] == "tuned-pot-odds"


def test_pot_odds_param_sweep_parser_accepts_jobs() -> None:
    args = build_parser().parse_args(["--jobs", "4", "--policy-family", "hybrid-pot-odds"])

    assert args.jobs == 4
    assert args.policy_family == "hybrid-pot-odds"


def test_pot_odds_param_sweep_reuses_deals_across_seats(monkeypatch) -> None:
    seen: list[tuple[int, int]] = []

    def fake_evaluate_policy_match(**kwargs):
        seen.append((kwargs["seed"], kwargs["model_player"]))
        return {
            "hands": kwargs["hands"],
            "model_player": kwargs["model_player"],
            "avg_utility_model": 0.0,
            "utility_stdev_model": 0.0,
            "utility_stderr_model": 0.0,
            "avg_utility_p0": 0.0,
            "utility_stdev_p0": 0.0,
            "utility_stderr_p0": 0.0,
            "avg_actions": 0.0,
            "folds": 0,
            "showdowns": 0,
            "seed": kwargs["seed"],
        }

    monkeypatch.setattr(
        sweep_holdem_pot_odds_params,
        "evaluate_policy_match",
        fake_evaluate_policy_match,
    )

    sweep_holdem_pot_odds_params.evaluate_param_config(
        2,
        (0.5, 0.7, 0.0),
        hands=1,
        seed=17,
        policy_family="pot-odds",
        opponent_policy="random",
        equity_sims=2,
        rollout_sims=None,
        model_players=(0, 1),
    )

    assert seen == [(200_023, 0), (200_023, 1)]

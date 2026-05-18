import pytest


pytest.importorskip("treys")

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
    assert metrics["best"]["hands"] == 4
    assert metrics["best"]["bet_threshold"] == 0.5

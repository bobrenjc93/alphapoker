import pytest

from alphapoker.sweep_holdem_equity_thresholds import parse_threshold_configs


def test_parse_threshold_configs() -> None:
    assert parse_threshold_configs("0.1,0.2,0.3;0.4,0.5,0.6") == [
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    ]


def test_parse_threshold_configs_rejects_bad_items() -> None:
    with pytest.raises(ValueError):
        parse_threshold_configs("0.1,0.2")

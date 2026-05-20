import pytest


pytest.importorskip("treys")

from alphapoker.holdem_self_play import build_parser, run  # noqa: E402


def test_holdem_self_play_smoke() -> None:
    args = build_parser().parse_args(["--hands", "5", "--seed", "3"])
    metrics = run(args)

    assert metrics["hands"] == 5
    assert metrics["showdowns"] + metrics["folds"] == 5
    assert metrics["avg_actions"] > 0


def test_holdem_self_play_equity_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "3",
            "--seed",
            "4",
            "--player0-policy",
            "equity",
            "--player1-policy",
            "random",
            "--equity-sims",
            "8",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 3
    assert metrics["player0_policy"] == "equity"


def test_holdem_self_play_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "3",
            "--seed",
            "5",
            "--player0-policy",
            "pot-odds",
            "--player1-policy",
            "random",
            "--equity-sims",
            "8",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 3
    assert metrics["player0_policy"] == "pot-odds"


def test_holdem_self_play_rollout_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "6",
            "--player0-policy",
            "rollout-pot-odds",
            "--player1-policy",
            "pot-odds",
            "--equity-sims",
            "2",
            "--rollout-sims",
            "2",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 2
    assert metrics["player0_policy"] == "rollout-pot-odds"
    assert metrics["rollout_sims"] == 2


def test_holdem_self_play_cached_rollout_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "11",
            "--player0-policy",
            "cached-rollout-pot-odds",
            "--player1-policy",
            "cached-pot-odds",
            "--equity-sims",
            "2",
            "--rollout-sims",
            "2",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 2
    assert metrics["player0_policy"] == "cached-rollout-pot-odds"
    assert metrics["rollout_sims"] == 2


def test_holdem_self_play_cached_tuned_rollout_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "12",
            "--player0-policy",
            "cached-tuned-rollout-pot-odds",
            "--player1-policy",
            "cached-tuned-pot-odds",
            "--equity-sims",
            "2",
            "--rollout-sims",
            "2",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 2
    assert metrics["player0_policy"] == "cached-tuned-rollout-pot-odds"
    assert metrics["rollout_sims"] == 2


def test_holdem_self_play_tuned_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "3",
            "--seed",
            "7",
            "--player0-policy",
            "tuned-pot-odds",
            "--player1-policy",
            "pot-odds",
            "--equity-sims",
            "4",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 3
    assert metrics["player0_policy"] == "tuned-pot-odds"


def test_holdem_self_play_cached_tuned_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "3",
            "--seed",
            "10",
            "--player0-policy",
            "cached-tuned-pot-odds",
            "--player1-policy",
            "cached-pot-odds",
            "--equity-sims",
            "4",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 3
    assert metrics["player0_policy"] == "cached-tuned-pot-odds"


def test_holdem_self_play_river_exact_tuned_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "3",
            "--seed",
            "12",
            "--player0-policy",
            "river-exact-tuned-pot-odds",
            "--player1-policy",
            "tuned-pot-odds",
            "--equity-sims",
            "4",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 3
    assert metrics["player0_policy"] == "river-exact-tuned-pot-odds"


def test_holdem_self_play_turn_river_exact_tuned_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "13",
            "--player0-policy",
            "turn-river-exact-tuned-pot-odds",
            "--player1-policy",
            "tuned-pot-odds",
            "--equity-sims",
            "4",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 2
    assert metrics["player0_policy"] == "turn-river-exact-tuned-pot-odds"


def test_holdem_self_play_tight_turn_river_exact_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "15",
            "--player0-policy",
            "tight-turn-river-exact-pot-odds",
            "--player1-policy",
            "tuned-pot-odds",
            "--equity-sims",
            "4",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 2
    assert metrics["player0_policy"] == "tight-turn-river-exact-pot-odds"


def test_holdem_self_play_balanced_turn_river_exact_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "16",
            "--player0-policy",
            "balanced-turn-river-exact-pot-odds",
            "--player1-policy",
            "tuned-pot-odds",
            "--equity-sims",
            "4",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 2
    assert metrics["player0_policy"] == "balanced-turn-river-exact-pot-odds"


def test_holdem_self_play_tight_range_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "17",
            "--player0-policy",
            "tight-range-pot-odds",
            "--player1-policy",
            "tight-turn-river-exact-pot-odds",
            "--equity-sims",
            "2",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 2
    assert metrics["player0_policy"] == "tight-range-pot-odds"


def test_holdem_self_play_tight_rollout_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "17",
            "--player0-policy",
            "tight-rollout-pot-odds",
            "--player1-policy",
            "tuned-pot-odds",
            "--equity-sims",
            "2",
            "--rollout-sims",
            "2",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 2
    assert metrics["player0_policy"] == "tight-rollout-pot-odds"


def test_holdem_self_play_tight_range_safe_rollout_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "1",
            "--seed",
            "23",
            "--player0-policy",
            "tight-range-safe-rollout-pot-odds",
            "--player1-policy",
            "tight-turn-river-exact-pot-odds",
            "--equity-sims",
            "1",
            "--rollout-sims",
            "1",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 1
    assert metrics["player0_policy"] == "tight-range-safe-rollout-pot-odds"


def test_holdem_self_play_tight_range_default_safe_rollout_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "1",
            "--seed",
            "24",
            "--player0-policy",
            "tight-range-default-safe-rollout-pot-odds",
            "--player1-policy",
            "tight-turn-river-exact-pot-odds",
            "--equity-sims",
            "2",
            "--rollout-sims",
            "1",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 1
    assert metrics["player0_policy"] == "tight-range-default-safe-rollout-pot-odds"


def test_holdem_self_play_balanced_rollout_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "18",
            "--player0-policy",
            "balanced-rollout-pot-odds",
            "--player1-policy",
            "tuned-pot-odds",
            "--equity-sims",
            "2",
            "--rollout-sims",
            "2",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 2
    assert metrics["player0_policy"] == "balanced-rollout-pot-odds"


def test_holdem_self_play_safe_rollout_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "19",
            "--player0-policy",
            "tight-safe-rollout-pot-odds",
            "--player1-policy",
            "tuned-pot-odds",
            "--equity-sims",
            "2",
            "--rollout-sims",
            "2",
            "--rollout-margin",
            "1.5",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 2
    assert metrics["player0_policy"] == "tight-safe-rollout-pot-odds"
    assert metrics["rollout_margin"] == 1.5


def test_holdem_self_play_hybrid_pot_odds_policy_smoke() -> None:
    args = build_parser().parse_args(
        [
            "--hands",
            "3",
            "--seed",
            "9",
            "--player0-policy",
            "hybrid-pot-odds",
            "--player1-policy",
            "pot-odds",
            "--equity-sims",
            "4",
        ]
    )
    metrics = run(args)

    assert metrics["hands"] == 3
    assert metrics["player0_policy"] == "hybrid-pot-odds"


def test_holdem_self_play_writes_metrics(tmp_path) -> None:
    out = tmp_path / "metrics.json"
    args = build_parser().parse_args(
        [
            "--hands",
            "2",
            "--seed",
            "8",
            "--out",
            str(out),
        ]
    )
    metrics = run(args)

    assert out.exists()
    assert metrics["utility_stderr_p0"] >= 0.0

"""Sweep fixed-limit Hold'em pot-odds policy parameters."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from alphapoker.holdem import (
    cached_pot_odds_equity_policy,
    hybrid_pot_odds_equity_policy,
    opponent_range_pot_odds_equity_policy,
    pot_odds_equity_policy,
    river_exact_pot_odds_equity_policy,
    turn_river_exact_pot_odds_equity_policy,
)
from alphapoker.holdem_evaluation import evaluate_policy_match, evaluate_policy_match_paired_seats
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES, make_policy
from alphapoker.train import write_json
from alphapoker.train_holdem_policy_gradient import (
    model_player_label,
    normalize_model_players,
    parse_model_players,
)

POT_ODDS_POLICY_FAMILIES = (
    "pot-odds",
    "cached-pot-odds",
    "river-exact-pot-odds",
    "turn-river-exact-pot-odds",
    "tight-range-pot-odds",
    "hybrid-pot-odds",
)


def parse_param_configs(configs: str) -> list[tuple[float, float, float]]:
    parsed = []
    for item in configs.split(";"):
        if not item.strip():
            continue
        parts = [float(part) for part in item.split(",")]
        if len(parts) != 3:
            raise ValueError("Each config must be bet,raise,call_margin")
        parsed.append((parts[0], parts[1], parts[2]))
    if not parsed:
        raise ValueError("At least one parameter config is required")
    return parsed


def _weighted_mean(metrics: list[dict[str, Any]], key: str) -> float:
    total_hands = sum(int(item["hands"]) for item in metrics)
    return (
        sum(float(item[key]) * int(item["hands"]) for item in metrics) / total_hands
        if total_hands
        else 0.0
    )


def _pooled_stdev(metrics: list[dict[str, Any]], mean_key: str, stdev_key: str) -> float:
    total_hands = sum(int(item["hands"]) for item in metrics)
    if total_hands <= 1:
        return 0.0
    mean = _weighted_mean(metrics, mean_key)
    sum_squares = 0.0
    for item in metrics:
        hands = int(item["hands"])
        stdev = float(item[stdev_key])
        item_mean = float(item[mean_key])
        sum_squares += max(0, hands - 1) * stdev * stdev
        sum_squares += hands * (item_mean - mean) * (item_mean - mean)
    return math.sqrt(sum_squares / (total_hands - 1))


def aggregate_seat_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if len(metrics) == 1:
        return metrics[0]
    total_hands = sum(int(item["hands"]) for item in metrics)
    model_stdev = _pooled_stdev(metrics, "avg_utility_model", "utility_stdev_model")
    p0_stdev = _pooled_stdev(metrics, "avg_utility_p0", "utility_stdev_p0")
    first = metrics[0]
    return {
        "hands": total_hands,
        "hands_per_model_player": first["hands"],
        "model_player": "both",
        "avg_utility_model": _weighted_mean(metrics, "avg_utility_model"),
        "utility_stdev_model": model_stdev,
        "utility_stderr_model": model_stdev / (total_hands**0.5) if total_hands else 0.0,
        "avg_utility_p0": _weighted_mean(metrics, "avg_utility_p0"),
        "utility_stdev_p0": p0_stdev,
        "utility_stderr_p0": p0_stdev / (total_hands**0.5) if total_hands else 0.0,
        "avg_actions": _weighted_mean(metrics, "avg_actions"),
        "folds": sum(int(item["folds"]) for item in metrics),
        "showdowns": sum(int(item["showdowns"]) for item in metrics),
        "opponent_policy": first["opponent_policy"],
        "equity_sims": first["equity_sims"],
        "opponent_equity_sims": first["opponent_equity_sims"],
        "opponent_rollout_sims": first["opponent_rollout_sims"],
        "rollout_margin": first["rollout_margin"],
        "policy_family": first["policy_family"],
        "bet_threshold": first["bet_threshold"],
        "raise_threshold": first["raise_threshold"],
        "call_margin": first["call_margin"],
        "seed": first["seed"],
        "seat_metrics": metrics,
    }


def make_candidate_policy(
    policy_family: str,
    rng: random.Random,
    *,
    equity_sims: int,
    bet_threshold: float,
    raise_threshold: float,
    call_margin: float,
):
    if policy_family == "pot-odds":
        return pot_odds_equity_policy(
            rng,
            simulations=equity_sims,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
        )
    if policy_family == "cached-pot-odds":
        return cached_pot_odds_equity_policy(
            simulations=equity_sims,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
        )
    if policy_family == "river-exact-pot-odds":
        return river_exact_pot_odds_equity_policy(
            simulations=equity_sims,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
        )
    if policy_family == "turn-river-exact-pot-odds":
        return turn_river_exact_pot_odds_equity_policy(
            simulations=equity_sims,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
        )
    if policy_family == "tight-range-pot-odds":
        def tight_baseline(_: random.Random):
            return turn_river_exact_pot_odds_equity_policy(
                simulations=equity_sims,
                bet_threshold=0.62,
                raise_threshold=0.84,
                call_margin=0.08,
            )

        return opponent_range_pot_odds_equity_policy(
            rng,
            simulations=equity_sims,
            opponent_policy_factory=tight_baseline,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
            cache_policy_matches=True,
        )
    if policy_family == "hybrid-pot-odds":
        return hybrid_pot_odds_equity_policy(
            rng,
            simulations=equity_sims,
            bet_threshold=bet_threshold,
            raise_threshold=raise_threshold,
            call_margin=call_margin,
        )
    raise ValueError(f"Unknown policy family: {policy_family}")


def evaluate_param_config(
    index: int,
    config: tuple[float, float, float],
    *,
    hands: int,
    seed: int,
    policy_family: str,
    opponent_policy: str,
    equity_sims: int,
    rollout_sims: int | None,
    model_players: tuple[int, ...],
    opponent_equity_sims: int | None = None,
    opponent_rollout_sims: int | None = None,
    rollout_margin: float = 1.0,
    paired_seats: bool = False,
) -> dict[str, Any]:
    bet_threshold, raise_threshold, call_margin = config
    resolved_opponent_equity_sims = (
        equity_sims if opponent_equity_sims is None else opponent_equity_sims
    )
    resolved_opponent_rollout_sims = (
        rollout_sims if opponent_rollout_sims is None else opponent_rollout_sims
    )
    if paired_seats:
        if model_players != (0, 1):
            raise ValueError("paired_seats requires model_players=(0, 1)")
        eval_seed = seed + index * 100_003
        result = {
            **evaluate_policy_match_paired_seats(
                model_policies=tuple(
                    make_candidate_policy(
                        policy_family,
                        random.Random(seed + index * 10_003 + model_player),
                        equity_sims=equity_sims,
                        bet_threshold=bet_threshold,
                        raise_threshold=raise_threshold,
                        call_margin=call_margin,
                    )
                    for model_player in (0, 1)
                ),
                opponent_policies=tuple(
                    make_policy(
                        opponent_policy,
                        random.Random(seed + 1_000_003 + index * 10_003 + model_player),
                        resolved_opponent_equity_sims,
                        resolved_opponent_rollout_sims,
                        rollout_margin,
                    )
                    for model_player in (0, 1)
                ),
                hands=hands,
                seed=eval_seed,
            ),
            "opponent_policy": opponent_policy,
            "equity_sims": equity_sims,
            "opponent_equity_sims": resolved_opponent_equity_sims,
            "opponent_rollout_sims": resolved_opponent_rollout_sims,
            "rollout_margin": rollout_margin,
            "policy_family": policy_family,
            "bet_threshold": bet_threshold,
            "raise_threshold": raise_threshold,
            "call_margin": call_margin,
            "paired_seats": True,
        }
        result["config_index"] = index
        return result

    seat_metrics = []
    for model_player in model_players:
        model_rng = random.Random(seed + index * 10_003 + model_player)
        opponent_rng = random.Random(seed + 1_000_003 + index * 10_003 + model_player)
        metrics = {
            **evaluate_policy_match(
                model_policy=make_candidate_policy(
                    policy_family,
                    model_rng,
                    equity_sims=equity_sims,
                    bet_threshold=bet_threshold,
                    raise_threshold=raise_threshold,
                    call_margin=call_margin,
                ),
                opponent_policy=make_policy(
                    opponent_policy,
                    opponent_rng,
                    resolved_opponent_equity_sims,
                    resolved_opponent_rollout_sims,
                    rollout_margin,
                ),
                hands=hands,
                seed=seed + index * 100_003,
                model_player=model_player,
            ),
            "opponent_policy": opponent_policy,
            "equity_sims": equity_sims,
            "opponent_equity_sims": resolved_opponent_equity_sims,
            "opponent_rollout_sims": resolved_opponent_rollout_sims,
            "rollout_margin": rollout_margin,
            "policy_family": policy_family,
            "bet_threshold": bet_threshold,
            "raise_threshold": raise_threshold,
            "call_margin": call_margin,
        }
        seat_metrics.append(metrics)
    result = aggregate_seat_metrics(seat_metrics)
    result["config_index"] = index
    result["paired_seats"] = False
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_players = normalize_model_players(args.model_player)
    configs = parse_param_configs(args.configs)
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    opponent_equity_sims = (
        args.equity_sims if args.opponent_equity_sims is None else args.opponent_equity_sims
    )
    opponent_rollout_sims = (
        args.rollout_sims if args.opponent_rollout_sims is None else args.opponent_rollout_sims
    )

    results = []
    if args.jobs == 1:
        for index, config in enumerate(configs):
            result = evaluate_param_config(
                index,
                config,
                hands=args.hands,
                seed=args.seed,
                policy_family=args.policy_family,
                opponent_policy=args.opponent_policy,
                equity_sims=args.equity_sims,
                rollout_sims=args.rollout_sims,
                opponent_equity_sims=opponent_equity_sims,
                opponent_rollout_sims=opponent_rollout_sims,
                rollout_margin=args.rollout_margin,
                model_players=model_players,
                paired_seats=args.paired_seats,
            )
            results.append(result)
            report_progress(args.progress, result)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(
                    evaluate_param_config,
                    index,
                    config,
                    hands=args.hands,
                    seed=args.seed,
                    policy_family=args.policy_family,
                    opponent_policy=args.opponent_policy,
                    equity_sims=args.equity_sims,
                    rollout_sims=args.rollout_sims,
                    opponent_equity_sims=opponent_equity_sims,
                    opponent_rollout_sims=opponent_rollout_sims,
                    rollout_margin=args.rollout_margin,
                    model_players=model_players,
                    paired_seats=args.paired_seats,
                )
                for index, config in enumerate(configs)
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                report_progress(args.progress, result)
        results.sort(key=lambda item: item["config_index"])

    best = max(results, key=lambda item: item["avg_utility_model"])
    payload = {
        "hands": args.hands,
        "seed": args.seed,
        "policy_family": args.policy_family,
        "opponent_policy": args.opponent_policy,
        "equity_sims": args.equity_sims,
        "opponent_equity_sims": opponent_equity_sims,
        "opponent_rollout_sims": opponent_rollout_sims,
        "rollout_margin": args.rollout_margin,
        "rollout_sims": args.rollout_sims,
        "jobs": args.jobs,
        "model_player": model_player_label(model_players),
        "paired_seats": args.paired_seats,
        "best": best,
        "results": results,
    }
    if args.out is not None:
        write_json(Path(args.out), payload)
    return payload


def report_progress(enabled: bool, result: dict[str, Any]) -> None:
    if not enabled:
        return
    print(
        f"config {result['config_index']}: "
        f"avg_utility_model={result['avg_utility_model']:.3f} "
        f"stderr={result['utility_stderr_model']:.3f}",
        file=sys.stderr,
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-family", choices=POT_ODDS_POLICY_FAMILIES, default="pot-odds")
    parser.add_argument("--opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--opponent-equity-sims", type=int)
    parser.add_argument("--opponent-rollout-sims", type=int)
    parser.add_argument("--rollout-margin", type=float, default=1.0)
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--paired-seats", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--configs",
        default="0.50,0.65,-0.05;0.58,0.72,0.0;0.66,0.82,0.05",
        help="Semicolon-separated bet,raise,call_margin triples.",
    )
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

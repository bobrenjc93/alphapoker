"""Sweep Hold'em MCCFR fallback min-strategy weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from alphapoker.evaluate_holdem_mccfr import evaluate_checkpoint
from alphapoker.evaluate_holdem_model import parse_model_players
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES
from alphapoker.train import write_json
from alphapoker.train_holdem_policy_gradient import model_player_label, normalize_model_players


def parse_min_strategy_weights(value: str) -> tuple[float, ...]:
    try:
        weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("min-strategy weights must be numeric") from error
    if not weights:
        raise argparse.ArgumentTypeError("at least one min-strategy weight is required")
    if any(weight < 0.0 for weight in weights):
        raise argparse.ArgumentTypeError("min-strategy weights must be non-negative")
    return weights


def evaluate_weight(
    *,
    checkpoint: Path,
    hands: int,
    seed: int,
    opponent_policy: str,
    fallback_policy: str,
    min_strategy_weight: float,
    equity_sims: int,
    rollout_sims: int | None,
    model_players: tuple[int, ...],
    eval_jobs: int,
    config_index: int,
) -> dict[str, Any]:
    metrics = evaluate_checkpoint(
        checkpoint=checkpoint,
        hands=hands,
        seed=seed,
        opponent_policy=opponent_policy,
        fallback_policy=fallback_policy,
        min_strategy_weight=min_strategy_weight,
        equity_sims=equity_sims,
        rollout_sims=rollout_sims,
        model_players=model_players,
        jobs=eval_jobs,
    )
    metrics["config_index"] = config_index
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_players = normalize_model_players(args.model_player)
    weights = (
        args.weights
        if isinstance(args.weights, tuple)
        else parse_min_strategy_weights(args.weights)
    )
    results = []
    for index, weight in enumerate(weights):
        result = evaluate_weight(
            checkpoint=args.checkpoint,
            hands=args.hands,
            seed=args.seed,
            opponent_policy=args.opponent_policy,
            fallback_policy=args.fallback_policy,
            min_strategy_weight=weight,
            equity_sims=args.equity_sims,
            rollout_sims=args.rollout_sims,
            model_players=model_players,
            eval_jobs=args.eval_jobs,
            config_index=index,
        )
        results.append(result)
        report_progress(args.progress, result)

    best = max(results, key=lambda item: item["avg_utility_model"])
    payload = {
        "checkpoint": str(args.checkpoint),
        "hands": args.hands,
        "seed": args.seed,
        "opponent_policy": args.opponent_policy,
        "fallback_policy": args.fallback_policy,
        "equity_sims": args.equity_sims,
        "rollout_sims": args.rollout_sims,
        "model_player": model_player_label(model_players),
        "eval_jobs": args.eval_jobs,
        "weights": list(weights),
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
        f"weight {result['min_strategy_weight']}: "
        f"avg_utility_model={result['avg_utility_model']:.3f} "
        f"stderr={result['utility_stderr_model']:.3f}",
        file=sys.stderr,
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--fallback-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument(
        "--weights",
        type=parse_min_strategy_weights,
        default=(0.0, 100.0, 500.0, 1000.0),
    )
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument("--eval-jobs", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

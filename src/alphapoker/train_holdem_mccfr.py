"""Train sampled abstract CFR for fixed-limit Hold'em."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alphapoker.evaluate_holdem_mccfr import evaluate_checkpoint
from alphapoker.evaluate_holdem_model import parse_model_players
from alphapoker.holdem_mccfr import (
    HoldemAbstractionCFRTrainer,
)
from alphapoker.holdem_self_play import HOLDEM_SELF_PLAY_POLICIES
from alphapoker.train import write_json


def normalize_model_players(value: int | str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    return parse_model_players(str(value))


def run(args: argparse.Namespace) -> dict[str, Any]:
    trainer = HoldemAbstractionCFRTrainer(
        seed=args.seed,
        cfr_plus=not args.vanilla_cfr,
        linear_averaging=not args.uniform_averaging,
        max_bets_per_round=args.max_bets_per_round,
        traversal=args.traversal,
        abstraction=args.abstraction,
    )
    result = trainer.train(args.iterations)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "holdem_mccfr.json"
    trainer.save_checkpoint(checkpoint)

    metrics: dict[str, Any] = {
        "iterations": result.iterations,
        "infosets": result.infosets,
        "sampled_game_value_p0": result.sampled_game_value_p0,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "algorithm": "cfr" if args.vanilla_cfr else "cfr_plus",
        "average_weighting": "uniform" if args.uniform_averaging else "linear",
        "max_bets_per_round": args.max_bets_per_round,
        "traversal": args.traversal,
        "abstraction": args.abstraction,
        "min_strategy_weight": args.min_strategy_weight,
        "checkpoint_saved": not args.discard_checkpoint,
    }

    if args.eval_hands > 0:
        eval_metrics = evaluate_checkpoint(
            checkpoint=checkpoint,
            hands=args.eval_hands,
            seed=args.seed + 10_000_019,
            opponent_policy=args.opponent_policy,
            fallback_policy=args.fallback_policy,
            min_strategy_weight=args.min_strategy_weight,
            equity_sims=args.equity_sims,
            rollout_sims=args.rollout_sims,
            model_players=normalize_model_players(args.model_player),
            jobs=args.eval_jobs,
        )
        metrics["evaluation"] = eval_metrics

    if args.discard_checkpoint:
        checkpoint.unlink(missing_ok=True)

    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vanilla-cfr", action="store_true")
    parser.add_argument("--uniform-averaging", action="store_true")
    parser.add_argument("--max-bets-per-round", type=int, default=4)
    parser.add_argument("--traversal", choices=["external", "full"], default="external")
    parser.add_argument("--abstraction", choices=["fine", "medium", "coarse"], default="coarse")
    parser.add_argument("--eval-hands", type=int, default=0)
    parser.add_argument("--opponent-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--fallback-policy", choices=HOLDEM_SELF_PLAY_POLICIES, default="pot-odds")
    parser.add_argument("--min-strategy-weight", type=float, default=0.0)
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--rollout-sims", type=int)
    parser.add_argument("--model-player", type=parse_model_players, default=(0,))
    parser.add_argument("--eval-jobs", type=int, default=1)
    parser.add_argument("--discard-checkpoint", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

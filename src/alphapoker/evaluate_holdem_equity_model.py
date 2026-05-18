"""Evaluate a Hold'em equity-prediction checkpoint as a threshold policy."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

from alphapoker.holdem import (
    BET,
    CALL,
    CHECK,
    FOLD,
    RAISE,
    FixedLimitHoldemState,
    HoldemPolicy,
    deal_fixed_limit_holdem,
    equity_threshold_policy,
    play_fixed_limit_holdem_hand,
    random_holdem_policy,
)
from alphapoker.holdem_features import encode_holdem_state
from alphapoker.train import write_json


def equity_model_policy_from_checkpoint(
    checkpoint_path: Path,
    *,
    bet_threshold: float = 0.58,
    raise_threshold: float = 0.72,
    call_threshold: float = 0.36,
) -> HoldemPolicy:
    import torch

    from alphapoker.holdem_model import HoldemEquityNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = HoldemEquityNet(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def select_action(state: FixedLimitHoldemState) -> str:
        features = torch.tensor([encode_holdem_state(state)], dtype=torch.float32)
        with torch.no_grad():
            equity = float(torch.sigmoid(model(features)).item())
        legal = state.legal_actions()
        if state.outstanding_call_amount() > 0:
            if RAISE in legal and equity >= raise_threshold:
                return RAISE
            if equity >= call_threshold:
                return CALL
            return FOLD
        if BET in legal and equity >= bet_threshold:
            return BET
        return CHECK

    return select_action


def make_opponent_policy(name: str, rng: random.Random, equity_sims: int) -> HoldemPolicy:
    if name == "random":
        return random_holdem_policy(rng)
    if name == "equity":
        return equity_threshold_policy(rng, simulations=equity_sims)
    raise ValueError(f"Unknown opponent policy: {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    deal_rng = random.Random(args.seed)
    opponent_rng = random.Random(args.seed + 1)
    policies = (
        equity_model_policy_from_checkpoint(args.checkpoint),
        make_opponent_policy(args.opponent_policy, opponent_rng, args.equity_sims),
    )

    utilities: list[float] = []
    total_actions = 0
    folds = 0
    showdowns = 0
    for _ in range(args.hands):
        terminal, actions = play_fixed_limit_holdem_hand(deal_fixed_limit_holdem(deal_rng), policies)
        utilities.append(terminal.utility(0))
        total_actions += len(actions)
        if terminal.showdown:
            showdowns += 1
        else:
            folds += 1

    utility_stdev = statistics.stdev(utilities) if len(utilities) > 1 else 0.0
    metrics = {
        "checkpoint": str(args.checkpoint),
        "hands": args.hands,
        "avg_utility_p0": sum(utilities) / args.hands if args.hands else 0.0,
        "utility_stdev_p0": utility_stdev,
        "utility_stderr_p0": utility_stdev / (args.hands**0.5) if args.hands else 0.0,
        "avg_actions": total_actions / args.hands if args.hands else 0.0,
        "folds": folds,
        "showdowns": showdowns,
        "opponent_policy": args.opponent_policy,
        "equity_sims": args.equity_sims,
        "seed": args.seed,
    }
    if args.out is not None:
        write_json(args.out, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", choices=["random", "equity"], default="random")
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


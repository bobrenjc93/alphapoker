"""Evaluate a Hold'em equity-prediction checkpoint as a threshold policy."""

from __future__ import annotations

import argparse
import json
import random
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
    equity_threshold_policy,
    pot_odds_equity_policy,
    random_holdem_policy,
)
from alphapoker.holdem_evaluation import evaluate_policy_match
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
    if name == "pot-odds":
        return pot_odds_equity_policy(rng, simulations=equity_sims)
    raise ValueError(f"Unknown opponent policy: {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    opponent_rng = random.Random(args.seed + 1)
    metrics = {
        "checkpoint": str(args.checkpoint),
        **evaluate_policy_match(
            model_policy=equity_model_policy_from_checkpoint(
                args.checkpoint,
                bet_threshold=args.bet_threshold,
                raise_threshold=args.raise_threshold,
                call_threshold=args.call_threshold,
            ),
            opponent_policy=make_opponent_policy(args.opponent_policy, opponent_rng, args.equity_sims),
            hands=args.hands,
            seed=args.seed,
            model_player=args.model_player,
        ),
        "opponent_policy": args.opponent_policy,
        "equity_sims": args.equity_sims,
        "bet_threshold": args.bet_threshold,
        "raise_threshold": args.raise_threshold,
        "call_threshold": args.call_threshold,
    }
    if args.out is not None:
        write_json(args.out, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", choices=["random", "equity", "pot-odds"], default="random")
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--model-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--bet-threshold", type=float, default=0.58)
    parser.add_argument("--raise-threshold", type=float, default=0.72)
    parser.add_argument("--call-threshold", type=float, default=0.36)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

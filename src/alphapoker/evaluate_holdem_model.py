"""Evaluate a trained fixed-limit Hold'em policy checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from alphapoker.holdem import (
    FixedLimitHoldemState,
    HoldemPolicy,
    equity_threshold_policy,
    random_holdem_policy,
)
from alphapoker.holdem_evaluation import evaluate_policy_match
from alphapoker.holdem_features import encode_holdem_state, holdem_legal_action_mask
from alphapoker.train import write_json


def model_policy_from_checkpoint(checkpoint_path: Path) -> HoldemPolicy:
    import torch

    from alphapoker.holdem_features import HOLDEM_CANONICAL_ACTIONS
    from alphapoker.holdem_model import HoldemPolicyNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = HoldemPolicyNet(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def select_action(state: FixedLimitHoldemState) -> str:
        features = torch.tensor([encode_holdem_state(state)], dtype=torch.float32)
        mask = torch.tensor(holdem_legal_action_mask(state), dtype=torch.bool)
        with torch.no_grad():
            logits = model(features).squeeze(0)
            logits = logits.masked_fill(~mask, -1e9)
            action_index = int(logits.argmax().item())
        return HOLDEM_CANONICAL_ACTIONS[action_index]

    return select_action


def make_opponent_policy(name: str, rng: random.Random, equity_sims: int) -> HoldemPolicy:
    if name == "random":
        return random_holdem_policy(rng)
    if name == "equity":
        return equity_threshold_policy(rng, simulations=equity_sims)
    raise ValueError(f"Unknown opponent policy: {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    opponent_rng = random.Random(args.seed + 1)
    metrics = {
        "checkpoint": str(args.checkpoint),
        **evaluate_policy_match(
            model_policy=model_policy_from_checkpoint(args.checkpoint),
            opponent_policy=make_opponent_policy(args.opponent_policy, opponent_rng, args.equity_sims),
            hands=args.hands,
            seed=args.seed,
            model_player=args.model_player,
        ),
        "opponent_policy": args.opponent_policy,
        "equity_sims": args.equity_sims,
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
    parser.add_argument("--model-player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

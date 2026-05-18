"""Evaluate a Hold'em policy checkpoint against held-out expert labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from alphapoker.holdem_dataset import generate_equity_policy_examples
from alphapoker.holdem_features import HOLDEM_CANONICAL_ACTIONS, adapt_holdem_features
from alphapoker.train import write_json


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from alphapoker.holdem_model import HoldemPolicyNet

    examples = generate_equity_policy_examples(
        hands=args.hands,
        seed=args.seed,
        equity_sims=args.equity_sims,
        expert_player=args.expert_player,
        expert_policy=args.expert_policy,
        opponent_policy=args.opponent_policy,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    features = torch.tensor(
        [adapt_holdem_features(example.features, input_dim) for example in examples],
        dtype=torch.float32,
    )
    targets = torch.tensor([example.action_index for example in examples], dtype=torch.long)
    masks = torch.tensor([example.legal_mask for example in examples], dtype=torch.bool)

    model = HoldemPolicyNet(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(features).masked_fill(~masks, -1e9)
        loss = F.cross_entropy(logits, targets)
        predictions = logits.argmax(dim=1)
        accuracy = float((predictions == targets).float().mean().cpu())

    target_action_counts = {
        action: int((targets == index).sum().item())
        for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
    }
    predicted_action_counts = {
        action: int((predictions == index).sum().item())
        for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
    }
    metrics: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "hands": args.hands,
        "examples": len(examples),
        "equity_sims": args.equity_sims,
        "expert_player": args.expert_player,
        "expert_policy": args.expert_policy,
        "opponent_policy": args.opponent_policy,
        "loss": float(loss.cpu()),
        "accuracy": accuracy,
        "target_action_counts": target_action_counts,
        "predicted_action_counts": predicted_action_counts,
        "seed": args.seed,
    }
    if args.out is not None:
        write_json(args.out, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--expert-player", type=int, choices=[0, 1])
    parser.add_argument("--expert-policy", choices=["equity", "pot-odds"], default="equity")
    parser.add_argument("--opponent-policy", choices=["equity", "pot-odds", "random"], default="equity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

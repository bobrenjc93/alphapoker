"""Distill the fixed-limit Hold'em equity policy into a neural policy."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from alphapoker.holdem_dataset import generate_equity_policy_examples
from alphapoker.holdem_features import HOLDEM_CANONICAL_ACTIONS
from alphapoker.train import write_json


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from alphapoker.holdem_model import HoldemPolicyNet

    behavior_policy = None
    if args.behavior_checkpoint is not None:
        from alphapoker.evaluate_holdem_model import model_policy_from_checkpoint

        behavior_policy = model_policy_from_checkpoint(args.behavior_checkpoint)

    examples = generate_equity_policy_examples(
        hands=args.hands,
        seed=args.seed,
        equity_sims=args.equity_sims,
        expert_player=args.expert_player,
        expert_policy=args.expert_policy,
        opponent_policy=args.opponent_policy,
        expert_behavior_policy=behavior_policy,
    )
    features = torch.tensor([example.features for example in examples], dtype=torch.float32)
    targets = torch.tensor([example.action_index for example in examples], dtype=torch.long)
    masks = torch.tensor([example.legal_mask for example in examples], dtype=torch.bool)

    torch.manual_seed(0)
    model = HoldemPolicyNet(input_dim=features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    final_loss = 0.0
    for _ in range(args.epochs):
        logits = model(features)
        masked_logits = logits.masked_fill(~masks, -1e9)
        loss = F.cross_entropy(masked_logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if final_loss < best_loss:
            best_loss = final_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    with torch.no_grad():
        best_logits = model(features).masked_fill(~masks, -1e9)
        predictions = best_logits.argmax(dim=1)
        train_accuracy = float((predictions == targets).float().mean().cpu())
    target_action_counts = {
        action: int((targets == index).sum().item())
        for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
    }
    predicted_action_counts = {
        action: int((predictions == index).sum().item())
        for index, action in enumerate(HOLDEM_CANONICAL_ACTIONS)
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "holdem_policy.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "canonical_actions": list(HOLDEM_CANONICAL_ACTIONS),
            "input_dim": features.shape[1],
        },
        checkpoint,
    )
    metrics: dict[str, Any] = {
        "hands": args.hands,
        "examples": len(examples),
        "equity_sims": args.equity_sims,
        "expert_player": args.expert_player,
        "expert_policy": args.expert_policy,
        "opponent_policy": args.opponent_policy,
        "epochs": args.epochs,
        "lr": args.lr,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "train_accuracy": train_accuracy,
        "target_action_counts": target_action_counts,
        "predicted_action_counts": predicted_action_counts,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
    }
    if args.behavior_checkpoint is not None:
        metrics["behavior_checkpoint"] = str(args.behavior_checkpoint)
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--equity-sims", type=int, default=8)
    parser.add_argument("--expert-player", type=int, choices=[0, 1])
    parser.add_argument("--expert-policy", choices=["equity", "pot-odds"], default="equity")
    parser.add_argument("--opponent-policy", choices=["equity", "pot-odds", "random"], default="equity")
    parser.add_argument("--behavior-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Train a fixed-limit Hold'em equity predictor."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from alphapoker.holdem_dataset import generate_equity_value_examples
from alphapoker.train import write_json


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from alphapoker.holdem_model import HoldemEquityNet

    examples = generate_equity_value_examples(
        hands=args.hands,
        seed=args.seed,
        equity_sims=args.equity_sims,
        player=args.player,
        opponent_policy=args.opponent_policy,
    )
    features = torch.tensor([example.features for example in examples], dtype=torch.float32)
    targets = torch.tensor([example.equity for example in examples], dtype=torch.float32)

    torch.manual_seed(0)
    model = HoldemEquityNet(input_dim=features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    final_loss = 0.0
    for _ in range(args.epochs):
        logits = model(features)
        predictions = torch.sigmoid(logits)
        loss = F.mse_loss(predictions, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if final_loss < best_loss:
            best_loss = final_loss
            best_state = copy.deepcopy(model.state_dict())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "holdem_equity.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": features.shape[1],
        },
        checkpoint,
    )
    metrics: dict[str, Any] = {
        "hands": args.hands,
        "examples": len(examples),
        "equity_sims": args.equity_sims,
        "player": args.player,
        "opponent_policy": args.opponent_policy,
        "epochs": args.epochs,
        "lr": args.lr,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
    }
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--equity-sims", type=int, default=16)
    parser.add_argument("--player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--opponent-policy", choices=["equity", "random"], default="random")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

